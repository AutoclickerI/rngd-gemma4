//! Resident STEP dispatchers share one device image across segment and attention launches.
//!
//! Each family packs its RSS, RSF, RFS and selected DSL attention bodies. The control
//! buffer's offset selects an opcode, and unused operand slots bind padding.
//! Opcodes 32–40 compose Full attention frames and 41–44 combine RSF with Full attention.
//! The carried images and member table are built by `tools/native/prebuilt/rebuild.sh`.
//!
//! A step selects a family when its resident segments match and the family covers its
//! attention paths. The wide family leaves three attention paths on standalone images.
//! Other launches, such as the prologue, head and media, also use standalone images.

use furiosa_opt_std::prelude::*;

use crate::axes::{Hqs, Rq};

#[path = "step_image_table.rs"]
mod table;

#[path = "step_image/persist.rs"]
pub(crate) mod persist;

#[path = "step_image/full_sequence.rs"]
pub(crate) mod full_sequence;

#[path = "step_image/launch_boundary.rs"]
pub(crate) mod launch_boundary;

#[path = "step_image/p999_g12_slots.rs"]
mod p999_g12_slots;
#[path = "step_image/p941_g12_slots.rs"]
mod p941_g12_slots;

type BoxError = Box<dyn std::error::Error>;

/// The path prefix of every step-family registry row.
pub const PATH_PREFIX: &str = "furiosa_opt_gemma4::host::step_image::";

/// The carried step families (`FRCZv1`), installed by `host::registry`.
pub(crate) const PAYLOAD: &[u8] = include_bytes!("../prebuilt/step.frcz");

fn family_members(family: &table::Family) -> impl Iterator<Item = (usize, usize)> + '_ {
    family.ops.iter().copied().enumerate().filter(|&(_, member)| member != table::NO_MEMBER)
}

fn standalone_wide_member(path: &str) -> bool {
    matches!(path,
        "furiosa_opt_gemma4::ops_dsl_attention::prompt_blocks::swa_prompt_fp8_4_4_1"
        | "furiosa_opt_gemma4::ops_dsl_attention::decode::w256_fp8::swa_decode_fp8_w256"
        | "furiosa_opt_gemma4::ops_dsl_attention::prompt_blocks::swa_prompt_fp8_4_3_1")
}

fn family_for(bucket: usize, rss: &str, rsf: &str, rfs: &str, attention: &[&str]) -> Option<usize> {
    table::FAMILIES.iter().position(|f| {
        let member = |op: usize| table::MEMBERS[f.ops[op]].path;
        f.bucket == bucket && member(0) == rss
            && member(1) == rsf
            && member(2) == rfs
            && attention.iter().all(|a| {
                family_members(f).any(|(_, m)| table::MEMBERS[m].path == *a)
                    || (bucket == 1536 && standalone_wide_member(a))
            })
    })
}

/// What a step at a resident bucket offers the chooser: its segment programs' definition paths (`rss` with its
/// `rfs`, `rsf`) and whether resident programs serve every segment.
pub(crate) struct StepKey<'a> {
    pub bucket: usize,
    pub all_resident: bool,
    pub rss: Option<(&'a str, Option<&'a str>)>,
    pub rsf: Option<&'a str>,
}

type Function = furiosa_opt_std::backend::npu::Function;
type Buffer = <furiosa_opt_std::runtime::Buffers as IntoIterator>::Item;

/// The loaded families: their functions, the member each registered function is, and the padding and control
/// buffers every routed launch binds.
struct Loaded {
    functions: Vec<Option<std::sync::Arc<Function>>>,
    /// Registered function (its `Arc` address) -> member index.
    member_of: std::collections::HashMap<usize, usize>,
    /// Per family: member index -> opcode.
    opcodes: Vec<std::collections::HashMap<usize, usize>>,
    padding: Buffer,
    control: Buffer,
    /// The tensors that own the padding and control buffers.
    _owners: (HbmTensor<bf16, crate::Chip, m![Rq, Hqs]>, HbmTensor<i32, crate::Chip, m![Rq]>),
}

thread_local! {
    /// Functions the engine loaded for a family member, by definition path (`register`).
    static REGISTERED: std::cell::RefCell<Vec<(&'static str, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
    static LOADED: std::cell::RefCell<Option<Loaded>> = const { std::cell::RefCell::new(None) };
    /// The family the current step's member launches go to.
    static ACTIVE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Forgets the registered functions and the loaded families (a new engine; call before its modules load).
pub fn reset() {
    launch_boundary::clear();
    persist::clear();
    {
        REGISTERED.with_borrow_mut(Vec::clear);
        LOADED.with_borrow_mut(|l| *l = None);
    }
    ACTIVE.set(None);
}

/// Records that the engine launches definition path `path` through `function` (the carried route of a resident
/// program, a DSL attention kernel); a family member's launches through it can then be routed.
pub fn register(path: &'static str, function: &std::sync::Arc<Function>) {
    if let Some(member) = table::MEMBERS.iter().position(|m| m.path == path) {
        let at = std::sync::Arc::as_ptr(function) as usize;
        let installed = LOADED.with_borrow_mut(|loaded| {
            if let Some(loaded) = loaded {
                loaded.member_of.insert(at, member);
                true
            } else {
                false
            }
        });
        if !installed {
            REGISTERED.with_borrow_mut(|r| r.push((path, at)));
        }
    }
}

/// Installs family metadata and padding/control buffers. Members may register
/// before this call or on their first launch; choosing a family still requires
/// the complete actual step key and attention paths.
pub async fn load(device: &mut Device) -> Result<(), BoxError> {
    use furiosa_opt_std::runtime::{Buffers, DeviceSend};
    let registered = REGISTERED.with_borrow_mut(std::mem::take);
    let mut member_of = std::collections::HashMap::new();
    for (path, at) in registered {
        if let Some(i) = table::MEMBERS.iter().position(|m| m.path == path) {
            member_of.insert(at, i);
        }
    }
    let mut functions: Vec<Option<std::sync::Arc<Function>>> = Vec::new();
    let mut opcodes: Vec<std::collections::HashMap<usize, usize>> = Vec::new();
    for f in table::FAMILIES {
        functions.push(None);
        opcodes.push(family_members(f).map(|(op, member)| (member, op)).collect());
    }
    let padding_owner: HbmTensor<bf16, crate::Chip, m![Rq, Hqs]> = HostTensor::<bf16, m![Rq, Hqs]>::zero().to_hbm(&mut device.pdma).await?;
    // A 512-aligned subregion makes opcodes 32..40 unambiguous even when
    // the device allocator only guarantees 256-byte allocation alignment.
    let control_owner: HbmTensor<i32, crate::Chip, m![Rq]> = HostTensor::<i32, m![Rq]>::zero().to_hbm(&mut device.pdma).await?;
    let one = |tensor: &dyn DeviceSend| -> Result<Buffer, BoxError> {
        let mut bound = Buffers::new();
        tensor.bind(&mut bound)?;
        match (bound.pop(), bound.len()) {
            (Some(b), 0) => Ok(b),
            _ => Err("step_image: a buffer tensor binds other than one buffer".into()),
        }
    };
    let padding = one(&padding_owner)?;
    let control = one(&control_owner)?;
    let align = control.addr().wrapping_neg() & 511;
    let control = control.slice(align..align + 512);
    if padding.size() < table::PADDING_BYTES || control.size() < 8 * 32 {
        return Err(format!("step_image: padding {} B, control {} B", padding.size(), control.size()).into());
    }
    LOADED.with_borrow_mut(|l| {
        *l = Some(Loaded { functions, member_of, opcodes, padding, control, _owners: (padding_owner, control_owner) })
    });
    Ok(())
}

/// Whether family metadata is loaded.
fn loaded() -> bool {
    return LOADED.with_borrow(Option::is_some);
}

/// Chooses the family the step's member launches go to (`None`: the stock images).
/// `attention` lists the definition paths of every attention launch the step's plans hold (`None`: a plan the
/// families cannot hold, such as an official one); it is asked only when the segments already match a family.
pub(crate) fn choose(key: StepKey<'_>, attention: impl FnOnce() -> Option<Vec<&'static str>>) -> Option<usize> {
    if !loaded() || !table::FAMILIES.iter().any(|f| f.bucket == key.bucket) {
        return None;
    }
    if !key.all_resident {
        return None;
    }
    let (Some((rss, Some(rfs))), Some(rsf)) = (key.rss, key.rsf) else { return None };
    family_for(key.bucket, rss, rsf, rfs, &[])?;
    family_for(key.bucket, rss, rsf, rfs, &attention()?)
}

/// Load the selected family if needed.
pub(crate) async fn prepare(device: &furiosa_opt_std::prelude::Device, family: Option<usize>) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(f) = family {
        let missing = LOADED.with_borrow(|l| {
            let l = l.as_ref().expect("chosen family must be registered");
            l.functions[f].is_none()
        });
        if missing {
            let row = table::FAMILIES[f].row;
            let function = std::sync::Arc::new(furiosa_opt_std::backend::npu::function(device, row, &[]).await?);
            LOADED.with_borrow_mut(|l| {
                let l = l.as_mut().expect("registered families");
                l.functions[f] = Some(function);
            });
        }
    }
    Ok(())
}

fn selected_function(l: &Loaded, family: usize) -> Option<&std::sync::Arc<Function>> {
    l.functions[family].as_ref()
}

/// The positional family header can describe several frame layouts. Check
/// the original function's full operand extents before putting its addresses
/// into any of those layouts.
fn frame_fits(
    family: &table::Family,
    op: usize,
    member: &table::Member,
    inputs: &furiosa_opt_std::runtime::Buffers,
    outputs: &furiosa_opt_std::runtime::Buffers,
) -> bool {
    let extents = match (family.name, op) {
        ("t1024_g12_p941_pair", 0) => p941_g12_slots::RSS,
        ("t1024_g12_p941_pair", 2) => p941_g12_slots::RFS,
        _ if family.bucket == 1536 => p999_g12_slots::OPERANDS[op],
        _ => return true,
    };
    extents.iter().enumerate().all(|(slot, &size)| {
        let bound = if outputs.is_empty() {
            inputs.get(slot)
        } else if let Some(i) = member.outputs.iter().position(|&output| output == slot) {
            outputs.get(i)
        } else {
            inputs.get(slot)
        };
        bound.is_some_and(|buffer| buffer.size() >= size)
    })
}

/// Sets the family the next member launches go to (`None` at a step's start and after its layers).
pub(crate) fn activate(family: Option<usize>) {
    assert!(launch_boundary::empty(), "flush pending RSF before changing the step family");
    assert!(persist::empty(), "flush pending RSS launches before changing the step family");
    ACTIVE.set(family);
}

/// Route an active family member to the family image, padding its arguments and binding the opcode.
pub(crate) fn route(
    function: std::sync::Arc<Function>,
    inputs: furiosa_opt_std::runtime::Buffers,
    outputs: furiosa_opt_std::runtime::Buffers,
) -> full_sequence::Frame {
    let Some(f) = ACTIVE.get() else { return (function, inputs, outputs) };
    let routed = LOADED.with_borrow(|l| {
        let l = l.as_ref()?;
        let m = *l.member_of.get(&(std::sync::Arc::as_ptr(&function) as usize))?;
        let op = *l.opcodes[f].get(&m)?;
        let (member, family) = (&table::MEMBERS[m], &table::FAMILIES[f]);
        let slots = family.slots;
        // The stock launch's own shape (`Signature::bind`): its inputs, and its outputs or none when they are inputs.
        let shaped = inputs.len() == member.inputs
            && if outputs.is_empty() { member.outputs.iter().all(|&o| o < member.inputs) } else { outputs.len() == member.outputs.len() };
        if !shaped {
            return None;
        }
        let buffer_at = |slot| {
            // As the stock bind: an output overrides the input of its slot.
            match member.outputs.iter().position(|&o| o == slot).filter(|_| !outputs.is_empty()) {
                Some(j) => Some(&outputs[j]),
                None => inputs.get(slot),
            }
        };
        if !frame_fits(family, op, member, &inputs, &outputs) {
            eprintln!("ENGINE_STEP_WIDE_BINDING opcode={op} does not fit its original extent");
            return None;
        }
        let mut args = furiosa_opt_std::runtime::Buffers::new();
        for slot in 0..slots - 1 {
            args.try_push(buffer_at(slot).unwrap_or(&l.padding).clone()).ok()?;
        }
        args.try_push(l.control.slice(8 * op..8 * op + 8)).ok()?;
        let function = selected_function(l, f)?.clone();
        Some((function, args))
    });
    match routed {
        Some((family, args)) => (family, args, furiosa_opt_std::runtime::Buffers::new()),
        None => (function, inputs, outputs),
    }
}

/// Clear family routing at the step boundary.
pub(crate) fn step_ended() {
    ACTIVE.set(None);
}

/// Family rows and member definition paths used by the startup registry check.
pub(crate) fn resolution_catalog() -> Vec<(&'static str, &'static str, Vec<&'static str>)> {
    table::FAMILIES.iter().map(|f| (f.name, f.row, family_members(f).map(|(_, i)| table::MEMBERS[i].path).collect())).collect()
}
