//! Batch adjacent RSS launches without moving the engine's between-layer hooks.
//! A drain flushes the pending batch before any host read or table reuse.
#[path = "p999_g12_persist.rs"]
mod wide_g12;
#[path = "p941_g12_persist.rs"]
mod narrow_g12;

struct Recipe {
    opcode: usize,
    count: usize,
    origins: &'static [(usize, usize)],
    // For every read operand: layer, original slot, compact binding, byte offset.
    checks: &'static [(usize, usize, usize, usize)],
    sizes: &'static [usize],
}

fn fits(r: &Recipe, count: usize, mut buffer: impl FnMut(usize, usize) -> Option<(usize, usize)>) -> bool {
    if count < r.count { return false; }
    r.checks.iter().all(|&(layer, slot, compact, offset)| {
        let Some(&(origin_layer, origin_slot)) = r.origins.get(compact) else { return false };
        let (Some((addr, size)), Some((base, _))) = (buffer(layer, slot), buffer(origin_layer, origin_slot)) else { return false };
        size >= r.sizes[slot] && base.checked_add(offset) == Some(addr)
    })
}

fn recipes_for(path: &str) -> &'static [Recipe] {
    if path == wide_g12::PATH { wide_g12::RECIPES }
    else if path == narrow_g12::PATH { narrow_g12::RECIPES }
    else { &[] }
}

// Only opcode zero uses the family's Q4 RSS loop frames.
fn rss_member(f: &super::table::Family, member: usize, opcode: usize) -> bool {
    opcode == 0 && f.ops.first() == Some(&member)
        && !recipes_for(super::table::MEMBERS[member].path).is_empty()
}

mod live {
    use super::*;
    use super::super::{ACTIVE, LOADED, Function, table as families};
    use furiosa_opt_std::runtime::Buffers;
    use std::cell::RefCell;
    use std::sync::Arc;

    pub(crate) struct Launch {
        pub function: Arc<Function>,
        pub inputs: Buffers,
        pub outputs: Buffers,
        // Preserve every original allocation until the loop completes, including
        // frames represented by a base plus a verified offset.
        pub owners: Vec<Buffers>,
    }
    struct Pending { family: usize, member: usize, function: Arc<Function>, inputs: Buffers }
    thread_local! {
        static PENDING: RefCell<Vec<Pending>> = const { RefCell::new(Vec::new()) };
    }

    const MAX_LAYERS: usize = 4;

    fn routed(function: Arc<Function>, inputs: Buffers, outputs: Buffers) -> Launch {
        let (function, inputs, outputs) = super::super::route(function, inputs, outputs);
        Launch { function, inputs, outputs, owners: Vec::new() }
    }

    pub(crate) fn clear() { PENDING.with_borrow_mut(Vec::clear); }
    pub(crate) fn empty() -> bool { PENDING.with_borrow(Vec::is_empty) }

    pub(crate) fn flush() -> Vec<Launch> {
        let mut pending = PENDING.with_borrow_mut(std::mem::take);
        let mut launches = Vec::new();
        while !pending.is_empty() {
            let family = pending[0].family;
            assert_eq!(ACTIVE.get(), Some(family), "persist batch crossed a step-family boundary");
            let member = pending[0].member;
            let rss_path = families::MEMBERS[member].path;
            let recipes = recipes_for(rss_path);
            let recipe = recipes.iter().filter(|r| r.count <= MAX_LAYERS).filter(|r| fits(r, pending.len(), |l,s| {
                let p = pending.get(l)?;
                if p.family != family || p.member != member || p.inputs.len() != r.sizes.len() { return None; }
                let b = p.inputs.get(s)?;
                Some((b.addr(), b.size()))
            })).max_by_key(|r| (r.count, usize::MAX-r.origins.len()));
            if let Some(r) = recipe {
                let launch = LOADED.with_borrow(|loaded| {
                    let loaded = loaded.as_ref().expect("active step family");
                    let f = &families::FAMILIES[family];
                    assert!(r.origins.len() < f.slots);
                    let mut inputs = Buffers::new();
                    for &(l,s) in r.origins { inputs.try_push(pending[l].inputs[s].clone()).expect("validated compact frame"); }
                    while inputs.len() + 1 < f.slots { inputs.try_push(loaded.padding.clone()).expect("family padding"); }
                    inputs.try_push(loaded.control.slice(8*r.opcode..8*r.opcode+8)).expect("family opcode");
                    Launch { function: super::super::selected_function(loaded, family).expect("prepared family").clone(), inputs, outputs: Buffers::new(), owners: Vec::new() }
                });
                let mut launch = launch;
                launch.owners = pending.drain(..r.count).map(|p|p.inputs).collect();
                launches.push(launch);
            } else {
                let p = pending.remove(0);
                launches.push(routed(p.function,p.inputs,Buffers::new()));
            }
        }
        launches
    }

    pub(crate) fn enqueue(function: Arc<Function>, inputs: Buffers, outputs: Buffers) -> Vec<Launch> {
        let candidate = ACTIVE.get().and_then(|family| {
            if !outputs.is_empty() { return None; }
            LOADED.with_borrow(|loaded| {
                let loaded = loaded.as_ref()?;
                let member = *loaded.member_of.get(&(Arc::as_ptr(&function) as usize))?;
                let opcode = *loaded.opcodes[family].get(&member)?;
                rss_member(&families::FAMILIES[family], member, opcode).then_some((family, member))
            })
        });
        if let Some((family, member)) = candidate {
            let mut ready = if PENDING.with_borrow(|p| p.last().is_some_and(|p| p.family != family || p.member != member)) {
                flush()
            } else { Vec::new() };
            PENDING.with_borrow_mut(|p| p.push(Pending { family, member, function, inputs }));
            if PENDING.with_borrow(Vec::len) >= MAX_LAYERS { ready.extend(flush()); }
            ready
        } else {
            let mut ready = flush();
            ready.push(routed(function, inputs, outputs));
            ready
        }
    }

}
pub(crate) use live::{Launch, clear, empty, enqueue, flush};
