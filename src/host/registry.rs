//! Installs our carried images into the binary's device-function registry.
//!
//! The runtime (`furiosa-opt-std` 0.9.0, `backend/npu/registry.rs`) finds every device function
//! in the `furiosa_kernels` section: at the first launch it bincode-decodes the section, from the
//! linker's `__start_furiosa_kernels`, as `Vec<(path, key, image)>` and ignores trailing bytes;
//! `backend::npu::function(device, path, key)` then selects a row by path and key and loads its
//! image. `cargo furiosa-opt build` appends those rows as one object at the end of the link.
//!
//! Each carried payload holds its rows LZMA-compressed: our own heavy kernels as prebuilt images
//! (`crate::host::prebuilt`, built by `tools/native/prebuilt/pack.py` from a build of this source), the
//! resident STEP images (`prebuilt/step.frcz`). [`PAYLOADS`] lists them.
//!
//! The merged registry lives in a zero-fill allocation whose symbols satisfy the runtime's registry ABI. The driver-produced
//! native rows remain unchanged in the executable's furiosa_kernels section; the
//! installer reads that section from /proc/self/exe, appends the decompressed carried
//! rows, and writes the allocation once before main. No carried decompression reserve
//! occupies the executable file. Installation failure is reported before serving.
//!
//! Wire formats:
//! - payload: `b"FRCZv1\0\0"` | u64 raw_len | u64 fnv1a64(raw) | `.lzma` stream of raw
//! - raw and registry: u64 count | count x (u64 path_len, path, u64 key_len, key_len x u64,
//!   u64 image_len, image)
use std::sync::OnceLock;

/// Also q4serve's q4cal artifact decoder (`host::engine::int8::q4cal`).
pub(crate) mod lzma;

const PAYLOAD_MAGIC: [u8; 8] = *b"FRCZv1\0\0";

/// One carried payload: a name for the log, the prefix every one of its rows' paths carries, and
/// its bytes (`FRCZv1`).
pub struct Payload {
    pub name: &'static str,
    pub prefix: &'static str,
    pub bytes: &'static [u8],
}

/// Payloads in registry merge order. The second holds the resident step images.
/// See `engine::step_image` for dispatch and binding.
pub const PAYLOADS: [Payload; 2] = [
    Payload { name: "prebuilt", prefix: crate::host::prebuilt::PATH_PREFIX, bytes: crate::host::prebuilt::PAYLOAD },
    Payload { name: "step_image", prefix: crate::host::engine::step_image::PATH_PREFIX, bytes: crate::host::engine::step_image::PAYLOAD },
];

const fn header_u64(p: &[u8], at: usize) -> u64 {
    u64::from_le_bytes([p[at], p[at + 1], p[at + 2], p[at + 3], p[at + 4], p[at + 5], p[at + 6], p[at + 7]])
}

/// Bytes of a payload's rows once decompressed (its header's `raw_len`).
pub const fn raw_len(p: &[u8]) -> usize {
    header_u64(p, 8) as usize
}

/// Bytes of every payload's rows once decompressed: the reserve.
const RESERVE_LEN: usize = {
    let mut sum = 0;
    let mut i = 0;
    while i < PAYLOADS.len() {
        sum += raw_len(PAYLOADS[i].bytes);
        i += 1;
    }
    sum
};

/// One registry row, borrowing its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row<'a> {
    pub path: &'a str,
    pub key: Vec<u64>,
    pub image: &'a [u8],
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at.checked_add(8)?)?.try_into().ok()?))
}

/// A registry at offset 0 of `b`: its rows and its end. A zero or implausible count is no
/// registry, so zero padding and the reserve's magic never decode as one.
pub fn decode_registry(b: &[u8]) -> Option<(Vec<Row<'_>>, usize)> {
    let count = u64_at(b, 0)?;
    if count == 0 || count > b.len() as u64 / 24 {
        return None;
    }
    let mut at = 8usize;
    let mut rows = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let n = u64_at(b, at)? as usize;
        let end = at.checked_add(8)?.checked_add(n)?;
        let path = std::str::from_utf8(b.get(at + 8..end)?).ok()?;
        at = end;
        let keys = u64_at(b, at)? as usize;
        at = at.checked_add(8)?;
        let mut key = Vec::with_capacity(keys.min(64));
        for _ in 0..keys {
            key.push(u64_at(b, at)?);
            at = at.checked_add(8)?;
        }
        let n = u64_at(b, at)? as usize;
        let end = at.checked_add(8)?.checked_add(n)?;
        let image = b.get(at + 8..end)?;
        at = end;
        rows.push(Row { path, key, image });
    }
    Some((rows, at))
}

/// Bytes of `rows` in the registry wire format.
pub fn encoded_len(rows: &[Row<'_>]) -> usize {
    8 + rows.iter().map(|r| 24 + r.path.len() + 8 * r.key.len() + r.image.len()).sum::<usize>()
}

/// Writes `rows` in the registry wire format to the start of `out` (at least [`encoded_len`] bytes); returns the length.
pub fn encode_registry_into(rows: &[Row<'_>], out: &mut [u8]) -> usize {
    let mut at = 0;
    let mut put = |b: &[u8]| {
        out[at..at + b.len()].copy_from_slice(b);
        at += b.len();
    };
    put(&(rows.len() as u64).to_le_bytes());
    for r in rows {
        put(&(r.path.len() as u64).to_le_bytes());
        put(r.path.as_bytes());
        put(&(r.key.len() as u64).to_le_bytes());
        for k in &r.key {
            put(&k.to_le_bytes());
        }
        put(&(r.image.len() as u64).to_le_bytes());
        put(r.image);
    }
    at
}

fn fnv1a64(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ *x as u64).wrapping_mul(0x100_0000_01b3))
}

/// A payload's rows, decompressed and checked against its header.
pub fn unpack_payload(p: &Payload) -> Result<Vec<u8>, String> {
    let b = p.bytes;
    if b.len() < 24 || b[..8] != PAYLOAD_MAGIC {
        return Err(format!("{}: not an FRCZv1 payload", p.name));
    }
    let raw = lzma::decode_alone(&b[24..], raw_len(b))?;
    if fnv1a64(&raw) != header_u64(b, 16) {
        return Err(format!("{}: rows do not match the header's checksum", p.name));
    }
    Ok(raw)
}

/// The carried rows of `raw`, checked: every path carries `prefix`, every key is empty. A payload
/// of no rows (a zero count) is allowed.
pub fn carried_with<'a>(raw: &'a [u8], prefix: &str) -> Result<Vec<Row<'a>>, String> {
    if raw.len() == 8 && u64_at(raw, 0) == Some(0) {
        return Ok(Vec::new());
    }
    let (rows, end) = decode_registry(raw).ok_or("payload rows do not decode")?;
    if end != raw.len() {
        return Err("payload rows end before the payload".into());
    }
    if let Some(r) = rows.iter().find(|r| !r.path.starts_with(prefix) || !r.key.is_empty()) {
        return Err(format!("unexpected payload row {}", r.path));
    }
    Ok(rows)
}

/// The outcome of the one installation.
#[derive(Clone, Debug)]
pub struct Installed {
    /// Carried rows now in the registry, every payload.
    pub carried: usize,
    /// Carried paths of our prebuilt kernels now in the registry.
    pub prebuilt: Vec<String>,
    /// Registry rows after the merge, driver's and carried.
    pub registry_rows: usize,
    /// Raw bytes of the carried rows, every payload.
    pub carried_bytes: usize,
    pub micros: u128,
}

static STATE: OnceLock<Result<Installed, String>> = OnceLock::new();

/// Installs the carried rows once (the constructor below normally has, before `main`); returns
/// the outcome of that installation.
pub fn install() -> &'static Result<Installed, String> {
    STATE.get_or_init(|| {
        let t = std::time::Instant::now();
        let result = install_once().map(|mut done| {
            done.micros = t.elapsed().as_micros();
            done
        });
        match &result {
            Ok(done) => eprintln!(
                "registry: installed {} carried rows ({} prebuilt, {} B) into the registry ({} rows) in {} us",
                done.carried,
                done.prebuilt.len(),
                done.carried_bytes,
                done.registry_rows,
                done.micros
            ),
            Err(why) => eprintln!("registry: not installed: {why}"),
        }
        result
    })
}

/// The registry as the runtime will decode it, read back from the section: (path, key) per row.
pub(crate) fn installed_rows() -> Result<Vec<(String, Vec<u64>)>, String> {
    install().as_ref().map_err(|why| why.clone())?;
    return section::rows();
}

/// A decode running on its own thread, or its result when no thread could be spawned (it then ran inline).
enum Pending<'s, T> {
    Thread(std::thread::ScopedJoinHandle<'s, Result<T, String>>),
    Done(Result<T, String>),
}

impl<T> Pending<'_, T> {
    fn join(self) -> Result<T, String> {
        match self {
            Pending::Thread(h) => h.join().unwrap_or_else(|_| Err("a registry decode thread panicked".into())),
            Pending::Done(r) => r,
        }
    }
}

/// Runs `f` on a scoped thread, or inline when no thread can be spawned (`f` is `Copy`, so it is still at hand).
fn pending<'s, 'e, T, F>(s: &'s std::thread::Scope<'s, 'e>, f: F) -> Pending<'s, T>
where
    F: FnOnce() -> Result<T, String> + Send + Copy + 'e,
    T: Send + 'e,
{
    match std::thread::Builder::new().name("registry-decode".into()).spawn_scoped(s, f) {
        Ok(h) => Pending::Thread(h),
        Err(_) => Pending::Done(f()),
    }
}

fn install_once() -> Result<Installed, String> {
    // Decode each payload and read the native registry in parallel. Install rows in payload order.
    static TABLE: [Payload; PAYLOADS.len()] = PAYLOADS;
    let (raws, native) = std::thread::scope(|s| {
        let native = pending(s, section::native_section);
        let raws: Vec<_> = TABLE.iter().map(|p| pending(s, move || unpack_payload(p))).collect();
        (raws.into_iter().map(Pending::join).collect::<Result<Vec<_>, _>>(), native.join())
    });
    let (raws, native) = (raws?, native?);
    let mut carried = Vec::new();
    for (p, raw) in PAYLOADS.iter().zip(&raws) {
        carried.extend(carried_with(raw, p.prefix).map_err(|why| format!("{}: {why}", p.name))?);
    }
    let prebuilt =
        carried.iter().filter(|r| r.path.starts_with(crate::host::prebuilt::PATH_PREFIX)).map(|r| r.path.to_string()).collect();
    let count = carried.len();
    let rows = section::merged(&native, carried)?;
    section::write(&rows)?;
    Ok(Installed { carried: count, prebuilt, registry_rows: rows.len(), carried_bytes: raws.iter().map(Vec::len).sum(), micros: 0 })
}

mod section {
    use super::{RESERVE_LEN, Row, decode_registry, encode_registry_into, encoded_len};

    // The runtime resolves these two symbols, not an ELF section header. Keep its
    // merged registry in zero-fill memory so carried rows do not also occupy a
    // file-backed reserve. The driver still emits its unchanged native rows into
    // furiosa_kernels; read that section from this executable before installing.
    // Native rows currently use less than 8 MiB. This source reserves 64 MiB for
    // them in addition to the exact carried length; unused pages stay untouched.
    const REGISTRY_CAPACITY: usize = RESERVE_LEN + 64 * 1024 * 1024;
    core::arch::global_asm!(
        r#"
        .pushsection .bss.furiosa_registry,"aw",@nobits
        .balign 4096
        .globl __start_furiosa_kernels
        .hidden __start_furiosa_kernels
        .type __start_furiosa_kernels,@object
        __start_furiosa_kernels:
        .zero {capacity}
        .size __start_furiosa_kernels, {capacity}
        .globl __stop_furiosa_kernels
        .hidden __stop_furiosa_kernels
        __stop_furiosa_kernels:
        .popsection
        "#,
        capacity = const REGISTRY_CAPACITY,
    );

    /// Runs `install` before `main`, so no launch can come first.
    #[used]
    #[unsafe(link_section = ".init_array")]
    static INSTALL_AT_START: extern "C" fn() = {
        extern "C" fn install_at_start() {
            let _ = super::install();
        }
        install_at_start
    };

    unsafe extern "C" {
        #[link_name = "__start_furiosa_kernels"]
        static START: u8;
        #[link_name = "__stop_furiosa_kernels"]
        static STOP: u8;
    }

    fn bounds() -> (usize, usize) {
        ((&raw const START) as usize, (&raw const STOP) as usize)
    }

    /// The section's registry rows (after `install`, the merged registry).
    pub(super) fn rows() -> Result<Vec<(String, Vec<u64>)>, String> {
        let (start, stop) = bounds();
        // SAFETY: the linker-defined bounds delimit the loaded section, which only `write` changes
        // (once, before this can run: `installed_rows` waits for `install`).
        let section = unsafe { std::slice::from_raw_parts(start as *const u8, stop - start) };
        let (rows, _) = decode_registry(section).ok_or("the section start is not a registry")?;
        Ok(rows.into_iter().map(|r| (r.path.to_string(), r.key)).collect())
    }

    /// The driver's rows in `region` (bytes beside the reserve): none when the region is empty or
    /// zero, else the first registry within the alignment padding at its start.
    fn driver_rows(region: &[u8]) -> Result<Vec<Row<'_>>, String> {
        if region.iter().all(|&b| b == 0) {
            return Ok(Vec::new());
        }
        for pad in 0..region.len().min(4096) {
            if let Some((rows, _)) = decode_registry(&region[pad..]) {
                return Ok(rows);
            }
            if region[pad] != 0 {
                break;
            }
        }
        Err("bytes beside the reserve are not a registry".into())
    }

    /// Read the driver's unchanged section, independently of the runtime symbols.
    /// Only this binary is read; no external payload, environment or build path.
    pub(super) fn native_section() -> Result<Vec<u8>, String> {
        use std::io::{Read, Seek, SeekFrom};
        let read = || -> Result<Vec<u8>, String> {
            let mut file = std::fs::File::open("/proc/self/exe").map_err(|e| e.to_string())?;
            fn bytes(file: &mut std::fs::File, offset: u64, len: usize) -> Result<Vec<u8>, String> {
                file.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
                let mut out = vec![0; len];
                file.read_exact(&mut out).map_err(|e| e.to_string())?;
                Ok(out)
            }
            let header = bytes(&mut file, 0, 64)?;
            if &header[..6] != b"\x7fELF\x02\x01" { return Err("expected little-endian ELF64".into()); }
            let u16_at = |b: &[u8], at| u16::from_le_bytes([b[at], b[at + 1]]) as usize;
            let shoff = super::u64_at(&header, 40).ok_or("missing section offset")?;
            let stride = u16_at(&header, 58);
            let count = u16_at(&header, 60);
            let names_index = u16_at(&header, 62);
            if stride != 64 || names_index >= count { return Err("unsupported section table".into()); }
            let table = bytes(&mut file, shoff, stride * count)?;
            let entry = |i| &table[i * stride..(i + 1) * stride];
            let range = |h: &[u8]| -> Result<(u64, usize), String> {
                Ok((super::u64_at(h, 24).ok_or("missing section offset")?,
                    usize::try_from(super::u64_at(h, 32).ok_or("missing section length")?).map_err(|e| e.to_string())?))
            };
            let (off, len) = range(entry(names_index))?;
            let names = bytes(&mut file, off, len)?;
            for i in 0..count {
                let h = entry(i);
                let name = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
                let name = names.get(name..).ok_or("section name out of bounds")?.split(|&b| b == 0).next().unwrap_or(&[]);
                if name == b"furiosa_kernels" {
                    let (off, len) = range(h)?;
                    return bytes(&mut file, off, len);
                }
            }
            Err("native furiosa_kernels section missing".into())
        };
        read().map_err(|e| format!("native registry in /proc/self/exe: {e}"))
    }

    /// Install the driver's rows first, followed by the carried rows.
    pub(super) fn merged<'a>(native: &'a [u8], carried: Vec<Row<'a>>) -> Result<Vec<Row<'a>>, String> {
        let mut rows = driver_rows(native)?;
        for c in &carried {
            if rows.iter().any(|r| r.path == c.path) {
                return Err(format!("{} is already in the registry", c.path));
            }
        }
        rows.extend(carried);
        Ok(rows)
    }

    /// Install once, before main and before the runtime reads its bound symbols: encodes `rows` straight into the
    /// zero-fill allocation (no intermediate copy).
    pub(super) fn write(rows: &[Row<'_>]) -> Result<(), String> {
        let (start, stop) = bounds();
        let len = encoded_len(rows);
        if len > stop - start {
            return Err(format!("merged registry of {len} B exceeds zero-fill capacity {} B", stop - start));
        }
        // SAFETY: these symbols bound our writable zero-fill allocation above; no reference into it is alive and
        // install's OnceLock serializes this write before any device launch.
        let out = unsafe { std::slice::from_raw_parts_mut(start as *mut u8, len) };
        if encode_registry_into(rows, out) != len {
            return Err("merged registry length mismatch".into());
        }
        let count = unsafe { (start as *const u64).read_unaligned() };
        if count != rows.len() as u64 {
            return Err("read-back of the merged registry failed".into());
        }
        Ok(())
    }

}
