//! Sliding and Full attention in the Opt DSL:
//! decode window variants and the window-keyed prefill kernels, over E4M3 KV pools. The engine launches
//! them through raw buffers (`host::engine::dsl`), like the transplanted kernels, so their query and output regions
//! are rows of the family's row space and their tables are words of the step's upload.
//!
//! Pools as the engine keeps them (`host::engine::tables`): Sliding `[Rs][8][256]`, Full `[Rf][512]`. The decode
//! kernels gather whole pool rows; the prefill kernels read the Sliding pool as `[Rs x 8][256]` rows of one KV head.

/// Decode E4M3 KV on fetch via the lookup table.
macro_rules! dsl_lut {
    (f8e4m3, $fetch:expr) => {
        $fetch.fetch_table_lookup::<bf16>()
    };
}

pub mod decode;
pub mod decode_full;
pub mod decode_groups;
pub mod prefill;
pub mod prefill_full;
pub mod prompt_blocks;
// Keep prompt-round expansions fixed for the kernel images.
pub mod prompt_pattn;
mod prompt_pattn_fallback;
/// The Sliding attention window `W` and the kernel geometry it fixes (one constant; module doc).
pub mod window;

#[cfg(feature = "compile-carried")]
pub mod paired;
