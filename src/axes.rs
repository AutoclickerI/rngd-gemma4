use furiosa_opt_std::prelude::*;

axes![
    H = 3840,
    L = 15360,
    W = 262144,
    Ns = 8,
    Gs = 2,
    Ds = 256,
    Qs = 4096,
    Ps = 2048,
    Ts = 1024,
    Gf = 16,
    Df = 512,
    Qf = 8192,
    Pf = 512,
    Tf = 512,
    C = 64,
    E = 32768,
    Rv = 6912,
    Mv = 3840,
    Ov = 3840,
    Pv = 1120,
    Aa = 640,
    Dummy2 = 2,
    Dummy8 = 8,
    Dummy256 = 256,
    // Token-packed step: token
    // buckets, the same rows as pairs (one row per cluster: `TokN = PairN x Pr`) and output rows.
    Tok8 = 8,
    Tok32 = 32,
    Tok128 = 128,
    Tok256 = 256,
    Tok384 = 384,
    Tok512 = 512,
    Tok640 = 640,
    Tok768 = 768,
    Tok896 = 896,
    Tok1024 = 1024,
    Pair4 = 4,
    Pr = 2,
    Out128 = 128,
    // KV pool per layer (`host::engine::tables`): Sliding `Ao` rings of `Ts` rows (rings 0 and 1
    // hold the padding slot 0 and up to 1,536 staging rows, the others are owner rings), Full
    // `Kp` pages of `Tf` rows (page 0 is the zero page holding the padding
    // slot, the last page takes padding writes). Indirect DMA keys one table axis, so the pools
    // are flat row axes: `Rs = Ao x Ts`, `Rf = Kp x Tf`.
    Ao = 102,
    Rs = 104448,
    MediaPool = 32768,
    Kp = 514,
    Rf = 263168,
    // Full attention page-slot variants (the step uses the smallest not below its longest row).
    Pg2 = 2,
    Pg5 = 5,
    Pg16 = 16,
    // Sampler: a logits row as 512 segments of 512, one segment per slice of both clusters.
    Nseg = 512,
    Lseg = 512,
    // Attention row space: a step
    // row's query or output `[16 heads x D]` as two kernel rows of 8 heads (`Hh` halves of `Hqs`
    // for Sliding, `Hqf` for Full), the per-family row space `Rq` the launches read and write,
    // and the size classes of the per-step gather-index and mask uploads (i32 words).
    Hh = 2,
    Hqs = 2048,
    Hqf = 4096,
    Rq = 8192,
    Iw16 = 65536,
    Iw17 = 131072,
    Iw18 = 262144,
    Iw19 = 524288,
    Iw20 = 1048576,
    Iw21 = 2097152,
    Iw22 = 4194304,
    Iw23 = 8388608,
    Iw24 = 16777216,
    // Half-octave word classes between them (√2 apart, rounded to 4,096 words): the merged single upload of
    // both families' words (`attention::Words`).
    Iw16h = 94208,
    Iw17h = 184320,
    Iw18h = 372736,
    Iw19h = 741376,
    Iw20h = 1482752,
    Iw21h = 2965504,
    Iw22h = 5931008,
    Iw23h = 11862016,
    // The stationary-weight core's per-slice constant copies: slice `s` reads copy `s % 16` of an `[Rc, H]`
    // RMSNorm vector.
    Rc = 16,
];

// Match the wide row geometry without adding a new axis to Opt's kernel cache keys.
pub use crate::ops_dsl_attention::prefill_full::w1536_fp8::W0 as Tok1536;
