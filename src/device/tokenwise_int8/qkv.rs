//! The Sliding QKV stage with rotated INT8 projections: input RMSNorm (gamma pre-multiplied by
//! the rotation's signs, exact), the H60 (x) H64 rotation and per-row INT8 quantization, the fused
//! Q/K/V INT8 contraction (8,192 rows: Q 4,096 | K 2,048 | V 2,048), then the head
//! phase (head norms, RoPE).
use furiosa_opt_std::prelude::*;

axes![Nq = 8192];

/// The fused Q/K/V weight rows' cluster.
pub(crate) type QkvCl = m![Nq / 4096];
