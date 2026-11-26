//! Rotated INT8 tokenwise building blocks.
//!
//! In a rotated INT8 linear, the bf16 input row is rotated in f32 by a seeded-sign Kronecker Hadamard and
//! narrowed to bf16 once; the rotated row is quantized per row to INT8 with
//! `s = max(amax / 127, 1e-30)` and `q = clamp(rne(x / s), -127, 127)`. Startup rotates and quantizes each output weight row
//! in the same basis. The contraction accumulates in i32 (exact,
//! so any K order or K split gives the same sum); the epilogue is
//! `bf16((f32(acc) * s_x) * w_n)` with `w_n = f32(s_w * g)` (g the NVFP4 global scale, 1 for
//! attention). Only the rotation's f32 summation order is implementation-defined.
pub(crate) mod core16;
pub(crate) mod ffn;
pub(crate) mod gemm;
pub(crate) mod gemm16;
pub(crate) mod passes;
pub(crate) mod post;
pub(crate) mod qkv;
pub(crate) mod rotate;
pub(crate) mod rotate_ff;
pub(crate) mod rotate_heads;
pub(crate) mod core16_full;
pub(crate) mod segments_full;
// INT8-Full (`int8::int8_full_kernels!` on): the Full QKV / O of the Full transitions on INT8.
// Keep the Full projection helpers fixed for the kernel images.
pub(crate) mod core8_full;
pub(crate) mod segments_full_i8;
pub(crate) mod heads_full1;
pub(crate) mod ffn_fast;
pub(crate) mod rotate2;
pub(crate) mod post2;
