//! Sliding projections that process larger token buckets. Fewer, larger transfers
//! reduce DMA queue pressure while preserving per-token arithmetic.

pub(crate) mod gemm;
pub(crate) mod heads;
// Keep QKV projection helper signatures fixed for the kernel images.
pub(crate) mod qkv;
// Keep O projection helper signatures fixed for the kernel images.
pub(crate) mod o;
pub(crate) mod rotq;
pub(crate) mod rot_heads;
