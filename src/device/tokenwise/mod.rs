//! Shared tokenwise normalization and head arithmetic. Rows are grouped in eights
//! on the TRF lanes. Each contraction keeps its K packet order and rounding points,
//! and row statistics reduce independently of the other tokens in the group.

pub(crate) mod common;
pub(crate) mod head;
