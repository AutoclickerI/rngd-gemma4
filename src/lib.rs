//! Gemma-4-12B-it on the Furiosa NPU: text, vision, and audio, targeting the Tensor
//! Contraction Processor via the `furiosa-opt-std` mapping-expression DSL. See the
//! crate's README for the device/host split and directory layout.

#![expect(clippy::type_complexity)]
#![feature(register_tool)]
#![feature(min_specialization)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

pub mod api;
pub mod axes;
pub mod host;
pub mod ops_audio_rows;
pub mod ops_glue;
pub mod ops_media;
pub mod ops_sampler;
pub mod ops_step;
pub mod ops_core;
pub mod ops_l0table;
pub mod ops_dsl_attention;
pub mod ops_tokenwise_int8;
pub mod ops_vision_rows;

pub(crate) mod device;

pub type Chip = m![1];

pub const LAYERS: usize = 48;
pub const EMBED_SCALE: f32 = 61.967_734;
pub const LOGIT_SOFTCAP: f32 = 30.0;
pub const EPS: f32 = 1e-6;
