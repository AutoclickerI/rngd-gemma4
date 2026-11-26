//! The server's worker admits requests and drives the token-packed `engine`.
//! `audio`, `image` and `tokenizer` prepare inputs. `generate` owns generation state
//! and emits text deltas as the engine retires rows.
//! `runtime` uploads RoPE tables.
pub mod audio;
pub mod engine;
pub mod generate;
pub mod image;
pub mod load;
pub mod media;
pub mod prebuilt;
mod preprocess;
pub mod registry;
pub mod runtime;
pub mod sampling;
pub mod tokenizer;
pub mod telemetry;
pub(crate) mod arrival;
pub mod worker;

pub(crate) mod envelope_lookup;
