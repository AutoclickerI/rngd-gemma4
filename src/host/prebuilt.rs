//! Installs carried device images in the registry before `main`.
//!
//! The launch macros select a carried registry row or a typed device function.
//! Both bind the same arguments in the same order. The reference build uses
//! `compile-carried`; kernels with individual compiler settings are rebuilt
//! separately through their recipes.
//!
//! The payload header checks decompression integrity. The startup registry
//! check reports missing rows before serving.

use furiosa_opt_std::__private::DeviceOutput;
use furiosa_opt_std::prelude::*;

/// The path prefix of every carried prebuilt row: `<prefix><definition path after the crate name>`.
pub const PATH_PREFIX: &str = "furiosa_opt_gemma4::prebuilt::";

/// The carried rows (`FRCZv1`), possibly none.
pub(crate) const PAYLOAD: &[u8] = include_bytes!("prebuilt/payload.frcz");

/// The Sliding window (`ops_dsl_attention::window::SLIDING_WINDOW`) the carried images were built at: the window
/// selects the resident, fused and prompt programs' shapes. The payload must match the compiled geometry.
pub const CARRIED_WINDOW: usize = 384;

/// Without `compile-carried` the server launches carried rows from their images: built at another window they would
/// take this build's tables at the wrong shapes. Rebuild and repack the carried set at this window (`tools/native/prebuilt/rebuild.sh`), or build with
/// `compile-carried`.
#[cfg(not(feature = "compile-carried"))]
const _: () = assert!(
    crate::ops_dsl_attention::window::SLIDING_WINDOW == CARRIED_WINDOW,
    "the carried images were built at another Sliding window (CARRIED_WINDOW): rebuild the carried set"
);

/// Defines the definition paths, registry paths and launch dispatch in payload order.
/// `stock` builds in the reference server, `knob` builds separately, and `pinned`
/// also uses the carried row at typed reference launches. `$` quotes macro variables.
macro_rules! carried_set {
    (@separate stock) => { false };
    (@separate knob) => { true };
    (@separate pinned) => { true };
    ($d:tt $($mode:ident $($seg:ident)::+),* $(,)?) => {
        /// Definition paths of every carried kernel, in payload order.
        pub const CARRIED: &[&str] = &[$(concat!("furiosa_opt_gemma4" $(, "::", stringify!($seg))+)),*];

        /// The registry path of each carried kernel's image ([`PATH_PREFIX`] + its definition path
        /// after the crate name), index for index with [`CARRIED`].
        pub const ROWS: &[&str] = &[$(concat!("furiosa_opt_gemma4::prebuilt" $(, "::", stringify!($seg))+)),*];

        const REFERENCE_ROWS: &[bool] = &[$(carried_set!(@separate $mode)),*];

        /// `launch_kernel!(how, device, path::to::device_fn, (args...))`: a launch of a kernel (a
        /// concrete `#[device]` fn returning `()`, its path relative to the crate root), from its
        /// carried image for a carried kernel in the server, else of the `#[device]` fn (module
        /// docs). Use `queued` for FIFO dispatch or `plain` for an awaited launch.
        macro_rules! launch_kernel {
            $( ($d how:ident, $d device:tt, $($seg)::+, $d args:tt $d(,)?) => {
                $crate::host::prebuilt::carried_kernel!($mode, $d how, $d device, ($($seg)::+), $d args)
            }; )*
            ($d how:ident, $d device:tt, $d($d p:ident)::+, $d args:tt $d(,)?) => {
                $crate::host::prebuilt::compiled_kernel!($d how, $d device, ($d($d p)::+), $d args)
            };
        }
        pub(crate) use launch_kernel;
    };
}

carried_set!($
    knob ops_core::resident::rss_q4_class512_t128_g1,
    knob ops_core::resident::rss_q4_class512_t128_g2,
    knob ops_core::resident::rss_q4_class512_t128_g4,
    knob ops_core::resident::rss_q4_class512_t128_g8,
    knob ops_core::resident::rss_q4_class512_t128_g16,
    knob ops_core::resident_fs::rfs_q4_class512_t128_g1,
    knob ops_core::resident_fs::rfs_q4_class512_t128_g2,
    knob ops_core::resident_fs::rfs_q4_class512_t128_g4,
    knob ops_core::resident_fs::rfs_q4_class512_t128_g8,
    knob ops_core::resident_fs::rfs_q4_class512_t128_g16,
    pinned ops_core::cohort_q4::q4::mid_ss_q4_t1024,
    pinned ops_core::cohort_q4::q4::mid_fs_q4_t1024,
    stock ops_core::ends::last_bf16_out128,
    stock ops_audio_rows::audio_rows,
    stock ops_vision_rows::vision_rows,
    stock ops_core::spm_clean::spm_clean,
    pinned ops_core::full_resident::rsf_int8_fp8kv_t1024_s8,
    stock ops_core::pre_int8_fp8kv_t1536,
    stock ops_core::full_i8::mid_sf_int8_fp8kv_t1536,
    stock ops_core::q4serve::mid_ss_t1536,
    stock ops_core::q4serve::mid_fs_t1536,
    stock ops_step::embed_rows_t1536,
    stock ops_step::embed_media_t1536,
    stock ops_core::full_resident::rsf_int8_fp8kv_t1536_s8,
    stock ops_sampler::feedback::resolve_t1536_three512,
    stock ops_step::wide_output::compact1536_out128,
    knob ops_core::resident::q4::rss_int8_fp8kv_t1536_g12_p999_pair,
    knob ops_core::resident_fs::q4::rfs_int8_fp8kv_t1536_g12_p999_pair,
    knob ops_core::resident::q4::rss_int8_fp8kv_t1024_g12_p941_pair,
    knob ops_core::resident_fs::q4::rfs_int8_fp8kv_t1024_g12_p941_pair,
);

/// `queued_kernel!(device, path::to::device_fn, (args...))`: `launch_kernel!(queued,...)`, the
/// form of `engine::queue::queued!`.
macro_rules! queued_kernel {
    ($($t:tt)*) => {
        $crate::host::prebuilt::launch_kernel!(queued, $($t)*)
    };
}
pub(crate) use queued_kernel;

/// `launch_kernel!`'s expansion for the `#[device]` fn (a kernel outside the carried set, or a build
/// that compiles the carried ones).
macro_rules! compiled_kernel {
    (queued, $device:tt, ($($seg:ident)::+), ($($arg:expr),* $(,)?)) => {
        $crate::host::engine::queue::queued!($device, $crate::$($seg)::+, ($($arg),*))
    };
    (plain, $device:tt, ($($seg:ident)::+), ($($arg:expr),* $(,)?)) => {
        ::furiosa_opt_std::runtime::launch($crate::$($seg)::+, (&mut *$device, $($arg,)*)).await
    };
}
pub(crate) use compiled_kernel;

/// Select the typed launch for reference builds except at pinned rows.
#[cfg(not(feature = "compile-carried"))]
macro_rules! carried_kernel {
    ($mode:ident, $how:ident, $device:tt, ($($seg:ident)::+), $args:tt) => {
        $crate::host::prebuilt::carried_launch!($how, $device, ($($seg)::+), $args)
    };
}
#[cfg(feature = "compile-carried")]
macro_rules! carried_kernel {
    (pinned, $how:ident, $device:tt, ($($seg:ident)::+), $args:tt) => {
        $crate::host::prebuilt::carried_launch!($how, $device, ($($seg)::+), $args)
    };
    ($mode:ident, $how:ident, $device:tt, ($($seg:ident)::+), $args:tt) => {
        $crate::host::prebuilt::compiled_kernel!($how, $device, ($($seg)::+), $args)
    };
}
pub(crate) use carried_kernel;

/// The carried launch of `launch_kernel!` (NPU only), each `how` as its compiled form: `queued`
/// through the launch queue when enabled and `plain` through [`run`].
macro_rules! carried_launch {
    (queued, $device:tt, ($($seg:ident)::+), ($($arg:expr),* $(,)?)) => {{
        const I: usize = $crate::host::prebuilt::index(concat!("furiosa_opt_gemma4" $(, "::", stringify!($seg))+));
        if $crate::host::engine::queue::queueing() {
            $crate::host::engine::queue::launch(&*$device, $crate::host::prebuilt::row::<I>(), &[], ($($arg,)*)).await
        } else {
            $crate::host::prebuilt::run::<I, (), _>((&mut *$device, $($arg,)*)).await
        }
    }};
    (plain, $device:tt, ($($seg:ident)::+), ($($arg:expr),* $(,)?)) => {{
        const I: usize = $crate::host::prebuilt::index(concat!("furiosa_opt_gemma4" $(, "::", stringify!($seg))+));
        $crate::host::prebuilt::run::<I, (), _>((&mut *$device, $($arg,)*)).await
    }};
}
pub(crate) use carried_launch;

/// Whether an NPU build uses the shipped images for all carried kernels.
pub const CARRIES: bool = cfg!(not(feature = "compile-carried"));
/// A `compile-carried` reference build still loads the knob-carried images, including Q4.
/// These images were compiled with per-kernel scheduling settings.
pub const REFERENCE_KNOBS: bool = cfg!(feature = "compile-carried");

/// The index of definition path `path` in [`CARRIED`]; a compile error (const panic) if it is not
/// carried.
pub const fn index(path: &str) -> usize {
    let mut i = 0;
    while i < CARRIED.len() {
        if eq(CARRIED[i].as_bytes(), path.as_bytes()) {
            return i;
        }
        i += 1;
    }
    panic!("not a carried kernel (host::prebuilt::carried_set)")
}

const fn eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// The registry path of carried kernel `I`'s image ([`ROWS`]).
pub const fn row<const I: usize>() -> &'static str {
    ROWS[I]
}

/// The registry path a raw launch of kernel `path` (its definition path) loads (`engine::fused8`'s
/// programs): its carried row where it is carried and this engine takes the carried images
/// ([`CARRIES`]), else `path` itself (the compiled fn, which a typed
/// site of the same kernel makes the build compile).
pub fn launch_path(path: &'static str) -> &'static str {
    let Some(i) = CARRIED.iter().position(|c| *c == path) else { return path };
    if CARRIES || (REFERENCE_KNOBS && REFERENCE_ROWS[i]) { ROWS[i] } else { path }
}

/// An argument tuple whose first element is the launch's `&mut Device`, as every `#[device]` fn's.
pub trait LeadingDevice {
    fn device(&self) -> &Device;
}

macro_rules! leading_device {
    ($($t:ident),*) => {
        impl<'d, $($t),*> LeadingDevice for (&'d mut Device, $($t,)*) {
            fn device(&self) -> &Device {
                &*self.0
            }
        }
    };
}

leading_device!();
leading_device!(T1);
leading_device!(T1, T2);
leading_device!(T1, T2, T3);
leading_device!(T1, T2, T3, T4);
leading_device!(T1, T2, T3, T4, T5);
leading_device!(T1, T2, T3, T4, T5, T6);
leading_device!(T1, T2, T3, T4, T5, T6, T7);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22);
leading_device!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23);

/// Launches carried kernel `I` (index into [`CARRIED`] / [`ROWS`]) from its installed image with
/// `args`, the `#[device]` fn's argument tuple (`&mut Device` first), returning `O`, the fn's return
/// type (inferred at the call site): the `#[device]` launch's statements with the carried path.
pub async fn run<const I: usize, O: DeviceOutput, A: DeviceSend + LeadingDevice>(args: A) -> Result<O, Error> {
    {
        let function = furiosa_opt_std::backend::npu::function(args.device(), ROWS[I], &[]).await?;
        let mut inputs = Buffers::new();
        DeviceSend::bind(&args, &mut inputs)?;
        let mut outputs = Buffers::new();
        <O as DeviceOutput>::alloc_into(&function, &mut outputs)?;
        function.run(&inputs, &outputs).await?;
        Ok(<O as DeviceOutput>::take(&mut outputs.into_iter()))
    }
}

/// The carried rows of [`ROWS`] missing from `installed` (registry paths).
fn missing_rows(installed: &[String]) -> Vec<&'static str> {
    ROWS.iter().copied().filter(|r| !installed.iter().any(|p| p == r)).collect()
}

/// The carried rows the installer put in the registry, against [`ROWS`]: an error names every
/// carried kernel the payload lacks (a site launching it would fail at its first launch). The engine
/// calls it at start, so a payload out of step with the source fails before serving.
pub fn check_installed() -> Result<(), String> {
    if !CARRIES && !REFERENCE_KNOBS {
        return Ok(());
    }
    let installed = crate::host::registry::install().as_ref().map_err(|why| format!("prebuilt: not installed: {why}"))?;
    let missing = missing_rows(&installed.prebuilt);
    if missing.is_empty() { Ok(()) } else { Err(format!("prebuilt: the payload lacks carried kernels {missing:?}")) }
}
