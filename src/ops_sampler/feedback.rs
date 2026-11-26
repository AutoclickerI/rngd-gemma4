//! Device token handoff. Tables use 32-byte rows for legal indirect DMA; only lane zero is live.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::*};
use crate::device::layout::{Cluster, Slice};

/// Publish the exact picks into request-owner slots. Nonparticipating rows write distinct scratch slots.
#[device(chip = 1)]
pub fn publish(device: &mut Device,
    picks: &HbmTensor<i32, Chip, m![Out128]>,
    owners: &HbmTensor<i32, Chip, m![Out128]>,
    table: &mut HbmTensor<i32, Chip, m![Tok256, Dummy8]>,
) {
    let x: DmTensor<i32, Chip, Cluster, Slice, m![Out128]> = picks.to_dm(&mut device.tdma);
    // Expand a dense scalar per row into a padded DMA row; no arithmetic conversion.
    let padded: DmTensor<i32, Chip, Cluster, Slice, m![Out128, 1 # 8]> = device.main.begin(x.view())
        .fetch::<m![Out128 / 8], m![Out128 % 8]>().collect::<m![Out128 / 8], m![Out128 % 8]>()
        .transpose::<m![Out128], m![1 # 8]>().commit_trim::<m![1 # 8]>().commit();
    let padded: DmTensor<i32, Chip, Cluster, Slice, m![Out128, Dummy8]> = unsafe { padded.reshape() };
    table.view_mut().scatter::<m![Tok256], m![Dummy8]>().by_byte_offsets(owners.view())
        .from_dm_view(&mut device.tdma, padded.view());
}

/// Each literal is a token id or -1 for a device pick. `sources` points at its owner slot in `previous`.
/// The resulting byte offsets are the exact buffer embed_rows reads; the host never supplies a feedback token.
fn resolve_body<T: AxisName, const P: usize>(device: &mut Device,
    previous: &HbmTensor<i32, Chip, m![Tok256, Dummy8]>,
    sources: &HbmTensor<i32, Chip, m![T]>,
    literals: &HbmTensor<i32, Chip, m![T, Dummy8]>,
    out: &mut HbmTensor<i32, Chip, m![T]>,
) {
    use BitReq::{Ignore, One};
    type C<T, const P: usize> = m![T / {P} # 2];
    type S<T, const P: usize> = m![T % {P} / 2 # 256];
    let picked: DmTensor<i32, Chip, C<T, P>, S<T, P>, m![T % 2, Dummy8]> = previous.gather::<m![Tok256], m![Dummy8]>()
        .by_byte_offsets(sources).to_dm(&mut device.tdma);
    let picked: VrfTensor<i32, Chip, C<T, P>, S<T, P>, m![T % 2, 1 # 8]> = device.sub
        .begin(picked.view().tile::<m![Dummy8], 1, m![T % 2, 1 # 8]>(0))
        .fetch::<m![T % 2], m![1 # 8]>().collect::<m![T % 2], m![1 # 8]>().to_vrf();
    let literals: DmTensor<i32, Chip, C<T, P>, S<T, P>, m![T % 2, Dummy8]> = literals.to_dm(&mut device.tdma);
    let offsets: DmTensor<i32, Chip, C<T, P>, S<T, P>, m![T % 2]> = device.main
        .begin(literals.view().tile::<m![Dummy8], 1, m![T % 2, 1 # 8]>(0))
        .fetch::<m![T % 2], m![1 # 8]>().collect::<m![T % 2], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Comparison([Cmp::Less(0), Cmp::False, Cmp::False, Cmp::False]))
        .vector_logic(LogicBinaryOpI32::BitAnd, Branched::imm(TagGuard::matches([One, Ignore, Ignore, Ignore]), 0))
        .vector_logic(LogicBinaryOpI32::BitOr, Branched::rf(TagGuard::matches([One, Ignore, Ignore, Ignore]), &picked))
        .vector_fxp(FxpBinaryOp::MulInt, (H::SIZE * 2) as i32)
        .vector_final().transpose::<m![1], m![T % 2 # 8]>().commit_trim::<m![T % 2]>().commit();
    offsets.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

macro_rules! resolver {
    ($name:ident, $t:ident, $p:literal) => {
        #[device(chip = 1)]
        pub fn $name(device: &mut Device, previous: &HbmTensor<i32, Chip, m![Tok256, Dummy8]>, sources: &HbmTensor<i32, Chip, m![$t]>,
            literals: &HbmTensor<i32, Chip, m![$t, Dummy8]>, out: &mut HbmTensor<i32, Chip, m![$t]>) {
            resolve_body::<$t, $p>(device, previous, sources, literals, out);
        }
    }
}
resolver!(resolve_t128, Tok128, 128);
resolver!(resolve_t1024, Tok1024, 512);

// A 1536-row resolver cannot use T/P=3 on a two-cluster grid. Keep the
// established 512-row placement and process three contiguous slabs in one
// device task. The small source-index copy is explicit: GatherByteOffsets in
// Opt0.9 accepts an owned HbmTensor, rather than an HbmTensorView.
#[cfg(feature = "compile-carried")]
type Slab1536 = m![Tok1536 / 512 = 1 # 3, Tok1536 % 512];
#[cfg(feature = "compile-carried")]
type Slab1536Literals = m![Tok1536 / 512 = 1 # 3, Tok1536 % 512, Dummy8];
#[cfg(feature = "compile-carried")]
type Slab1536Out = m![Tok1536 / 512 = 1 #{!} 3, Tok1536 % 512];

#[cfg(feature = "compile-carried")]
fn resolve_slab1536(
    device: &mut Device,
    previous: &HbmTensor<i32, Chip, m![Tok256, Dummy8]>,
    sources: HbmTensorView<'_, i32, Chip, Slab1536>,
    literals: HbmTensorView<'_, i32, Chip, Slab1536Literals>,
    out: HbmTensorViewMut<'_, i32, Chip, Slab1536Out>,
) {
    use BitReq::{Ignore, One};
    type C = m![1 # 2];
    type S = m![Tok1536 % 512 / 2 # 256];
    let source_dm: DmTensor<i32, Chip, C, S, m![Tok1536 % 2]> = sources.to_dm(&mut device.tdma);
    let source_hbm: HbmTensor<i32, Chip, m![Tok1536 % 512]> = source_dm.to_hbm(&mut device.tdma);
    let picked: DmTensor<i32, Chip, C, S, m![Tok1536 % 2, Dummy8]> = previous
        .gather::<m![Tok256], m![Dummy8]>()
        .by_byte_offsets(&source_hbm).to_dm(&mut device.tdma);
    let picked: VrfTensor<i32, Chip, C, S, m![Tok1536 % 2, 1 # 8]> = device.sub
        .begin(picked.view().tile::<m![Dummy8], 1, m![Tok1536 % 2, 1 # 8]>(0))
        .fetch::<m![Tok1536 % 2], m![1 # 8]>()
        .collect::<m![Tok1536 % 2], m![1 # 8]>().to_vrf();
    let literals: DmTensor<i32, Chip, C, S, m![Tok1536 % 2, Dummy8]> = literals.to_dm(&mut device.tdma);
    let offsets: DmTensor<i32, Chip, C, S, m![Tok1536 % 2]> = device.main
        .begin(literals.view().tile::<m![Dummy8], 1, m![Tok1536 % 2, 1 # 8]>(0))
        .fetch::<m![Tok1536 % 2], m![1 # 8]>()
        .collect::<m![Tok1536 % 2], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Comparison([Cmp::Less(0), Cmp::False, Cmp::False, Cmp::False]))
        .vector_logic(LogicBinaryOpI32::BitAnd, Branched::imm(TagGuard::matches([One, Ignore, Ignore, Ignore]), 0))
        .vector_logic(LogicBinaryOpI32::BitOr, Branched::rf(TagGuard::matches([One, Ignore, Ignore, Ignore]), &picked))
        .vector_fxp(FxpBinaryOp::MulInt, (H::SIZE * 2) as i32)
        .vector_final().transpose::<m![1], m![Tok1536 % 2 # 8]>()
        .commit_trim::<m![Tok1536 % 2]>().commit();
    offsets.view().to_hbm_view(&mut device.tdma, out);
}

#[cfg(feature = "compile-carried")]
#[device(chip = 1)]
pub fn resolve_t1536_three512(
    device: &mut Device,
    previous: &HbmTensor<i32, Chip, m![Tok256, Dummy8]>,
    sources: &HbmTensor<i32, Chip, m![Tok1536]>,
    literals: &HbmTensor<i32, Chip, m![Tok1536, Dummy8]>,
    out: &mut HbmTensor<i32, Chip, m![Tok1536]>,
) {
    macro_rules! slab {
        ($pass:literal) => {
            resolve_slab1536(device, previous,
                sources.view().tile::<m![Tok1536 / 512], 1, Slab1536>($pass),
                literals.view().tile::<m![Tok1536 / 512], 1, Slab1536Literals>($pass),
                out.view_mut().tile::<m![Tok1536 / 512], 1, Slab1536Out>($pass));
        };
    }
    slab!(0);
    slab!(1);
    slab!(2);
}
