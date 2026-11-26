//! Exact residual compaction into the existing W4 head's Tok1024 input ABI.
//! Only the requested emitting rows cross HBM; the head receives compact row
//! indices. Its O/FFN/norm/head arithmetic and image bytes stay unchanged.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::*};

#[device(chip=1)]
pub fn compact1536_out128(
    device:&mut Device,
    source:&HbmTensor<bf16,Chip,m![Tok1536,H]>,
    rows:&HbmTensor<i32,Chip,m![Out128]>,
    out:&mut HbmTensor<bf16,Chip,m![Tok1024,H]>,
) {
    type C=m![1 # 2];
    type S=m![Out128 / 2 # 256];
    let idx:DmTensor<i32,Chip,C,S,m![Out128 % 2]>=rows.to_dm(&mut device.tdma);
    let idx:SpmTensor<i32,Chip,C,m![1 # 4],m![Out128]>=idx.to_spm(&mut device.tdma);
    let selected:DmTensor<bf16,Chip,C,S,m![Out128 % 2,H]>=source
        .gather::<m![Tok1536],m![H]>().by_positions::<m![Out128]>(&idx).to_dm(&mut device.tdma);
    let compact:HbmTensor<bf16,Chip,m![Out128,H]>=selected.to_hbm(&mut device.tdma);
    // Materialize before changing the temporary's axis, never reshape a view.
    let compact:HbmTensor<bf16,Chip,m![Tok1024=128,H]>=unsafe{compact.reshape()};
    compact.view().to_hbm_view(&mut device.tdma,
        out.view_mut().tile::<m![Tok1024],128,m![Tok1024=128 #{!}1024,H]>(0));
}
