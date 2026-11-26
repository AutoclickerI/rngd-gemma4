//! One ordered launch for a layer's standalone Full-attention bodies.
//! Original input/output buffers remain owned by the composed launch.

pub(crate) fn enabled() -> bool {
    super::ACTIVE.get().is_some_and(|f| {
        let family = &super::table::FAMILIES[f];
        matches!(family.bucket, 1024 | 1536)
    })
}

pub(crate) type Frame = (
    std::sync::Arc<super::Function>,
    furiosa_opt_std::runtime::Buffers,
    furiosa_opt_std::runtime::Buffers,
);

pub(crate) fn compose(frames: &[Frame]) -> Option<Frame> {
    use furiosa_opt_std::runtime::Buffers;
    if !enabled() { return None; }
    if let Some(composed) = super::launch_boundary::compose(frames) { return Some(composed); }
    let f = super::ACTIVE.get()?;
    super::LOADED.with_borrow(|loaded| {
        let l = loaded.as_ref()?;
        let family = &super::table::FAMILIES[f];
        if frames.len() < 2 || frames.len() > 10 || frames.len() * 10 >= family.slots { return None; }
        let mut args = Buffers::new();
        for (function, inputs, outputs) in frames {
            let m = *l.member_of.get(&(std::sync::Arc::as_ptr(function) as usize))?;
            let member = &super::table::MEMBERS[m];
            if !(member.path.contains("::prefill_full::") || member.path.contains("::decode_full::"))
                || inputs.len() != member.inputs || outputs.len() != 1 || member.outputs.len() != 1
                || member.outputs[0] != member.inputs || member.inputs + 1 > 9 { return None; }
            let op = *l.opcodes[f].get(&m)?;
            if !super::frame_fits(family, op, member, inputs, outputs) {
                return None;
            }
            for slot in 0..9 {
                let buffer = if slot < member.inputs { inputs[slot].clone() }
                    else if slot == member.inputs { outputs[0].clone() }
                    else { l.padding.clone() };
                args.try_push(buffer).ok()?;
            }
            args.try_push(l.control.slice(8 * op..8 * op + 8)).ok()?;
        }
        while args.len() + 1 < family.slots { args.try_push(l.padding.clone()).ok()?; }
        let op = 30 + frames.len();
        args.try_push(l.control.slice(8 * op..8 * op + 8)).ok()?;
        let function = super::selected_function(l, f)?.clone();
        Some((function, args, Buffers::new()))
    })
}
