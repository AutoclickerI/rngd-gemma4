//! Keep an RSF frame until the immediately following Full-attention batch.
//! Composed frames run the RSF member before their Full-attention bodies.
//! Any intervening launch or host drain flushes RSF normally.

mod live {
    use super::super::{ACTIVE, LOADED, Function, table, full_sequence::Frame, persist::Launch};
    use furiosa_opt_std::runtime::Buffers;
    use std::{cell::RefCell, sync::Arc};

    thread_local! { static PENDING: RefCell<Option<(usize, Frame)>> = const { RefCell::new(None) }; }

    pub(crate) fn accepts(function: &Arc<Function>, inputs: &Buffers, outputs: &Buffers) -> bool {
        ACTIVE.get().is_some_and(|f| outputs.is_empty()
            && LOADED.with_borrow(|l| l.as_ref().is_some_and(|l| {
                let family = &table::FAMILIES[f];
                l.member_of.get(&(Arc::as_ptr(function) as usize)).is_some_and(|&m| {
                    l.opcodes[f].get(&m).is_some_and(|&op| {
                        op == 1
                            && inputs.len() == table::MEMBERS[m].inputs
                            && super::super::frame_fits(family, op, &table::MEMBERS[m], inputs, outputs)
                    })
                })
            })))
    }

    pub(crate) fn hold(frame: Frame) {
        assert!(accepts(&frame.0, &frame.1, &frame.2));
        PENDING.with_borrow_mut(|p| {
            assert!(p.is_none(), "flush previous RSF before holding another");
            *p = Some((ACTIVE.get().expect("RSF family"), frame));
        });
    }

    pub(crate) fn clear() { PENDING.with_borrow_mut(|p| *p = None); }
    pub(crate) fn empty() -> bool { PENDING.with_borrow(Option::is_none) }

    pub(crate) fn flush() -> Vec<Launch> {
        PENDING.with_borrow_mut(Option::take).map_or_else(Vec::new, |(f, (function, inputs, outputs))| {
            assert_eq!(ACTIVE.get(), Some(f), "RSF crossed step-family boundary");
            let (function, inputs, outputs) = super::super::route(function, inputs, outputs);
            vec![Launch { function, inputs, outputs, owners: Vec::new() }]
        })
    }

    pub(crate) fn compose(frames: &[Frame]) -> Option<Frame> {
        let f = ACTIVE.get()?;
        if frames.is_empty() || frames.len() > 4 { return None; }
        let result = PENDING.with_borrow(|p| {
            let (pending_f, (original, rsf, _)) = p.as_ref()?;
            if *pending_f != f { return None; }
            LOADED.with_borrow(|loaded| {
                let l = loaded.as_ref()?;
                let family = &table::FAMILIES[f];
                let m = *l.member_of.get(&(Arc::as_ptr(original) as usize))?;
                let offset = table::MEMBERS[m].inputs;
                if rsf.len() != offset || offset + 10 * frames.len() >= family.slots { return None; }
                let mut args = rsf.clone();
                for (function, inputs, outputs) in frames {
                    let m = *l.member_of.get(&(Arc::as_ptr(function) as usize))?;
                    let member = &table::MEMBERS[m];
                    if !(member.path.contains("::prefill_full::") || member.path.contains("::decode_full::"))
                        || inputs.len() != member.inputs || outputs.len() != 1
                        || member.outputs != [member.inputs] || member.inputs >= 9 { return None; }
                    let op = *l.opcodes[f].get(&m)?;
                    if !super::super::frame_fits(family, op, member, inputs, outputs) { return None; }
                    for slot in 0..9 {
                        let buffer = if slot < member.inputs { inputs[slot].clone() }
                            else if slot == member.inputs { outputs[0].clone() }
                            else { l.padding.clone() };
                        args.try_push(buffer).ok()?;
                    }
                    args.try_push(l.control.slice(8*op..8*op+8)).ok()?;
                }
                while args.len()+1 < family.slots { args.try_push(l.padding.clone()).ok()?; }
                let op = 40 + frames.len();
                args.try_push(l.control.slice(8*op..8*op+8)).ok()?;
                let function = super::super::selected_function(l, f)?.clone();
                Some((function, args, Buffers::new()))
            })
        });
        if result.is_some() {
            clear();
        }
        result
    }
}

pub(crate) use live::{accepts, hold, clear, empty, flush, compose};
