//! Poll independent table uploads together. The runtime keeps DMA transfers in
//! flight concurrently, and writes of at most 2 KiB post in their first poll.

use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use furiosa_opt_std::prelude::*;

use crate::Chip;

/// One host image on its way into a device tensor.
pub type Write<'a> = Pin<Box<dyn Future<Output = Result<(), Error>> + 'a>>;

/// A device buffer (or a region of one), as a launch binds it.
pub type Buffer = <furiosa_opt_std::runtime::Buffers as IntoIterator>::Item;

thread_local! {
    /// Per table tensor whose bytes this step went out inside another upload: its buffer's address and size and the
    /// region holding the bytes (`alias`).
    static ALIASES: std::cell::RefCell<Vec<(usize, usize, Buffer)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The one buffer `tensor` binds.
fn buffer_of(tensor: &dyn furiosa_opt_std::runtime::DeviceSend) -> Result<Buffer, Error> {
    let mut bound = furiosa_opt_std::runtime::Buffers::new();
    tensor.bind(&mut bound)?;
    assert_eq!(bound.len(), 1, "a table tensor binds one buffer");
    Ok(bound.pop().unwrap())
}

/// Launches through the queue bind `region` for `tensor` from now on (`substitute`); `region` is its size.
pub fn alias(tensor: &dyn furiosa_opt_std::runtime::DeviceSend, region: Buffer) -> Result<(), Error> {
    let own = buffer_of(tensor)?;
    assert_eq!(own.size(), region.size(), "an alias region is its tensor's size");
    ALIASES.with_borrow_mut(|a| {
        a.retain(|(at, _, _)| *at != own.addr());
        a.push((own.addr(), own.size(), region));
    });
    Ok(())
}

/// `tensor` binds itself again (its bytes went out into it).
pub fn unalias(tensor: &dyn furiosa_opt_std::runtime::DeviceSend) -> Result<(), Error> {
    if ALIASES.with_borrow(|a| a.is_empty()) {
        return Ok(());
    }
    let own = buffer_of(tensor)?;
    ALIASES.with_borrow_mut(|a| a.retain(|(at, _, _)| *at != own.addr()));
    Ok(())
}

/// Replaces every bound buffer that is an aliased table tensor's by its region.
pub fn substitute(buffers: &mut furiosa_opt_std::runtime::Buffers) {
    ALIASES.with_borrow(|a| {
        if a.is_empty() {
            return;
        }
        for b in buffers.iter_mut() {
            if let Some((_, _, region)) = a.iter().find(|(at, size, _)| *at == b.addr() && *size == b.size()) {
                *b = region.clone();
            }
        }
    });
}

/// Forgets every alias (a new engine).
pub fn clear_aliases() {
    ALIASES.with_borrow_mut(|a| a.clear());
}

/// `host` written into the existing device tensor `into` (the backend's `to_hbm_into`, what `to_hbm(..).output(into)`
/// runs), as a future that owns the host image until the write lands.
pub fn write<'a, D: MaterializableScalar + 'a, E: M + 'a>(device: &'a Device, host: HostTensor<D, E>, into: &'a mut HbmTensor<D, Chip, E>) -> Write<'a> {
    Box::pin(async move { <furiosa_opt_std::backend::Npu as furiosa_opt_std::backend::Backend>::to_hbm_into(&host, &device.pdma, into).await })
}

/// Awaits every write, each polled from the start (so every transfer is issued before any is awaited). On a
/// failure the others still run to their end (a DMA reads its host image until it lands); the first error is
/// returned.
pub async fn join(mut writes: Vec<Write<'_>>) -> Result<(), Error> {
    let mut done = vec![false; writes.len()];
    let mut first: Option<Error> = None;
    std::future::poll_fn(|cx| {
        let mut pending = false;
        for (w, d) in writes.iter_mut().zip(done.iter_mut()) {
            if *d {
                continue;
            }
            match w.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    *d = true;
                    if let Err(why) = result {
                        first.get_or_insert(why);
                    }
                }
                Poll::Pending => pending = true,
            }
        }
        if pending { Poll::Pending } else { Poll::Ready(()) }
    })
    .await;
    match first {
        Some(why) => Err(why),
        None => Ok(()),
    }
}
