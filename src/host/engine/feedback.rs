//! Request-local feedback metadata: one packed write, two completion-fenced banks.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::*};
use super::{RowInput, queue};

fn fits<T: AxisName>(inputs: &[RowInput<'_>]) -> bool {
    T::SIZE <= Tok1536::SIZE && T::SIZE % Dummy8::SIZE == 0 && inputs.len() <= T::SIZE
}

/// Same padding, owner byte offsets and replicated literals as the serial path.
fn fill<T: AxisName>(words: &mut [i32], inputs: &[RowInput<'_>]) {
    assert!(fits::<T>(inputs));
    assert_eq!(words.len(), T::SIZE * (1 + Dummy8::SIZE));
    words.fill(0);
    let (sources, literals) = words.split_at_mut(T::SIZE);
    for (r, input) in inputs.iter().enumerate() {
        let (source, literal) = match input {
            RowInput::DeviceToken(owner) => ((*owner * Dummy8::SIZE * 4) as i32, -1),
            _ => (0, input.token() as i32),
        };
        sources[r] = source;
        literals[r * Dummy8::SIZE..(r + 1) * Dummy8::SIZE].fill(literal);
    }
}

pub(crate) struct Bank<T: AxisName> {
    packed: HbmTensor<i32, Chip, m![T, 9]>,
    pub sources: HbmTensor<i32, Chip, m![T]>,
    pub literals: HbmTensor<i32, Chip, m![T, Dummy8]>,
    host: Vec<i32>,
    complete: Option<queue::Fence>,
}

impl<T: AxisName> Bank<T> {
    async fn new(device: &mut Device) -> Result<Self, Error> {
        Ok(Self {
            packed: HostTensor::<i32, m![T, 9]>::zero().to_hbm(&mut device.pdma).await?,
            sources: HostTensor::<i32, m![T]>::zero().to_hbm(&mut device.pdma).await?,
            literals: HostTensor::<i32, m![T, Dummy8]>::zero().to_hbm(&mut device.pdma).await?,
            host: vec![0; T::SIZE * (1 + Dummy8::SIZE)], complete: None,
        })
    }

    async fn write(&mut self, device: &mut Device, inputs: &[RowInput<'_>]) -> Result<(), Error> {
        // Retire through this bank's preceding reader, leaving later work queued.
        if let Some(fence) = self.complete.take() { queue::drain_through(&fence).await?; }
        fill::<T>(&mut self.host, inputs);
        let host = HostTensor::<i32, m![T, 9]>::from_vec(std::mem::take(&mut self.host));
        let result = host.to_hbm(&mut device.pdma).output(&mut self.packed).await;
        self.host = host.into_vec();
        result?;
        {
            use furiosa_opt_std::runtime::DeviceSend;
            let mut buffers = furiosa_opt_std::runtime::Buffers::new();
            self.packed.bind(&mut buffers)?;
            let buffer = buffers.pop().expect("feedback frame binds one buffer");
            super::uploads::alias(&self.sources, buffer.slice(0..T::SIZE * 4))?;
            super::uploads::alias(&self.literals, buffer.slice(T::SIZE * 4..T::SIZE * 36))?;
        }
        // CPU has no region binding; materialize identical views for its oracle.
        Ok(())
    }

    pub fn submitted(&mut self) { self.complete = Some(queue::fence()); }
}

pub(crate) struct Metadata<T: AxisName> { banks: [Bank<T>; 2], next: usize }

impl<T: AxisName> Metadata<T> {
    pub fn fits(inputs: &[RowInput<'_>]) -> bool { fits::<T>(inputs) }
    pub async fn new(device: &mut Device) -> Result<Self, Error> {
        Ok(Self { banks: [Bank::new(device).await?, Bank::new(device).await?], next: 0 })
    }
    pub async fn prepare(&mut self, device: &mut Device, inputs: &[RowInput<'_>]) -> Result<&mut Bank<T>, Error> {
        let bank = self.next; self.next ^= 1;
        self.banks[bank].write(device, inputs).await?;
        Ok(&mut self.banks[bank])
    }
}
