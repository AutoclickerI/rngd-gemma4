//! Physical dense-row permutation. Kept dependency-free for host-only verification.

#[derive(Clone, Copy, Debug)]
pub struct RowSpan {
    pub rows: usize,
    pub emits: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowOrder {
    /// Includes the logical padding rows, so every table is permuted bijectively.
    pub logical_to_physical: Vec<usize>,
    pub physical_to_logical: Vec<usize>,
    /// A row-role body may run only when every consumed logit is in the final 128 rows.
    /// More than 128 emitters retain the exact original layout instead.
    pub emitting_in_tail: bool,
}

impl RowOrder {
    pub fn of(spans: &[RowSpan], bucket: usize) -> Result<Self, &'static str> {
        if spans.iter().any(|s| s.rows == 0) { return Err("a span has at least one row"); }
        let live = spans.iter().try_fold(0usize, |n, s| n.checked_add(s.rows)).ok_or("step row count overflow")?;
        if live > bucket { return Err("step rows exceed their bucket"); }
        let emitting = spans.iter().filter(|s| s.emits).count();
        let mut logical_to_physical: Vec<usize> = (0..bucket).collect();
        if emitting > 0 && emitting <= 128 {
            let (mut source, mut body, mut tail) = (0, 0, bucket - emitting);
            for span in spans {
                for j in 0..span.rows {
                    let target = if span.emits && j + 1 == span.rows {
                        let target = tail;
                        tail += 1;
                        target
                    } else {
                        let target = body;
                        body += 1;
                        target
                    };
                    logical_to_physical[source] = target;
                    source += 1;
                }
            }
            // Put padding between KV-only rows and the emitting tail, including underfilled buckets.
            for target in &mut logical_to_physical[live..] { *target = body; body += 1; }
        }
        let mut physical_to_logical = vec![0; bucket];
        for (logical, &physical) in logical_to_physical.iter().enumerate() {
            physical_to_logical[physical] = logical;
        }
        Ok(Self { logical_to_physical, physical_to_logical, emitting_in_tail: emitting <= 128 })
    }

    /// `logical` covers the whole bucket, padding included.
    pub fn physical<T: Clone>(&self, logical: &[T]) -> Vec<T> {
        assert_eq!(logical.len(), self.physical_to_logical.len());
        self.physical_to_logical.iter().map(|&i| logical[i].clone()).collect()
    }
}
