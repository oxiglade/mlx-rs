use super::*;

#[derive(Clone)]
pub(crate) struct Recurrent {
    spec: LayerCacheSpec,
    offset: usize,
    state: (Array, Array),
    /// Where the staged step began.
    start: usize,
    /// The states after the staged step's earlier rows, when it may be committed row by row.
    rows: Vec<(Array, Array)>,
    pending: Vec<(usize, (Array, Array))>,
}

impl Recurrent {
    /// A negative size in the layout's shape is an axis that holds the tokens seen, up to
    /// that many.
    pub(super) fn new(spec: &LayerCacheSpec) -> Result<Self, CacheError> {
        let [first, second] = spec
            .recurrent_state
            .as_ref()
            .ok_or_else(|| invalid("recurrent layer without state shapes"))?;
        let empty = |(shape, dtype): &(Vec<i32>, Dtype)| {
            let shape: Vec<_> = shape.iter().map(|&size| size.max(0)).collect();
            zeros_dtype(&shape, *dtype)
        };
        Ok(Self {
            spec: spec.clone(),
            offset: 0,
            start: 0,
            rows: Vec::new(),
            pending: Vec::new(),
            state: (empty(first)?, empty(second)?),
        })
    }
    pub(super) fn from_state(
        spec: &LayerCacheSpec,
        offset: usize,
        state: (Array, Array),
    ) -> Result<Self, CacheError> {
        let [first, second] = spec
            .recurrent_state
            .as_ref()
            .ok_or_else(|| invalid("recurrent layer without state shapes"))?;
        for (array, (shape, dtype)) in [(&state.0, first), (&state.1, second)] {
            let held = dimension(offset)?;
            let shaped = array.ndim() == shape.len()
                && shape.iter().zip(array.shape()).all(|(&expected, &actual)| {
                    actual
                        == if expected < 0 {
                            held.min(-expected)
                        } else {
                            expected
                        }
                });
            if !shaped || array.dtype() != *dtype {
                return Err(CacheError::FingerprintMismatch);
            }
        }
        Ok(Self {
            spec: spec.clone(),
            offset,
            state,
            start: offset,
            rows: Vec::new(),
            pending: Vec::new(),
        })
    }
    /// `states` are those after the step's last rows, in order, ending with the last row's.
    pub(super) fn advance(
        &mut self,
        mut states: Vec<(Array, Array)>,
        count: usize,
    ) -> Result<(), CacheError> {
        self.start = self.offset;
        self.offset = self
            .offset
            .checked_add(count)
            .ok_or_else(|| invalid("position overflow"))?;
        self.state = states
            .pop()
            .filter(|_| states.len() < count)
            .ok_or_else(|| {
                invalid("recurrent step needs its last state and at most one per row")
            })?;
        self.rows = states;
        self.pending.clear();
        Ok(())
    }
    pub(super) fn submitted(&mut self, staged: &Self) {
        self.pending.retain(|(end, _)| *end <= staged.start);
        let first = staged.offset - staged.rows.len();
        self.pending
            .extend((first..).zip(staged.rows.iter().cloned()));
        self.pending.push((staged.offset, staged.state.clone()));
    }
}

impl sealed::Sealed for Recurrent {}
impl LayerCache for Recurrent {
    fn recurrent_mut(&mut self) -> Option<&mut Self> {
        Some(self)
    }
    fn update_and_fetch(&mut self, _: Array, _: Array) -> Result<(Array, Array), CacheError> {
        Err(invalid("recurrent layers do not store K/V"))
    }
    fn info(&self, layer: usize) -> CacheInfo {
        CacheInfo {
            layer,
            kind: CacheKind::Recurrent,
            processed_tokens: self.offset,
            retained_prefix: 0..0,
            retained_positions: self.offset..self.offset,
            capacity: 0,
        }
    }
    fn arrays(&self) -> Result<(Array, Array), CacheError> {
        Ok(self.state.clone())
    }
    fn logical_arrays(&self) -> Result<(Array, Array), CacheError> {
        Err(invalid("recurrent layers do not store K/V"))
    }
    fn clone_box(&self) -> Box<dyn LayerCache> {
        Box::new(self.clone())
    }
    fn snapshot_box(&self) -> Result<Box<dyn LayerCache>, CacheError> {
        Ok(Box::new(Self {
            rows: Vec::new(),
            pending: Vec::new(),
            ..self.clone()
        }))
    }
    fn set_length(&mut self, length: usize) -> Result<(), CacheError> {
        if length != self.offset {
            self.state = self
                .pending
                .iter()
                .find(|(end, _)| *end == length)
                .ok_or_else(|| invalid("recurrent state was not submitted"))?
                .1
                .clone();
            self.offset = length;
            self.pending.retain(|(end, _)| *end > length);
        }
        Ok(())
    }
    fn fingerprint(&self) -> LayerFingerprint {
        LayerFingerprint {
            spec: self.spec.clone(),
            kind: CacheKind::Recurrent,
            capacity: 0,
            keep_prefix: 0,
            processed_tokens: self.offset,
            length: 0,
            index: 0,
            keys_shape: self.state.0.shape().to_vec(),
            values_shape: self.state.1.shape().to_vec(),
            keys_dtype: self.state.0.dtype(),
            values_dtype: self.state.1.dtype(),
        }
    }
}
