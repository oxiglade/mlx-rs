use super::*;

#[derive(Clone)]
pub(crate) struct Recurrent {
    spec: LayerCacheSpec,
    offset: usize,
    state: (Array, Array),
    pending: Vec<(usize, (Array, Array))>,
}

impl Recurrent {
    pub(super) fn new(spec: &LayerCacheSpec) -> Result<Self, CacheError> {
        Ok(Self {
            spec: spec.clone(),
            offset: 0,
            pending: Vec::new(),
            state: (
                zeros_dtype(&[1, 3, 8192], spec.dtype)?,
                zeros_dtype(&[32, 128, 128], Dtype::Float32)?,
            ),
        })
    }
    pub(super) fn advance(&mut self, next: (Array, Array), count: usize) -> Result<(), CacheError> {
        self.offset = self
            .offset
            .checked_add(count)
            .ok_or_else(|| invalid("position overflow"))?;
        self.state = next;
        self.pending.clear();
        Ok(())
    }
    pub(super) fn submitted(&mut self, staged: &Self) {
        self.pending.retain(|(end, _)| *end < staged.offset);
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
