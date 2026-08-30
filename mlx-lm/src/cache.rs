use mlx_rs::{error::Exception, ops::concatenate, Array};

// TODO: somehow move quantized methods to a separate trait?
pub trait KeyValueCache {
    fn is_quantized(&self) -> bool {
        false
    }

    /// Returns the group size used for quantization. `None` if not quantized.
    fn group_size(&self) -> Option<i32> {
        None
    }

    /// Returns the number of bits used for quantization. `None` if not quantized.
    fn bits(&self) -> Option<i32> {
        None
    }

    fn offset(&self) -> i32;

    fn max_size(&self) -> Option<i32>;

    fn update_and_fetch(&mut self, keys: Array, values: Array)
        -> Result<(Array, Array), Exception>;
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl ConcatKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        match (self.keys.take(), self.values.take()) {
            (Some(k), Some(v)) => {
                self.keys = Some(concatenate(&[k, keys], -2)?);
                self.values = Some(concatenate(&[v, values], -2)?);
            }
            _ => {
                self.keys = Some(keys);
                self.values = Some(values);
            }
        }
        let shape = self.keys.as_ref().expect("Keys cannot be None").shape();
        self.offset = shape[shape.len() - 2];

        Ok((
            self.keys.clone().expect("Keys cannot be None"),
            self.values.clone().expect("Values cannot be None"),
        ))
    }
}

/// TODO: A generic KV Cache
pub struct DefaultKeyValueCache {}
