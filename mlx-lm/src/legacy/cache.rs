use mlx_rs::{error::Exception, Array};
pub trait KeyValueCache {
    fn offset(&self) -> i32;
    fn max_size(&self) -> Option<i32>;
    fn update_and_fetch(&mut self, keys: Array, values: Array)
        -> Result<(Array, Array), Exception>;
}
