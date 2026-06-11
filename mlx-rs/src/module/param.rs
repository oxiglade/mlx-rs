use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use crate::{nested::NestedValue, Array};

use super::ModuleParameters;

/// Trait for a module parameter.
pub trait Parameter {
    /// Total number of parameters in this module/parameter.
    fn count(&self) -> usize;

    /// Freeze the parameter.
    fn freeze(&mut self, recursive: bool);

    /// Unfreeze the parameter.
    fn unfreeze(&mut self, recursive: bool);

    /// Check if the parameter is frozen. Returns `None` if the parameter is a module that has no
    /// parameters.
    fn is_frozen(&self) -> Option<bool>;

    /// Get the parameter as a nested value.
    fn as_nested_value(&self) -> NestedValue<Rc<str>, &Array>;

    /// Get the parameter as a mutable nested value.
    fn as_nested_value_mut(&mut self) -> NestedValue<Rc<str>, &mut Array>;

    /// Get the parameter as a nested value if it is trainable.
    fn as_trainable_nested_value(&self) -> Option<NestedValue<Rc<str>, &Array>>;
}

/// A simple wrapper for a module parameter.
#[derive(Debug, Clone)]
pub struct Param<T> {
    /// The value of the parameter.
    pub value: T,

    /// Whether the parameter is frozen.
    ///
    /// This is no longer public because it should be accessed through the `Parameter` trait.
    is_frozen: bool,
}

impl<T> Param<T> {
    /// Create a new `Param`
    pub fn new(value: T) -> Self {
        Self {
            value,
            is_frozen: false,
        }
    }
}

impl<T> From<T> for Param<T> {
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}

impl<T> Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for Param<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T> AsRef<T> for Param<T> {
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T> AsMut<T> for Param<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl Parameter for Param<Array> {
    fn count(&self) -> usize {
        1
    }

    fn freeze(&mut self, _recursive: bool) {
        self.is_frozen = true;
    }

    fn unfreeze(&mut self, _recursive: bool) {
        self.is_frozen = false;
    }

    fn is_frozen(&self) -> Option<bool> {
        Some(self.is_frozen)
    }

    fn as_nested_value<'a>(&self) -> NestedValue<Rc<str>, &Array> {
        NestedValue::Value(&self.value)
    }

    fn as_nested_value_mut<'a>(&mut self) -> NestedValue<Rc<str>, &mut Array> {
        NestedValue::Value(&mut self.value)
    }

    fn as_trainable_nested_value<'a>(&self) -> Option<NestedValue<Rc<str>, &Array>> {
        if self.is_frozen {
            None
        } else {
            Some(NestedValue::Value(&self.value))
        }
    }
}

impl Parameter for Param<Option<Array>> {
    fn count(&self) -> usize {
        self.value.as_ref().map_or(0, |_| 1)
    }

    fn freeze(&mut self, _recursive: bool) {
        self.is_frozen = true;
    }

    fn unfreeze(&mut self, _recursive: bool) {
        self.is_frozen = false;
    }

    fn is_frozen(&self) -> Option<bool> {
        Some(self.is_frozen)
    }

    fn as_nested_value(&self) -> NestedValue<Rc<str>, &Array> {
        match &self.value {
            Some(array) => NestedValue::Value(array),
            // An empty map entry will be ignored during flattening
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }

    fn as_nested_value_mut(&mut self) -> NestedValue<Rc<str>, &mut Array> {
        match &mut self.value {
            Some(array) => NestedValue::Value(array),
            // An empty map entry will be ignored during flattening
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }

    fn as_trainable_nested_value(&self) -> Option<NestedValue<Rc<str>, &Array>> {
        if self.is_frozen {
            None
        } else {
            self.value.as_ref().map(NestedValue::Value)
        }
    }
}

impl<M> Parameter for Option<M>
where
    M: ModuleParameters,
{
    fn count(&self) -> usize {
        self.as_ref().map_or(0, |m| m.count())
    }

    fn freeze(&mut self, recursive: bool) {
        if let Some(m) = self.as_mut() {
            m.freeze(recursive);
        }
    }

    fn unfreeze(&mut self, recursive: bool) {
        if let Some(m) = self.as_mut() {
            m.unfreeze(recursive);
        }
    }

    fn is_frozen(&self) -> Option<bool> {
        self.as_ref().and_then(|m| m.is_frozen())
    }

    fn as_nested_value(&self) -> NestedValue<Rc<str>, &Array> {
        match self {
            Some(m) => m.as_nested_value(),
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }

    fn as_nested_value_mut(&mut self) -> NestedValue<Rc<str>, &mut Array> {
        match self {
            Some(m) => m.as_nested_value_mut(),
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }

    fn as_trainable_nested_value(&self) -> Option<NestedValue<Rc<str>, &Array>> {
        match self {
            Some(m) => m.as_trainable_nested_value(),
            None => None,
        }
    }
}

impl<T> Parameter for T
where
    T: ModuleParameters,
{
    fn count(&self) -> usize {
        self.num_parameters()
    }

    fn freeze(&mut self, recursive: bool) {
        self.freeze_parameters(recursive);
    }

    fn unfreeze(&mut self, recursive: bool) {
        self.unfreeze_parameters(recursive);
    }

    fn is_frozen(&self) -> Option<bool> {
        self.all_frozen()
    }

    fn as_nested_value(&self) -> NestedValue<Rc<str>, &Array> {
        self.parameters().into()
    }

    fn as_nested_value_mut(&mut self) -> NestedValue<Rc<str>, &mut Array> {
        self.parameters_mut().into()
    }

    fn as_trainable_nested_value(&self) -> Option<NestedValue<Rc<str>, &Array>> {
        Some(self.trainable_parameters().into())
    }
}
