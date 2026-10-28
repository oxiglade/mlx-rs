#![allow(unexpected_cfgs)]

#[cfg(not(omit_external))]
pub use fixture_macros::*;

macro_rules! builder {
    () => {
        pub struct WidgetBuilder;
        impl WidgetBuilder {
            pub fn build(self) -> Widget { Widget }
        }
    };
}
#[cfg(not(omit_builder))]
builder!();

#[derive(Clone)]
pub struct Widget;

mod private {
    pub struct Original;
}
#[cfg(not(omit_reexport))]
pub use private::Original as Renamed;

#[doc(hidden)]
pub use private::Original as HiddenAlias;

pub fn visible_alias(_: private::Original) {}

pub fn r#type() {}

pub struct VisibleTuple(#[doc(hidden)] pub i32);

#[cfg(not(omit_hidden))]
#[doc(hidden)]
pub fn hidden() {}

#[cfg(all(feature = "extra", not(omit_gated)))]
pub fn gated() {}

fn private_function() {}

pub trait Inspect {
    fn inspect(&self);
}

impl<T: Inspect> Inspect for &mut T {
    fn inspect(&self) { (**self).inspect(); }
}


pub type WidgetAlias = Widget;

pub mod visible_name {
    pub struct Same;
}

#[doc(hidden)]
pub mod hidden_name {
    pub struct Same;
}


#[derive(Clone)]
pub struct Generic<T>(pub T);
pub type SpecializedAlias<'a> = Generic<&'a mut Widget>;

pub struct SendProbe {
    #[cfg(change_send)]
    _marker: std::rc::Rc<()>,
}

pub struct SyncProbe {
    #[cfg(change_sync)]
    _marker: std::cell::Cell<()>,
}
