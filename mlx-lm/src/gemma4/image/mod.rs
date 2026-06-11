//! Gemma 4 vision (SigLIP) tower. Active behind the `image` feature. The
//! multimodal glue that binds this tower to the text model lives in
//! `crate::gemma4::vlm`.

pub mod config;
pub mod multimodal;
pub mod processor;
pub mod vision;
