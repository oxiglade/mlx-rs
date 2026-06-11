//! Gemma 4 multimodal glue: the single VLM adapter + processor + weight loader
//! that bind the text model to the optional vision (and, behind the `audio`
//! feature, audio) towers. Active behind the `image` feature.

pub mod adapter;
pub mod weights;
