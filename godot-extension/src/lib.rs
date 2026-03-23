//! whisper_cpp_gdext — GDExtension bindings for whisper.cpp
//!
//! Exposes a `WhisperCpp` class to Godot 4 that wraps whisper-rs for
//! on-device speech recognition.
//!
//! # Threading Model
//! Inference runs on a dedicated background thread.  Results are sent back
//! to the main thread via `Gd::call_deferred` to keep Godot's scene tree
//! safe.

use godot::prelude::*;
use godot::classes::Node;
use std::sync::{Arc, Mutex};
use std::thread;

mod whisper_node;

use whisper_node::WhisperCpp;

/// Entry-point required by Godot's GDExtension loader.
struct WhisperCppExtension;

#[gdextension]
unsafe impl ExtensionLibrary for WhisperCppExtension {}
