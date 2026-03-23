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

mod whisper_node;

/// Entry-point required by Godot's GDExtension loader.
struct WhisperCppExtension;

#[gdextension]
unsafe impl ExtensionLibrary for WhisperCppExtension {}
