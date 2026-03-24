//! WhisperCpp — Godot Node wrapping whisper-rs for speech recognition.
//!
//! ## Usage (GDScript)
//! ```gdscript
//! @onready var whisper: WhisperCpp = $WhisperCpp
//!
//! func _ready():
//!     whisper.transcription_complete.connect(_on_transcription)
//!     whisper.transcription_error.connect(_on_error)
//!     whisper.load_model("user://models/ggml-base.en.bin")
//!
//! func record_done(audio: PackedByteArray):
//!     whisper.transcribe(audio)
//!
//! func _on_transcription(text: String):
//!     print("Got:", text)
//!
//! func _on_error(msg: String):
//!     push_error(msg)
//! ```

use godot::prelude::*;
use godot::classes::Node;
use std::sync::{Arc, Mutex};
use std::thread;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
use enigo::{Enigo, Key, Keyboard, Settings};

/// GDExtension Node for whisper.cpp speech recognition.
///
/// Signals
/// -------
/// * `transcription_complete(text: String)` — emitted on the main thread when
///   transcription finishes successfully.
/// * `transcription_error(msg: String)` — emitted on the main thread when an
///   error occurs during loading or transcription.
/// * `model_loaded(path: String)` — emitted after the model is ready.
#[derive(GodotClass)]
#[class(base = Node)]
pub struct WhisperCpp {
    base: Base<Node>,

    /// Shared whisper context, protected by a mutex so the background thread
    /// and the main thread can both access it safely.
    context: Arc<Mutex<Option<WhisperContext>>>,

    /// Whether a transcription job is currently running.
    is_transcribing: bool,

    /// Transcription language (BCP-47 code, e.g. "en").
    #[var]
    pub language: GString,

    /// Number of CPU threads to use for inference.
    #[var]
    pub threads: i32,
}

#[godot_api]
impl INode for WhisperCpp {
    fn init(base: Base<Node>) -> Self {
        godot_print!("[WhisperCpp] initialised");
        WhisperCpp {
            base,
            context: Arc::new(Mutex::new(None)),
            is_transcribing: false,
            language: GString::from("en"),
            threads: 4,
        }
    }
}

#[godot_api]
impl WhisperCpp {
    /// Emitted when transcription completes successfully.
    #[signal]
    fn transcription_complete(text: GString);

    /// Emitted when an error occurs during model loading or transcription.
    #[signal]
    fn transcription_error(msg: GString);

    /// Emitted after a model is successfully loaded.
    #[signal]
    fn model_loaded(model_path: GString);

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /// Load a whisper.cpp GGML model file.
    ///
    /// This is a **synchronous** call that blocks until the model is loaded.
    /// For large models consider calling from a background thread; the signal
    /// `model_loaded` fires on completion.
    ///
    /// Returns `true` on success, `false` on failure (error signal emitted).
    #[func]
    pub fn load_model(&mut self, path: GString) -> bool {
        let path_str = path.to_string();
        godot_print!("[WhisperCpp] loading model: {path_str}");

        match WhisperContext::new_with_params(&path_str, WhisperContextParameters::default()) {
            Ok(ctx) => {
                let mut guard = self.context.lock().unwrap();
                *guard = Some(ctx);
                drop(guard);
                godot_print!("[WhisperCpp] model loaded ok");
                self.base_mut()
                    .emit_signal("model_loaded", &[path.to_variant()]);
                true
            }
            Err(e) => {
                let msg = format!("Failed to load model '{path_str}': {e}");
                godot_error!("{msg}");
                self.base_mut()
                    .emit_signal("transcription_error", &[GString::from(msg).to_variant()]);
                false
            }
        }
    }

    /// Unload the currently loaded model and free its memory.
    #[func]
    pub fn unload_model(&mut self) {
        let mut guard = self.context.lock().unwrap();
        *guard = None;
        godot_print!("[WhisperCpp] model unloaded");
    }

    /// Returns `true` if a model is currently loaded.
    #[func]
    pub fn is_model_loaded(&self) -> bool {
        self.context.lock().unwrap().is_some()
    }

    /// Returns `true` if a transcription is currently in progress.
    #[func]
    pub fn is_transcribing(&self) -> bool {
        self.is_transcribing
    }

    /// Transcribe raw PCM audio data.
    ///
    /// `audio_buffer` — raw 16-bit signed PCM samples, **mono, 16 000 Hz**,
    /// packed as little-endian bytes (i.e. a `PackedByteArray` where every two
    /// bytes form one i16 sample).
    ///
    /// Transcription runs on a **background thread**; the signal
    /// `transcription_complete` or `transcription_error` fires on the main
    /// thread when done.
    ///
    /// Returns `true` if the job was queued, `false` if already busy or no
    /// model is loaded.
    #[func]
    pub fn transcribe(&mut self, audio_buffer: PackedByteArray) -> bool {
        if self.is_transcribing {
            godot_warn!("[WhisperCpp] transcribe() called while already transcribing — ignoring");
            return false;
        }

        if !self.is_model_loaded() {
            let msg = "No model loaded — call load_model() first".to_string();
            godot_error!("{msg}");
            self.base_mut()
                .emit_signal("transcription_error", &[GString::from(msg).to_variant()]);
            return false;
        }

        self.is_transcribing = true;

        // Convert PackedByteArray (raw i16 LE bytes) → Vec<f32> normalised to [-1, 1]
        let bytes = audio_buffer.to_vec();
        let samples: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| {
                let s = i16::from_le_bytes([c[0], c[1]]);
                s as f32 / i16::MAX as f32
            })
            .collect();

        let ctx_arc = Arc::clone(&self.context);
        let language = self.language.to_string();
        let threads = self.threads;

        // Grab a Gd<Self> handle for call_deferred
        let mut self_gd = self.base_mut().clone().cast::<WhisperCpp>();

        thread::spawn(move || {
            let result = transcribe_on_thread(ctx_arc, samples, language, threads);
            match result {
                Ok(text) => {
                    let text_gstring = GString::from(text);
                    self_gd.call_deferred(
                        "emit_signal",
                        &[
                            GString::from("transcription_complete").to_variant(),
                            text_gstring.to_variant(),
                        ],
                    );
                    self_gd.call_deferred("_set_transcribing_false", &[]);
                }
                Err(e) => {
                    let msg = GString::from(e);
                    self_gd.call_deferred(
                        "emit_signal",
                        &[
                            GString::from("transcription_error").to_variant(),
                            msg.to_variant(),
                        ],
                    );
                    self_gd.call_deferred("_set_transcribing_false", &[]);
                }
            }
        });

        true
    }

    /// Internal helper — resets `is_transcribing` flag on main thread.
    #[func]
    pub fn _set_transcribing_false(&mut self) {
        self.is_transcribing = false;
    }

    // -------------------------------------------------------------------------
    // Input Simulation (cross-platform via enigo)
    // -------------------------------------------------------------------------

    /// Type text using the system keyboard input simulation.
    ///
    /// Works on Linux, macOS, and Windows via the `enigo` crate.
    /// Returns `true` on success, `false` on failure.
    #[func]
    pub fn type_text(&mut self, text: GString) -> bool {
        let text_str = text.to_string();
        if text_str.is_empty() {
            return false;
        }

        match Enigo::new(&Settings::default()) {
            Ok(mut enigo) => {
                godot_print!("[WhisperCpp] typing: {text_str}");
                enigo.text(&text_str).unwrap_or_else(|e| {
                    godot_error!("[WhisperCpp] type_text error: {e}");
                });
                true
            }
            Err(e) => {
                godot_error!("[WhisperCpp] failed to create Enigo: {e}");
                false
            }
        }
    }

    /// Press a key by name (e.g., "Return", "Escape", "Tab", "a", "F1").
    ///
    /// Uses enigo's cross-platform key mapping. Returns `true` on success.
    #[func]
    pub fn press_key(&mut self, key_name: GString) -> bool {
        let key_str = key_name.to_string();
        if key_str.is_empty() {
            return false;
        }

        match Enigo::new(&Settings::default()) {
            Ok(mut enigo) => {
                // Map common key names to enigo Key enum
                let key = match key_str.to_lowercase().as_str() {
                    "return" | "enter" => Key::Return,
                    "escape" | "esc" => Key::Escape,
                    "tab" => Key::Tab,
                    "space" => Key::Space,
                    "backspace" | "back" => Key::Backspace,
                    "delete" | "del" => Key::Delete,
                    "insert" => Key::Insert,
                    "home" => Key::Home,
                    "end" => Key::End,
                    "pageup" | "page_up" => Key::PageUp,
                    "pagedown" | "page_down" => Key::PageDown,
                    "up" => Key::UpArrow,
                    "down" => Key::DownArrow,
                    "left" => Key::LeftArrow,
                    "right" => Key::RightArrow,
                    "f1" => Key::F1,
                    "f2" => Key::F2,
                    "f3" => Key::F3,
                    "f4" => Key::F4,
                    "f5" => Key::F5,
                    "f6" => Key::F6,
                    "f7" => Key::F7,
                    "f8" => Key::F8,
                    "f9" => Key::F9,
                    "f10" => Key::F10,
                    "f11" => Key::F11,
                    "f12" => Key::F12,
                    s if s.len() == 1 => {
                        // Single character - try as layout key
                        let c = s.chars().next().unwrap();
                        Key::Unicode(c)
                    }
                    _ => {
                        godot_warn!("[WhisperCpp] unknown key: {key_str}");
                        return false;
                    }
                };

                godot_print!("[WhisperCpp] pressing key: {key_str}");
                enigo.key(key, enigo::Direction::Click).unwrap_or_else(|e| {
                    godot_error!("[WhisperCpp] press_key error: {e}");
                });
                true
            }
            Err(e) => {
                godot_error!("[WhisperCpp] failed to create Enigo: {e}");
                false
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Internals
// -----------------------------------------------------------------------------

/// Runs whisper inference on the calling thread.
///
/// Designed to be called from a `thread::spawn` closure.
fn transcribe_on_thread(
    ctx_arc: Arc<Mutex<Option<WhisperContext>>>,
    samples: Vec<f32>,
    language: String,
    threads: i32,
) -> Result<String, String> {
    let guard = ctx_arc
        .lock()
        .map_err(|e| format!("mutex poisoned: {e}"))?;

    let ctx = guard
        .as_ref()
        .ok_or_else(|| "context was unloaded before transcription finished".to_string())?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(&language));
    params.set_n_threads(threads);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    let mut state = ctx
        .create_state()
        .map_err(|e| format!("failed to create state: {e}"))?;

    state
        .full(params, &samples)
        .map_err(|e| format!("transcription failed: {e}"))?;

    let num_segments = state.full_n_segments().map_err(|e| e.to_string())?;
    let mut result = String::new();
    for i in 0..num_segments {
        let seg = state
            .full_get_segment_text(i)
            .map_err(|e| e.to_string())?;
        result.push_str(&seg);
        result.push(' ');
    }

    Ok(result.trim().to_string())
}
