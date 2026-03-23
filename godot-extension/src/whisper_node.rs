//! WhisperCpp — Godot Node wrapping whisper-rs for speech recognition.

use godot::prelude::*;
use godot::classes::{Node, INode};
use std::sync::{Arc, Mutex};
use std::thread;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

/// GDExtension Node for whisper.cpp speech recognition.
///
/// Signals
/// -------
/// * `transcription_complete(text: String)` — emitted when transcription finishes.
/// * `transcription_error(msg: String)` — emitted when an error occurs.
/// * `model_loaded(path: String)` — emitted after the model is ready.
#[derive(GodotClass)]
#[class(base = Node)]
pub struct WhisperCpp {
    base: Base<Node>,

    /// Shared whisper context.
    context: Arc<Mutex<Option<WhisperContext>>>,

    /// Whether a transcription job is currently running.
    is_transcribing: bool,

    /// Shared result slot — background thread writes here, _process polls it.
    pending_result: Arc<Mutex<Option<Result<String, String>>>>,

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
            pending_result: Arc::new(Mutex::new(None)),
            language: GString::from("en"),
            threads: 4,
        }
    }

    fn process(&mut self, _delta: f64) {
        let result = {
            let mut guard = self.pending_result.lock().unwrap();
            guard.take()
        };
        if let Some(outcome) = result {
            self.is_transcribing = false;
            match outcome {
                Ok(text) => {
                    self.base_mut().emit_signal(
                        "transcription_complete",
                        &[GString::from(text.as_str()).to_variant()],
                    );
                }
                Err(msg) => {
                    self.base_mut().emit_signal(
                        "transcription_error",
                        &[GString::from(msg.as_str()).to_variant()],
                    );
                }
            }
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

    /// Load a whisper.cpp GGML model file (synchronous).
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
                    .emit_signal("transcription_error", &[GString::from(msg.as_str()).to_variant()]);
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

    /// Transcribe raw PCM audio (mono 16 kHz i16 LE bytes).
    /// Runs on a background thread; result fires via signal on next _process tick.
    #[func]
    pub fn transcribe(&mut self, audio_buffer: PackedByteArray) -> bool {
        if self.is_transcribing {
            godot_warn!("[WhisperCpp] transcribe() called while already transcribing — ignoring");
            return false;
        }

        if !self.is_model_loaded() {
            let msg = "No model loaded — call load_model() first";
            godot_error!("{msg}");
            self.base_mut()
                .emit_signal("transcription_error", &[GString::from(msg).to_variant()]);
            return false;
        }

        self.is_transcribing = true;

        let bytes = audio_buffer.to_vec();
        let samples: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| {
                let s = i16::from_le_bytes([c[0], c[1]]);
                s as f32 / i16::MAX as f32
            })
            .collect();

        let ctx_arc = Arc::clone(&self.context);
        let result_arc = Arc::clone(&self.pending_result);
        let language = self.language.to_string();
        let threads = self.threads;

        thread::spawn(move || {
            let outcome = transcribe_on_thread(ctx_arc, samples, language, threads);
            let mut guard = result_arc.lock().unwrap();
            *guard = Some(outcome);
        });

        true
    }
}

// -----------------------------------------------------------------------------
// Internals
// -----------------------------------------------------------------------------

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

    let num_segments = state.full_n_segments();
    let mut result = String::new();
    for i in 0..num_segments {
        if let Some(seg) = state.get_segment(i) {
            let text = seg.to_str().map_err(|e| e.to_string())?;
            result.push_str(text);
            result.push(' ');
        }
    }

    Ok(result.trim().to_string())
}
