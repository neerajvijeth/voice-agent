"""
voice_agent/stt/transcriber.py

Speech-to-text using faster-whisper (CTranslate2 backend).

Why faster-whisper over openai-whisper?
  - 2-4× faster on CPU due to CTranslate2 int8 quantisation
  - Lower memory footprint
  - Same accuracy as original Whisper
  - Streaming segment output available

Latency notes:
  - Model load: ~1-3 s (done once at startup)
  - Transcription of 2 s audio on CPU (base model, int8):  ~400-800 ms
  - Transcription of 2 s audio on GPU (base model):        ~50-150 ms
  - Use 'tiny' model for <300 ms on CPU (lower accuracy)
  - Use 'small' model for better accuracy (1-2 s on CPU)
"""

import time
import numpy as np
from faster_whisper import WhisperModel
from config import (
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE, WHISPER_BEAM_SIZE, WHISPER_VAD_FILTER,
    SAMPLE_RATE
)


class Transcriber:
    def __init__(self):
        print(f"[STT] Loading Whisper '{WHISPER_MODEL}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE}) ...")
        t0 = time.time()
        self.model = WhisperModel(
            WHISPER_MODEL,
            device       = WHISPER_DEVICE,
            compute_type = WHISPER_COMPUTE_TYPE,
        )
        print(f"[STT] Model loaded in {time.time()-t0:.1f}s")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a numpy array of int16 PCM audio.
        Returns the transcribed text (stripped, lowercase optional).

        audio: np.ndarray shape (N,) dtype int16
        """
        # Whisper expects float32 normalised to [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0

        t0 = time.time()
        segments, info = self.model.transcribe(
            audio_f32,
            language      = WHISPER_LANGUAGE,
            beam_size     = WHISPER_BEAM_SIZE,
            vad_filter    = WHISPER_VAD_FILTER,
            word_timestamps = False,
        )

        # Collect all segment texts
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text.strip())

        result = " ".join(text_parts).strip()
        elapsed = time.time() - t0
        print(f"[STT] '{result}' ({elapsed:.2f}s, lang={info.language})")
        return result

    def transcribe_bytes(self, raw_bytes: bytes) -> str:
        """Convenience: transcribe from raw int16 PCM bytes."""
        audio = np.frombuffer(raw_bytes, dtype=np.int16)
        return self.transcribe(audio)
