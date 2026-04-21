"""
voice_agent/tts/synthesizer.py

Text-to-speech with three backends:
  1. Piper    – fully local, best quality, requires model download
  2. edge-tts – free Microsoft cloud TTS, excellent quality, needs internet
  3. pyttsx3  – system TTS, works everywhere offline, robotic voice

All backends return raw 16-bit PCM bytes that can be fed to audio/playback.py.

Latency notes:
  - Piper:    ~30-80 ms synthesis latency per chunk (very fast, fully local)
  - edge-tts: ~150-400 ms per chunk (network RTT + synthesis)
  - pyttsx3:  ~20-60 ms but uses low-quality system voices

The TTS is called per-chunk from the LLM streamer, not on the full response,
so the first audio starts playing ~150-500 ms after the first LLM sentence.
"""

import io
import os
import sys
import time
import tempfile
import subprocess
import numpy as np
from config import (
    TTS_BACKEND, PIPER_EXECUTABLE, PIPER_MODEL_PATH, PIPER_CONFIG_PATH,
    PIPER_SAMPLE_RATE, EDGE_TTS_VOICE, SAMPLE_RATE
)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_synthesizer():
    """Return the configured TTS backend instance."""
    backend = TTS_BACKEND.lower()
    if backend == "piper":
        return PiperTTS()
    elif backend == "edge":
        return EdgeTTS()
    elif backend == "pyttsx3":
        return Pyttsx3TTS()
    else:
        raise ValueError(f"Unknown TTS_BACKEND: {TTS_BACKEND}")


# ── Base class ────────────────────────────────────────────────────────────────

class BaseTTS:
    plays_directly = False   # True if backend plays audio itself (e.g. pyttsx3)

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """
        Returns (raw_int16_pcm_bytes, sample_rate).
        Must be implemented by subclasses.
        """
        raise NotImplementedError


# ── Piper TTS ─────────────────────────────────────────────────────────────────

class PiperTTS(BaseTTS):
    """
    Uses the piper binary via subprocess.
    Piper reads from stdin (text) and writes raw PCM to stdout.
    This is fast because we don't write/read temp files.
    """

    def __init__(self):
        if not os.path.exists(PIPER_MODEL_PATH):
            print(f"[TTS] ERROR: Piper model not found at {PIPER_MODEL_PATH}")
            print("[TTS] Download with setup script or see README.")
        else:
            print(f"[TTS] Piper ready ({PIPER_MODEL_PATH})")

    def synthesize(self, text: str) -> tuple[bytes, int]:
        t0 = time.time()
        cmd = [
            PIPER_EXECUTABLE,
            "--model",        PIPER_MODEL_PATH,
            "--config",       PIPER_CONFIG_PATH,
            "--output_raw",   # raw PCM to stdout
        ]
        try:
            result = subprocess.run(
                cmd,
                input          = text.encode(),
                capture_output = True,
                timeout        = 10,
            )
            if result.returncode != 0:
                print(f"[TTS] Piper error: {result.stderr.decode()}")
                return b"", PIPER_SAMPLE_RATE
            print(f"[TTS] Piper synthesized {len(text)} chars in {time.time()-t0:.2f}s")
            return result.stdout, PIPER_SAMPLE_RATE
        except FileNotFoundError:
            print("[TTS] Piper binary not found. Install piper or change TTS_BACKEND.")
            return b"", PIPER_SAMPLE_RATE
        except subprocess.TimeoutExpired:
            print("[TTS] Piper timed out.")
            return b"", PIPER_SAMPLE_RATE


# ── edge-tts ──────────────────────────────────────────────────────────────────

class EdgeTTS(BaseTTS):
    """
    Uses Microsoft Edge TTS via the `edge-tts` Python package.
    Free, high quality, requires internet.
    Returns MP3 which we decode to PCM with pydub/ffmpeg.

    Uses a persistent asyncio event loop to avoid the overhead of
    creating/tearing down event loops per call (which causes 20-30s
    hangs when called from background threads).
    """

    def __init__(self):
        import threading
        import asyncio

        try:
            import edge_tts  # noqa
            print(f"[TTS] edge-tts ready (voice: {EDGE_TTS_VOICE})")
        except ImportError:
            print("[TTS] edge-tts not installed. Run: pip install edge-tts")

        # Persistent event loop running in a dedicated thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._loop_thread.start()

    def synthesize(self, text: str) -> tuple[bytes, int]:
        import asyncio
        t0 = time.time()
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_synthesize(text), self._loop
            )
            pcm, sr = future.result(timeout=15)
            print(f"[TTS] edge-tts synthesized in {time.time()-t0:.2f}s")
            return pcm, sr
        except Exception as e:
            print(f"[TTS] edge-tts error: {e}")
            return b"", 24000

    async def _async_synthesize(self, text: str) -> tuple[bytes, int]:
        import edge_tts
        communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
        mp3_buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buf.write(chunk["data"])
        mp3_buf.seek(0)
        return self._mp3_to_pcm(mp3_buf.read())

    def _mp3_to_pcm(self, mp3_bytes: bytes) -> tuple[bytes, int]:
        """Decode MP3 to int16 PCM using pydub (requires ffmpeg)."""
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
            seg = seg.set_channels(1).set_frame_rate(24000).set_sample_width(2)
            return seg.raw_data, 24000
        except ImportError:
            # Fallback: write MP3 to temp file, decode with ffmpeg directly
            return self._ffmpeg_decode(mp3_bytes)

    def _ffmpeg_decode(self, mp3_bytes: bytes) -> tuple[bytes, int]:
        """Decode MP3 to PCM using ffmpeg subprocess."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(mp3_bytes)
            tmp_mp3 = f.name
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_mp3,
                    "-f", "s16le", "-ar", "24000", "-ac", "1",
                    "-loglevel", "error",
                    "pipe:1"
                ],
                capture_output=True, timeout=10
            )
            return result.stdout, 24000
        finally:
            os.unlink(tmp_mp3)


# ── pyttsx3 ───────────────────────────────────────────────────────────────────

class Pyttsx3TTS(BaseTTS):
    """
    System TTS via pyttsx3.
    Works everywhere without any downloads.
    Voice quality is robotic but functional.
    """

    def __init__(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 175)   # slightly faster than default
            voices = self._engine.getProperty("voices")
            # Prefer a female voice if available
            for v in voices:
                if "female" in v.name.lower() or "zira" in v.id.lower() or "samantha" in v.id.lower():
                    self._engine.setProperty("voice", v.id)
                    break
            print("[TTS] pyttsx3 ready")
        except ImportError:
            print("[TTS] pyttsx3 not installed. Run: pip install pyttsx3")
            self._engine = None

    plays_directly = True   # pyttsx3 plays through system speakers directly

    def synthesize(self, text: str) -> tuple[bytes, int]:
        if not self._engine:
            return b"", 22050
        t0 = time.time()
        # Play directly through system speakers — most reliable on macOS
        self._engine.say(text)
        self._engine.runAndWait()
        print(f"[TTS] pyttsx3 synthesized in {time.time()-t0:.2f}s")
        return b"ok", 22050   # non-empty marker; audio already played


# ── Helpers ───────────────────────────────────────────────────────────────────

def _audio_file_to_pcm(file_path: str, target_sr: int = 22050) -> tuple[bytes, int]:
    """Read any audio file (WAV, AIFF, etc.) and return raw int16 PCM bytes + sample rate.
    Uses ffmpeg directly instead of pydub (which is broken on Python 3.14 due to missing audioop).
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", file_path,
            "-f", "s16le", "-ar", str(target_sr), "-ac", "1",
            "-loglevel", "error",
            "pipe:1"
        ],
        capture_output=True, timeout=10
    )
    return result.stdout, target_sr


def _wav_to_pcm(wav_path: str) -> tuple[bytes, int]:
    """Read a WAV file and return raw int16 PCM bytes + sample rate."""
    import wave
    with wave.open(wav_path, "rb") as wf:
        sr       = wf.getframerate()
        n_frames = wf.getnframes()
        raw      = wf.readframes(n_frames)
    return raw, sr


def pcm_bytes_to_numpy(raw: bytes, dtype=np.int16) -> np.ndarray:
    return np.frombuffer(raw, dtype=dtype)
