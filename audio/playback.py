"""
voice_agent/audio/playback.py

Plays PCM audio through the default output device using sounddevice.
Supports chunked/streaming playback so the first TTS chunk plays
before synthesis of later chunks is done.
"""

import numpy as np
import sounddevice as sd
from config import SAMPLE_RATE, PLAYBACK_DTYPE


def play_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """
    Blocking playback of a complete audio array (int16 or float32).
    Resamples if necessary.
    """
    if audio.dtype != np.int16:
        # Normalise float to int16
        audio = (audio * 32767).astype(np.int16)

    sd.play(audio, samplerate=sample_rate, blocking=True)


def play_audio_bytes(raw_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    """Play raw 16-bit PCM bytes."""
    audio = np.frombuffer(raw_bytes, dtype=np.int16)
    play_audio(audio, sample_rate)


class StreamingPlayer:
    """
    Queues and plays audio chunks as they arrive from TTS.
    Usage:
        player = StreamingPlayer()
        player.start()
        player.feed(chunk1_bytes)
        player.feed(chunk2_bytes)
        player.finish()   # blocks until playback done
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self._sample_rate = sample_rate
        self._buffer      = []

    def start(self):
        self._buffer = []

    def feed(self, raw_bytes: bytes):
        """Add a chunk of raw 16-bit PCM to the queue."""
        self._buffer.append(np.frombuffer(raw_bytes, dtype=np.int16))

    def finish(self):
        """Concatenate and play all buffered audio."""
        if not self._buffer:
            return
        audio = np.concatenate(self._buffer)
        sd.play(audio, samplerate=self._sample_rate, blocking=True)
        self._buffer = []

    def stop(self):
        """Interrupt current playback (for barge-in)."""
        sd.stop()
