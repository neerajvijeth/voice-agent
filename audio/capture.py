"""
voice_agent/audio/capture.py

Microphone capture using sounddevice.
Streams audio into a queue in VAD-friendly frame sizes.

Design notes:
- We use sounddevice's InputStream with a callback (runs in a separate C thread).
- Frames are placed on a thread-safe queue.
- The main async loop drains the queue and feeds frames to the VAD.
- We intentionally DO NOT capture during playback to avoid the agent
  hearing itself – the pipeline sets `is_playing` flag to pause input.
"""

import queue
import threading
import numpy as np
import sounddevice as sd
from config import SAMPLE_RATE, CHANNELS, VAD_FRAME_SAMPLES, DTYPE


class MicCapture:
    def __init__(self):
        self._q          = queue.Queue()
        self._stream     = None
        self._running    = False
        self._remainder  = np.array([], dtype=DTYPE)  # leftover samples < frame size
        self.is_playing  = False   # set True externally during TTS playback

    # ── sounddevice callback (called from audio thread) ──────────────────────
    def _callback(self, indata, frames, time_info, status):
        if self.is_playing:
            return  # discard mic input while agent is speaking → no echo
        self._q.put(indata.copy())

    # ── public API ────────────────────────────────────────────────────────────
    def start(self):
        """Open the default input device and begin streaming."""
        self._running = True
        self._stream  = sd.InputStream(
            samplerate  = SAMPLE_RATE,
            channels    = CHANNELS,
            dtype       = DTYPE,
            blocksize   = VAD_FRAME_SAMPLES,  # ask for exactly one VAD frame
            callback    = self._callback,
            latency     = "low",
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def read_frame(self) -> bytes | None:
        """
        Return the next complete VAD frame as raw bytes, or None if no
        data is available yet.  Call this in a loop from the main thread.
        """
        try:
            chunk = self._q.get_nowait()
        except queue.Empty:
            return None

        # chunk shape: (VAD_FRAME_SAMPLES, 1) for mono
        samples = chunk.flatten()

        # Append to any leftover from last call
        combined = np.concatenate([self._remainder, samples])

        if len(combined) < VAD_FRAME_SAMPLES:
            self._remainder = combined
            return None

        frame           = combined[:VAD_FRAME_SAMPLES]
        self._remainder = combined[VAD_FRAME_SAMPLES:]
        return frame.tobytes()

    def flush_queue(self):
        """Discard buffered audio – call after we start playing back."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._remainder = np.array([], dtype=DTYPE)
