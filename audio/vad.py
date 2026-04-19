"""
voice_agent/audio/vad.py

Voice Activity Detection using webrtcvad.
Collects audio frames, emits complete utterances when silence is detected.
"""

import collections
import webrtcvad
import numpy as np
from config import (
    SAMPLE_RATE, VAD_FRAME_MS, VAD_FRAME_SAMPLES, VAD_FRAME_BYTES,
    VAD_AGGRESSIVENESS, VAD_SILENCE_FRAMES, MAX_UTTERANCE_SEC
)


class VADProcessor:
    """
    State machine that takes raw 16-bit PCM audio frames and yields
    complete utterances as numpy arrays.

    States:
        IDLE       → waiting for speech to begin
        SPEAKING   → accumulating audio during speech
    """

    def __init__(self, on_utterance_callback):
        """
        on_utterance_callback: callable(np.ndarray[int16]) called when
        an utterance is complete.
        """
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.on_utterance = on_utterance_callback

        self._speaking      = False
        self._silence_count = 0
        self._audio_buf     = []   # list of np.ndarray frames
        self._max_frames    = int(MAX_UTTERANCE_SEC * 1000 / VAD_FRAME_MS)

        # Ring buffer of recent frames kept for pre-speech context
        # (captures the first syllable that started before VAD triggered)
        self._pre_roll = collections.deque(maxlen=8)  # ~240 ms context

    def process_frame(self, frame_bytes: bytes) -> bool:
        """
        Feed one VAD_FRAME_MS chunk of raw 16-bit PCM bytes.
        Returns True if we are currently in a speech segment.
        """
        if len(frame_bytes) != VAD_FRAME_BYTES:
            return self._speaking

        is_speech = self.vad.is_speech(frame_bytes, SAMPLE_RATE)
        frame_np  = np.frombuffer(frame_bytes, dtype="int16").copy()

        if not self._speaking:
            self._pre_roll.append(frame_np)
            if is_speech:
                self._speaking      = True
                self._silence_count = 0
                # Prepend pre-roll so we don't miss the utterance start
                self._audio_buf = list(self._pre_roll) + []
                self._audio_buf.append(frame_np)
        else:
            self._audio_buf.append(frame_np)
            if is_speech:
                self._silence_count = 0
            else:
                self._silence_count += 1
                # End utterance on enough silence OR max duration
                if (self._silence_count >= VAD_SILENCE_FRAMES or
                        len(self._audio_buf) >= self._max_frames):
                    self._flush()

        return self._speaking

    def _flush(self):
        if self._audio_buf:
            audio = np.concatenate(self._audio_buf)
            self.on_utterance(audio)
        self._audio_buf     = []
        self._speaking      = False
        self._silence_count = 0
        self._pre_roll.clear()

    def force_flush(self):
        """Call this to end an utterance manually (e.g. push-to-talk release)."""
        self._flush()

    @property
    def is_speaking(self) -> bool:
        return self._speaking


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Normalise int16 PCM to float32 in [-1, 1] for Whisper."""
    return audio.astype(np.float32) / 32768.0
