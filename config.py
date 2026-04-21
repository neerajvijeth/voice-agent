"""
voice_agent/config.py
Central configuration for all pipeline stages.
Tune these values to balance latency vs quality on your hardware.
"""

import os

# ─────────────────────────────────────────
# AUDIO CAPTURE
# ─────────────────────────────────────────
SAMPLE_RATE        = 16000   # Hz – Whisper and WebRTC VAD both want 16 kHz
CHANNELS           = 1       # mono
DTYPE              = "int16" # 16-bit PCM – required by webrtcvad

# VAD processes audio in fixed frames. Valid options: 10 / 20 / 30 ms
# 30 ms = slightly more latency but fewer false triggers
VAD_FRAME_MS       = 30
VAD_FRAME_SAMPLES  = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480 samples
VAD_FRAME_BYTES    = VAD_FRAME_SAMPLES * 2  # int16 → 2 bytes/sample

# How aggressive the VAD is (0 = least, 3 = most aggressive noise filtering)
VAD_AGGRESSIVENESS = 3

# Consecutive silent frames needed before we consider speech ended
# 30 ms × 17 = 510 ms of trailing silence → end of utterance (was 25 = 750ms)
VAD_SILENCE_FRAMES = 17

# Maximum utterance length before we force a cut (prevents runaway recording)
MAX_UTTERANCE_SEC  = 15

# ─────────────────────────────────────────
# SPEECH-TO-TEXT  (faster-whisper)
# ─────────────────────────────────────────
# Model sizes and rough WER/latency on CPU (i5/i7 class):
#   tiny   → ~0.3-0.5 s, decent accuracy
#   base   → ~0.5-0.9 s, good accuracy  ← recommended MVP default
#   small  → ~1.5-2.5 s, better accuracy
#   medium → ~4-6 s, not great for live use on CPU
WHISPER_MODEL        = "base.en"
WHISPER_DEVICE       = "cuda"        # GPU acceleration (RTX 5050)
WHISPER_COMPUTE_TYPE = "float16"     # float16 on GPU for best speed
WHISPER_LANGUAGE     = "en"          # hardcoded to skip auto-detect penalty
WHISPER_BEAM_SIZE    = 1             # greedy decoding – fastest
WHISPER_VAD_FILTER   = False         # relying on pipeline WebRTC VAD instead

# ─────────────────────────────────────────
# LLM  (Google Gemini API)
# ─────────────────────────────────────────
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")  # loaded from .env
GEMINI_MODEL     = "gemini-2.5-flash"    # with thinking disabled for low latency
GEMINI_THINKING_BUDGET = 0               # 0 = no thinking tokens, fast TTFT
LLM_MAX_TOKENS   = 256             # concise voice responses don't need more
LLM_TEMPERATURE  = 0.7

# ─────────────────────────────────────────
# RAG  (Retrieval Augmented Generation)
# ─────────────────────────────────────────
RAG_ENABLED             = True
RAG_DOCUMENTS_DIR       = "./documents/"    # drop .docx/.pptx files here
RAG_TOP_K               = 3                 # number of chunks to retrieve per query
RAG_SIMILARITY_THRESHOLD = 0.35             # minimum cosine similarity to include a chunk
RAG_CHUNK_SIZE          = 512               # characters per chunk
RAG_CHUNK_OVERLAP       = 64                # overlap between adjacent chunks
RAG_EMBEDDING_MODEL     = "all-MiniLM-L6-v2"  # fast, 384-dim embeddings

# ─────────────────────────────────────────
# PostgreSQL + pgvector
# ─────────────────────────────────────────
PG_HOST     = "localhost"
PG_PORT     = 5432
PG_DBNAME   = "voice_agent"
PG_USER     = "naren"
PG_PASSWORD = "voiceagent"

# ─────────────────────────────────────────
# TEXT-TO-SPEECH
# ─────────────────────────────────────────
# Supported backends (in order of recommended preference):
#   "piper"   – best quality, fully local, requires model download
#   "edge"    – free Microsoft Azure voices, needs internet, great quality
#   "pyttsx3" – system TTS, no download, robotic voice, works everywhere
TTS_BACKEND = "edge"

# Piper settings (used when TTS_BACKEND = "piper")
PIPER_EXECUTABLE  = "piper"        # path to piper binary or just "piper" if in PATH
PIPER_MODEL_PATH  = "./models/piper/en_US-lessac-medium.onnx"
PIPER_CONFIG_PATH = "./models/piper/en_US-lessac-medium.onnx.json"
PIPER_SAMPLE_RATE = 22050

# Edge TTS settings (used when TTS_BACKEND = "edge")
EDGE_TTS_VOICE    = "en-US-AriaNeural"   # natural, fast
# Other good voices: en-US-GuyNeural, en-GB-SoniaNeural

# ─────────────────────────────────────────
# AGENT PERSONA
# ─────────────────────────────────────────
SYSTEM_PROMPT = """\
You are Aria, a fast voice assistant. Rules:
- Maximum 1-2 sentences per reply. Be ultra-concise.
- Answer directly. No filler words, no preamble, no "Certainly!" or "Great question!".
- Speak naturally. No markdown, no bullet points, no symbols.
- When document context is provided, prefer answering from it. If the context doesn't cover the question, use your own knowledge instead.
- Do NOT mention which source, slide, or document you are referencing. Just answer the question.
"""

AGENT_NAME = "Aria"   # how the agent refers to itself

# ─────────────────────────────────────────
# INTERACTION MODE
# ─────────────────────────────────────────
# "vad"  – always-listening with voice activity detection (recommended)
# "ptt"  – push-to-talk; press ENTER to start/stop recording (more reliable)
INTERACTION_MODE = "vad"

# ─────────────────────────────────────────
# PLAYBACK
# ─────────────────────────────────────────
OUTPUT_SAMPLE_RATE = 22050   # matches Piper; edge-tts uses 24000 (auto-resampled)
PLAYBACK_DTYPE     = "int16"
