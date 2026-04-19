# 🎙️ Local Low-Latency Voice Agent — MVP v1.0

A fully local, free, conversational voice AI that runs on your laptop.
Speak naturally → agent transcribes → thinks → speaks back.

---

## SECTION 1 — Architecture

```
Microphone
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  CAPTURE (sounddevice)                                      │
│  30ms PCM frames at 16kHz                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  VAD  (webrtcvad)                                           │
│  Detects speech start/end. Emits complete utterances.       │
│  ~750ms trailing silence → end of utterance                 │
└──────────────────────────┬──────────────────────────────────┘
                           │  np.ndarray int16 audio
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STT  (faster-whisper, base model, int8 CPU)                │
│  Transcribes utterance to text                              │
│  CPU latency: ~500-900ms for 2s audio                       │
└──────────────────────────┬──────────────────────────────────┘
                           │  "what time is it"
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM  (Ollama – llama3.2:3b, streaming)                     │
│  Generates response token by token                          │
│  TTFT on CPU: ~0.5-1.5s    GPU: ~0.1-0.3s                  │
│  Chunks split at sentence boundaries                        │
└──────────────────────────┬──────────────────────────────────┘
                           │  "It's about 3 PM." → TTS now
                           ▼                "Let me check." → TTS next
┌─────────────────────────────────────────────────────────────┐
│  TTS  (edge-tts or Piper)                                   │
│  Synthesises each chunk as it arrives from LLM              │
│  Chunk 1 starts playing before LLM finishes                 │
└──────────────────────────┬──────────────────────────────────┘
                           │  int16 PCM audio
                           ▼
                      Speakers / Headphones
```

**Why this architecture?**
- Local-first: STT and LLM run 100% on-device (Whisper + Ollama)
- Streaming pipeline: TTS starts on first sentence, not full response
- No cloud dependencies (edge-tts is optional cloud TTS; swap to Piper for full offline)
- Simple single-process Python: easy to debug, no microservice overhead
- Modular: swap any component with a one-line config change

---

## SECTION 2 — Stack

| Layer | Tool | Why |
|-------|------|-----|
| Audio capture | `sounddevice` | Low-latency, cross-platform, simple API |
| VAD | `webrtcvad` | Google's WebRTC VAD, <1ms per frame, C extension |
| STT | `faster-whisper` | 2-4× faster than original Whisper, int8 quantised |
| LLM | `Ollama + llama3.2:3b` | Free, local, fast 3B model, streaming API |
| TTS | `edge-tts` (default) | Free Microsoft TTS, natural voice, no account needed |
| TTS (local) | `Piper` | Best fully-offline voice, ~50ms synthesis |
| Audio output | `sounddevice` | Same lib as capture |
| Concurrency | `threading + queue` | Simple, no async overhead in hot path |

**Alternatives considered:**
- Vosk instead of Whisper: faster but lower accuracy
- llama.cpp directly instead of Ollama: harder to manage models
- Coqui TTS: good quality but slower than Piper
- silero-vad: more accurate than webrtcvad but needs PyTorch (~200MB)

---

## SECTION 3 — Latency Breakdown & Optimisations

### Where latency comes from:

```
[VAD silence timeout]   +750ms   ← you must stop talking before we transcribe
[STT transcription]     +500ms   ← faster-whisper base model on CPU
[LLM first token]       +700ms   ← llama3.2:3b first token on CPU
[TTS synthesis]         +250ms   ← edge-tts or piper
[Audio playback start]  +30ms    ← sounddevice buffer
─────────────────────────────────
TOTAL REALISTIC LATENCY  ~2.2s  on a modern CPU
                          ~1.0s  with a mid-range GPU
```

### Optimisations implemented:

1. **Chunk size = 30ms** — smallest VAD-compatible frame, balances CPU use vs. responsiveness
2. **int8 quantisation** — faster-whisper with int8 cuts STT time by ~50% vs fp32
3. **beam_size=1** — greedy decoding, fastest possible, minimal accuracy loss for short utterances
4. **LLM streaming** — first TTS chunk starts as soon as the first LLM sentence is complete
5. **Sentence-boundary chunking** — TTS starts on `. ! ?` not on full response
6. **Mic muted during playback** — prevents echo, avoids wasted STT cycles on agent voice
7. **Pre-roll buffer** — VAD captures 240ms before speech detected → no clipped first syllables
8. **Short system prompt responses** — forced to 1-3 sentences → less tokens to generate
9. **Small context window (2048)** — faster Ollama attention computation
10. **VAD_SILENCE_FRAMES=25** (750ms) — good balance between cutting too early vs. adding latency

### Perceived latency tricks:
- Keep responses to 1-2 sentences → audio starts faster
- Use `"tiny"` Whisper model if you need <300ms STT (accuracy trade-off)
- Use Piper TTS locally (no network RTT)
- Run Ollama on GPU if available → 5-10× faster TTFT

### If you have a GPU (NVIDIA):
```python
# config.py changes:
WHISPER_DEVICE       = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
```
And Ollama automatically uses GPU when available.

---

## SECTION 4 — Project Structure

```
voice_agent/
├── main.py                  ← entry point (CLI flags: --mode --model --tts --llm)
├── config.py                ← ALL settings in one place
├── requirements.txt
├── setup.sh                 ← automated setup (Linux/macOS)
│
├── audio/
│   ├── capture.py           ← mic → PCM frames via sounddevice
│   ├── vad.py               ← webrtcvad utterance detector
│   └── playback.py          ← PCM → speakers
│
├── stt/
│   └── transcriber.py       ← faster-whisper wrapper
│
├── llm/
│   └── responder.py         ← Ollama streaming client + chunk splitter
│
├── tts/
│   └── synthesizer.py       ← piper / edge-tts / pyttsx3 backends
│
├── agent/
│   └── conversation.py      ← pipeline orchestrator
│
├── tools/
│   └── benchmark.py         ← per-stage latency measurement
│
└── models/
    └── piper/               ← piper binary + voice model (after setup.sh)
```

---

## SECTION 5 — Installation

### Prerequisites

**macOS:**
```bash
brew install python portaudio ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y python3 python3-pip portaudio19-dev ffmpeg
```

**Windows:**
```powershell
winget install Python.Python.3.11
winget install Gyan.FFmpeg
# Download and install portaudio: http://www.portaudio.com/
```

### Python dependencies
```bash
cd voice_agent
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Ollama (local LLM)
```bash
# Linux/macOS:
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download

# Pull the model:
ollama pull llama3.2:3b
```

### Whisper model
Auto-downloads on first run (~150 MB for `base`). Or pre-download:
```bash
python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
```

### Piper TTS (optional – best local voice)
```bash
chmod +x setup.sh && ./setup.sh
# This downloads the Piper binary + en_US-lessac-medium model
# Then set in config.py: TTS_BACKEND = "piper"
```

---

## SECTION 6 — Running

**Terminal 1 – Start Ollama:**
```bash
ollama serve
```

**Terminal 2 – Start agent:**
```bash
source .venv/bin/activate
python main.py
```

**Flags:**
```bash
python main.py --mode ptt      # push-to-talk (more reliable on noisy setups)
python main.py --model tiny    # fastest STT (~200ms, less accurate)
python main.py --tts pyttsx3   # offline TTS, no ffmpeg needed
python main.py --llm mistral   # use mistral:7b instead (smarter, slower)
```

---

## SECTION 7 — Testing & Measuring Latency

### Quick latency check:
```bash
python tools/benchmark.py
```
This prints per-stage timings without needing a microphone.

### Manual latency test:
1. Wear headphones (prevents echo)
2. Say "What is two plus two?"
3. Start a timer when you stop speaking
4. Stop timer when you hear the first word of the response
5. Target: <2.5s on CPU, <1.0s on GPU

### Stability test:
```bash
# Run for 5 minutes, check for crashes or memory leaks
python main.py 2>&1 | tee agent.log
```

### Checking Ollama is using GPU:
```bash
ollama ps   # shows loaded model and whether GPU VRAM is used
```

### Push-to-talk for a clean test:
```bash
python main.py --mode ptt
```
This eliminates VAD uncertainty – just press ENTER to start/stop recording.

---

## SECTION 8 — Upgrade Path

### A. Add barge-in (interrupt while agent is speaking)
```python
# In agent/conversation.py, add a listener thread during playback.
# If new speech detected → call mic.stop() + sd.stop()
# Requires silero-vad (better accuracy for detecting speech during playback)
```

### B. Add memory / conversation history
```python
# LLMResponder already has self.history list
# Add persistence: save/load from JSON on disk
# Or use a vector store (ChromaDB) for long-term semantic memory
```

### C. Add telephony (Twilio/Vonage)
- Replace MicCapture with a WebSocket audio stream from Twilio Media Streams
- Replace playback with streaming TTS audio back over the WebSocket
- Wrap in FastAPI for the webhook endpoint

### D. Add business tools (function calling)
```python
# In responder.py, add Ollama tool_use support:
# Define functions like get_weather(), search_crm(), create_ticket()
# Ollama supports OpenAI-compatible function calling
```

### E. Better models (when ready)
```python
# config.py changes:
OLLAMA_MODEL  = "llama3.1:8b"    # smarter, slower
WHISPER_MODEL = "small"           # better accuracy, ~2x slower STT
TTS_BACKEND   = "piper"           # best local voice
```

### F. Production upgrade
- Wrap in FastAPI with WebSocket for a web UI
- Use asyncio throughout (replace threading with asyncio queues)
- Add Redis for conversation state
- Containerise with Docker

---

## SECTION 9 — Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'webrtcvad'` | `pip install webrtcvad-wheels` (pre-built) |
| `PortAudio not found` | `brew install portaudio` or `apt install portaudio19-dev` |
| `Ollama connection refused` | Run `ollama serve` in another terminal |
| `Model not found` | Run `ollama pull llama3.2:3b` |
| `edge-tts MP3 decode error` | Install ffmpeg: `brew install ffmpeg` |
| Agent hears itself | Use headphones; `is_playing` flag is set in capture.py |
| VAD too sensitive | Increase `VAD_AGGRESSIVENESS` to 3 in config.py |
| VAD misses speech | Decrease `VAD_AGGRESSIVENESS` to 1 |
| Response too slow | Switch to `--model tiny` and `OLLAMA_MODEL = "phi3:mini"` |

---

## SECTION 10 — Engineering Recommendations

### Build first:
1. Get the pipeline running with `--mode ptt` first (eliminates VAD complexity)
2. Test STT accuracy with `tools/benchmark.py`
3. Switch to `--mode vad` when the basic loop works

### What to keep simple:
- Don't add a web UI yet — terminal is fine for an MVP
- Don't add async/await everywhere — threading is simpler and fast enough
- Don't use a vector DB for memory yet — the history list in `LLMResponder` is enough

### What actually moves the latency needle:
1. **GPU** — single biggest improvement (5-10×)
2. **Whisper tiny model** — halves STT time, acceptable accuracy loss
3. **Piper TTS** — eliminates network RTT from edge-tts
4. **Short responses** — the system prompt forces this; don't change it

### Realistic numbers on a modern laptop (no GPU):
- VAD silence detection: 750ms (fixed, unavoidable)
- STT (base): 500-900ms
- LLM TTFT (3b): 500-1500ms
- TTS (edge-tts): 200-400ms
- **Total: ~2-3.5 seconds** (feels acceptable for conversational use)

### With a mid-range GPU (RTX 3060/4060):
- STT: 50-150ms
- LLM TTFT: 100-300ms
- TTS: 50-150ms (Piper)
- **Total: ~1-1.6 seconds** (feels natural and responsive)
