#!/usr/bin/env bash
# voice_agent/setup.sh
#
# One-shot setup script for the local voice agent.
# Run with:  chmod +x setup.sh && ./setup.sh
#
# Tested on: Ubuntu 22.04, macOS 13+
# Windows users: use setup.ps1 (see comments at bottom) or WSL2

set -e

echo ""
echo "═══════════════════════════════════════════════"
echo "  Voice Agent – Setup Script"
echo "═══════════════════════════════════════════════"
echo ""

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "[1/6] Checking system dependencies..."

OS="$(uname -s)"

if [ "$OS" = "Linux" ]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3 python3-pip python3-venv \
        portaudio19-dev \
        ffmpeg \
        curl wget
    echo "  ✓ Linux packages installed"
elif [ "$OS" = "Darwin" ]; then
    if ! command -v brew &> /dev/null; then
        echo "  Homebrew not found. Install from https://brew.sh"
        exit 1
    fi
    brew install portaudio ffmpeg python3 curl wget 2>/dev/null || true
    echo "  ✓ macOS packages installed"
else
    echo "  Windows: run setup.ps1 or use WSL2."
    exit 1
fi

# ── 2. Python virtual environment ────────────────────────────────────────────
echo ""
echo "[2/6] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
echo "  ✓ venv at .venv/"

# ── 3. Python packages ────────────────────────────────────────────────────────
echo ""
echo "[3/6] Installing Python dependencies..."
pip install -r requirements.txt -q
echo "  ✓ Python packages installed"

# ── 4. Ollama (local LLM runtime) ────────────────────────────────────────────
echo ""
echo "[4/6] Installing Ollama..."
if command -v ollama &> /dev/null; then
    echo "  ✓ Ollama already installed"
else
    if [ "$OS" = "Linux" ]; then
        curl -fsSL https://ollama.com/install.sh | sh
    elif [ "$OS" = "Darwin" ]; then
        echo "  Download Ollama from: https://ollama.com/download"
        echo "  Or: brew install ollama"
    fi
fi

echo ""
echo "  Pulling llama3.2:3b model (≈2 GB, takes a few minutes)..."
echo "  (This runs in background – press Ctrl+C to skip, pull later manually)"
ollama pull llama3.2:3b || echo "  ⚠ Pull failed – run manually: ollama pull llama3.2:3b"

# ── 5. Whisper model (auto-downloaded by faster-whisper on first run) ─────────
echo ""
echo "[5/6] Whisper model..."
echo "  The 'base' Whisper model (~150 MB) will auto-download on first run."
echo "  To pre-download it now:"
python3 -c "
from faster_whisper import WhisperModel
print('  Downloading base model...')
WhisperModel('base', device='cpu', compute_type='int8')
print('  ✓ Whisper base model ready')
" || echo "  ⚠ Could not pre-download – will download on first run"

# ── 6. Piper TTS (optional, best local voice quality) ────────────────────────
echo ""
echo "[6/6] Piper TTS (optional – best local voice quality)..."
mkdir -p models/piper

if [ "$OS" = "Linux" ]; then
    PIPER_RELEASE="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
elif [ "$OS" = "Darwin" ]; then
    PIPER_RELEASE="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_macos_x64.tar.gz"
fi

if [ -n "$PIPER_RELEASE" ]; then
    echo "  Downloading Piper binary..."
    wget -q "$PIPER_RELEASE" -O /tmp/piper.tar.gz || curl -sL "$PIPER_RELEASE" -o /tmp/piper.tar.gz
    tar -xzf /tmp/piper.tar.gz -C models/piper/ --strip-components=1
    chmod +x models/piper/piper
    echo "  ✓ Piper binary at models/piper/piper"

    echo "  Downloading Piper voice model (en_US-lessac-medium)..."
    VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
    wget -q "${VOICE_BASE}/en_US-lessac-medium.onnx" \
         -O models/piper/en_US-lessac-medium.onnx || true
    wget -q "${VOICE_BASE}/en_US-lessac-medium.onnx.json" \
         -O models/piper/en_US-lessac-medium.onnx.json || true
    echo "  ✓ Piper voice model downloaded"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  To run the voice agent:"
echo "    1. Start Ollama:  ollama serve"
echo "    2. In another terminal:"
echo "       source .venv/bin/activate"
echo "       python main.py"
echo ""
echo "  Quick options:"
echo "    python main.py --mode ptt    # push-to-talk"
echo "    python main.py --model tiny  # fastest (less accurate)"
echo "    python main.py --tts piper   # local TTS (if model downloaded)"
echo "═══════════════════════════════════════════════"

# ── Windows PowerShell instructions (comment block) ──────────────────────────
: <<'WINDOWS_INSTRUCTIONS'
Windows Setup (PowerShell):
  1. Install Python 3.11+ from https://python.org
  2. Install ffmpeg: winget install Gyan.FFmpeg
  3. Install Ollama: https://ollama.com/download
  4. Open PowerShell in this folder:
       python -m venv .venv
       .venv\Scripts\activate
       pip install -r requirements.txt
       ollama pull llama3.2:3b
       python main.py
WINDOWS_INSTRUCTIONS
