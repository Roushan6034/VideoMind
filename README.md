# 🎬 VideoMind: Multimodal AI Video & Audio Intelligence Copilot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?style=for-the-badge&logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Orchestration-LangChain%20LCEL-1C3C3C?style=for-the-badge&logo=chainlink" alt="LangChain" />
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB%20(HNSW)-purple?style=for-the-badge" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Speech%20AI-OpenAI%20Whisper%20%2B%20Sarvam-orange?style=for-the-badge" alt="Whisper" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

An enterprise-grade, multimodal meeting intelligence and video analysis copilot. Transcribe, analyze, summarize, extract key decisions, and converse with multi-hour videos and live audio using a state-of-the-art **Hybrid RAG Pipeline** (BM25 + ChromaDB + Reciprocal Rank Fusion + Cross-Encoder Re-Ranking).

---

## 🌟 Key Highlights & Engineering Breakthroughs

- 🎙️ **3-Way Multi-Source Ingestion:**
  - **Option 1: YouTube URLs** with automatic timed-text caption scraping (~1.4s) and headless `yt-dlp` stream extraction.
  - **Option 2: Local File Upload** (`.mp4`, `.mp3`, `.wav`, `.m4a`).
  - **Option 3: Live Microphone Recording** directly in the browser via HTML5 `st.audio_input`.
- ⚡ **Zero-FFmpeg Pure-Python Audio Engine:** Custom RIFF/WAVE binary parser (`load_wav_numpy`) using standard `wave` and `numpy`. Directly feeds normalized 16kHz float32 NumPy arrays into OpenAI Whisper, **completely bypassing external system FFmpeg dependencies**.
- 🧠 **Indic & Hinglish Speech Recognition:** Integrated with **Sarvam AI (`saaras:v2.5`)** with automated 25-second audio slicing and boundary stitching for code-mixed Indian English.
- 🚀 **Fast 4-Second Intelligence Extraction:** Replaced legacy 2-minute Map-Reduce with **Multi-Segment Temporal Sampling** ($T[0:K/3] + T[\text{mid}] + T[\text{end}]$), extracting Summary, Action Items, Key Decisions, and Sentiment in **4.1s**.
- 🔬 **State-of-the-Art Hybrid RAG Engine:**
  - **Dense Semantic Retrieval:** ChromaDB with `all-MiniLM-L6-v2` embeddings (384-d) on an HNSW graph.
  - **Sparse Lexical Retrieval:** In-memory `rank_bm25` (BM25Okapi) for exact keyword/code retrieval (e.g. `Errno 30`).
  - **Reciprocal Rank Fusion (RRF, $k=60$):** Merges dense and sparse rankings without score calibration distortion.
  - **Cross-Encoder Re-Ranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` evaluates query-chunk pairs with full bidirectional cross-attention, filtering 15 candidates down to the **Top 3 ultra-high-precision chunks**.
  - **Two-Stage Anaphora Contextualizer:** Resolves conversational pronouns (*"how did they fix it?"* $\rightarrow$ *"how did the team fix the ChromaDB latency bug?"*).
- 📄 **Executive PDF Briefing Generator:** Compiles structured intelligence into publication-ready PDF reports with corporate headers, pagination ("Page X of Y"), and Latin-1 Unicode sanitization using `fpdf2`.

---

## 🏗️ Master Architectural Pipeline

```
  [ USER INPUT ]
   ├── Option 1: YouTube Link (https://youtube.com/...)
   ├── Option 2: Upload Media File (.mp4, .mp3, .wav, .m4a)
   └── Option 3: 🎙️ Live Browser Microphone Recording
                               │
                               ▼
  [ STAGE 1: INGESTION & AUDIO EXTRACTION ]
   ├── YouTube: Try instant Subtitle Scrape (~1.4s) ──> (If missing) ──> yt-dlp Audio Stream
   ├── File Upload: Temporary file persistence in /downloads
   └── Microphone: Web browser audio buffer saved as 16-bit WAV
                               │
                               ▼
  [ STAGE 2: PURE-PYTHON ZERO-FFMPEG TRANSCRIPTION ]
   ├── Load WAV directly into 16kHz float32 NumPy array (load_wav_numpy)
   ├── Mode A (Standard): OpenAI Whisper CPU (No FFmpeg subprocess!)
   └── Mode B (Indic/Hinglish): Sarvam AI API (Auto 25s chunking & stitching)
                               │
                               ▼
             ┌─────────────────┴─────────────────┐
             │                                   │
             ▼                                   ▼
  [ STAGE 3: INSTANT INTELLIGENCE ]    [ STAGE 4: HYBRID RAG INDEXING ]
   ├── Multi-Segment Temporal Sampling  ├── Recursive Character Splitting
   │   (0-33%, 50%, 66-100%)            │   (1000 chars, 200 overlap)
   ├── Single-Call Gemini 1.5 Flash     ├── Dense Vector Embeddings (MiniLM-L6)
   └── Outputs:                         │   Persisted to ChromaDB (HNSW)
       ├── Executive Summary            └── Sparse BM25 Index fitted in memory
       ├── Action Items Table
       ├── Key Technical Decisions
       └── Sentiment Analysis
             │                                   │
             ▼                                   ▼
  [ STAGE 5: PDF REPORT EXPORT ]       [ STAGE 5: CONVERSATIONAL RAG CHAT ]
   ├── Subclassed FPDF2 Report          ├── Reactive st.chat_input
   ├── Clean Latin-1 typography         ├── Multi-Turn Anaphora Contextualizer
   └── Downloadable Corporate PDF       ├── Hybrid RRF + Cross-Encoder Re-rank
                                        └── Grounded, Humanized LLM Streaming
```

---

## 🔄 The 7-Stage RAG Data Lifecycle

```
[Raw Transcript] ──> [Chunking: 1000 chars, 200 overlap] ──> [all-MiniLM-L6-v2 Embeddings]
                                                                        │
                                                                        ▼
[User Question] ──> [Query Contextualizer] ──> [ChromaDB HNSW + BM25 Lexical (Parallel)]
                                                                        │
                                                                        ▼
                                                   [Reciprocal Rank Fusion (k=60)]
                                                                        │
                                                                        ▼
                                                   [Cross-Encoder Re-Ranker (Top 3)]
                                                                        │
                                                                        ▼
                                                   [Closed-World LLM Generation]
```

---

## 📁 Repository Structure

```
ai-video-assistant/
├── app.py                      # Reactive Streamlit frontend & chat interface
├── main.py                     # Headless CLI entry point
├── requirements.txt            # Production dependencies
├── .env.example                # Template for API keys
├── .gitignore                  # Git exclusions (vector_db, downloads, .env)
├── core/
│   ├── transcriber.py          # Pure-Python WAV decoding, Whisper, Sarvam ASR
│   ├── vector_store.py         # HybridVectorStore (BM25 + ChromaDB + Cross-Encoder)
│   ├── rag_engine.py           # Conversational memory & query contextualizer
│   ├── extractor.py            # Fast 4-second multi-segment temporal extractor
│   └── summarizer.py           # Map-Reduce hierarchical summarizer
└── utils/
    ├── audio_processor.py      # yt-dlp audio downloader & 25-second audio slicer
    └── pdf_generator.py        # FPDF2 corporate PDF report generator
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- Git installed.

### 2. Clone the Repository
```bash
git clone https://github.com/Roushan6034/VideoMind.git
cd VideoMind
```

### 3. Create & Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Keys
Copy the example environment file:
```bash
cp .env.example .env
```
Open `.env` and add your Google Gemini API key (free tier eligible):
```env
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: Sarvam AI API for Indic/Hinglish speech
SARVAM_API_KEY=your_sarvam_api_key_here
```

### 6. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
