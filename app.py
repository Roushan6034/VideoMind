import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import time
import re
import importlib
import streamlit as st
from dotenv import load_dotenv

import utils.audio_processor
import utils.pdf_generator
import core.transcriber
import core.summarizer
import core.extractor
import core.rag_engine

# Ensure changes to submodules are dynamically picked up by running Streamlit process
importlib.reload(utils.audio_processor)
importlib.reload(utils.pdf_generator)
importlib.reload(core.transcriber)
importlib.reload(core.summarizer)
importlib.reload(core.extractor)
importlib.reload(core.rag_engine)

from utils.audio_processor import process_input, fetch_youtube_subtitles
from utils.pdf_generator import generate_pdf_report
from core.transcriber import transcribe_all, translate_transcript_to_english
from core.summarizer import summarize, generate_title
from core.extractor import extract_all_insights
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #09090e;
    --surface: #12121b;
    --surface-2: #191926;
    --border: #232334;
    --border-hover: #3b3b55;
    --accent: #7c3aed;
    --accent-glow: #a78bfa;
    --accent-2: #06b6d4;
    --text: #ececf4;
    --text-muted: #8b8ba7;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(circle at 15% 15%, rgba(124, 58, 237, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.06) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.15;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 10%, #d8b4fe 55%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
    margin-top: 0.5rem;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: all 0.25s ease;
}

.card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 6px 24px rgba(124, 58, 237, 0.12);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(to bottom, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent-glow);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1.25rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    transition: border-color 0.2s ease;
}

[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
    border-color: var(--border-hover) !important;
}

.stButton > button {
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--border) !important;
}

.stButton > button:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.25) !important;
    transform: translateY(-1px);
}

.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.65rem 1rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-pending  { background: var(--border); }
.dot-active   { background: var(--warning); box-shadow: 0 0 10px var(--warning); animation: pulse 1s infinite; }
.dot-done     { background: var(--success); box-shadow: 0 0 10px var(--success); }

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.85); }
}

.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1.2rem;
    max-height: 480px;
    overflow-y: auto;
    padding-right: 0.5rem;
}

.chat-msg {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}

.chat-bubble {
    padding: 0.9rem 1.2rem;
    border-radius: 14px;
    font-size: 0.88rem;
    line-height: 1.68;
    max-width: 85%;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.user-bubble {
    background: linear-gradient(135deg, #4c1d95, #6d28d9);
    border: 1px solid rgba(167, 139, 250, 0.4);
    color: #ffffff;
    align-self: flex-end;
    border-bottom-right-radius: 3px;
    box-shadow: 0 4px 16px rgba(109, 40, 217, 0.25);
}

.bot-bubble {
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text);
    align-self: flex-start;
    border-bottom-left-radius: 3px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

.chat-label {
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--text-muted);
}

.user-label { align-self: flex-end; }
.bot-label  { align-self: flex-start; }

.badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}

.badge-purple { background: rgba(124, 58, 237, 0.15); border: 1px solid var(--accent); color: var(--accent-glow); }
.badge-cyan   { background: rgba(6, 182, 212, 0.15); border: 1px solid var(--accent-2); color: var(--accent-2); }
.badge-green  { background: rgba(16, 185, 129, 0.15); border: 1px solid var(--success); color: var(--success); }

.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem;
    font-size: 0.82rem;
    line-height: 1.8;
    color: #c8c8dd;
    max-height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
    font-family: 'JetBrains Mono', monospace;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input Source</span>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Choose Input Source",
        ["📺 YouTube URL", "📁 Local Media File", "🎙️ Record Live Microphone"],
        index=0,
        label_visibility="collapsed"
    )

    source = ""
    if input_mode == "📺 YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    elif input_mode == "📁 Local Media File":
        uploaded_file = st.file_uploader(
            "Upload Audio or Video",
            type=["wav", "mp3", "mp4", "m4a", "mov", "mkv"],
            help="Upload an audio or video recording from your computer."
        )
        if uploaded_file:
            from utils.audio_processor import DOWNLOAD_DIR
            saved_path = os.path.join(DOWNLOAD_DIR, uploaded_file.name)
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source = saved_path
            st.success(f"Loaded: {uploaded_file.name}")
        else:
            local_path = st.text_input("Or enter local file path", placeholder="/path/to/meeting.mp4")
            source = local_path.strip()
    elif input_mode == "🎙️ Record Live Microphone":
        st.markdown("<div style='font-size:0.8rem;color:var(--text-muted);margin-bottom:0.4rem'>Click record, speak into your mic, then stop:</div>", unsafe_allow_html=True)
        audio_record = st.audio_input("Microphone Input", label_visibility="collapsed")
        if audio_record:
            from utils.audio_processor import DOWNLOAD_DIR
            mic_path = os.path.join(DOWNLOAD_DIR, "live_mic_recording.wav")
            with open(mic_path, "wb") as f:
                f.write(audio_record.read())
            source = mic_path
            st.audio(mic_path)
            st.success("Audio recorded! Ready to analyse.")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    st.markdown("---")
    st.markdown('<span class="badge badge-cyan">API Configuration</span>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "API Key (Gemini or Mistral)",
        value=os.getenv("GOOGLE_API_KEY") or os.getenv("MISTRAL_API_KEY") or "",
        type="password"
    )
    if api_key_input:
        if api_key_input.startswith("AQ.") or api_key_input.startswith("AIza"):
            os.environ["GOOGLE_API_KEY"] = api_key_input
        else:
            os.environ["MISTRAL_API_KEY"] = api_key_input

    st.markdown("---")
    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please provide a YouTube URL, upload an audio/video file, or record microphone audio first.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with st.status("🚀 Processing Video Intelligence Pipeline...", expanded=True) as status_box:
                transcript = ""
                update_step("audio", "active")
                
                if source.startswith("http://") or source.startswith("https://"):
                    status_box.write("🔊 Checking YouTube subtitles and audio stream...")
                    subtitles = fetch_youtube_subtitles(source, language)
                    if subtitles:
                        update_step("audio", "done")
                        update_step("transcript", "active")
                        transcript = subtitles
                        status_box.write(f"📝 Retrieved {len(transcript):,} characters of transcript directly.")
                        update_step("transcript", "done")
                    else:
                        status_box.write("🔊 Downloading and chunking audio with yt-dlp...")
                        chunks = process_input(source)
                        update_step("audio", "done")
                        update_step("transcript", "active")
                        status_box.write(f"📝 Transcribing audio chunks using Whisper ({language})...")
                        transcript = transcribe_all(chunks, language)
                        update_step("transcript", "done")
                else:
                    status_box.write("🔊 Processing recorded audio / local media...")
                    chunks = process_input(source)
                    update_step("audio", "done")
                    update_step("transcript", "active")
                    status_box.write(f"📝 Transcribing audio using Whisper ({language})...")
                    transcript = transcribe_all(chunks, language)
                    update_step("transcript", "done")

                if not transcript or not transcript.strip():
                    raise ValueError("No speech could be transcribed from the input. Please ensure clear audio is present.")

                status_box.write("🏷️ Generating session title...")
                update_step("title", "active")
                title = generate_title(transcript)
                update_step("title", "done")

                status_box.write("📋 Generating executive summary...")
                update_step("summary", "active")
                summary = summarize(transcript)
                update_step("summary", "done")

                status_box.write("🔍 Extracting action items, key decisions, and questions...")
                update_step("extract", "active")
                insights = extract_all_insights(transcript)
                action_items = insights["action_items"]
                decisions = insights["key_decisions"]
                questions = insights["open_questions"]
                update_step("extract", "done")

                status_box.write("🧠 Indexing transcript into ChromaDB for RAG chat...")
                update_step("rag", "active")
                rag_chain = build_rag_chain(transcript)
                update_step("rag", "done")

                status_box.update(label="✅ Analysis complete!", state="complete", expanded=False)

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            st.rerun()

        except Exception as e:
            for k in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            st.error(f"❌ Error during processing: {e}")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Title banner + PDF Export Action
    t_col1, t_col2 = st.columns([3, 1], gap="medium")
    with t_col1:
        st.markdown(f"""
        <div class="card" style="margin-bottom:0.5rem">
            <div class="card-title">📌 Session Title</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:var(--text)">
                {r['title']}
            </div>
        </div>""", unsafe_allow_html=True)
    with t_col2:
        pdf_bytes = generate_pdf_report(
            title=r.get("title", "Meeting Report"),
            summary=r.get("summary", ""),
            action_items=r.get("action_items", ""),
            key_decisions=r.get("key_decisions", ""),
            open_questions=r.get("open_questions", "")
        )
        safe_fname = re.sub(r'[^a-zA-Z0-9_-]', '_', r.get('title', 'meeting'))[:30] or "meeting_report"
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📄 Export Report (PDF)",
            data=pdf_bytes,
            file_name=f"{safe_fname}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )

    # Top row: summary + transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown('<div class="card-title">📋 Summary</div>', unsafe_allow_html=True)
            st.markdown(r['summary'])

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            has_hindi = bool(re.search(r"[\u0900-\u097F]", r["transcript"]))
            display_text = r["transcript"]
            if has_hindi:
                trans_view = st.radio(
                    "Transcript Language",
                    ["Original Transcript", "English Translation (Fast Preview)"],
                    index=0,
                    horizontal=True,
                    key="transcript_view_choice"
                )
                if trans_view == "English Translation (Fast Preview)":
                    if "translated_transcript" not in r:
                        with st.spinner("Translating preview into English..."):
                            r["translated_transcript"] = translate_transcript_to_english(r["transcript"])
                    display_text = r["translated_transcript"]
            st.markdown(f'<div class="transcript-box">{display_text}</div>', unsafe_allow_html=True)

    # Second row: action items | decisions | questions
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        with st.container(border=True):
            st.markdown('<div class="card-title" style="color:#34d399">✅ Action Items</div>', unsafe_allow_html=True)
            st.markdown(r['action_items'])

    with c2:
        with st.container(border=True):
            st.markdown('<div class="card-title" style="color:#38bdf8">🔑 Key Decisions</div>', unsafe_allow_html=True)
            st.markdown(r['key_decisions'])

    with c3:
        with st.container(border=True):
            st.markdown('<div class="card-title" style="color:#fbbf24">❓ Open Questions</div>', unsafe_allow_html=True)
            st.markdown(r['open_questions'])

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.25rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

    # Chat history display
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">👤 You</span>
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">✨ AI Partner</span>
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:1.5rem;margin-bottom:0.8rem">
            <div style="font-size:1.8rem;margin-bottom:0.3rem">💬</div>
            <div style="color:var(--text);font-weight:600;font-size:0.95rem;margin-bottom:0.2rem">Ask Anything About This Session</div>
            <div style="color:var(--text-muted);font-size:0.8rem">Click a quick prompt below or type your own:</div>
        </div>""", unsafe_allow_html=True)

        q1, q2, q3 = st.columns(3)
        with q1:
            if st.button("💡 Problems & Solutions?", use_container_width=True):
                st.session_state["pending_query"] = "What were the main problems faced and how were they solved?"
                st.rerun()
        with q2:
            if st.button("📋 3-Bullet Briefing?", use_container_width=True):
                st.session_state["pending_query"] = "Give me a quick 3-bullet executive briefing of this session."
                st.rerun()
        with q3:
            if st.button("🛠️ Tech Stack Used?", use_container_width=True):
                st.session_state["pending_query"] = "What technologies, tools, and libraries were used or discussed?"
                st.rerun()

    # Native chat input
    user_query = st.chat_input("Ask a question about this video or meeting...")
    if not user_query and st.session_state.get("pending_query"):
        user_query = st.session_state.pop("pending_query")

    if user_query and user_query.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_query.strip(), chat_history=st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user",      "content": user_query.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty landing state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem 1rem 2rem;text-align:center">
        <div style="font-size:3.5rem;margin-bottom:0.75rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:var(--text);margin-bottom:0.4rem">
            Welcome to AI Video Assistant
        </div>
        <div style="color:var(--text-muted);font-size:0.95rem;max-width:540px;line-height:1.7;margin-bottom:2rem">
            Transform hours of meetings, technical workshops, and tutorials into structured intelligence, actionable next steps, and a grounded conversation partner.
        </div>
    </div>""", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        with st.container(border=True):
            st.markdown("""
            <div style="font-size:1.6rem;margin-bottom:0.5rem">🎙️</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;color:var(--text);margin-bottom:0.3rem">1. Multi-Source Ingestion</div>
            <div style="color:var(--text-muted);font-size:0.82rem;line-height:1.6">
                Paste any YouTube URL, upload an MP4/WAV recording, or record live audio straight from your browser microphone.
            </div>
            """, unsafe_allow_html=True)

    with f2:
        with st.container(border=True):
            st.markdown("""
            <div style="font-size:1.6rem;margin-bottom:0.5rem">⚡</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;color:var(--text);margin-bottom:0.3rem">2. Deep Extraction</div>
            <div style="color:var(--text-muted);font-size:0.82rem;line-height:1.6">
                Automatically extracts executive summaries, concrete action items, technical architecture decisions, and open challenges.
            </div>
            """, unsafe_allow_html=True)

    with f3:
        with st.container(border=True):
            st.markdown("""
            <div style="font-size:1.6rem;margin-bottom:0.5rem">🧠</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;color:var(--text);margin-bottom:0.3rem">3. Conversational RAG</div>
            <div style="color:var(--text-muted);font-size:0.82rem;line-height:1.6">
                Chat with the recording using Hybrid BM25 + Cross-Encoder re-ranking. Export complete reports to PDF with one click.
            </div>
            """, unsafe_allow_html=True)
