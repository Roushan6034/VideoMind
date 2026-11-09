import os
import re
import requests
import whisper
from pydub import AudioSegment

SARVAM_PIECE_SECONDS = 25
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None

def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model

import wave
import numpy as np

def load_wav_numpy(file_path: str):
    """Loads a WAV file into a 16kHz float32 NumPy array without needing FFmpeg."""
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        
        if sampwidth == 2:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
            
        if framerate != 16000:
            duration = len(samples) / framerate
            target_len = int(duration * 16000)
            samples = np.interp(
                np.linspace(0, len(samples), target_len, endpoint=False),
                np.arange(len(samples)),
                samples
            ).astype(np.float32)
            
        return samples

def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()
    try:
        samples = load_wav_numpy(chunk_path)
        result = model.transcribe(samples, task="transcribe")
        return result["text"]
    except Exception:
        result = model.transcribe(chunk_path, task="transcribe")
        return result["text"]

def _send_to_sarvam(piece_path: str) -> str:
    headers = {"api-subscription-key": SARVAM_API_KEY}
    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"Sarvam error {response.status_code}: {response.text}")
        response.raise_for_status()

    return response.json().get("transcript", "")

def transcribe_chunk_sarvam(chunk_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start:start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)

def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription complete.")
    return full_transcript.strip()

def translate_transcript_to_english(transcript: str, max_chars: int = 25000) -> str:
    if not re.search(r"[\u0900-\u097F]", transcript):
        return transcript

    from core.summarizer import get_llm
    llm = get_llm()
    
    sample = transcript[:max_chars]
    prompt = (
        "Translate the following meeting transcript into natural, clear, fluent English. "
        "Preserve the discussion flow and all technical points. "
        "Only return the English translation, nothing else:\n\n" + sample
    )
    res = llm.invoke(prompt)
    translated = "".join(b.get("text", "") for b in res.content if isinstance(b, dict)) if isinstance(res.content, list) else str(res.content)
    
    if len(transcript) > max_chars:
        translated += "\n\n--- [Note: Translated the first 25,000 characters for quick reading. The entire meeting is indexed in the RAG Chat below!] ---"
        
    return translated.strip()
