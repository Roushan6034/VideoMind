import os
import re
import yt_dlp
from pydub import AudioSegment

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})",
        r"^([0-9A-Za-z_-]{11})$"
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return ""

def fetch_youtube_subtitles(url: str, language: str = "english") -> str:
    video_id = extract_video_id(url)
    if not video_id:
        return ""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        target_langs = ["en", "en-US", "en-GB", "en-IN"] if language == "english" else ["hi", "en"]
        try:
            items = list(transcript_list.find_transcript(target_langs).fetch())
            return " ".join(item.text for item in items)
        except Exception:
            pass

        for t in transcript_list:
            if t.is_translatable:
                items = list(t.translate("en").fetch())
                return " ".join(item.text for item in items)
                
        for t in transcript_list:
            items = list(t.fetch())
            return " ".join(item.text for item in items)
    except Exception:
        pass
    return ""

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename

def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
        print("Chunking audio...")
        return chunk_audio(wav_path)
    
    # If file is already a WAV file (e.g. from st.audio_input or direct upload)
    if source.lower().endswith(".wav"):
        try:
            return chunk_audio(source)
        except Exception:
            return [source]

    print("Detected local media file. Converting to WAV...")
    wav_path = convert_to_wav(source)
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
