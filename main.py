import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

from dotenv import load_dotenv
from utils.audio_processor import process_input, fetch_youtube_subtitles
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all_insights
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Video Assistant...")

    transcript = ""
    if source.startswith("http://") or source.startswith("https://"):
        subtitles = fetch_youtube_subtitles(source, language)
        if subtitles:
            transcript = subtitles
        else:
            chunks = process_input(source)
            transcript = transcribe_all(chunks, language)
    else:
        chunks = process_input(source)
        transcript = transcribe_all(chunks, language)

    print(f"Raw transcription preview: {transcript[:300]}...")

    title = generate_title(transcript)
    summary = summarize(transcript)
    insights = extract_all_insights(transcript)
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": insights["action_items"],
        "key_decisions": insights["key_decisions"],
        "open_questions": insights["open_questions"],
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")
