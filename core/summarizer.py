import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_llm():
    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if mistral_key and not mistral_key.startswith("your_"):
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model="mistral-small-latest", mistral_api_key=mistral_key, temperature=0.3)
    
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if google_key and not google_key.startswith("your_"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=google_key)
        
    raise ValueError("No valid API key found. Please set MISTRAL_API_KEY or GOOGLE_API_KEY in .env or the sidebar.")

def split_transcript(transcript: str) -> list:
    if len(transcript) <= 25000:
        return [transcript]
    chunk_size = max(15000, len(transcript) // 5)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=500
    )
    return splitter.split_text(transcript)

def summarize(transcript: str) -> str:
    llm = get_llm()
    chunks = split_transcript(transcript)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a seasoned technical director writing an executive briefing from a meeting/video transcript.\n"
            "Write in natural, clear, fluent English. Avoid robotic clichés like 'In this video' or 'It is important to note'.\n"
            "Structure your summary as follows:\n\n"
            "**Overview:** 2 crisp sentences summarizing the big picture.\n\n"
            "**Key Discussion Points:**\n"
            "- Bold headline: Clear, high-value takeaway.\n"
            "- Bold headline: Clear, high-value takeaway.\n\n"
            "**Outcome / Bottom Line:** One impactful concluding sentence on where things stand.",
        ),
        ("human", "{text}"),
    ])
    combined_chain = combined_prompt | llm | StrOutputParser()

    if len(chunks) == 1:
        return combined_chain.invoke({"text": chunks[0]})

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely in clear English."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)
    return combined_chain.invoke({"text": combined})

def generate_title(transcript: str) -> str:
    llm = get_llm()
    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Based on the meeting transcript, generate a short professional meeting title "
            "(max 8 words) in English. Only return the title, nothing else.",
        ),
        ("human", "{text}"),
    ])
    title_chain = title_prompt | llm | StrOutputParser()
    return title_chain.invoke({"text": transcript[:3000]}).strip()
