import os
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.summarizer import get_llm

def build_chain(system_prompt: str):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    return prompt | llm | StrOutputParser()

def _prepare_sample(transcript: str, max_chars: int = 20000) -> str:
    """Takes a representative sample across the entire transcript (beginning, middle, and end)."""
    if len(transcript) <= max_chars:
        return transcript
    
    part_size = max_chars // 3
    start = transcript[:part_size]
    mid_start = (len(transcript) // 2) - (part_size // 2)
    middle = transcript[mid_start:mid_start + part_size]
    end = transcript[-part_size:]
    return f"{start}\n\n[... transcript continues ...]\n\n{middle}\n\n[... transcript continues ...]\n\n{end}"

def extract_all_insights(transcript: str, max_chars: int = 20000) -> dict:
    """Extracts Action Items, Key Decisions, and Open Questions in a single, fast (3-5s) LLM call."""
    sample = _prepare_sample(transcript, max_chars=max_chars)
    chain = build_chain(
        "You are an expert video and meeting intelligence analyst.\n"
        "Analyze the provided transcript and extract all three sections in clear English.\n"
        "Format your answer with these exact three markdown headings:\n\n"
        "### 1. Action Items & Implementation Steps\n"
        "- Concrete tasks, next steps, or code implementation steps discussed.\n\n"
        "### 2. Key Decisions & Architectural Choices\n"
        "- Major decisions, library choices, design choices, and core takeaways made.\n\n"
        "### 3. Open Questions, Challenges & Edge Cases\n"
        "- Unresolved questions, technical challenges, and limitations discussed.\n\n"
        "Extract detailed, practical points discussed. Never leave sections empty."
    )
    res = chain.invoke({"text": sample})
    
    sections = {
        "action_items": "",
        "key_decisions": "",
        "open_questions": ""
    }
    p1 = re.search(r"###\s*1\..*?(?=###\s*2\.|\Z)", res, re.DOTALL)
    p2 = re.search(r"###\s*2\..*?(?=###\s*3\.|\Z)", res, re.DOTALL)
    p3 = re.search(r"###\s*3\..*", res, re.DOTALL)
    
    if p1: sections["action_items"] = re.sub(r"^###\s*1\.[^\n]*\n", "", p1.group(0)).strip()
    if p2: sections["key_decisions"] = re.sub(r"^###\s*2\.[^\n]*\n", "", p2.group(0)).strip()
    if p3: sections["open_questions"] = re.sub(r"^###\s*3\.[^\n]*\n", "", p3.group(0)).strip()
    
    if not sections["action_items"] and not sections["key_decisions"]:
        sections["action_items"] = res
        sections["key_decisions"] = "See action items and takeaways above."
        sections["open_questions"] = "See technical details and challenges above."
        
    return sections

def extract_action_items(transcript: str) -> str:
    return extract_all_insights(transcript)["action_items"]

def extract_key_decisions(transcript: str) -> str:
    return extract_all_insights(transcript)["key_decisions"]

def extract_questions(transcript: str) -> str:
    return extract_all_insights(transcript)["open_questions"]
