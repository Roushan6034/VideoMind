import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.summarizer import get_llm
from core.vector_store import build_vector_store, load_vector_store, hybrid_search_and_rerank

def format_docs(docs) -> str:
    return "\n\n".join([doc.page_content for doc in docs])

def contextualize_query(question: str, chat_history: list = None) -> str:
    if not chat_history:
        return question

    llm = get_llm()
    formatted_history = "\n".join([
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content')}"
        for m in chat_history[-4:]
    ])

    rephrase_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        ),
        ("human", "Chat History:\n{history}\n\nLatest Question: {question}\n\nStandalone Question:")
    ])
    chain = rephrase_prompt | llm | StrOutputParser()
    try:
        standalone = chain.invoke({"history": formatted_history, "question": question}).strip()
        return standalone if standalone else question
    except Exception:
        return question

def build_rag_chain(transcript: str):
    return build_vector_store(transcript)

def load_rag_chain():
    return load_vector_store()

def ask_question(rag_chain, question: str, chat_history: list = None) -> str:
    standalone_query = contextualize_query(question, chat_history)
    print(f"Original Question: {question}")
    print(f"Search Query: {standalone_query}")

    # Step 1 & 2: Hybrid Search (BM25 + Dense RRF) + Cross-Encoder Re-Ranking
    try:
        docs = hybrid_search_and_rerank(standalone_query, top_k=4, candidate_k=15)
        context = format_docs(docs)
    except Exception as e:
        print(f"Hybrid search warning ({e}), falling back to vector retriever")
        try:
            store = load_vector_store()
            docs = store.similarity_search(standalone_query, k=4)
            context = format_docs(docs)
        except Exception:
            return "I could not find this information in the meeting transcript."

    llm = get_llm()
    messages = [
        (
            "system",
            "You are an intelligent, thoughtful meeting partner and tech analyst who was in the room for this recording.\n"
            "Your personality is warm, clear, concise, and direct—like an experienced senior engineer talking to a peer.\n\n"
            "Core Guidelines:\n"
            "1. Speak naturally: Avoid robotic boilerplate like 'Based on the provided context' or 'As an AI'. Get straight to the answer.\n"
            "2. Grounded & Honest: Stay strictly grounded in the transcript and conversation history. If a specific point was not discussed, say so warmly and naturally (e.g., 'That wasn't covered in this session—they primarily focused on...').\n"
            "3. Formatting: Use bold lead-in phrases and clean bullet points for readability.\n"
            "4. Follow-up Intelligence: Seamlessly connect follow-up questions ('how did you solve it?', 'why?') to what you already discussed."
        )
    ]
    if chat_history:
        for msg in chat_history[-6:]:
            role = "human" if msg.get("role") == "user" else "ai"
            messages.append((role, msg.get("content", "")))

    messages.append(("human", "Meeting Transcript Context:\n{context}\n\nQuestion: {question}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    print(f"Answer: {answer}")
    return answer
