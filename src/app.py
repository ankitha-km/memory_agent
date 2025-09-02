import os
import uuid
import ollama
import chromadb
from chromadb.utils import embedding_functions

# ====== CONFIG ======
MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")  # change to "phi3" if you want smaller
DB_PATH = "./chroma_db"
COLLECTION_NAME = "memories"

# ====== VECTOR DB (PERSISTENT) ======
# Uses a small local encoder (free) to embed memory sentences
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)

def save_memories(user_id: str, facts: list[str]) -> None:
    if not facts:
        return
    ids = [str(uuid.uuid4()) for _ in facts]
    metas = [{"user_id": user_id} for _ in facts]
    collection.add(ids=ids, documents=facts, metadatas=metas)

def retrieve_memories(user_id: str, message: str, k: int = 5) -> list[str]:
    res = collection.query(
        query_texts=[message],
        n_results=k,
        where={"user_id": user_id},
    )
    return res.get("documents", [[]])[0]

def llm_chat(system: str, user: str) -> str:
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp["message"]["content"].strip()

def extract_memories_from(text: str) -> list[str]:
    """
    Ask the LLM to extract crisp, reusable facts.
    """
    system = (
        "You extract concise first-person facts useful later (preferences, profile, plans, constraints). "
        "Return each fact as a single bullet line starting with '- '. If none, return 'NONE'. "
        "Do not include private/secret data or long sentences."
    )
    content = llm_chat(system, text)
    facts = []
    for line in content.splitlines():
        s = line.strip()
        if s.startswith("-"):
            facts.append(s.lstrip("- ").strip())
    return facts

def respond(user_id: str, message: str) -> str:
    # 1) retrieve relevant memories
    memories = retrieve_memories(user_id, message, k=6)
    memory_block = "\n".join(f"- {m}" for m in memories) if memories else "- (none)"

    # 2) answer with memory context
    system = (
        "You are a helpful assistant with long-term memory. "
        "Use the provided user memories if relevant. Don't invent facts. "
        "If unsure, ask a short clarifying question."
    )
    user = f"Known user memories:\n{memory_block}\n\nUser message:\n{message}"
    reply = llm_chat(system, user)

    # 3) extract and save new memories from the user's message (not from the bot reply)
    new_facts = extract_memories_from(message)
    save_memories(user_id, new_facts)

    return reply

def forget_all(user_id: str | None = None) -> None:
    if user_id is None:
        # wipe the whole collection
        client.delete_collection(COLLECTION_NAME)
        global collection
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn
        )
    else:
        # wipe only one user's memories
        collection.delete(where={"user_id": user_id})

if __name__ == "__main__":
    print("🧠 Memory Agent (local, free). Type 'exit' to quit. Type 'forget all' or 'forget me'.")
    user_id = "ankita"  # change per user if you want multi-user
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not msg:
            continue
        lower = msg.lower()
        if lower in {"exit", "quit"}:
            break
        if lower == "forget all":
            forget_all()
            print("Bot: Cleared ALL memories.")
            continue
        if lower in {"forget me", "reset"}:
            forget_all(user_id)
            print("Bot: Cleared your memories.")
            continue

        bot = respond(user_id, msg)
        print(f"Bot: {bot}\n")
