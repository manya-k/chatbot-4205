import json
import time
import base64
import re
from pathlib import Path
from langchain_ollama import OllamaLLM                      
from langchain_community.vectorstores import Chroma          
from langchain_community.embeddings import OllamaEmbeddings  
from langchain.text_splitter import RecursiveCharacterTextSplitter 
                  

# configs
NOTES_DIR    = Path("knowledge_base/notes")
PHOTOS_DIR   = Path("knowledge_base/photos")
EPISODES_DIR = Path("knowledge_base/episodes")
EPISODES_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "llava"

# embeddings set-up
llm = OllamaLLM(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# vector db set up for both system a and system b using chroma as it is simple and file based
chunk_store = Chroma(collection_name="system_a_chunks", embedding_function=embeddings,
    persist_directory="knowledge_base/chroma_db")

episode_store = Chroma(collection_name="system_b_episodes", embedding_function=embeddings,
    persist_directory="knowledge_base/chroma_db")


# TODO: add imag processeing and episode processing

# System A: Chunk Ingestion
def ingest_chunks(destination, text):

    print(f"\n  [System A] Chunking {destination}...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,chunk_overlap=200, separators=["\n\n", "\n", ". ", " "]
    )

    # storing it as retrievable onjects with meta data so that it is easier to 
    # evaluate results at the end.
    docs = splitter.create_documents(
        texts=[text],metadatas=[{"destination": destination, "type": "chunk"}]
    )

    # Adding more meta data to add more labels inh case required for debugging later 
    for idx, doc in enumerate(docs):
        doc.metadata["chunk_index"] = idx
        doc.metadata["source"]      = destination

    # while chromadb can autogenerate ids, but thimight help in filtering as the
    # ids are now identify by their location.. which means easy filtering can be
    # applied - if unnecesary, might remove
    ids = []
    for idx in range(len(docs)):
        ids.append(f"{destination.upper()}_CHUNK_{idx:03d}")
    chunk_store.add_documents(docs, ids=ids)

    print(f"  ✓ {len(docs)} chunks stored for {destination}")
    return docs

def run_ingestion():
    print("\n" + "="*55)
    print("  MEMORYLENS INGESTION")
    print("="*55)
    print(f"  Model: {MODEL}")

    # Verify Ollama
    try:
        llm.invoke("hi")
        print(f"  ✓ Ollama + LangChain connected")
    except Exception as e:
        print(f"  ✗ Connection error: {e}")
        return

    note_files = list(NOTES_DIR.glob("*.txt"))
    if not note_files:
        print(f"  ✗ No .txt files in {NOTES_DIR}")
        return

    for note_file in sorted(note_files):
        dest = note_file.stem
        text = note_file.read_text(encoding="utf-8")

        print(f"\n{'='*55}")
        print(f"  {dest.upper()}")
        print(f"{'='*55}")

        # system A 
        ingest_chunks(dest, text)


    print(f"\n{'='*55}")
    print(f"  DONE")

if __name__ == "__main__":
    run_ingestion()
