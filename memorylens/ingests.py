import json
import time
import base64
import re
from pathlib import Path
from langchain_ollama import OllamaLLM                      
from langchain_community.vectorstores import Chroma          
from langchain_community.embeddings import OllamaEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document                      
from langchain_core.tools import tool       
from PIL import Image
import io


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

# Destination aliases
DESTINATION_ALIASES = {
    "indo":      "india",
    "sing":      "singapore",
    "bali":      "bali",
    "india":     "india",
    "dubai":     "dubai",
    "singapore": "singapore",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".heic", ".HEIC"}

# TODO: add imag processeing and episode processing
 
# Filename Parse
def parse_photo_filename(filepath):
    stem       = filepath.stem
    stem_clean = stem.replace("-", " ")
    parts      = stem_clean.split("_", 1)

    raw_destination = parts[0].strip().lower()
    location        = parts[1].strip() if len(parts) > 1 else stem_clean
    location        = location.replace("_", " ").strip()
    destination     = DESTINATION_ALIASES.get(raw_destination, raw_destination)
    hint            = location

    return destination, location, hint
 
 
def get_all_photos():
    if not PHOTOS_DIR.exists():
        return []

    photos = []
    for f in PHOTOS_DIR.iterdir():
        if f.suffix.lower() in {ext.lower() for ext in SUPPORTED_EXTENSIONS} or f.suffix == "":
            photos.append(f)
    return sorted(photos)
 
 
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def safe_id(text):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text).upper()
 
 
# ── System A: Chunck Ingestion 
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


# System B: Episode Extraction 

def extract_episodes(destination, raw_text):
    print(f"\n  [System B] Extracting episodes for {destination}...")

    prompt = f"""Convert these personal travel notes into a JSON array of structured memory episodes.
Each episode represents one distinct activity, meal, visit, or day event.

Return ONLY a valid JSON array. No markdown. No explanation. No preamble.

Each episode object must have exactly these fields:
- episode_id: short unique string e.g. BALI_DAY2_BEACH
- destination: string
- day_number: integer
- location: string (specific place name)
- title: string (one sentence summary of the event)
- description: string (2-3 sentences describing what happened)
- cost_aud: number (cost in AUD, 0 if free)
- tags: array of strings chosen from: beach, food, temple, hiking, culture, budget, crowded, peaceful, market, nightlife, nature, city, luxury, art, history, wildlife, adventure, relaxing
- constraint_relevant: string describing any user preference or constraint revealed, or null

Travel notes for {destination}:
{raw_text}

Return the JSON array now:"""

    raw = llm.invoke(prompt)
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        print(f"  ✗ No JSON array found")
        return []

    try:
        episodes = json.loads(raw[start:end])
        print(f"  ✓ {len(episodes)} episodes extracted")
        return episodes
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON error: {e}")
        print(f"  Raw response:\n{raw[start:end][:500]}")
        return []



def add_temporal_links(episodes):
    """
    Adds temporal links between the previous nad the next episode to establish a timelone
    and adds order to the episodews
    """
    episodes.sort(key=lambda e: e.get("day_number") or 0)
    for i, ep in enumerate(episodes):
        if i > 0: 
            ep["linked_previous"] = episodes[i-1]["episode_id"]
        else: 
            ep["linked_previous"] = None

        if i < len(episodes) - 1:
            ep["linked_next"] = episodes[i+1]["episode_id"]
        else:
            ep["linked_next"] = None
    return episodes


def add_thematic_links(all_episodes):
    tag_index = {}
    for ep in all_episodes:
        # add related tags
        # the idea is that prompting llm to struc ture the data will provide a 
        # a json response where one of thefeilds will be "tags"
        # for example "tags": [beach, nature, peaceful] and then this function
        # will add links to other episodes which have these tags for example:
        # [{"beach": ["bali_day1, india_day7"]}, {"nature":[]}]
        for tag in ep.get("tags", []):
            # adds its own tag
            tag_index.setdefault(tag, []).append(ep["episode_id"])

    for ep in all_episodes:
        # find other similar tags
        linked = []
        for tag in ep.get("tags", []):
            for eid in tag_index.get(tag, []):
                if eid != ep["episode_id"] and eid not in linked:
                    linked.append(eid)
        ep["linked_theme"] = linked[:5]
    return all_episodes


def store_text_episodes(episodes):
    docs = []
    ids  = []

    for ep in episodes:
        eid = ep.get("episode_id", "")
        if not eid:
            continue
        
        # storing all the datat in the epsisode vector db
        doc_text = f"{ep.get('title', '')}. {ep.get('description', '')}"
        if ep.get("constraint_relevant"):
            doc_text += f" User preference noted: {ep['constraint_relevant']}"

        docs.append(Document(
            page_content=doc_text,
            metadata={
                "episode_id": eid,
                "destination": ep.get("destination", ""),
                "day_number": ep.get("day_number", 0),
                "location": ep.get("location", ""),
                "emotion": ep.get("emotion", ""),
                "cost_aud": str(ep.get("cost_aud") or ""),
                "tags": json.dumps(ep.get("tags", [])),
                "linked_previous": ep.get("linked_previous") or "",
                "linked_next": ep.get("linked_next") or "",
                "linked_theme": json.dumps(ep.get("linked_theme", [])),
                "constraint_relevant": ep.get("constraint_relevant") or "",
                "type": "episode"
            }
        ))
        ids.append(eid)

    if docs:
        episode_store.add_documents(docs, ids=ids)

    print(f"  ✓ {len(docs)} text episodes stored")


# image ingestio

def describe_image(image_path, location, hint):
    import ollama  # direct call needed for image support

    prompt = f"""You are describing a personal travel photo for a memory retrieval system.
This photo was taken at: {location}
Context clue: {hint}

Describe this photo in exactly 3 sentences:
1. The main scene or subject visible in the photo
2. Key visual details — colours, textures, lighting, atmosphere
3. What kind of place, moment, or experience this captures

Be specific and vivid. Start directly with the description."""

    try:
        img_b64  = encode_image_base64(image_path)
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"    ✗ Vision error: {e}")
        return f"A travel photo taken at {location}."



def ingest_images():
    print(f"\n{'='*55}")
    print(f"  IMAGE INGESTION")
    print(f"{'='*55}")

    photos = get_all_photos()
    if not photos:
        print("  No photos found")
        return []

    print(f"  Found {len(photos)} photos")
    stored_episodes = []

    for photo_path in photos:
        print(f"\n  → {photo_path.name}")
        destination, location, hint = parse_photo_filename(photo_path)
        eid = "IMG_" + safe_id(photo_path.stem)

        print(f"    Describing with LLaVA...")
        description = describe_image(photo_path, location, hint)
        print(f"    ✓ {description[:80]}...")

        doc_text = f"Photo from {location} in {destination}. Location: {destination}, {location}. {description}"

        # LangChain Document for the image episode
        doc = Document(
            page_content=doc_text,
            metadata={
                "episode_id":          eid,
                "destination":         destination,
                "location":            location,
                "type":                "image_episode",
                "image_filename":      photo_path.name,
                "image_description":   description,
                "tags":                json.dumps(hint.split()),
                "linked_previous":     "",
                "linked_next":         "",
                "linked_theme":        json.dumps([]),
                "emotion":             "",
                "cost_aud":            "",
                "constraint_relevant": ""
            }
        )

        episode_store.add_documents([doc], ids=[eid])

        # now store the same image in the chunk store as well but without the 
        # structure - as flat chunks
        chunk_img_doc = Document(
            page_content=doc_text,
            metadata={
                "destination": destination,
                "type":        "chunk",
                "source":      "image",
                "location":    location
            }
        )
        chunk_store.add_documents([chunk_img_doc], ids=[f"CHUNK_{eid}"])

        stored_episodes.append({
            "episode_id":  eid,
            "destination": destination,
            "location":    location,
            "filename":    photo_path.name,
            "description": description
        })

        time.sleep(2)

    print(f"\n  ✓ {len(stored_episodes)} image episodes stored")
    return stored_episodes


@tool
def search_chunks(query: str) -> str:
    """
    Search System A — retrieves raw text chunks.
    Used by the baseline variant of the chatbot.
    """
    results = chunk_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])


@tool
def search_episodes(query: str) -> str:
    """
    Search System B — retrieves structured episodes with metadata.
    Used by the episodic memory variant of the chatbot.
    """
    results = episode_store.similarity_search(query, k=3)
    context = []
    for doc in results:
        meta = doc.metadata
        context.append(
            f"[{meta.get('destination','').upper()} — {meta.get('location','')}]\n"
            f"{doc.page_content}\n"
            f"Emotion: {meta.get('emotion','')}\n"
            f"Cost: ${meta.get('cost_aud','?')} AUD\n"
            f"Linked next: {meta.get('linked_next','')}"
        )
    return "\n\n---\n\n".join(context)

@tool
def search_images(query: str) -> str:
    """
    Search image episodes by visual description.
    Used for cross-modal queries.
    """
    results = episode_store.similarity_search(
        query,
        k=3,
        filter={"type": "image_episode"}
    )
    context = []
    for doc in results:
        meta = doc.metadata
        context.append(
            f"[Photo: {meta.get('location','')}]\n"
            f"{meta.get('image_description','')}\n"
            f"File: {meta.get('image_filename','')}"
        )
    return "\n\n---\n\n".join(context)


# Main Pipeline 
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

    all_episodes = []

    for note_file in sorted(note_files):
        dest = note_file.stem
        text = note_file.read_text(encoding="utf-8")

        print(f"\n{'='*55}")
        print(f"  {dest.upper()}")
        print(f"{'='*55}")

        # system A 
        ingest_chunks(dest, text)

        # system B
        episodes = extract_episodes(dest, text)
        if episodes:
            episodes = add_temporal_links(episodes)
            all_episodes.extend(episodes)
            out = EPISODES_DIR / f"{dest}_episodes.json"
            out.write_text(json.dumps(episodes, indent=2))
            print(f"  ✓ Saved to {out}")

        time.sleep(1)

    if all_episodes:
        print(f"\n  Linking {len(all_episodes)} episodes thematically...")
        all_episodes = add_thematic_links(all_episodes)
        store_text_episodes(all_episodes)
        graph_doc = EPISODES_DIR / "episode_graph.json"
        graph_doc.write_text(json.dumps(all_episodes, indent=2))
        print(f"  ✓ Episode graph saved")

    image_episodes = ingest_images()
    img_out        = EPISODES_DIR / "image_episodes.json"
    img_out.write_text(json.dumps(image_episodes, indent=2))

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"{'='*55}")
    print(f"  Text episodes : {len(all_episodes)}")
    print(f"  Image episodes: {len(image_episodes)}")
    print(f"\n  Tools ready for LangGraph agent:")
    print(f"  - search_chunks(query)")
    print(f"  - search_episodes(query)")
    print(f"  - search_images(query)")
    print(f"\n  Next: python chatbot.py")


if __name__ == "__main__":
    run_ingestion()
