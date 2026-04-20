# MemoryLens — Episodic Memory Travel Agent

## Research Question
"Does structuring retrieved context as temporally-linked episodes with user attribute metadata 
satisfy more user-stated constraints across multi-turn queries than retrieving fixed-size 
text chunks from the same raw data?"

## Project Structure
```
memorylens/
├── knowledge_base/
│   ├── notes/          ← your raw travel notes (.txt files)
│   ├── photos/         ← your travel photos (.jpg/.png)
│   ├── episodes/       ← auto-generated episode JSON files
│   └── chroma_db/      ← vector database (auto-created)
├── ingest.py           ← Step 1: build the knowledge base
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your API key
Create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

### 3. Run ingestion
```bash
python ingest.py
```

This will:
- Split your notes into fixed chunks → stored in System A (ChromaDB)
- Extract structured episodes via LLM → stored in System B (ChromaDB)
- Add temporal links (previous/next) between episodes
- Add thematic links across destinations
- Save all episodes to knowledge_base/episodes/

## What Gets Created

### System A (Baseline — Chunks)
Raw text split into 300-word chunks, no metadata, no links.

### System B (Innovation — Episodes)
Structured episodes with:
- Temporal links (previous → next within a trip)
- Thematic links (same theme across destinations)  
- Emotion tags
- Cost metadata
- Constraint-relevant fields (user preferences revealed in notes)python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
