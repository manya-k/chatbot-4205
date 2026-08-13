# MemoryLens

A research-focused personal travel assistant that compares three different memory retrieval architectures to answer user questions about their own travel experiences.

##  Purpose

MemoryLens is an experimental system designed to answer the research question:

**Does structuring retrieved context as temporally linked episodic memories with user attribute metadata improve personalised conversational performance, measured through constraint satisfaction rate (CSR), relevance, keyword match accuracy, and multi-turn consistency, when compared to retrieving fixed-size flat chunks from the same raw travel dataset?**

The system evaluates three variants:
- **Variant 0 (Plain LLM)**: Base flat chunks retrieval without memory
- **Variant A (Chunk Retrieval)**: Flat text chunks + AgentState memory 
- **Variant B (Episodic Memory)**: Structured episodes with temporal/thematic links + memory 

## What It Does

MemoryLens processes personal travel data in two parallel systems:

### **System A: Chunk Ingestion**
- Splits travel notes into 1200-character overlapping chunks
- Stores in ChromaDB for semantic similarity search
- Simple, flat retrieval baseline

### **System B: Episode Extraction**
- Uses LLM to structure travel notes into JSON episodes with:
  - Episode ID, destination, day number, location
  - Title, description, cost (AUD)
  - Tags (beach, food, temple, hiking, culture, etc.)
  - User constraint tracking
- Creates **temporal links** (previous/next episodes by day)
- Creates **thematic links** (episodes sharing tags)
- Stores structured episodes in ChromaDB with rich metadata

### **Image Processing**
- Scans travel photos directory
- Generates vivid descriptions using vision LLM (LLaVA)
- Stores as image episodes with location context

## Quick Start

### Prerequisites

```bash
# Download LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llava
```

### Directory Structure Setup
```
knowledge_base/
├── notes/              # Place .txt travel notes here
├── photos/             # Place travel photos (.jpg, .png, .heic)
├── episodes/           # Generated episode JSON files (auto-created)
└── chroma_db/          # Vector store (auto-created)
```

## How to Run

### 1. **Ingest Data**
Convert raw travel notes and photos into structured episodes:

In a new terminal, run 

```bash
ollama serve
```

```bash
# Install dependencies
pip install -r requirements.txt

# create a virtual environment
python3 -m venv new-venv    
source venv/bin/activate
```

```bash
# build the database
python ingests.py
```

**What `ingests.py` does:**
- Reads all `.txt` files from `knowledge_base/notes/`
- Extracts and chunks text (System A)
- Structures episodes with temporal/thematic links (System B)
- Generates photo descriptions and stores images
- Saves JSON outputs:
  - `{destination}_episodes.json` — extracted episodes per destination
  - `episode_graph.json` — complete episode graph with all links
  - `image_episodes.json` — image-based episodes


### 2. **Run Chatbot (Interactive Mode)**
Have multi-turn conversations with the agent:

```bash
# Default: Variant B (Episodic Memory)
cd memorylens
python chatbot.py --mode episodes

# Or choose variant:
python chatbot.py --mode plain      # Variant 0: Base LLM only
python chatbot.py --mode chunks     # Variant A: Chunk retrieval + memory
python chatbot.py --mode episodes   # Variant B: Episode retrieval + memory
```

**Commands in chat:**
- `quit` — Exit the conversation
- `memory` — Display stored user preferences this session
- `clear` — Reset conversation and memory

**Example session:**
```
MEMORYLENS — VARIANT B: Episodic Memory
Type 'quit' to exit, 'memory' to see preferences, 'clear' to reset

You: I hate crowded tourist spots
Assistant: I've noted your preference for avoiding crowded places.
[1.23s — episodes]

You: Plan me 3 days in Bali
Assistant: Based on your preference for avoiding crowds, I'd recommend:
- Day 1: Ubud traditional market early morning...
[2.15s — episodes]
```

### 3. **Run Evaluation (Benchmark)**
Systematically test all variants on 24 test queries:

```bash
# Test all 3 variants
python evaluate.py

# Or test one variant only
python evaluate.py --mode chunks
python evaluate.py --mode episodes
python evaluate.py --mode plain
```

**What it evaluates:**
- **24 queries** across 4 families:
  - 6 Factual queries (e.g., "How much did I spend in Bali?")
  - 6 Cross-Modal (e.g., "Show me a photo of a temple")
  - 6 Multi-Hop (e.g., "Which trip had best food-to-cost ratio?")
  - 6 Conversational (multi-turn with constraints)

**Metrics per query:**
- **Constraint Satisfaction Rate (CSR)**: Did agent honor user preferences?
- **Relevance (1-5)**: How personal and specific is the answer?
- **Keyword Match**: What percentage of expected keywords appear?
- **Latency**: Response time in seconds

**Output files in `evaluation/` folder:**
- `results.json` — Full detailed results for all queries


**Example output:**
```
================================================================================
  RESULTS SUMMARY
================================================================================
  Metric                                V0        VA        VB
  ───────────────────────────────────────────────────────────
  Avg Relevance (1-5)                 2.50      3.17      3.42
  Avg Keyword Match                   0.45      0.68      0.75
  Avg Final CSR                        N/A       0.85      0.92
  Avg Turn CSR                         N/A       0.80      0.89
  Avg Latency (s)                     1.23      1.45      1.52
```

## Project Structure

```
memorylens/
├── README.md                          # This file
├── ingests.py                         # Data ingestion pipeline
├── chatbot.py                         # Conversational agent
├── evaluate.py                        # Automated benchmark suite
└── knowledge_base/
    ├── notes/                         # Place travel notes here (.txt)
    ├── photos/                        # Place travel photos here
    ├── episodes/                      # Generated JSON episodes (auto)
    ├── chroma_db/                     # Vector database files (auto)
    └── venv/                          # Virtual environment (if used)
```

## Testing & Validation

### Test Categories

**1. Factual Queries** (F1-F6)
- Direct fact retrieval from travel notes
- Example: "How much did I spend per day in Bali?"
- Measures: Keyword accuracy, numerical precision

**2. Cross-Modal Queries** (C1-C6)
- Combine text and image retrieval
- Example: "Find me a photo I took of a temple"
- Measures: Multi-source integration

**3. Multi-Hop Queries** (M1-M6)
- Require reasoning across multiple episodes/destinations
- Example: "Which trip had the best food-to-cost ratio?"
- Measures: Reasoning depth, comparison ability

**4. Conversational Queries** (V1-V6)
- Multi-turn with user constraint stated in turn 1
- Example Turn 1: "I hate crowded places"
- Turn 2-3: Must honor constraint
- Measures: Memory retention, constraint satisfaction



## Configuration

### Key Parameters

**ingests.py:**
```python
MODEL = "llava"                    # LLM model to use
CHUNK_SIZE = 1200                  # System A chunk size
CHUNK_OVERLAP = 200                # Overlap between chunks
```

**chatbot.py:**
```python
MODEL = "llava"                    # LLM model
K = 5                              # Number of similar results to retrieve
MEMORY_SIZE = "last 3 turns"       # Conversation history window
```

**evaluate.py:**
```python
TEST_SET = [...]                   # 24 test queries
K = 5                              # Retrieval count (same for all variants)
OUTPUT_DIR = Path("evaluation")    # Results output directory
```


### Vector database already exists
To start fresh:
```bash
rm -rf knowledge_base/chroma_db
python ingests.py  # Rebuild from scratch
```

## AI Declaration:
chatGPT was used to receieve generic help with concepts. For instance, it was used to understand LangGraph and how the usage works
a copy of the entire history with chatGPT is attached in the references


## References
- LangChain: https://python.langchain.com/
- LangGraph: https://python.langchain.com/docs/langgraph/
- ChromaDB: https://www.trychroma.com/
- Ollama: https://ollama.ai/
- ChatGPT: https://chatgpt.com/share/6a05b898-0a9c-83ec-a768-9d4fa482af6a
