# 🚀 FusionRAG: Local Multimodal Agentic RAG

FusionRAG is a fully local, completely free, self-correcting AI system built for Apple Silicon (M-series Macs). It acts as a **Multi-Agent Orchestrator** capable of:

* 📄 Reading PDFs
* 🌐 Searching the live internet
* 🎥 Processing YouTube videos (Audio + Vision simultaneously)

---

# 🛠️ Prerequisites & System Setup

Before running the Python code, install the core dependencies on your Mac.

## 1. Install FFmpeg (Audio Processing)

Whisper AI requires `ffmpeg` to process media files.

```bash
brew install ffmpeg
```

---

## 2. Install & Configure Ollama (Local LLM Engine)

Download and install **Ollama for Mac**, then pull the required models:

### 🧠 Core Model (Text, Routing, Reasoning)

```bash
ollama pull llama3.2
```

### 👁️ Vision Model (Image + Video Understanding)

```bash
ollama pull llama3.2-vision
```

---

## 3. Setup Python Environment

It is recommended to use an isolated environment to avoid dependency conflicts.

```bash
conda create -n rag_env python=3.11
conda activate rag_env
pip install -r requirements.txt
```

---

# 🏃‍♂️ Running the System

FusionRAG is built in phases. Run the script corresponding to your use case.

---

## 📘 Phase 1: Standard PDF RAG

**File:** `main_p1.py`

### What it does:

* Reads a PDF document
* Splits it into chunks
* Converts chunks into embeddings
* Stores them in ChromaDB
* Retrieves relevant chunks to answer queries

### Run:

```bash
python main_p1.py
```

---

## 🤖 Phase 2 & 3: Agentic "Corrective" RAG (CRAG) + Web Routing

**File:** `main_p2.py`

### What it does:

* Builds a multi-agent system using **LangGraph**

### Components:

* **Router** → Decides between local DB vs internet
* **Web Search** → Uses DuckDuckGo (ddgs) for live scraping
* **Critic** → Evaluates responses for hallucinations

  * Can trigger up to **3 rewrites** if needed

### Run:

```bash
python main_p2.py
```

---

## 🎥 Phase 4: Multimodal Agentic RAG (Flagship)

**File:** `main_multimodal.py`

### What it does:

Processes YouTube videos using both **audio and visual understanding**.

### Pipeline:

#### 🎧 Audio Lane

* Downloads video
* Transcribes using Whisper
* Produces timestamped speech

#### 👁️ Vision Lane

* Extracts frames every 5 seconds
* Uses Llama-3.2-Vision for:

  * Scene understanding
  * OCR text extraction

#### 🔗 Fusion Layer

* Combines audio + vision into **Super-Chunks**
* Each chunk represents a 5-second interval

#### 🤖 Agent System

* Stores Super-Chunks in ChromaDB
* Uses LangGraph with:

  * Router
  * Writer
  * Critic
* Enables conversational querying over video content

### Run:

```bash
python main_multimodal.py
```

---

# 🧠 Architecture Overview

### 🔹 Embeddings

* Model: `all-MiniLM-L6-v2` (HuggingFace)
* Optimized for fast local inference

### 🔹 Vector Store

* **ChromaDB** (fully local)

### 🔹 Orchestration

* **LangGraph** manages agent workflow and state transitions

### 🔹 Agent Components

* Router
* Web Searcher
* Retriever
* Generator
* Critic (Self-correction loop)

---

# ✅ Summary

FusionRAG is a powerful, fully local RAG system combining:

* Multi-agent reasoning
* Self-correction
* Multimodal understanding (text, audio, vision)

All without relying on paid APIs 🚀
