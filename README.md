<div align="center">

# 📊 Financial Report Assistant

### Evidence-focused financial-report retrieval with Docling, FAISS, BM25, and RAG

[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docling](https://img.shields.io/badge/Docling-Document%20AI-6F42C1)](https://docling-project.github.io/docling/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF)](https://github.com/facebookresearch/faiss)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9)](https://docs.astral.sh/uv/)
![Status](https://img.shields.io/badge/Status-Retrieval%20Evaluation-F4B942)

Convert long annual reports into structured, searchable evidence and retrieve
the passages needed for grounded financial analysis.

</div>



## 🧭 Table of contents

- [About the project](#-about-the-project)
- [Current features](#-current-features)
- [Architecture](#️-architecture)
- [Technology](#️-technology)
- [Project structure](#-project-structure)
- [Getting started](#-getting-started)
- [Run the pipeline](#-run-the-pipeline)
- [Evaluate retrieval](#-evaluate-retrieval)
- [Design decisions](#-design-decisions)
- [Roadmap](#️-roadmap)
- [Data and privacy](#-data-and-privacy)
- [Disclaimer](#️-disclaimer)

## 💡 About the project

Annual reports contain valuable financial information, but they are often
hundreds of pages long and difficult to explore manually. This project builds
the retrieval foundation for an assistant that can answer questions while
keeping every claim connected to evidence in the source report.

Example questions include:

- 📈 What were the company's main sources of revenue?
- 💰 How did operating expenses change from the previous year?
- ⚠️ Which risks did management identify?
- 🔎 Which passages and pages support the answer?

### 🎯 Project goals

| Goal | Why it matters |
|---|---|
| Preserve document structure | Headings, tables, and sections provide essential context |
| Retain page provenance | Users must be able to verify answers in the source report |
| Combine semantic and keyword search | Financial questions contain both concepts and exact terminology |
| Evaluate retrieval | Search quality should be measured instead of guessed |
| Handle uncertainty | The finished assistant should refuse unsupported conclusions |


## ⚙️ Architecture

```text
📄 Financial-report PDF
          │
          ▼
🧠 Docling conversion
          │
          ├──► 📝 Markdown for human inspection
          └──► 🧩 Structured Docling JSON
                          │
                          ▼
               ✂️ Structure-aware chunks
                 + headings and page metadata
                          │
                          ▼
               🔢 Normalized local embeddings
                          │
                          ▼
                  🗂️ FAISS vector index
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
 🔍 Semantic vector search         🔤 BM25 keyword search
          └───────────────┬───────────────┘
                          ▼
              🔀 Reciprocal Rank Fusion
                          │
                          ▼
              📚 Ranked evidence passages
                          │
                          ▼
          💬 Grounded answers with citations
                    (next milestone)
```



## 🛠️ Technology

| Tool | Purpose |
|---|---|
| [Python 3.13+](https://www.python.org/) | Application and data-processing language |
| [uv](https://docs.astral.sh/uv/) | Environment and dependency management |
| [Docling](https://docling-project.github.io/docling/) | PDF layout, tables, text, and provenance extraction |
| [Sentence Transformers](https://www.sbert.net/) | Local document and query embeddings |
| [FAISS](https://github.com/facebookresearch/faiss) | Exact vector similarity search |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | Lexical keyword retrieval |
| Git and GitHub | Version control and portfolio hosting |

## 📁 Project structure

```text
financial_report_assistant/
├── data/                           # Local PDFs; not committed
├── evaluation/
│   └── retrieval_questions.json    # Manually verified relevance labels
├── output/                         # Generated artifacts; not committed
├── src/
│   └── financial_report_assistant/
│       ├── ingest.py               # Convert PDF to Markdown and Docling JSON
│       ├── chunk_document.py       # Create page-aware JSONL chunks
│       ├── build_index.py          # Generate embeddings and build FAISS index
│       ├── search.py               # Vector-search baseline
│       ├── hybrid_search.py        # FAISS + BM25 + RRF retrieval
│       └── evaluate_retrieval.py   # Compare retrieval strategies
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## 🚀 Getting started

### Prerequisites

- [Git](https://git-scm.com/downloads)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### 1. Clone the repository

```powershell
git clone https://github.com/gautamikh/financial-report-assistant.git
cd financial-report-assistant
```

### 2. Install the locked dependencies

```powershell
uv sync
```

`uv` creates and manages the virtual environment automatically.

### 3. Add a financial report

Download a public annual report from an official investor-relations website or
from [SEC EDGAR](https://www.sec.gov/search-filings), then place it in `data/`.

The examples below use:

```text
data/alphabet_2025_annual_report.pdf
```


## ▶️ Run the pipeline

Run all commands from the repository root.

### 1. Convert the PDF

```powershell
uv run python .\src\financial_report_assistant\ingest.py `
  .\data\alphabet_2025_annual_report.pdf `
  --output-dir .\output
```

This creates readable Markdown and structured Docling JSON. Inspect the
Markdown against the original report before continuing, especially tables,
negative numbers, units, footnotes, and reading order.

### 2. Create structure-aware chunks

```powershell
uv run python .\src\financial_report_assistant\chunk_document.py `
  .\output\alphabet_2025_annual_report.json `
  --output-path .\output\alphabet_2025_chunks.jsonl `
  --company "Alphabet" `
  --fiscal-year 2025 `
  --document-type "annual-report"
```

Each chunk includes its text, contextualized embedding text, headings, page
numbers, document references, token count, and report metadata.

### 3. Build the FAISS index

```powershell
uv run python .\src\financial_report_assistant\build_index.py `
  .\output\alphabet_2025_chunks.jsonl `
  --output-dir .\output\alphabet_2025_index
```

The index directory contains:

```text
index.faiss       # Normalized vectors
metadata.jsonl    # Chunk data in matching FAISS row order
manifest.json     # Model, dimensions, metric, and index configuration
```

### 4. Run hybrid retrieval

```powershell
uv run python .\src\financial_report_assistant\hybrid_search.py `
  "What were Alphabet's main sources of revenue?" `
  --index-dir .\output\alphabet_2025_index `
  --candidate-k 20 `
  --top-k 5
```

The command prints the fused rank, vector and BM25 ranks, similarity scores,
chunk ID, page numbers, section headings, and retrieved passage.

## 🧪 Evaluate retrieval

`evaluation/retrieval_questions.json` stores manually verified questions and
relevant chunk IDs. Only verified, answerable questions with relevance labels
are included in the automated evaluation.

```powershell
uv run python .\src\financial_report_assistant\evaluate_retrieval.py `
  --index-dir .\output\alphabet_2025_index `
  --questions .\evaluation\retrieval_questions.json `
  --candidate-k 20 `
  --top-k 5
```

The evaluator compares vector, BM25, and hybrid rankings using:

| Metric | Meaning |
|---|---|
| Hit Rate@5 | Fraction of questions with at least one relevant chunk in the top five |
| Recall@5 | Fraction of all labelled relevant chunks found in the top five |
| MRR | Average reciprocal rank of the first relevant result |

Unanswerable questions are reserved for a later evaluation of the assistant's
ability to refuse unsupported answers.



## 🗺️ Roadmap

| Phase | Milestone | Status |
|---:|---|:---:|
| 1 | Convert and validate a financial-report PDF | ✅ Complete |
| 2 | Create structure-aware chunks with page metadata | ✅ Complete |
| 3 | Generate local embeddings and build a FAISS index | ✅ Complete |
| 4 | Implement vector, BM25, and hybrid retrieval | ✅ Complete |
| 5 | Build and label the retrieval evaluation set | 🚧 In progress |
| 6 | Generate evidence-grounded answers with citations | ⏳ Next |
| 7 | Evaluate answer faithfulness and unsupported-question refusal | ⏳ Planned |
| 8 | Add a Streamlit user interface | ⏳ Planned |
| 9 | Support multiple companies and reporting periods | ⏳ Planned |

## ⚖️ Disclaimer

This project is for educational and informational purposes only. It does not
provide financial, investment, tax, or legal advice. Any generated answer must
be verified against its cited source document.

---

<div align="center">

Built as a hands-on project for learning document AI, information retrieval,
RAG evaluation, and responsible financial-data analysis. 🚀

</div>
