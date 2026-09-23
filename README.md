<div align="center">

# 📊 Financial Report Assistant

### Evidence-focused analysis of financial reports with Docling and RAG

[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docling](https://img.shields.io/badge/Docling-Document%20AI-6F42C1)](https://docling-project.github.io/docling/)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9)](https://docs.astral.sh/uv/)
![Status](https://img.shields.io/badge/Status-Early%20Development-F4B942)

Convert long company reports into structured, searchable documents and build
toward answers that remain traceable to their original evidence.

</div>

> [!IMPORTANT]
> **Current status:** PDF ingestion is implemented. Chunking, retrieval,
> answer generation, citations, evaluation, and the user interface are the next
> development stages.

## 🧭 Table of contents

- [About the project](#-about-the-project)
- [Current features](#-current-features)
- [Planned features](#-planned-features)
- [How it works](#️-how-it-works)
- [Technology](#️-technology)
- [Project structure](#-project-structure)
- [Getting started](#-getting-started)
- [Verify the conversion](#-verify-the-conversion)
- [Roadmap](#️-roadmap)
- [Data and privacy](#-data-and-privacy)
- [Disclaimer](#️-disclaimer)

## 💡 About the project

Annual reports and regulatory filings contain valuable information, but they
are often hundreds of pages long and difficult to search manually. The
Financial Report Assistant aims to make those documents easier to explore while
keeping every answer connected to verifiable source evidence.

The finished assistant should answer questions such as:

- 📈 What were the main drivers of revenue growth?
- 💰 How did operating expenses change from the previous year?
- ⚠️ Which risks did management identify?
- 🔎 Which passages and pages support the answer?

### 🎯 Project goals

| Goal | Why it matters |
|---|---|
| Preserve document structure | Headings, tables, and sections provide essential context |
| Retain page provenance | Users must be able to verify an answer in the source report |
| Ground every answer | Fluent but unsupported financial claims are not acceptable |
| Evaluate retrieval | Search quality should be measured instead of guessed |
| Handle uncertainty | The assistant should admit when the report lacks enough evidence |

## ✅ Current features

- [x] Accept a local financial-report PDF
- [x] Validate that the input exists and uses the PDF format
- [x] Convert the report with Docling
- [x] Export readable Markdown for manual inspection
- [x] Export structured Docling JSON for downstream processing
- [x] Preserve source structure and provenance for later citations
- [x] Exclude source reports, generated output, environments, and secrets from Git

## 🚧 Planned features

- [ ] Structure-aware document chunking
- [ ] Page and section metadata for citations
- [ ] Local text embeddings
- [ ] Semantic and hybrid retrieval
- [ ] Evidence-grounded answer generation
- [ ] Insufficient-evidence handling
- [ ] Streamlit user interface
- [ ] Retrieval and answer-quality evaluation
- [ ] Multiple reports, companies, and reporting periods

## ⚙️ How it works

### Current ingestion pipeline

```text
📄 Financial-report PDF
          │
          ▼
🧠 Docling document conversion
          │
          ├──► 📝 Markdown for human inspection
          │
          └──► 🧩 Structured JSON for chunking and provenance
```

### Planned RAG pipeline

```text
🧩 Structured document
          │
          ▼
✂️ Page-aware chunks
          │
          ▼
🔢 Embeddings and retrieval
          │
          ▼
💬 LLM answer with source citations
```

> [!NOTE]
> The language model will be the presentation layer—not the source of financial
> facts. Answers must come from retrieved report evidence.

## 🛠️ Technology

| Tool | Purpose |
|---|---|
| [Python 3.13+](https://www.python.org/) | Application and data-processing language |
| [uv](https://docs.astral.sh/uv/) | Python version, environment, and dependency management |
| [Docling](https://docling-project.github.io/docling/) | PDF conversion, layout understanding, tables, and provenance |
| Git | Local version control |
| GitHub | Remote repository and project portfolio |

Additional dependencies will be introduced only when their corresponding
features are implemented.

## 📁 Project structure

```text
financial_report_assistant/
├── data/                       # Local source reports (not committed)
├── output/                     # Generated Markdown and JSON (not committed)
├── src/
│   └── financial_report_assistant/
│       ├── __init__.py
│       └── ingest.py           # Docling ingestion command
├── tests/                      # Automated tests will be added here
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## 🚀 Getting started

### Prerequisites

Install the following tools:

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

`uv` creates and manages the project virtual environment automatically.

### 3. Add a financial report

Download a public annual report from an official company investor-relations
website or from [SEC EDGAR](https://www.sec.gov/search-filings). Place the PDF
inside `data/`:

```text
data/annual_report.pdf
```

> [!TIP]
> Start with a text-based PDF in which you can select words with your cursor.
> Scanned reports may require additional OCR configuration.

The `data/` directory is intentionally excluded from Git.

### 4. Convert the report

Run the command from the repository root:

```powershell
uv run python .\src\financial_report_assistant\ingest.py .\data\annual_report.pdf
```

To choose a different output directory:

```powershell
uv run python .\src\financial_report_assistant\ingest.py `
    .\data\annual_report.pdf `
    --output-dir .\output
```

Successful conversion creates:

```text
output/annual_report.md
output/annual_report.json
```

## 🔍 Verify the conversion

A successful command does not guarantee that every table or page was
interpreted correctly. Compare the generated Markdown with the original PDF
before building retrieval on top of it.

### Inspection checklist

- [ ] Section headings appear in the correct order
- [ ] Paragraphs follow the original reading order
- [ ] Financial table columns match their correct years
- [ ] Currency symbols and units are preserved
- [ ] Negative values remain negative
- [ ] Footnotes remain associated with the correct content
- [ ] The structured JSON contains page provenance

## 🗺️ Roadmap

| Phase | Milestone | Status |
|---:|---|:---:|
| 1 | Convert and validate a financial-report PDF | ✅ Complete |
| 2 | Create structure-aware chunks with page metadata | 🚧 Next |
| 3 | Generate local embeddings | ⏳ Planned |
| 4 | Implement semantic and hybrid retrieval | ⏳ Planned |
| 5 | Add grounded answers and citations | ⏳ Planned |
| 6 | Build an evaluation dataset | ⏳ Planned |
| 7 | Add a web interface | ⏳ Planned |
| 8 | Support company and report comparisons | ⏳ Planned |


## ⚖️ Disclaimer

This project is for educational and informational purposes only. It does not
provide financial, investment, tax, or legal advice. Generated answers must
always be verified against the cited source documents.

---

<div align="center">

Built as a hands-on project for learning document AI, RAG, evaluation, and
responsible financial-data analysis. 🚀

</div>
