# Financial Report Assistant

An evidence-focused financial document analysis project built with Python and
[Docling](https://github.com/docling-project/docling). The application is being
developed into a retrieval-augmented generation (RAG) assistant that will answer
questions about company financial reports and show the source evidence behind
each answer.

> **Project status:** Early development. PDF ingestion is implemented. Chunking,
> semantic retrieval, answer generation, citations, evaluation, and the user
> interface are planned next.

## Why this project?

Annual reports and regulatory filings contain valuable information, but they
are long and difficult to search manually. This project aims to make those
documents easier to explore while keeping answers traceable to the original
report.

The finished assistant should be able to answer questions such as:

- What were the main drivers of revenue growth?
- How did operating expenses change from the previous year?
- Which risks did management identify?
- What evidence in the report supports the answer?

This project is intended for document analysis and learning. It does not provide
investment advice.

## Current features

- Accepts a local PDF financial report
- Validates that the input exists and is a PDF
- Converts the report with Docling
- Exports readable Markdown for manual inspection
- Exports structured Docling JSON for downstream processing and provenance
- Keeps source reports and generated output out of Git through `.gitignore`

## Planned features

- Structure-aware document chunking
- Page and section metadata for citations
- Local text embeddings
- Semantic and hybrid retrieval
- Evidence-grounded answer generation
- Insufficient-evidence handling
- A Streamlit user interface
- Retrieval and answer-quality evaluation
- Support for multiple reports, companies, and reporting periods

## Current processing pipeline

```text
Financial-report PDF
        |
        v
Docling document conversion
        |
        +--> Markdown for human inspection
        |
        +--> Structured JSON for chunking and provenance
```

The planned RAG stages will extend this pipeline:

```text
Structured document
        |
        v
Page-aware chunks
        |
        v
Embeddings and retrieval
        |
        v
LLM answer with source citations
```

## Technology

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for Python and dependency management
- [Docling](https://docling-project.github.io/docling/) for document conversion
- Git and GitHub for version control

More dependencies will be introduced only when their corresponding feature is
implemented.

## Project structure

```text
financial_report_assistant/
|-- data/                       # Local source reports (not committed)
|-- output/                     # Generated Markdown and JSON (not committed)
|-- src/
|   `-- financial_report_assistant/
|       |-- __init__.py
|       `-- ingest.py           # Docling ingestion command
|-- tests/                      # Automated tests will be added here
|-- .gitignore
|-- .python-version
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Getting started

### Prerequisites

Install:

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
inside `data/`, for example:

```text
data/annual_report.pdf
```

The `data/` directory is intentionally excluded from Git.

### 4. Convert the report

From the repository root, run:

```powershell
uv run python .\src\financial_report_assistant\ingest.py .\data\annual_report.pdf
```

To select a different output directory:

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

## Verifying the conversion

Before using the document in a RAG pipeline, compare the generated Markdown
with the original PDF. Check:

- Section headings and reading order
- Financial tables and their column order
- Currency symbols, units, and negative values
- Footnotes
- Page provenance in the structured JSON

A successful conversion command does not guarantee that every table or page was
interpreted correctly. Extraction quality must be verified before retrieval is
built on top of it.

## Development roadmap

1. Convert and validate a financial-report PDF
2. Create structure-aware chunks with page metadata
3. Generate local embeddings
4. Implement semantic retrieval
5. Add grounded answer generation and citations
6. Build an evaluation dataset
7. Add a web interface
8. Support report and company comparisons

## Data and privacy

The repository does not include source financial reports, generated extraction
files, virtual environments, or `.env` files. Never commit API keys or other
secrets.

## Disclaimer

This project is for educational and informational purposes only. It is not
financial, investment, tax, or legal advice. Generated answers must be verified
against the cited source documents.
