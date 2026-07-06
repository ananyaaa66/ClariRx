# ClariRx

**Understand your prescriptions and lab reports — in plain language.**

ClariRx is a web application that helps users understand their medical prescriptions and lab reports by extracting structured data via OCR, looking it up against a curated knowledge base, and generating clear Hindi/English explanations.

## Target Users
- Elderly patients in India
- First-time chronic-disease patients
- Families who struggle with doctor's handwriting and jargon-heavy lab reports

## Tech Stack
| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| OCR | PaddleOCR (primary) + LLM Vision (baseline comparison) |
| Extraction | LLM structured extraction (MVP) + Fine-tuned BioBERT NER (depth) |
| Explanation | LLM grounded in curated knowledge base |
| Reminders | APScheduler |
| Frontend | React (Vite) |
| Eval | Custom Python scripts → CSV/Markdown reports |

## Project Status
🚧 Under active development — Phase 0 & 1

## Results (to be filled after eval)
| Module | Metric | Method A | Method B |
|---|---|---|---|
| OCR | CER | PaddleOCR: _TBD_ | LLM Vision: _TBD_ |
| Extraction | P/R/F1 | LLM: _TBD_ | BioBERT: _TBD_ |
| End-to-End | Accuracy | _TBD_ | — |

## Setup
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# fill in your API keys in .env
```

## License
This project is for educational/portfolio purposes. No real patient data is used.
