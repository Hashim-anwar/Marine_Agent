# MarineWise AI

A simple Streamlit MVP for marine engine troubleshooting and technician training.

## What it does

- **Troubleshooting Agent:** searches uploaded marine-engine manuals with FAISS and answers from the retrieved pages.
- **Technical Training:** generates training material as PDF, PowerPoint, and/or Word.
- **Quiz Generator:** creates a quiz PDF with an answer key.
- **Score an Assessment:** OCRs a JPG/PNG assessment and scores it against an answer key. If the score is below 50%, it creates a remedial PowerPoint.
- **Learning:** explains RAG, Advanced RAG, chunks, and FAISS.
- **Optional web fallback:** if the answer is not in the manuals, the troubleshooting page can use Groq's browser-search tool and labels the answer as web-sourced.

## Important safety note

This is an MVP, not a certified marine engineering system. Always verify troubleshooting steps, limits, isolation procedures, and safety requirements against the engine manufacturer's current documentation and vessel procedures.

## 1. Download the five files

Put these five files in one folder:

```text
MarineWise-AI/
├── app.py
├── agents.py
├── rag.py
├── requirements.txt
└── README.md
```

## 2. Install Python

Use Python **3.11 or 3.12** for the easiest beginner setup.

Check your version:

```bash
python --version
```

## 3. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install the libraries

```bash
pip install -r requirements.txt
```

The first manual search also downloads the small `all-MiniLM-L6-v2` embedding model from Sentence Transformers.

### Assessment OCR note

The assessment scorer uses `pytesseract`, which also needs the **Tesseract OCR application** installed on your computer. If OCR is unavailable, the app will show a friendly error instead of crashing.

## 5. Add your Groq API key

Never put the key inside the Python files.

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_KEY"
```

### macOS / Linux

```bash
export GROQ_API_KEY="YOUR_GROQ_KEY"
```

Or create `.streamlit/secrets.toml` locally:

```toml
GROQ_API_KEY = "YOUR_GROQ_KEY"
```

Do **not** commit that file to GitHub.

The app uses the Groq model:

```text
openai/gpt-oss-120b
```

## 6. Run locally

```bash
streamlit run app.py
```

Your browser should open the Streamlit app.

## 7. Add manuals

In the left sidebar you can either:

1. Upload one or more PDF manuals, **or**
2. Paste a public Google Drive PDF link.

Both are optional. The app does **not** download from Google Drive when it starts.

Click **Build / Rebuild FAISS Index** when you are ready. The index is kept in `st.session_state` for the current app session.

For Google Drive, the PDF needs to be accessible to the person/server running the app (for example, "Anyone with the link").

## 8. Push to GitHub

Create a new GitHub repository, then in your project folder:

```bash
git init
git add app.py agents.py rag.py requirements.txt README.md
git commit -m "Initial MarineWise AI MVP"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

Do not commit `.streamlit/secrets.toml` or any API key.

## 9. Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud.
2. Create a new app.
3. Select your GitHub repository.
4. Select branch `main`.
5. Set the main file to `app.py`.
6. Deploy.
7. Open the app's **Secrets** settings.
8. Add:

```toml
GROQ_API_KEY = "YOUR_GROQ_KEY"
```

9. Save/redeploy.

The app reads the key from `st.secrets` first and the environment as a fallback.

## 10. First test

For a quick smoke test:

1. Open the app.
2. Upload a small text-based marine PDF.
3. Click **Build / Rebuild FAISS Index**.
4. Open **Troubleshooting Agent**.
5. Enter the manufacturer, model, and a known alarm/symptom from the manual.
6. Confirm that the response shows the retrieved manual filename and page.
7. Open **Technical Training** and generate a small PDF/PPT.
8. Open **Learning** to see the RAG/FAISS explanation.

## MVP architecture

```text
PDF manual
   ↓
PyMuPDF extracts page text
   ↓
900-character overlapping chunks
   ↓
Sentence Transformer embeddings
   ↓
FAISS vector index
   ↓
Top matching manual pages
   ↓
Groq GPT-OSS 120B
   ↓
Troubleshooting / training / quiz
```

## Beginner notes

- `rag.py` handles PDFs, chunks, embeddings, FAISS search, and Google Drive download.
- `agents.py` contains the CrewAI agent role definitions.
- `app.py` contains all Streamlit pages and document-generation code so there are as few files as possible.
- The actual LLM calls use the **Groq Python client**, as requested.
- The optional web fallback uses Groq's built-in browser search for the same GPT-OSS 120B model.
