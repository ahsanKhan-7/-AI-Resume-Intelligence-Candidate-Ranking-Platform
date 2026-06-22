# 🧠 AI Resume Intelligence & Candidate Ranking Platform

> *"Teaching a machine to read resumes faster than your eyes glaze over at the 47th one."*

A practical, end-to-end **AI/ML system** that takes a Job Description and a pile of resumes, and hands you back a ranked, explainable shortlist — complete with missing-skill gaps and AI-generated interview questions. Built as a hands-on portfolio project to actually *use* the ML/NLP stack instead of just reading about it.

No database. No Docker. No 47-microservice fever dream. Just clean Python, real ML, and a Streamlit dashboard that runs on your laptop in under 2 minutes.

---

## ✨ What it actually does

| Step | What's happening under the hood |
|---|---|
| 📄 Upload a JD | Paste text or upload a `.txt`/`.pdf` |
| 📤 Upload resumes | Multiple PDFs at once |
| 🔍 Extract | `pdfplumber` pulls clean text out of each resume |
| 🛠️ Skill matching | Keyword-matches against a curated skills dictionary |
| 📐 Lexical similarity | TF-IDF + Cosine Similarity (the "do the words match" check) |
| 🧬 Semantic similarity | Sentence-BERT embeddings + Cosine Similarity (the "do the *meanings* match" check) |
| 🌳 Ranking | An XGBoost regressor scores every candidate |
| 🔬 Explainability | SHAP shows *exactly* why each candidate scored what they scored |
| 🎤 Interview prep | Gemini generates tailored interview questions per candidate |
| 📊 Dashboard | Everything lands in a clean Streamlit UI |

It's basically a tiny, honest version of what real ATS (Applicant Tracking System) ranking engines do — minus the enterprise bloat.

---

## 🧱 Tech Stack

- **UI:** Streamlit
- **ML/NLP:** scikit-learn (TF-IDF), sentence-transformers (Sentence-BERT), XGBoost, SHAP
- **PDF parsing:** pdfplumber
- **LLM:** Google Gemini (`gemini-1.5-flash`)
- **Storage:** `st.session_state` only — nothing touches disk, nothing persists after you close the tab 🔒
- **Package manager:** `uv`

---

## 📂 Project Structure

```
resume_ranker/
├── app.py             # Streamlit UI - all 5 pages, sidebar navigation
├── core.py             # All the ML/NLP logic - the actual brain
├── skills_list.json   # Curated skills dictionary used for matching
└── pyproject.toml     # Dependencies (managed via uv)
```

Four files. That's it. No `services/`, no `controllers/`, no `factory_factory.py`. 😄

---

## 🚀 Running it yourself

```bash
# 1. Clone / unzip the project, then cd into it
cd resume_ranker

# 2. Create the virtual environment with uv
uv venv

# 3. Activate it
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # Mac/Linux

# 4. Install dependencies
uv sync

# 5. (Optional) Add your Gemini API key for AI-generated interview questions
#    Create a .env file in the project root with:
#    GEMINI_API_KEY=your_key_here
#    Get a free key at https://aistudio.google.com/apikey
#    (Skip this and the app still works - it just falls back to template questions)

# 6. Run it
uv run streamlit run app.py
```

Opens automatically at `http://localhost:8501`. Upload a JD, throw some resumes at it, and watch it rank them.

---


## 🔒 Privacy by design

Everything lives in `st.session_state` for the duration of your browser session  no database, no saved files, no logs of who uploaded what. Close the tab, and it's gone. Refresh the page, and you start clean. This was a deliberate design choice, not a missing feature.

---

## 🛣️ Ideas for going further

- Swap the synthetic training label for real historical hiring outcomes (if you ever have them)
- Add support for `.docx` resumes
- Let recruiters tune the scoring weights live from the sidebar
- Persist sessions per-recruiter with proper auth (out of scope for this version, on purpose)

---

## 🙋 Why this project exists

Built as a hands-on AI/ML portfolio project — to actually wire together NLP, classic ML, gradient boosting, and explainability in one coherent pipeline instead of doing five disconnected tutorials. If you're a recruiter reading this: yes, I know exactly why every line of code is here. Ask me anything. 😎
