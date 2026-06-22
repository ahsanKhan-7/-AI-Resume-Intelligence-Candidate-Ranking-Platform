import json
import re
from pathlib import Path

import numpy as np
import pdfplumber
import shap
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SKILLS_LIST_PATH = Path(__file__).resolve().parent / "skills_list.json"
SENTENCE_BERT_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-1.5-flash"


WEIGHT_SEMANTIC_SIMILARITY = 0.40
WEIGHT_TFIDF_SIMILARITY = 0.15
WEIGHT_SKILLS_MATCH = 0.30
WEIGHT_EXPERIENCE = 0.15

FEATURE_NAMES = ["semantic_similarity", "tfidf_similarity", "skills_match_pct", "experience_score"]

MAX_RELEVANT_EXPERIENCE_YEARS = 10  
_sbert_model = None


# 1. PDF TEXT EXTRACTION

def extract_text_from_pdf(uploaded_file) -> str:
   
    pages_text = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
    except Exception as exc:
        print(f"[core.extract_text_from_pdf] failed: {exc}")
        return ""

    raw_text = "\n".join(pages_text)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return "\n".join(lines)


def guess_candidate_name(resume_text: str, fallback_filename: str) -> str:
    
    if not resume_text:
        return fallback_filename

    first_line = resume_text.splitlines()[0].strip()
    looks_like_name = (
        0 < len(first_line) <= 40
        and "@" not in first_line
        and not any(char.isdigit() for char in first_line)
    )
    return first_line if looks_like_name else fallback_filename



# 2. SKILL EXTRACTION

def load_skills_list() -> list:
    with open(SKILLS_LIST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [skill.lower() for skill in data["skills"]]


def extract_skills(text: str, skills_list: list) -> list:
    
    if not text:
        return []
    text_lower = text.lower()
    found = []
    for skill in skills_list:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(found)


def get_missing_skills(jd_skills: list, resume_skills: list) -> list:
    return sorted(set(jd_skills) - set(resume_skills))


def compute_skills_match_pct(jd_skills: list, resume_skills: list) -> float:
   
    if not jd_skills:
        return 0.0
    overlap = set(jd_skills) & set(resume_skills)
    return round(100 * len(overlap) / len(jd_skills), 2)



# 3. EXPERIENCE EXTRACTION

def extract_experience_years(text: str) -> float:
   
    if not text:
        return 0.0
    pattern = r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)\b"
    matches = re.findall(pattern, text.lower())
    if not matches:
        return 0.0
    return max(float(m) for m in matches)


def normalize_experience_score(years: float) -> float:
   
    return round(min(years / MAX_RELEVANT_EXPERIENCE_YEARS, 1.0), 3)



# 4 & 5. SIMILARITY: TF-IDF (lexical) + SENTENCE-BERT (semantic)

def compute_tfidf_similarity(jd_text: str, resume_text: str) -> float:
    
    if not jd_text or not resume_text:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(float(similarity), 4)


def get_sbert_model() -> SentenceTransformer:
   
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(SENTENCE_BERT_MODEL_NAME)
    return _sbert_model


def compute_semantic_similarity(jd_text: str, resume_text: str) -> float:
    
    if not jd_text or not resume_text:
        return 0.0
    model = get_sbert_model()
    embeddings = model.encode([jd_text, resume_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(similarity), 4)



# 6. XGBOOST RANKING MODEL + SHAP EXPLAINABILITY

def build_synthetic_labels(feature_matrix: np.ndarray) -> np.ndarray:
    """
    IMPORTANT (and worth saying out loud in an interview): we don't have
    thousands of historical "this candidate got hired / didn't" labels to
    train a real supervised ranking model on. The honest, standard approach
    for a project like this is to:

      1. Define what "a good match" means via a transparent weighted
         formula (semantic similarity + lexical similarity + skills match
         + experience) - this is our synthetic label.
      2. Train XGBoost to LEARN that scoring function from the features.

    Why bother training a model instead of just using the formula directly?
    Because this is what makes SHAP meaningful: SHAP explains which
    features the trained model relied on most for each prediction, which
    is the actual "Explainable AI" deliverable for this project. In a real
    production system, step 1's synthetic labels would be replaced by
    real historical hiring outcomes.
    """
    semantic = feature_matrix[:, 0]
    tfidf = feature_matrix[:, 1]
    skills_pct = feature_matrix[:, 2] / 100.0  
    experience = feature_matrix[:, 3]

    raw_score = (
        WEIGHT_SEMANTIC_SIMILARITY * semantic
        + WEIGHT_TFIDF_SIMILARITY * tfidf
        + WEIGHT_SKILLS_MATCH * skills_pct
        + WEIGHT_EXPERIENCE * experience
    )
    return raw_score * 100  


def train_ranking_model(feature_matrix: np.ndarray):
    """
    Trains an XGBoost regressor on the current batch of candidates.

    Parameters
    ----------
    feature_matrix : np.ndarray of shape (n_candidates, 4), columns in the
                      order defined by FEATURE_NAMES.

    Returns
    -------
    (model, predicted_scores) : trained XGBRegressor + a (n_candidates,)
                                 array of final scores (0-100, rounded).
    """
    labels = build_synthetic_labels(feature_matrix)

    model = xgb.XGBRegressor(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(feature_matrix, labels)

    predicted_scores = model.predict(feature_matrix)
    predicted_scores = np.clip(predicted_scores, 0, 100)
    return model, np.round(predicted_scores, 2)


def explain_candidates(model, feature_matrix: np.ndarray) -> list:
    """
    Runs SHAP explainability with a safe fallback.
    If SHAP fails because of XGBoost compatibility issues,
    the app will continue working instead of crashing.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_matrix)

        explanations = []
        for row in shap_values:
            explanations.append(
                {
                    name: round(float(val), 3)
                    for name, val in zip(FEATURE_NAMES, row)
                }
            )

        return explanations

    except Exception as e:
        print(f"[SHAP ERROR] {e}")

        explanations = []

        for row in feature_matrix:
            explanations.append(
                {
                    "semantic_similarity": round(float(row[0]), 3),
                    "tfidf_similarity": round(float(row[1]), 3),
                    "skills_match_pct": round(float(row[2]), 3),
                    "experience_score": round(float(row[3]), 3),
                }
            )

        return explanations


# 7. GEMINI INTERVIEW QUESTION GENERATION
 
def generate_interview_questions(jd_text: str, candidate_name: str, matched_skills: list,
                                  missing_skills: list, api_key: str) -> list:
    
    if not api_key:
        return _fallback_interview_questions(matched_skills, missing_skills)

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        prompt = f"""You are a technical interviewer preparing questions for {candidate_name}.

Job description:
{jd_text[:2000]}

Candidate's matched skills: {', '.join(matched_skills) if matched_skills else 'None clearly listed'}
Skills required by the JD but not found in the resume: {', '.join(missing_skills) if missing_skills else 'None'}

Generate exactly 5 interview questions:
- 3 should probe real depth on the candidate's matched skills (not generic definitions).
- 2 should check whether the missing skills are a genuine gap (e.g. "Have you worked with X in any capacity, even informally?").

Return ONLY a numbered list of 5 questions, no preamble or extra commentary."""

        response = model.generate_content(prompt)
        questions = _parse_numbered_list(response.text)
        return questions if questions else _fallback_interview_questions(matched_skills, missing_skills)

    except Exception as exc:
        print(f"[core.generate_interview_questions] Gemini call failed: {exc}")
        return _fallback_interview_questions(matched_skills, missing_skills)


def _parse_numbered_list(text: str) -> list:
    """Pulls clean question strings out of Gemini's numbered-list response."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    questions = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned:
            questions.append(cleaned)
    return questions[:5]


def _fallback_interview_questions(matched_skills: list, missing_skills: list) -> list:
    """Simple template-based fallback so the UI always has something to
    show even without a valid Gemini API key."""
    questions = []
    for skill in matched_skills[:3]:
        questions.append(f"Can you walk me through a project where you applied {skill} and a challenge you faced?")
    for skill in missing_skills[:2]:
        questions.append(f"Our JD mentions {skill} - have you had any exposure to it, even informally?")
    while len(questions) < 5:
        questions.append("Tell me about a project you're most proud of and why.")
    return questions[:5]
