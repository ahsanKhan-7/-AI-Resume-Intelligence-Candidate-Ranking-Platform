import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

import core

load_dotenv() 
st.set_page_config(page_title="AI Resume Intelligence Platform", page_icon="🧠", layout="wide")




defaults = {
    "job_title": "",
    "jd_text": "",
    "jd_skills": [],
    "candidates": [],          
    "shap_explanations": {},   
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.sidebar.title("🧠 Resume Ranker")
page = st.sidebar.radio(
    "Navigate",
    [
        "1. Job Description Upload",
        "2. Resume Upload",
        "3. Candidate Ranking Dashboard",
        "4. Candidate Details",
        "5. Interview Question Generator",
    ],
)

with st.sidebar.expander("🔒 Privacy note"):
    st.caption(
     
    )
    if st.button("🗑️ Clear all session data now"):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()


# PAGE 1: JOB DESCRIPTION UPLOAD

if page == "1. Job Description Upload":
    st.title("📄 Job Description Upload")
    st.caption("Paste the JD text directly, or upload a .txt/.pdf file. Nothing leaves this session.")

    job_title = st.text_input("Job Title", value=st.session_state.job_title,
                               placeholder="e.g. Data Scientist - Internship")

    uploaded_jd_file = st.file_uploader("Optional: upload a JD file (.txt or .pdf)", type=["txt", "pdf"])
    jd_text_input = st.text_area("Or paste the Job Description text here", height=250)

    if st.button("💾 Save Job Description", use_container_width=True):
        final_jd_text = ""
        if uploaded_jd_file is not None:
            if uploaded_jd_file.name.endswith(".pdf"):
                final_jd_text = core.extract_text_from_pdf(uploaded_jd_file)
            else:
                final_jd_text = uploaded_jd_file.read().decode("utf-8", errors="ignore")
        elif jd_text_input.strip():
            final_jd_text = jd_text_input.strip()

        if not job_title.strip() or not final_jd_text.strip():
            st.error("Please provide both a job title and the JD text (typed or uploaded).")
        else:
            st.session_state.job_title = job_title.strip()
            st.session_state.jd_text = final_jd_text
            st.session_state.jd_skills = core.extract_skills(final_jd_text, core.load_skills_list())

         
            st.session_state.candidates = []
            st.session_state.shap_explanations = {}

            st.success("✅ Job Description saved for this session. Go to **Resume Upload** next.")

    if st.session_state.jd_text:
        st.divider()
        st.subheader(f"Current Active Job: {st.session_state.job_title}")
        st.text_area("JD Text", st.session_state.jd_text, height=150, disabled=True)
        st.write("**Skills detected in JD:**", ", ".join(st.session_state.jd_skills) or "None detected")


# PAGE 2: RESUME UPLOAD

elif page == "2. Resume Upload":
    st.title("📤 Resume Upload")

    if not st.session_state.jd_text:
        st.warning("⚠️ Please save a Job Description first (Page 1).")
    else:
        st.caption(f"Uploading resumes against: **{st.session_state.job_title}**")
        uploaded_resumes = st.file_uploader(
            "Upload one or more resumes (PDF)", type=["pdf"], accept_multiple_files=True
        )

        if st.button("⚙️ Process Resumes", use_container_width=True, disabled=not uploaded_resumes):
            jd_text = st.session_state.jd_text
            jd_skills = st.session_state.jd_skills
            skills_list = core.load_skills_list()

            raw_candidates = []
            with st.spinner("Extracting text, skills, and computing similarity scores..."):
                for file in uploaded_resumes:
                    resume_text = core.extract_text_from_pdf(file)
                    if not resume_text:
                        st.warning(f"⚠️ Could not extract text from **{file.name}** - skipping.")
                        continue

                    name = core.guess_candidate_name(resume_text, file.name)
                    resume_skills = core.extract_skills(resume_text, skills_list)
                    missing_skills = core.get_missing_skills(jd_skills, resume_skills)
                    experience_years = core.extract_experience_years(resume_text)

                    tfidf_sim = core.compute_tfidf_similarity(jd_text, resume_text)
                    semantic_sim = core.compute_semantic_similarity(jd_text, resume_text)
                    skills_match_pct = core.compute_skills_match_pct(jd_skills, resume_skills)
                    experience_score = core.normalize_experience_score(experience_years)

                    raw_candidates.append({
                        "filename": file.name,
                        "candidate_name": name,
                        "resume_text": resume_text,
                        "matched_skills": resume_skills,
                        "missing_skills": missing_skills,
                        "experience_years": experience_years,
                        "tfidf_similarity": tfidf_sim,
                        "semantic_similarity": semantic_sim,
                        "skills_match_pct": skills_match_pct,
                        "experience_score": experience_score,
                    })

            if not raw_candidates:
                st.error("No resumes could be processed. Please check the uploaded files.")
            else:
                with st.spinner("Training ranking model and computing SHAP explanations..."):
                    feature_matrix = np.array([
                        [c["semantic_similarity"], c["tfidf_similarity"], c["skills_match_pct"], c["experience_score"]]
                        for c in raw_candidates
                    ])
                    model, final_scores = core.train_ranking_model(feature_matrix)
                    explanations = core.explain_candidates(model, feature_matrix)

              
                new_candidates = []
                new_explanations = {}
                for idx, (candidate, final_score, explanation) in enumerate(
                    zip(raw_candidates, final_scores, explanations)
                ):
                    candidate["id"] = idx
                    candidate["final_score"] = float(final_score)
                    new_candidates.append(candidate)
                    new_explanations[idx] = explanation

                st.session_state.candidates = new_candidates
                st.session_state.shap_explanations = new_explanations

                st.success(f"✅ Processed {len(new_candidates)} resume(s). Go to **Candidate Ranking Dashboard**.")


# PAGE 3: CANDIDATE RANKING DASHBOARD

elif page == "3. Candidate Ranking Dashboard":
    st.title("📊 Candidate Ranking Dashboard")

    if not st.session_state.jd_text:
        st.warning("⚠️ Please save a Job Description first (Page 1).")
    elif not st.session_state.candidates:
        st.info("No candidates processed yet. Go to **Resume Upload** (Page 2).")
    else:
        
        ranked = sorted(st.session_state.candidates, key=lambda c: c["final_score"], reverse=True)

        table_rows = [{
            "Candidate Name": c["candidate_name"],
            "Similarity Score": round(c["semantic_similarity"], 3),
            "Skills Match %": c["skills_match_pct"],
            "Experience Score": round(c["experience_score"], 3),
            "Final Ranking Score": round(c["final_score"], 2),
        } for c in ranked]

        df = pd.DataFrame(table_rows)
        df.insert(0, "Rank", range(1, len(df) + 1))
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.bar_chart(df.set_index("Candidate Name")["Final Ranking Score"])


# PAGE 4: CANDIDATE DETAILS

elif page == "4. Candidate Details":
    st.title("🔍 Candidate Details")

    if not st.session_state.jd_text:
        st.warning("⚠️ Please save a Job Description first (Page 1).")
    elif not st.session_state.candidates:
        st.info("No candidates processed yet. Go to **Resume Upload** (Page 2).")
    else:
        name_to_candidate = {c["candidate_name"]: c for c in st.session_state.candidates}
        selected_name = st.selectbox("Select a candidate", list(name_to_candidate.keys()))
        candidate = name_to_candidate[selected_name]

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Score", f"{candidate['final_score']:.1f}/100")
        col2.metric("Similarity Score", f"{candidate['semantic_similarity']:.2f}")
        col3.metric("Experience", f"{candidate['experience_years']:.1f} yrs")

        st.subheader("✅ Matched Skills")
        st.write(", ".join(candidate["matched_skills"]) or "None detected")

        st.subheader("❌ Missing Skills (required by JD, not found in resume)")
        st.write(", ".join(candidate["missing_skills"]) or "None - full skills match!")

        st.subheader("🧠 Why this score? (SHAP Explanation)")
        explanation = st.session_state.shap_explanations.get(candidate["id"])
        if explanation is None:
            st.caption(
                "SHAP explanation isn't available for this candidate. "
                "Re-run **Process Resumes** on Page 2 to regenerate it."
            )
        else:
            fig, ax = plt.subplots(figsize=(6, 3))
            features = list(explanation.keys())
            values = list(explanation.values())
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
            ax.barh(features, values, color=colors)
            ax.set_xlabel("Contribution to Final Score (SHAP value)")
            ax.axvline(0, color="black", linewidth=0.8)
            st.pyplot(fig)
            st.caption(
              
            )

        with st.expander("📄 View extracted resume text"):
            st.text(candidate["resume_text"][:3000])


# PAGE 5: INTERVIEW QUESTION GENERATOR

elif page == "5. Interview Question Generator":
    st.title("🎤 Interview Question Generator")

    api_key_input = st.sidebar.text_input(
        "Gemini API Key", value=st.session_state.gemini_api_key, type="password",
        help="Or set GEMINI_API_KEY in a .env file to avoid pasting it here.",
    )
    st.session_state.gemini_api_key = api_key_input

    if not st.session_state.jd_text:
        st.warning("⚠️ Please save a Job Description first (Page 1).")
    elif not st.session_state.candidates:
        st.info("No candidates processed yet. Go to **Resume Upload** (Page 2).")
    else:
        name_to_candidate = {c["candidate_name"]: c for c in st.session_state.candidates}
        selected_name = st.selectbox("Select a candidate", list(name_to_candidate.keys()))
        candidate = name_to_candidate[selected_name]

        if not st.session_state.gemini_api_key:
            st.info(
                "ℹ️ No Gemini API key provided - questions will be generated from a "
                "simple template instead. Add a key in the sidebar for AI-generated questions.",
                icon="ℹ️",
            )

        if st.button("✨ Generate Interview Questions", use_container_width=True):
            with st.spinner("Generating questions..."):
                questions = core.generate_interview_questions(
                    jd_text=st.session_state.jd_text,
                    candidate_name=candidate["candidate_name"],
                    matched_skills=candidate["matched_skills"],
                    missing_skills=candidate["missing_skills"],
                    api_key=st.session_state.gemini_api_key,
                )
            st.subheader(f"Questions for {candidate['candidate_name']}")
            for i, question in enumerate(questions, start=1):
                st.write(f"**{i}.** {question}")
