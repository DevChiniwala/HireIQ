import streamlit as st
from langchain_groq import ChatGroq
import os
from utils import (
    extract_key_requirements,
    score_candidate_explainable,
    generate_interview_questions,
    extract_pdf_text,
    create_candidate_rag_retriever,
    ask_rag_question,
    generate_email_templates,
)
import time
import json

st.set_page_config(
    page_title="HireIQ | AI-Powered Hiring Intelligence",
    page_icon="🧠",
    layout="wide",
)

# ─── LLM Initialisation ───────────────────────────────────────────────────────
if "llm" not in st.session_state:
    try:
        st.session_state.llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.1,
            api_key=st.secrets["GROQ_API_KEY"],
        )
    except (KeyError, FileNotFoundError):
        st.error(
            "🔴 GROQ_API_KEY not found. "
            "Create `.streamlit/secrets.toml` and add: `GROQ_API_KEY = 'your-key'`"
        )
        st.stop()

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');

    :root {
        --bg-color: #0D1117;
        --card-bg-color: #161B22;
        --border-color: #30363D;
        --text-color: #E2E8F0;
        --subtle-text-color: #94A3B8;
        --accent-color: #007BFF;
        --accent-glow: rgba(0, 123, 255, 0.3);
        --font-main: 'Inter', sans-serif;
        --font-display: 'Playfair Display', serif;
    }

    html, body, [class*="st-"] { font-family: var(--font-main); color: var(--text-color); }
    .stApp {
        background-color: var(--bg-color);
        background-image: radial-gradient(var(--border-color) 0.5px, transparent 0.5px);
        background-size: 15px 15px;
    }

    .main-content-wrapper { max-width: 1200px; margin: auto; padding: 2rem; }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in-up { animation: fadeInUp 0.9s ease-out forwards; }
    .staggered { opacity: 0; animation: fadeInUp 0.7s ease-out forwards; }

    /* Header */
    .header { text-align: center; margin: 2rem 0 3.5rem; }
    .header h1 {
        font-family: var(--font-display);
        font-size: 5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #007BFF, #00C6FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        opacity: 0;
        animation: fadeInUp 1s ease-out 0.2s forwards;
    }
    .header p {
        color: var(--subtle-text-color);
        font-size: 1.2rem;
        margin-top: 0.4rem;
        opacity: 0;
        animation: fadeInUp 1s ease-out 0.5s forwards;
    }

    /* Cards */
    .input-card {
        background: var(--card-bg-color);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2.5rem;
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .input-card:focus-within {
        border-color: var(--accent-color);
        box-shadow: 0 0 20px var(--accent-glow);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.2s ease-in-out !important;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0   var(--accent-glow); }
        70%  { box-shadow: 0 0 0 10px rgba(0,123,255,0); }
        100% { box-shadow: 0 0 0 0   rgba(0,123,255,0); }
    }
    .primary-btn > button {
        background-color: var(--accent-color) !important;
        color: white !important;
        border: none !important;
        animation: pulse 2s infinite;
    }
    .primary-btn > button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 25px var(--accent-glow) !important;
        animation: none;
    }
    .secondary-btn > button {
        background: transparent !important;
        border: 1px solid var(--border-color) !important;
        color: var(--subtle-text-color) !important;
    }
    .secondary-btn > button:hover {
        background: var(--card-bg-color) !important;
        border-color: var(--text-color) !important;
        color: var(--text-color) !important;
    }

    /* Section header */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
    }

    /* Progress bar colour */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #007BFF, #00C6FF);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid var(--border-color); }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem; padding: 1rem; }
    .stTabs [aria-selected="true"] { color: var(--accent-color) !important; border-bottom-color: var(--accent-color) !important; }

    /* Expanders */
    .stExpander { border: none !important; background: rgba(0,0,0,0.2); border-radius: 8px; }

    /* Candidate display */
    .candidate-name { font-size: 1.7rem; font-weight: 700; color: #FFFFFF; margin: 0; }

    /* XAI items */
    .xai-item { border-left: 3px solid; padding-left: 1rem; margin-bottom: 1rem; }
    .xai-met  { border-color: #28a745; }
    .xai-gap  { border-color: #dc3545; }

    /* Score badges */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-high { background: rgba(40,167,69,0.15); color:#28a745; border:1px solid #28a745; }
    .badge-mid  { background: rgba(255,193,7,0.15);  color:#ffc107; border:1px solid #ffc107; }
    .badge-low  { background: rgba(220,53,69,0.15);  color:#dc3545; border:1px solid #dc3545; }

    /* Chat bubbles */
    .chat-bubble { padding: 0.85rem 1rem; border-radius: 10px; margin-bottom: 0.75rem; max-width: 82%; word-wrap: break-word; }
    .chat-bubble.user      { background: var(--accent-color); color: #fff; margin-left: auto; border-bottom-right-radius: 0; }
    .chat-bubble.assistant { background: #1e2530; color: var(--text-color); border-bottom-left-radius: 0; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = "upload"
    st.session_state.candidates = []
    st.session_state.key_requirements = []
    st.session_state.chat_histories = {}
    st.session_state.rag_retrievers = {}
    st.session_state.saved_job_description = ""
    st.session_state.saved_resume_files = []
    st.session_state.generated_emails = {}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def clamp_score(raw) -> int:
    try:
        return max(0, min(int(raw), 100))
    except (TypeError, ValueError):
        return 0

def badge_class(score: int) -> str:
    if score >= 75: return "badge-high"
    if score >= 50: return "badge-mid"
    return "badge-low"

def extract_job_title(jd: str) -> str:
    for line in jd.splitlines():
        s = line.strip()
        if s:
            return s[:80]
    return "the position"

# ─── Callbacks ────────────────────────────────────────────────────────────────
def proceed_to_weighting():
    if not st.session_state.saved_job_description.strip():
        st.warning("⚠️ Please paste a Job Description before proceeding.")
        return
    if not st.session_state.saved_resume_files:
        st.warning("⚠️ Please upload at least one PDF resume.")
        return
    with st.spinner("🧠 Extracting key requirements from Job Description…"):
        try:
            reqs = extract_key_requirements(
                st.session_state.saved_job_description, st.session_state.llm
            )
            if reqs and isinstance(reqs, list) and len(reqs) > 0:
                st.session_state.key_requirements = reqs
                st.session_state.step = "weighting"
            else:
                st.error("❗ Could not extract requirements. Try a more detailed job description.")
        except Exception as e:
            st.error(f"AI extraction failed: {e}")


def run_final_analysis(weighted_reqs, resume_files, job_description):
    with st.spinner("🔬 Deep-analysing all candidates…"):
        # Parse PDFs
        resumes = []
        for f in resume_files:
            text = extract_pdf_text(f)
            if text and text.strip():
                resumes.append({"text": text, "filename": f.name})
            else:
                st.warning(f"⚠️ Could not extract text from `{f.name}` — skipping.")

        if not resumes:
            st.error("❌ No readable PDFs found. Please re-upload valid resumes.")
            return

        results = []
        bar = st.progress(0.0, "Starting…")

        for i, res in enumerate(resumes):
            try:
                bar.progress((i + 1) / len(resumes), f"Analysing {res['filename']}…")
                score_data = score_candidate_explainable(
                    job_description, res["text"], weighted_reqs, st.session_state.llm
                )
                d = score_data.model_dump()
                d["filename"] = res["filename"]
                results.append(d)
            except Exception as e:
                st.warning(f"⚠️ Scoring failed for `{res['filename']}`: {e}")
                results.append({
                    "name": f"Error: {res['filename']}",
                    "overall_score": 0,
                    "summary": f"AI could not process this resume. Error: {e}",
                    "requirement_analysis": [],
                    "filename": res["filename"],
                })
            time.sleep(0.5)

        bar.empty()

        st.session_state.candidates = sorted(results, key=lambda x: x["overall_score"], reverse=True)

        # Build RAG indices
        st.session_state.rag_retrievers = {}
        st.session_state.chat_histories = {}
        for c in st.session_state.candidates:
            if "Error:" in c["name"]:
                continue
            name = c["name"]
            src = next((r for r in resumes if r["filename"] == c.get("filename")), None)
            if src:
                try:
                    st.session_state.rag_retrievers[name] = create_candidate_rag_retriever(
                        src["text"], src["filename"]
                    )
                    st.session_state.chat_histories[name] = []
                except Exception as e:
                    st.warning(f"⚠️ RAG index failed for {name}: {e}")

        st.session_state.step = "results"


def go_back():
    st.session_state.step = "upload"
    st.session_state.key_requirements = []


def trigger_analysis():
    weighted_reqs = {
        req: {
            "importance": st.session_state[f"imp_{req}"],
            "knockout": st.session_state[f"ko_{req}"],
        }
        for req in st.session_state.key_requirements
    }
    run_final_analysis(
        weighted_reqs,
        st.session_state.saved_resume_files,
        st.session_state.saved_job_description,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE SHELL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-content-wrapper fade-in-up">', unsafe_allow_html=True)
st.markdown(
    "<div class='header'>"
    "<h1>HireIQ</h1>"
    "<p>AI-Powered Hiring Intelligence &nbsp;·&nbsp; Screen Smarter. Hire Faster. Explain Every Decision.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == "upload":
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Step 1 &nbsp;·&nbsp; Provide Your Data</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<h5>📝 Job Description</h5>", unsafe_allow_html=True)
        st.session_state.saved_job_description = st.text_area(
            "JD",
            value=st.session_state.saved_job_description,
            placeholder="Paste the full job description here…",
            height=300,
            label_visibility="collapsed",
        )
    with col2:
        st.markdown("<h5>👥 Upload Candidate Resumes (PDF)</h5>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Resumes",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        # Only update if new files were actually selected — avoids losing files on re-run
        if uploaded:
            st.session_state.saved_resume_files = uploaded
        if st.session_state.saved_resume_files:
            st.caption(f"✅ {len(st.session_state.saved_resume_files)} resume(s) ready")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    st.button("🔍 Analyse Requirements →", on_click=proceed_to_weighting, use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Weighting
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == "weighting":
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Step 2 &nbsp;·&nbsp; Define What Matters Most</h2>", unsafe_allow_html=True)
    st.info("🤖 AI has extracted the requirements below. Set their weight and flag any automatic disqualifiers.")

    for i, req in enumerate(st.session_state.key_requirements):
        st.markdown(f'<div class="staggered" style="animation-delay:{i*80}ms">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([5, 2, 1])
        with c1:
            st.write(f"▸ {req}")
        with c2:
            st.selectbox(
                "Weight",
                ["Normal", "Important", "Critical"],
                key=f"imp_{req}",
                index=1,
                label_visibility="collapsed",
            )
        with c3:
            st.checkbox(
                "KO?",
                key=f"ko_{req}",
                help="✅ If checked: missing this requirement auto-disqualifies the candidate.",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        st.button("⬅️ Go Back & Edit", on_click=go_back, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        st.button("🚀 Run Final Analysis", on_click=trigger_analysis, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Results
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == "results":
    r1, r2 = st.columns([6, 1])
    with r1:
        st.success(f"✅ Analysis complete — {len(st.session_state.candidates)} candidate(s) ranked.")
    with r2:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        st.button("🔄 Start Over", on_click=go_back, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    tabs = st.tabs(["🏆 Leaderboard", "🤝 Compare Candidates", "✉️ Email Drafts"])

    # ── Leaderboard ──────────────────────────────────────
    with tabs[0]:
        if not st.session_state.candidates:
            st.info("No candidates processed. Go back and try again.")

        for rank, cand in enumerate(st.session_state.candidates, start=1):
            name = cand["name"]
            score = clamp_score(cand.get("overall_score", 0))
            is_error = "Error:" in name

            st.markdown('<div class="input-card" style="margin-bottom:1.5rem;">', unsafe_allow_html=True)

            h1, h2, h3 = st.columns([1, 5, 2])
            with h1:
                st.markdown(f"<h2 style='color:#4A5568;margin:0;line-height:1;'>#{rank}</h2>", unsafe_allow_html=True)
            with h2:
                st.markdown(f"<p class='candidate-name'>{name}</p>", unsafe_allow_html=True)
            with h3:
                st.markdown(
                    f"<div style='text-align:right;padding-top:6px;'>"
                    f"<span class='badge {badge_class(score)}'>{score} / 100</span></div>",
                    unsafe_allow_html=True,
                )

            # FIX: divide by 100 so st.progress receives 0.0–1.0
            st.progress(score / 100.0)
            st.markdown(
                f"<p style='color:var(--subtle-text-color);margin-top:0.6rem;'>{cand['summary']}</p>",
                unsafe_allow_html=True,
            )

            if not is_error:
                # XAI
                with st.expander("📊 Detailed XAI Requirement Analysis"):
                    req_list = cand.get("requirement_analysis", [])
                    if not req_list:
                        st.info("No requirement data available.")
                    for r in req_list:
                        if r["match_status"]:
                            st.markdown(
                                f"<div class='xai-item xai-met'><b>✅ Met:</b> {r['requirement']}"
                                f"<br><small><i>Evidence: \"{r['evidence']}\"</i></small></div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div class='xai-item xai-gap'><b>❌ Gap:</b> {r['requirement']}"
                                f"<br><small><i>Reason: {r['evidence']}</i></small></div>",
                                unsafe_allow_html=True,
                            )

                # Interview questions
                if st.button("🎯 Generate Interview Questions", key=f"iq_{name}_{rank}"):
                    with st.spinner("Generating tailored questions…"):
                        qs = generate_interview_questions(
                            cand["name"],
                            cand["summary"],
                            st.session_state.saved_job_description,
                            st.session_state.llm,
                        )
                    qc1, qc2 = st.columns(2)
                    with qc1:
                        st.markdown("**🗣️ Behavioral**")
                        for q in qs.behavioral:
                            st.markdown(f"- {q}")
                    with qc2:
                        st.markdown("**⚙️ Technical**")
                        for q in qs.technical:
                            st.markdown(f"- {q}")

                # RAG chat
                st.markdown("<hr style='border-color:var(--border-color);margin:1.5rem 0;'>", unsafe_allow_html=True)
                st.markdown("<h5>💬 Chat About This Candidate</h5>", unsafe_allow_html=True)
                chat_box = st.container(height=230)
                with chat_box:
                    for msg in st.session_state.chat_histories.get(name, []):
                        st.markdown(
                            f"<div class='chat-bubble {msg['role']}'>{msg['content']}</div>",
                            unsafe_allow_html=True,
                        )

                if prompt := st.chat_input(f"Ask about {name}…", key=f"chat_{name}_{rank}"):
                    retriever = st.session_state.rag_retrievers.get(name)
                    if retriever:
                        st.session_state.chat_histories[name].append({"role": "user", "content": prompt})
                        with chat_box:
                            st.markdown(f"<div class='chat-bubble user'>{prompt}</div>", unsafe_allow_html=True)
                            with st.spinner("Thinking…"):
                                answer = ask_rag_question(retriever, prompt, st.session_state.llm)
                            st.session_state.chat_histories[name].append({"role": "assistant", "content": answer})
                            st.markdown(f"<div class='chat-bubble assistant'>{answer}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("RAG index not available for this candidate.")

            st.markdown("</div>", unsafe_allow_html=True)

    # ── Compare ──────────────────────────────────────────
    with tabs[1]:
        valid_names = [c["name"] for c in st.session_state.candidates if "Error:" not in c["name"]]
        if not valid_names:
            st.warning("No valid candidates available to compare.")
        else:
            selected = st.multiselect("Select 2 or more candidates to compare:", valid_names, key="compare_list")
            if len(selected) >= 2:
                lookup = {c["name"]: c for c in st.session_state.candidates}
                cols = st.columns(len(selected))
                for idx, sel_name in enumerate(selected):
                    d = lookup[sel_name]
                    s = clamp_score(d.get("overall_score", 0))
                    with cols[idx]:
                        st.markdown(f"<h4 style='margin-bottom:0.3rem;'>{sel_name}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<span class='badge {badge_class(s)}'>{s} / 100</span>", unsafe_allow_html=True)
                        st.progress(s / 100.0)   # FIX: 0.0–1.0
                        st.markdown("<br>**AI Summary**", unsafe_allow_html=True)
                        st.markdown(f"<p style='color:var(--subtle-text-color);font-size:0.88rem;'>{d['summary']}</p>", unsafe_allow_html=True)
                        met   = [r for r in d.get("requirement_analysis", []) if r["match_status"]]
                        unmet = [r for r in d.get("requirement_analysis", []) if not r["match_status"]]
                        if met:
                            st.markdown("**✅ Met**")
                            for r in met:
                                st.markdown(f"<small>• {r['requirement']}</small>", unsafe_allow_html=True)
                        if unmet:
                            st.markdown("**❌ Missing**")
                            for r in unmet:
                                st.markdown(f"<small>• {r['requirement']}</small>", unsafe_allow_html=True)
                        st.markdown("<hr style='border-color:var(--border-color);'>", unsafe_allow_html=True)
            elif len(selected) == 1:
                st.info("Select at least one more candidate to compare.")

    # ── Email Drafts ─────────────────────────────────────
    with tabs[2]:
        st.markdown("<h3 class='section-header'>✉️ Email Generation Centre</h3>", unsafe_allow_html=True)
        valid = [c for c in st.session_state.candidates if "Error:" not in c["name"]]
        if not valid:
            st.warning("⚠️ No valid candidates — cannot generate emails.")
        else:
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown("<h5>⚙️ Configuration</h5>", unsafe_allow_html=True)
                num_invite = st.slider("Top candidates to invite", 1, len(valid), min(3, len(valid)))
                min_score  = st.slider("Minimum score to invite", 0, 100, 70)
            with ec2:
                st.markdown("<h5>📅 Interview Scheduling</h5>", unsafe_allow_html=True)
                idate = st.date_input("Interview Date")
                itime = st.time_input("Interview Time")

            if st.button("✉️ Generate All Emails", use_container_width=True, type="primary"):
                with st.spinner("✍️ Crafting personalised emails…"):
                    title = extract_job_title(st.session_state.saved_job_description)
                    dt_str = f"{idate.strftime('%A, %B %d, %Y')} at {itime.strftime('%I:%M %p')}"
                    st.session_state.generated_emails = generate_email_templates(
                        valid, {"title": title}, num_invite, min_score, dt_str, st.session_state.llm
                    )

            if st.session_state.get("generated_emails"):
                st.markdown("<hr style='border-color:var(--border-color);margin:2rem 0;'>", unsafe_allow_html=True)
                inv_col, rej_col = st.columns(2)
                with inv_col:
                    st.markdown("<h4>✅ Interview Invitations</h4>", unsafe_allow_html=True)
                    for em in st.session_state.generated_emails.get("invitations", []):
                        with st.expander(f"To: {em['name']}", expanded=True):
                            st.code(em["email_body"], language=None)
                    if not st.session_state.generated_emails.get("invitations"):
                        st.info("No candidates met the score threshold.")
                with rej_col:
                    st.markdown("<h4>❌ Rejection Emails</h4>", unsafe_allow_html=True)
                    for em in st.session_state.generated_emails.get("rejections", []):
                        with st.expander(f"To: {em['name']}", expanded=True):
                            st.code(em["email_body"], language=None)
                    if not st.session_state.generated_emails.get("rejections"):
                        st.info("All candidates were invited — no rejections needed.")

st.markdown("</div>", unsafe_allow_html=True)
