import streamlit as st
from langchain_groq import ChatGroq
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

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HireIQ | AI-Powered Hiring Intelligence",
    page_icon="🧠",
    layout="wide",
)

# ─── LLM Init ─────────────────────────────────────────────────────────────────
if "llm" not in st.session_state:
    try:
        st.session_state.llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0.1,
            api_key=st.secrets["GROQ_API_KEY"],
        )
    except (KeyError, FileNotFoundError):
        st.error("🔴 GROQ_API_KEY not found. Add it to `.streamlit/secrets.toml`")
        st.stop()

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');
:root {
    --bg:     #0D1117; --card:   #161B22; --border: #30363D;
    --text:   #E2E8F0; --muted:  #94A3B8; --accent: #007BFF;
    --glow:   rgba(0,123,255,0.25);
}
html,body,[class*="st-"]{font-family:'Inter',sans-serif;color:var(--text);}
.stApp{background-color:var(--bg);background-image:radial-gradient(var(--border) 0.5px,transparent 0.5px);background-size:15px 15px;}
.block-container{padding-top:2rem!important;}
@keyframes up{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.hiq-header{text-align:center;margin-bottom:2.5rem;}
.hiq-header h1{font-family:'Playfair Display',serif;font-size:5rem;font-weight:700;background:linear-gradient(135deg,#007BFF,#00C6FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:up 1s ease-out 0.2s both;}
.hiq-header p{color:var(--muted);font-size:1.15rem;animation:up 1s ease-out 0.5s both;}
.hiq-sh{font-size:1.4rem;font-weight:600;border-bottom:1px solid var(--border);padding-bottom:.75rem;margin-bottom:1.25rem;}
.stButton>button{border-radius:8px;padding:12px 24px;font-weight:600;transition:all .2s ease!important;}
@keyframes pulse{0%{box-shadow:0 0 0 0 var(--glow);}70%{box-shadow:0 0 0 10px rgba(0,123,255,0);}100%{box-shadow:0 0 0 0 rgba(0,123,255,0);}}
.pbtn>button{background:var(--accent)!important;color:#fff!important;border:none!important;animation:pulse 2s infinite;}
.pbtn>button:hover{transform:scale(1.02);animation:none;box-shadow:0 0 20px var(--glow)!important;}
.sbtn>button{background:transparent!important;border:1px solid var(--border)!important;color:var(--muted)!important;}
.sbtn>button:hover{border-color:var(--text)!important;color:var(--text)!important;}
.stProgress>div>div>div>div{background:linear-gradient(90deg,#007BFF,#00C6FF);}
.stTabs [data-baseweb="tab-list"]{border-bottom:2px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-size:1rem;padding:1rem;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important;}
.stExpander{border:none!important;background:rgba(0,0,0,.2);border-radius:8px;}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;font-size:.88rem;}
.bh{background:rgba(40,167,69,.15);color:#28a745;border:1px solid #28a745;}
.bm{background:rgba(255,193,7,.15);color:#ffc107;border:1px solid #ffc107;}
.bl{background:rgba(220,53,69,.15);color:#dc3545;border:1px solid #dc3545;}
.shortlist-pill{display:inline-block;background:rgba(0,123,255,.15);color:#007BFF;border:1px solid #007BFF;border-radius:20px;padding:3px 12px;font-size:.82rem;font-weight:600;margin:2px;}
.skill-tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.82rem;font-weight:500;margin:2px;}
.skill-match{background:rgba(40,167,69,.15);color:#28a745;border:1px solid #28a745;}
.skill-missing{background:rgba(220,53,69,.15);color:#dc3545;border:1px solid #dc3545;}
.xai{border-left:3px solid;padding:.5rem 1rem;margin-bottom:.75rem;border-radius:0 6px 6px 0;}
.xai-y{border-color:#28a745;background:rgba(40,167,69,.05);}
.xai-n{border-color:#dc3545;background:rgba(220,53,69,.05);}
.bubble{padding:.75rem 1rem;border-radius:10px;margin-bottom:.6rem;max-width:82%;word-wrap:break-word;}
.bubble.user{background:var(--accent);color:#fff;margin-left:auto;border-bottom-right-radius:0;}
.bubble.assistant{background:#1e2530;color:var(--text);border-bottom-left-radius:0;}
.cname{font-size:1.6rem;font-weight:700;color:#fff;margin:0;}
#MainMenu,footer{visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session State ─────────────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.update(
        {
            "step": "upload",
            "candidates": [],
            "key_requirements": [],
            "chat_histories": {},
            "rag_retrievers": {},
            "saved_jd": "",
            "saved_files": [],
            "generated_emails": {},
            "shortlist": [],
        }
    )
if "shortlist" not in st.session_state:
    st.session_state.shortlist = []


# ─── Helpers ──────────────────────────────────────────────────────────────────
def clamp(v):
    try:
        return max(0, min(int(v), 100))
    except:
        return 0


def badge(score):
    cls = "bh" if score >= 75 else ("bm" if score >= 50 else "bl")
    return f"<span class='badge {cls}'>{score} / 100</span>"


def decision(score):
    if score >= 75:
        return "🟢 Strong Hire"
    if score >= 60:
        return "🟡 Consider"
    return "🔴 Reject"


def match_label(score):
    if score >= 75:
        return "🟢 High Match"
    if score >= 60:
        return "🟡 Medium Match"
    return "🔴 Low Match"


def job_title(jd):
    for line in jd.splitlines():
        s = line.strip()
        if s:
            return s[:80]
    return "the position"


# ─── Callbacks ────────────────────────────────────────────────────────────────
def go_to_weighting():
    if not st.session_state.saved_jd.strip():
        st.warning("Please paste a Job Description.")
        return
    if not st.session_state.saved_files:
        st.warning("Please upload at least one PDF resume.")
        return
    with st.spinner("🧠 Analysing job description…"):
        try:
            reqs = extract_key_requirements(
                st.session_state.saved_jd, st.session_state.llm
            )
            if reqs:
                st.session_state.key_requirements = reqs
                st.session_state.step = "weighting"
            else:
                st.error("Could not extract requirements. Add more detail to the JD.")
        except Exception as e:
            st.error(f"Extraction failed: {e}")


def go_back():
    st.session_state.step = "upload"
    st.session_state.key_requirements = []


def run_analysis():
    weighted = {
        req: {
            "importance": st.session_state[f"imp_{req}"],
            "knockout": st.session_state[f"ko_{req}"],
        }
        for req in st.session_state.key_requirements
    }
    with st.spinner("🔬 Scoring all candidates…"):
        resumes = []
        for f in st.session_state.saved_files:
            text = extract_pdf_text(f)
            if text and text.strip():
                resumes.append({"text": text, "filename": f.name})
            else:
                st.warning(f"Could not read `{f.name}` — skipping.")
        if not resumes:
            st.error("No readable PDFs. Please re-upload valid resumes.")
            return

        results = []
        bar = st.progress(0.0, "Starting…")
        for i, res in enumerate(resumes):
            bar.progress((i + 1) / len(resumes), f"Scoring {res['filename']}…")
            try:
                data = score_candidate_explainable(
                    st.session_state.saved_jd,
                    res["text"],
                    weighted,
                    st.session_state.llm,
                )
                d = data.model_dump()
                d["filename"] = res["filename"]
                results.append(d)
            except Exception as e:
                st.warning(f"Could not score `{res['filename']}`: {e}")
                results.append(
                    {
                        "name": f"Error: {res['filename']}",
                        "overall_score": 0,
                        "summary": str(e),
                        "requirement_analysis": [],
                        "filename": res["filename"],
                    }
                )
            time.sleep(0.4)
        bar.empty()

        st.session_state.candidates = sorted(
            results, key=lambda x: x["overall_score"], reverse=True
        )

        st.session_state.rag_retrievers = {}
        st.session_state.chat_histories = {}
        for c in st.session_state.candidates:
            if "Error:" in c["name"]:
                continue
            src = next((r for r in resumes if r["filename"] == c.get("filename")), None)
            if src:
                try:
                    st.session_state.rag_retrievers[c["name"]] = (
                        create_candidate_rag_retriever(src["text"], src["filename"])
                    )
                    st.session_state.chat_histories[c["name"]] = []
                except Exception as e:
                    st.warning(f"RAG index failed for {c['name']}: {e}")

        st.session_state.step = "results"


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="hiq-header">
  <h1>HireIQ</h1>
  <p>AI-Powered Hiring Intelligence &nbsp;·&nbsp; Screen Smarter. Hire Faster. Explain Every Decision.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == "upload":

    st.markdown(
        "<div class='hiq-sh'>Step 1 &nbsp;·&nbsp; Provide Your Data</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**📝 Job Description**")
        st.session_state.saved_jd = st.text_area(
            "jd_input",
            value=st.session_state.saved_jd,
            placeholder="Paste the full job description here…",
            height=320,
            label_visibility="collapsed",
        )
    with col2:
        st.markdown("**👥 Upload Candidate Resumes (PDF)**")
        new_files = st.file_uploader(
            "pdf_upload",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if new_files:
            st.session_state.saved_files = new_files
        if st.session_state.saved_files:
            st.success(f"✅ {len(st.session_state.saved_files)} resume(s) loaded")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="pbtn">', unsafe_allow_html=True)
    st.button(
        "🔍 Analyse Requirements →", on_click=go_to_weighting, use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — WEIGHTING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == "weighting":

    st.markdown(
        "<div class='hiq-sh'>Step 2 &nbsp;·&nbsp; Define What Matters Most</div>",
        unsafe_allow_html=True,
    )
    st.info("🤖 AI extracted these requirements. Set importance and flag knock-outs.")

    for req in st.session_state.key_requirements:
        c1, c2, c3 = st.columns([5, 2, 1])
        with c1:
            st.write(f"▸ {req}")
        with c2:
            st.selectbox(
                "Importance",
                ["Normal", "Important", "Critical"],
                key=f"imp_{req}",
                index=1,
                label_visibility="collapsed",
            )
        with c3:
            st.checkbox(
                "KO?",
                key=f"ko_{req}",
                help="If checked: missing this requirement auto-disqualifies the candidate.",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="sbtn">', unsafe_allow_html=True)
        st.button("⬅️ Go Back", on_click=go_back, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="pbtn">', unsafe_allow_html=True)
        st.button(
            "🚀 Run Final Analysis", on_click=run_analysis, use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == "results":

    # Top bar
    ta, tb = st.columns([6, 1])
    with ta:
        st.success(
            f"✅ Analysis complete — **{len(st.session_state.candidates)}** candidate(s) ranked."
        )
    with tb:
        st.markdown('<div class="sbtn">', unsafe_allow_html=True)
        st.button("🔄 Start Over", on_click=go_back, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── FEATURE 1: Recruiter Dashboard Metrics ────────────────────────────────
    scores = [
        c["overall_score"]
        for c in st.session_state.candidates
        if "Error:" not in c["name"]
    ]

    if scores:
        avg_score = round(sum(scores) / len(scores), 1)
        strong_matches = len([s for s in scores if s >= 75])

        st.markdown("## 📊 Hiring Insights")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Candidates", len(scores))
        with m2:
            st.metric("Average Score", avg_score)
        with m3:
            st.metric("Strong Matches", strong_matches)

    # ── Top candidate banner ──────────────────────────────────────────────────
    if st.session_state.candidates:
        top = st.session_state.candidates[0]
        top_score = clamp(top["overall_score"])
        st.info(
            f"🏆 **Top Candidate:** {top['name']} — {top_score} / 100  ·  {decision(top_score)}"
        )

        # ── FEATURE 2: Why Top Candidate Won ─────────────────────────────────
        with st.expander("🏆 Why was this candidate ranked #1?"):
            best_key = "top_candidate_reason"
            if best_key not in st.session_state:
                with st.spinner("Analyzing top candidate..."):
                    explanation = st.session_state.llm.invoke(
                        f"""You are a senior hiring director.

Job Description:
{st.session_state.saved_jd}

Top Candidate:
{top['name']}

Candidate Summary:
{top['summary']}

Candidate Score:
{top_score}

Explain:
- Why this candidate ranked highest
- Biggest strengths
- Hiring advantages
- Potential risks
- Final recommendation

Keep it concise and executive-level."""
                    )
                    st.session_state[best_key] = explanation.content
            st.write(st.session_state[best_key])

    # ── Shortlist bar ─────────────────────────────────────────────────────────
    if st.session_state.shortlist:
        pills = "".join(
            f"<span class='shortlist-pill'>⭐ {n}</span>"
            for n in st.session_state.shortlist
        )
        st.markdown(
            f"<div style='margin-bottom:1rem'>📋 <b>Shortlisted:</b> {pills}</div>",
            unsafe_allow_html=True,
        )

        # ── FEATURE 5: Download Shortlist ─────────────────────────────────────
        shortlist_text = "\n".join(st.session_state.shortlist)
        st.download_button(
            "📥 Download Shortlist",
            shortlist_text,
            file_name="shortlist.txt",
        )

    tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "🤝 Compare", "✉️ Emails"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — LEADERBOARD
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        if not st.session_state.candidates:
            st.info("No candidates found. Go back and try again.")
        else:
            st.caption("📊 Candidates ranked by AI score (highest → lowest)")

            for rank, cand in enumerate(st.session_state.candidates, start=1):
                name = cand["name"]
                score = clamp(cand.get("overall_score", 0))
                is_err = "Error:" in name

                with st.container():
                    st.markdown("---")

                    # Rank / Name / Badge
                    r1, r2, r3 = st.columns([1, 5, 2])
                    with r1:
                        st.markdown(
                            f"<h2 style='color:#4A5568;margin:0'>#{rank}</h2>",
                            unsafe_allow_html=True,
                        )
                    with r2:
                        st.markdown(
                            f"<p class='cname'>{name}</p>", unsafe_allow_html=True
                        )
                    with r3:
                        st.markdown(
                            f"<div style='text-align:right;padding-top:8px'>{badge(score)}</div>",
                            unsafe_allow_html=True,
                        )

                    # Progress bar
                    st.progress(score / 100.0)

                    # Match + Decision
                    ml, dl = st.columns(2)
                    with ml:
                        st.markdown(f"**Match:** {match_label(score)}")
                    with dl:
                        st.markdown(f"**Decision:** {decision(score)}")

                    # Summary
                    st.markdown(
                        f"<p style='color:var(--muted);margin-top:.5rem'>{cand['summary']}</p>",
                        unsafe_allow_html=True,
                    )

                    if not is_err:
                        req_analysis = cand.get("requirement_analysis", [])

                        # SKILL INSIGHTS
                        matched = [
                            r["requirement"] for r in req_analysis if r["match_status"]
                        ][:3]
                        missing = [
                            r["requirement"]
                            for r in req_analysis
                            if not r["match_status"]
                        ][:3]
                        if matched or missing:
                            si1, si2 = st.columns(2)
                            with si1:
                                if matched:
                                    st.markdown("**🔥 Top matched skills:**")
                                    tags = " ".join(
                                        f"<span class='skill-tag skill-match'>{s}</span>"
                                        for s in matched
                                    )
                                    st.markdown(tags, unsafe_allow_html=True)
                            with si2:
                                if missing:
                                    st.markdown("**⚠️ Key gaps:**")
                                    tags = " ".join(
                                        f"<span class='skill-tag skill-missing'>{s}</span>"
                                        for s in missing
                                    )
                                    st.markdown(tags, unsafe_allow_html=True)

                        st.markdown(
                            "<div style='height:.4rem'></div>", unsafe_allow_html=True
                        )

                        # SHORTLIST BUTTON
                        if name not in st.session_state.shortlist:
                            if st.button("⭐ Add to Shortlist", key=f"short_{rank}"):
                                st.session_state.shortlist.append(name)
                                st.success(f"✅ {name} added to shortlist!")
                                st.rerun()
                        else:
                            st.markdown("⭐ **In Shortlist**")

                        # WHY THIS SCORE — cached so it only calls the API once per candidate
                        with st.expander("🧠 Why this score?"):
                            if score < 75:

    with st.expander("❌ Why not selected?"):

        reject_key = f"reject_reason_{name}"

        if reject_key not in st.session_state:

            with st.spinner("Analyzing candidate gaps..."):

                response = st.session_state.llm.invoke(
                    f"""
You are a senior recruiter.

Job Description:
{st.session_state.saved_jd}

Candidate Summary:
{cand.get('summary', '')}

Candidate Score:
{score}

Explain:

- missing skills
- weaknesses
- hiring concerns
- major gaps

Keep it concise and professional.
"""
                )

                st.session_state[reject_key] = response.content

        st.write(st.session_state[reject_key])
                            cache_key = f"explain_{name}"
                            if cache_key not in st.session_state:
                                with st.spinner("Analysing…"):
                                    try:
                                        resp = st.session_state.llm.invoke(
                                            f"""You are a senior hiring manager.

Job Description:
{st.session_state.saved_jd}

Candidate Summary:
{cand.get('summary', '')}

Candidate Score: {score}/100

Give a structured explanation:
- Why this score?
- Top strengths (bullet points)
- Weaknesses / risks
- Missing skills
- Final hiring recommendation (Hire / Maybe / Reject)

Keep it concise and practical."""
                                        )
                                        st.session_state[cache_key] = resp.content
                                    except Exception as e:
                                        st.session_state[cache_key] = (
                                            f"Could not generate explanation: {e}"
                                        )
                            st.write(st.session_state[cache_key])

                        # ── FEATURE 3: Why Not Selected ───────────────────────
                        if score < 75:
                            with st.expander("❌ Why not selected?"):
                                reject_key = f"reject_reason_{name}"
                                if reject_key not in st.session_state:
                                    with st.spinner("Analyzing weaknesses..."):
                                        response = st.session_state.llm.invoke(
                                            f"""Explain why this candidate may not be selected.

Candidate:
{cand['summary']}

Score:
{score}

Job Description:
{st.session_state.saved_jd}

Give:
- missing skills
- concerns
- gaps
- hiring risks

Keep it professional."""
                                        )
                                        st.session_state[reject_key] = response.content
                                st.write(st.session_state[reject_key])

                        # XAI REQUIREMENT ANALYSIS
                        with st.expander("📊 Full XAI Requirement Analysis"):
                            if not req_analysis:
                                st.info("No requirement data available.")
                            for r in req_analysis:
                                if r["match_status"]:
                                    st.markdown(
                                        f"<div class='xai xai-y'>✅ <b>{r['requirement']}</b>"
                                        f"<br><small><i>Evidence: \"{r['evidence']}\"</i></small></div>",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(
                                        f"<div class='xai xai-n'>❌ <b>{r['requirement']}</b>"
                                        f"<br><small><i>Reason: {r['evidence']}</i></small></div>",
                                        unsafe_allow_html=True,
                                    )

                        # INTERVIEW QUESTIONS
                        if st.button(
                            "🎯 Generate Interview Questions", key=f"iq_{rank}"
                        ):
                            with st.spinner("Generating…"):
                                qs = generate_interview_questions(
                                    cand["name"],
                                    cand["summary"],
                                    st.session_state.saved_jd,
                                    st.session_state.llm,
                                )
                            qa, qb = st.columns(2)
                            with qa:
                                st.markdown("**🗣️ Behavioral**")
                                for q in qs.behavioral:
                                    st.markdown(f"- {q}")
                            with qb:
                                st.markdown("**⚙️ Technical**")
                                for q in qs.technical:
                                    st.markdown(f"- {q}")

                        # RAG CHAT
                        st.markdown("---")
                        st.markdown(f"**💬 Chat about {name}**")
                        chat_area = st.container(height=220)
                        with chat_area:
                            for msg in st.session_state.chat_histories.get(name, []):
                                role = msg["role"]
                                content = msg["content"]
                                st.markdown(
                                    f"<div class='bubble {role}'>{content}</div>",
                                    unsafe_allow_html=True,
                                )

                        if prompt := st.chat_input(
                            f"Ask about {name}…", key=f"ci_{rank}"
                        ):
                            retriever = st.session_state.rag_retrievers.get(name)
                            if retriever:
                                st.session_state.chat_histories[name].append(
                                    {"role": "user", "content": prompt}
                                )
                                with chat_area:
                                    st.markdown(
                                        f"<div class='bubble user'>{prompt}</div>",
                                        unsafe_allow_html=True,
                                    )
                                    with st.spinner("Thinking…"):
                                        ans = ask_rag_question(
                                            retriever, prompt, st.session_state.llm
                                        )
                                    st.session_state.chat_histories[name].append(
                                        {"role": "assistant", "content": ans}
                                    )
                                    st.markdown(
                                        f"<div class='bubble assistant'>{ans}</div>",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.warning(
                                    "RAG index not available for this candidate."
                                )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — COMPARE
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        valid = [
            c["name"] for c in st.session_state.candidates if "Error:" not in c["name"]
        ]
        if not valid:
            st.warning("No valid candidates to compare.")
        else:
            selected = st.multiselect("Select 2 or more candidates:", valid)
            if len(selected) >= 2:
                lookup = {c["name"]: c for c in st.session_state.candidates}
                cols = st.columns(len(selected))
                for i, sel in enumerate(selected):
                    d = lookup[sel]
                    s = clamp(d.get("overall_score", 0))
                    with cols[i]:
                        st.markdown(f"**{sel}**")
                        st.markdown(badge(s), unsafe_allow_html=True)
                        st.progress(s / 100.0)
                        st.markdown(f"**Match:** {match_label(s)}")
                        st.markdown(f"**Decision:** {decision(s)}")
                        st.markdown(
                            f"<p style='color:var(--muted);font-size:.88rem'>{d['summary']}</p>",
                            unsafe_allow_html=True,
                        )
                        met = [
                            r
                            for r in d.get("requirement_analysis", [])
                            if r["match_status"]
                        ]
                        unmet = [
                            r
                            for r in d.get("requirement_analysis", [])
                            if not r["match_status"]
                        ]
                        if met:
                            st.markdown("**✅ Met**")
                            for r in met:
                                st.markdown(
                                    f"<small>• {r['requirement']}</small>",
                                    unsafe_allow_html=True,
                                )
                        if unmet:
                            st.markdown("**❌ Missing**")
                            for r in unmet:
                                st.markdown(
                                    f"<small>• {r['requirement']}</small>",
                                    unsafe_allow_html=True,
                                )
                        st.markdown("---")

                # ── FEATURE 4: AI Comparison Engine ──────────────────────────
                if st.button("🤖 AI Compare Candidates"):
                    compare_data = ""
                    for sel in selected:
                        d = lookup[sel]
                        compare_data += f"""
Candidate: {sel}

Summary:
{d['summary']}

Score:
{d['overall_score']}
"""
                    compare_prompt = f"""Compare these candidates for the following role.

Job Description:
{st.session_state.saved_jd}

Candidates:
{compare_data}

Give:
- strongest candidate
- biggest strengths
- biggest weaknesses
- hiring recommendation
- final ranking

Keep it concise and recruiter-focused."""

                    with st.spinner("Comparing candidates..."):
                        response = st.session_state.llm.invoke(compare_prompt)

                    st.markdown("## 🤖 AI Comparison Report")
                    st.write(response.content)

            elif len(selected) == 1:
                st.info("Select at least one more candidate.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — EMAILS
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### ✉️ Email Generation Centre")
        valid_cands = [
            c for c in st.session_state.candidates if "Error:" not in c["name"]
        ]
        if not valid_cands:
            st.warning("No valid candidates to email.")
        else:
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown("**⚙️ Configuration**")
                n_invite = st.slider(
                    "Top candidates to invite",
                    1,
                    len(valid_cands),
                    min(3, len(valid_cands)),
                )
                min_sc = st.slider("Minimum score to invite", 0, 100, 70)
            with ec2:
                st.markdown("**📅 Interview Scheduling**")
                idate = st.date_input("Interview Date")
                itime = st.time_input("Interview Time")

            if st.button(
                "✉️ Generate All Emails", use_container_width=True, type="primary"
            ):
                with st.spinner("Drafting personalised emails…"):
                    dt = f"{idate.strftime('%A, %B %d, %Y')} at {itime.strftime('%I:%M %p')}"
                    st.session_state.generated_emails = generate_email_templates(
                        valid_cands,
                        {"title": job_title(st.session_state.saved_jd)},
                        n_invite,
                        min_sc,
                        dt,
                        st.session_state.llm,
                    )

            if st.session_state.get("generated_emails"):
                st.markdown("---")
                ic, rc = st.columns(2)
                with ic:
                    st.markdown("#### ✅ Invitations")
                    for em in st.session_state.generated_emails.get("invitations", []):
                        with st.expander(f"To: {em['name']}", expanded=True):
                            st.code(em["email_body"], language=None)
                    if not st.session_state.generated_emails.get("invitations"):
                        st.info("No candidates met the score threshold.")
                with rc:
                    st.markdown("#### ❌ Rejections")
                    for em in st.session_state.generated_emails.get("rejections", []):
                        with st.expander(f"To: {em['name']}", expanded=True):
                            st.code(em["email_body"], language=None)
                    if not st.session_state.generated_emails.get("rejections"):
                        st.info("All candidates were invited — no rejections needed.")
