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
import json
import sqlite3
import hashlib
from datetime import datetime
from functools import lru_cache

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet

    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HireIQ | AI-Powered Hiring Intelligence",
    page_icon="🧠",
    layout="wide",
)

# ════════════════════════════════════════════════════════════════════════════
# MULTI-RECRUITER ACCOUNT SYSTEM
# ════════════════════════════════════════════════════════════════════════════
RECRUITER_ACCOUNTS = {
    "admin": {"password": "hireiq", "role": "Admin", "name": "Admin Recruiter"},
    "hr1": {"password": "hr2026", "role": "Recruiter", "name": "HR Recruiter 1"},
    "hiring": {"password": "hire2026", "role": "Manager", "name": "Hiring Manager"},
}


# ════════════════════════════════════════════════════════════════════════════
# SESSION SECURITY SIMULATION (JWT-style token)
# ════════════════════════════════════════════════════════════════════════════
def generate_session_token(username: str) -> str:
    payload = f"{username}:{datetime.now().date()}:hireiq_secret"
    return hashlib.sha256(payload.encode()).hexdigest()[:16].upper()


def verify_session_token(username: str, token: str) -> bool:
    return token == generate_session_token(username)


# ════════════════════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect("hireiq.db")
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT, job_name TEXT, name TEXT,
        score INTEGER, summary TEXT, recruiter TEXT, ts TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS notes (
        candidate TEXT, note TEXT, recruiter TEXT, ts TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS email_log (
        candidate TEXT, email_type TEXT, recruiter TEXT, ts TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS candidate_memory (
        candidate TEXT, skills TEXT, recruiter TEXT, ts TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS interview_evaluations (
        candidate TEXT, score_text TEXT, recruiter TEXT, ts TEXT)""")
    conn.commit()
    return conn


def save_candidates_to_db(candidates, job_name="", recruiter=""):
    conn = init_db()
    cur = conn.cursor()
    ts = str(datetime.now())
    for c in candidates:
        if "Error:" not in c["name"]:
            cur.execute(
                "INSERT INTO candidates (job_name, name, score, summary, recruiter, ts) VALUES (?,?,?,?,?,?)",
                (job_name, c["name"], c["overall_score"], c["summary"], recruiter, ts),
            )
            cur.execute(
                "INSERT INTO candidate_memory (candidate, skills, recruiter, ts) VALUES (?,?,?,?)",
                (c["name"], c["summary"][:500], recruiter, ts),
            )
    conn.commit()
    conn.close()


def save_note_to_db(candidate_name, note, recruiter=""):
    conn = init_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notes (candidate, note, recruiter, ts) VALUES (?,?,?,?)",
        (candidate_name, note, recruiter, str(datetime.now())),
    )
    conn.commit()
    conn.close()


def log_email_to_db(candidate, email_type, recruiter=""):
    conn = init_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO email_log (candidate, email_type, recruiter, ts) VALUES (?,?,?,?)",
        (candidate, email_type, recruiter, str(datetime.now())),
    )
    conn.commit()
    conn.close()


def save_interview_eval_to_db(candidate, eval_text, recruiter=""):
    conn = init_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO interview_evaluations (candidate, score_text, recruiter, ts) VALUES (?,?,?,?)",
        (candidate, eval_text, recruiter, str(datetime.now())),
    )
    conn.commit()
    conn.close()


def get_historical_stats():
    try:
        conn = init_db()
        cur = conn.cursor()
        cur.execute("SELECT AVG(score) FROM candidates")
        avg = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM candidates")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(DISTINCT job_name) FROM candidates WHERE job_name != ''"
        )
        roles = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM email_log")
        emails = cur.fetchone()[0]
        conn.close()
        return round(avg or 0, 1), total, roles, emails
    except Exception:
        return 0.0, 0, 0, 0


def search_candidate_memory(query):
    try:
        conn = init_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT candidate, skills, recruiter, ts FROM candidate_memory WHERE skills LIKE ? LIMIT 10",
            (f"%{query}%",),
        )
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


# ════════════════════════════════════════════════════════════════════════════
# LLM INIT
# ════════════════════════════════════════════════════════════════════════════
if "llm" not in st.session_state:
    try:
        st.session_state.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=st.secrets["GROQ_API_KEY"],
        )
    except (KeyError, FileNotFoundError):
        st.error("🔴 GROQ_API_KEY not found. Add it to `.streamlit/secrets.toml`")
        st.stop()

# ════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');
:root {
    --bg:#0D1117; --card:#161B22; --border:#30363D;
    --text:#E2E8F0; --muted:#94A3B8; --accent:#007BFF; --glow:rgba(0,123,255,0.25);
}
html,body,[class*="st-"]{font-family:'Inter',sans-serif;color:var(--text);}
.stApp{background-color:var(--bg);background-image:radial-gradient(var(--border) 0.5px,transparent 0.5px);background-size:15px 15px;}
.block-container{padding-top:2rem!important;}
@keyframes up{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.hiq-header{text-align:center;margin-bottom:2.5rem;}
.hiq-header h1{font-family:'Playfair Display',serif;font-size:5rem;font-weight:700;background:linear-gradient(135deg,#007BFF,#00C6FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:up 1s ease-out 0.2s both;}
.hiq-header p{color:var(--muted);font-size:1.15rem;animation:up 1s ease-out 0.5s both;}
.hiq-sh{font-size:1.4rem;font-weight:600;border-bottom:1px solid var(--border);padding-bottom:.75rem;margin-bottom:1.25rem;}
.login-card{max-width:420px;margin:4rem auto;background:#161B22;border:1px solid #30363D;border-radius:16px;padding:2.5rem;}
.role-badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.78rem;font-weight:600;background:rgba(0,123,255,.15);color:#007BFF;border:1px solid #007BFF;margin-left:8px;}
.token-badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:.72rem;font-weight:600;background:rgba(40,167,69,.1);color:#28a745;border:1px solid #28a745;font-family:monospace;}
.rec-card{background:#161B22;border:1px solid #30363D;border-radius:12px;padding:1.2rem;margin-bottom:.75rem;}
.rec-rank{font-size:2rem;font-weight:700;color:#007BFF;}
.mem-card{background:#0d1117;border:1px solid #30363D;border-radius:8px;padding:.8rem 1rem;margin-bottom:.5rem;}
.forecast-card{background:rgba(0,123,255,.05);border:1px solid rgba(0,123,255,.2);border-radius:10px;padding:1rem;margin:.5rem 0;}
.notif-item{background:rgba(0,123,255,.07);border-left:3px solid var(--accent);border-radius:0 6px 6px 0;padding:.4rem .8rem;margin-bottom:.4rem;font-size:.83rem;}
.stage-badge{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.8rem;font-weight:600;margin:2px;background:rgba(0,123,255,.15);color:#007BFF;border:1px solid #007BFF;}
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
.tag-chip{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.8rem;font-weight:600;margin:2px;}
.tag-high{background:rgba(40,167,69,.2);color:#28a745;border:1px solid #28a745;}
.tag-review{background:rgba(255,193,7,.2);color:#ffc107;border:1px solid #ffc107;}
.tag-technical{background:rgba(0,123,255,.2);color:#007BFF;border:1px solid #007BFF;}
.tag-final{background:rgba(156,39,176,.2);color:#9c27b0;border:1px solid #9c27b0;}
.tag-rejected{background:rgba(220,53,69,.2);color:#dc3545;border:1px solid #dc3545;}
.xai{border-left:3px solid;padding:.5rem 1rem;margin-bottom:.75rem;border-radius:0 6px 6px 0;}
.xai-y{border-color:#28a745;background:rgba(40,167,69,.05);}
.xai-n{border-color:#dc3545;background:rgba(220,53,69,.05);}
.bubble{padding:.75rem 1rem;border-radius:10px;margin-bottom:.6rem;max-width:82%;word-wrap:break-word;}
.bubble.user{background:var(--accent);color:#fff;margin-left:auto;border-bottom-right-radius:0;}
.bubble.assistant{background:#1e2530;color:var(--text);border-bottom-left-radius:0;}
.cname{font-size:1.6rem;font-weight:700;color:#fff;margin:0;}
.action-tag{display:inline-block;padding:3px 12px;border-radius:8px;font-size:.83rem;font-weight:600;background:rgba(0,123,255,.12);color:#007BFF;border:1px solid rgba(0,123,255,.3);}
.activity-item{padding:.4rem .8rem;border-left:2px solid var(--accent);margin-bottom:.4rem;font-size:.88rem;color:var(--muted);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stFileUploaderDropzoneButton"]{font-size:0!important;color:transparent!important;}
[data-testid="stFileUploaderDropzoneButton"] *{font-size:0!important;color:transparent!important;}
[data-testid="stFileUploaderDropzoneButton"]::after{content:"Browse files";font-size:.88rem;font-weight:600;color:#E2E8F0;}
</style>
""",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
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
            "job_name": "",
            "activity_log": [],
            "authenticated": False,
            "current_user": "",
            "current_user_role": "",
            "current_user_name": "",
            "session_token": "",
        }
    )
for _k, _d in [
    ("shortlist", []),
    ("job_name", ""),
    ("activity_log", []),
    ("authenticated", False),
    ("current_user", ""),
    ("current_user_role", ""),
    ("current_user_name", ""),
    ("session_token", ""),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _d


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
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


def next_action(score):
    if score >= 85:
        return "📞 Schedule final interview"
    if score >= 70:
        return "🧠 Conduct technical assessment"
    if score >= 50:
        return "📋 Review manually"
    return "❌ Reject candidate"


@lru_cache(maxsize=64)
def cached_label(score: int) -> str:
    if score >= 85:
        return "⭐ Strong Hire"
    if score >= 70:
        return "✅ Potential Hire"
    if score >= 50:
        return "⚠️ Borderline"
    return "❌ Weak Match"


def job_title(jd):
    for line in jd.splitlines():
        s = line.strip()
        if s:
            return s[:80]
    return "the position"


def llm_cached(key, prompt):
    if key not in st.session_state:
        try:
            resp = st.session_state.llm.invoke(prompt)
            st.session_state[key] = resp.content
        except Exception as e:
            st.session_state[key] = f"Could not generate: {e}"
    return st.session_state[key]


def log_activity(msg: str):
    user = st.session_state.get("current_user", "system")
    ts = datetime.now().strftime("%H:%M")
    st.session_state.activity_log.append(f"[{ts}] [{user}] {msg}")


def tag_css_class(tag: str) -> str:
    return {
        "High Priority": "tag-high",
        "Needs Review": "tag-review",
        "Technical Round": "tag-technical",
        "Final Interview": "tag-final",
        "Rejected": "tag-rejected",
    }.get(tag, "tag-review")


def save_session_data():
    return json.dumps(
        {
            "timestamp": str(datetime.now()),
            "job_name": st.session_state.get("job_name", ""),
            "recruiter": st.session_state.get("current_user_name", ""),
            "job_description": st.session_state.saved_jd,
            "candidates": st.session_state.candidates,
            "shortlist": st.session_state.shortlist,
        },
        indent=2,
    )


def generate_pdf_report(text: str) -> str:
    path = "/tmp/hireiq_report.pdf"
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    doc.build([Paragraph(text.replace("\n", "<br/>"), styles["BodyText"])])
    return path


def build_recommendation_engine(candidates):
    valid = [c for c in candidates if "Error:" not in c["name"]]
    top3 = sorted(valid, key=lambda x: x["overall_score"], reverse=True)[:3]
    recs = []
    for i, c in enumerate(top3):
        score = clamp(c["overall_score"])
        recs.append(
            {
                "rank": i + 1,
                "name": c["name"],
                "score": score,
                "confidence": max(60, min(98, score - i * 3)),
                "summary": c["summary"],
            }
        )
    return recs


# ════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ════════════════════════════════════════════════════════════════════════════
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
            st.toast(f"⚡ Queued AI evaluation task for {res['filename']}")
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
        save_candidates_to_db(
            st.session_state.candidates,
            st.session_state.job_name,
            st.session_state.current_user,
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

        log_activity(
            f"Analysis complete — {len(results)} candidate(s) scored & saved to DB"
        )
        st.session_state.step = "results"


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — WORKSPACE + NOTIFICATIONS + PRODUCTIVITY
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.authenticated:
    st.sidebar.markdown("## 🏢 Recruiter Workspace")
    workspace = st.sidebar.selectbox(
        "Active Team",
        ["AI Hiring Team", "Backend Recruitment", "Executive Hiring", "Campus Hiring"],
        label_visibility="collapsed",
    )
    st.sidebar.success(f"📂 Workspace: **{workspace}**")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"👤 **{st.session_state.current_user_name}**")
    st.sidebar.markdown(f"🔑 Role: `{st.session_state.current_user_role}`")
    if st.session_state.get("session_token"):
        st.sidebar.markdown(
            f"🔒 Token: <span class='token-badge'>{st.session_state.session_token}</span>",
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")

    # FEATURE: Recruiter Notification Center
    st.sidebar.markdown("## 🔔 Notifications")
    cand_count = len(st.session_state.get("candidates", []))
    shortlist_count = len(st.session_state.get("shortlist", []))
    notifications = [
        (
            f"{cand_count} candidate(s) in current session"
            if cand_count
            else "No candidates loaded yet"
        ),
        (
            f"{shortlist_count} candidate(s) shortlisted"
            if shortlist_count
            else "Shortlist is empty"
        ),
        "🤖 AI caching active — responses instant after first load",
    ]
    # Add dynamic notification if strong candidate found
    if st.session_state.get("candidates"):
        top_score = clamp(st.session_state.candidates[0].get("overall_score", 0))
        if top_score >= 90:
            notifications.insert(0, f"🔥 Top candidate scored {top_score}/100!")
        elif top_score >= 75:
            notifications.insert(0, f"✅ Strong top candidate — {top_score}/100")
    for note in notifications:
        st.sidebar.markdown(
            f"<div class='notif-item'>{note}</div>", unsafe_allow_html=True
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Quick Stats**")
    hist_avg, hist_total, hist_roles, hist_emails = get_historical_stats()
    st.sidebar.metric("Total Evaluated", hist_total)
    st.sidebar.metric("Emails Sent", hist_emails)

    # FEATURE: Recruiter Productivity Analytics (sidebar)
    if st.session_state.get("candidates"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ⚡ My Productivity")
        total_reviews = len(st.session_state.candidates)
        shortlisted_cnt = len(st.session_state.shortlist)
        st.sidebar.metric("Profiles Reviewed", total_reviews)
        st.sidebar.metric("Candidates Shortlisted", shortlisted_cnt)
        if total_reviews:
            efficiency = round((shortlisted_cnt / total_reviews) * 100, 1)
            st.sidebar.metric("Shortlisting Efficiency", f"{efficiency}%")

# ════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    st.markdown(
        """
    <div class="hiq-header" style="margin-top:3rem">
      <h1>HireIQ</h1>
      <p>AI-Powered Hiring Intelligence Platform</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    _, lc, _ = st.columns([1, 1.2, 1])
    with lc:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.markdown("### 🔐 Recruiter Login")
        user = st.text_input("Username", placeholder="admin / hr1 / hiring")
        pwd = st.text_input("Password", type="password", placeholder="••••••••")
        if st.button("Login", use_container_width=True):
            account = RECRUITER_ACCOUNTS.get(user)
            if account and account["password"] == pwd:
                token = generate_session_token(user)
                st.session_state.update(
                    {
                        "authenticated": True,
                        "current_user": user,
                        "current_user_role": account["role"],
                        "current_user_name": account["name"],
                        "session_token": token,
                    }
                )
                log_activity(f"Authenticated — token: {token}")
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.markdown(
            """<br><small style='color:#94A3B8'>
        Demo accounts:<br>admin / hireiq &nbsp;·&nbsp; hr1 / hr2026 &nbsp;·&nbsp; hiring / hire2026
        </small>""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="hiq-header">
  <h1>HireIQ</h1>
  <p>AI-Powered Hiring Intelligence &nbsp;·&nbsp; Screen Smarter. Hire Faster. Explain Every Decision.</p>
</div>
""",
    unsafe_allow_html=True,
)

hdr1, hdr2 = st.columns([8, 1])
with hdr1:
    role_badge = f"<span class='role-badge'>{st.session_state.current_user_role}</span>"
    token_badge = (
        f"<span class='token-badge'>🔒 {st.session_state.session_token}</span>"
    )
    st.markdown(
        f"⚡ AI caching · 🗄 SQLite · 👤 **{st.session_state.current_user_name}** {role_badge} &nbsp; {token_badge}",
        unsafe_allow_html=True,
    )
with hdr2:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.session_token = ""
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.step == "upload":
    st.markdown(
        "<div class='hiq-sh'>Step 1 &nbsp;·&nbsp; Provide Your Data</div>",
        unsafe_allow_html=True,
    )
    st.session_state.job_name = st.text_input(
        "📌 Job Role",
        value=st.session_state.job_name,
        placeholder="e.g. Senior AI Engineer",
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**📝 Job Description**")
        st.session_state.saved_jd = st.text_area(
            "Job Description",
            value=st.session_state.saved_jd,
            placeholder="Paste the full job description here…",
            height=320,
            label_visibility="collapsed",
        )
    with col2:
        st.markdown("**👥 Upload Candidate Resumes (PDF)**")
        new_files = st.file_uploader(
            "Candidate PDFs",
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

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — WEIGHTING
# ════════════════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — RESULTS
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == "results":

    ta, tb, tc = st.columns([5, 1, 1])
    with ta:
        role_label = (
            f" — **{st.session_state.job_name}**" if st.session_state.job_name else ""
        )
        st.success(
            f"✅ Analysis complete{role_label} — **{len(st.session_state.candidates)}** candidate(s) ranked."
        )
    with tb:
        st.download_button(
            "💾 Save Session",
            save_session_data(),
            file_name=f"hireiq_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            use_container_width=True,
        )
    with tc:
        st.markdown('<div class="sbtn">', unsafe_allow_html=True)
        st.button("🔄 Start Over", on_click=go_back, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # EXECUTIVE ANALYTICS
    # ════════════════════════════════════════════════════════════════════════
    scores = [
        c["overall_score"]
        for c in st.session_state.candidates
        if "Error:" not in c["name"]
    ]
    if scores:
        avg_score = round(sum(scores) / len(scores), 1)
        strong_matches = len([s for s in scores if s >= 75])
        hire_ready = len([s for s in scores if s >= 80])

        st.markdown("## 📊 Executive Hiring Insights")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Candidates", len(scores))
        with m2:
            st.metric("Average Score", avg_score)
        with m3:
            st.metric("Strong Matches", strong_matches)
        with m4:
            st.metric("Hire-Ready", hire_ready)

        high = len([s for s in scores if s >= 75])
        mid = len([s for s in scores if 50 <= s < 75])
        low = len([s for s in scores if s < 50])
        st.markdown("### 🧠 Recruitment Snapshot")
        st.markdown(
            f"- ✅ Strong candidates: **{high}**\n- ⚠️ Medium-fit: **{mid}**\n- ❌ Weak candidates: **{low}**"
        )

        st.markdown("## 📈 AI Hiring Trends")
        strong_ratio = round(
            (len([s for s in scores if s >= 75]) / len(scores)) * 100, 1
        )
        tr1, tr2 = st.columns(2)
        with tr1:
            st.metric("Strong Talent Ratio", f"{strong_ratio}%")
        with tr2:
            st.metric("Average Talent Score", avg_score)
        if strong_ratio < 30:
            st.warning(
                "⚠️ Low high-quality candidate density. Consider expanding sourcing channels."
            )
        elif strong_ratio < 60:
            st.info("ℹ️ Moderate talent pipeline. A few strong candidates are present.")
        else:
            st.success(
                "✅ Healthy talent pipeline detected. Strong candidate pool available."
            )

        st.markdown("## 🔮 Hiring Forecast Intelligence")
        predicted_hires = max(1, int(len(scores) * 0.25))
        time_to_hire_days = max(7, 30 - int(strong_ratio / 5))
        pipeline_health = (
            "🟢 Strong"
            if strong_ratio >= 60
            else ("🟡 Moderate" if strong_ratio >= 30 else "🔴 Weak")
        )
        fp1, fp2, fp3 = st.columns(3)
        with fp1:
            st.metric("Predicted Successful Hires", predicted_hires)
        with fp2:
            st.metric("Est. Time to Hire (days)", time_to_hire_days)
        with fp3:
            st.metric("Pipeline Health", pipeline_health)
        if predicted_hires < 2:
            st.warning("⚠️ Pipeline may require broader sourcing or relaxed criteria.")
        else:
            st.success(
                f"✅ Healthy projected pipeline — {predicted_hires} hire(s) predicted."
            )

        # FEATURE: Recruiter Productivity Analytics (main page)
        st.markdown("## ⚡ Recruiter Productivity")
        total_reviews = len(st.session_state.candidates)
        shortlisted_cnt = len(st.session_state.shortlist)
        rp1, rp2, rp3 = st.columns(3)
        with rp1:
            st.metric("Profiles Reviewed", total_reviews)
        with rp2:
            st.metric("Candidates Shortlisted", shortlisted_cnt)
        with rp3:
            if total_reviews:
                efficiency = round((shortlisted_cnt / total_reviews) * 100, 1)
                st.metric("Shortlisting Efficiency", f"{efficiency}%")

        st.markdown("### 📈 Score Distribution")
        st.bar_chart(
            {
                c["name"]: c["overall_score"]
                for c in st.session_state.candidates
                if "Error:" not in c["name"]
            }
        )

    # Historical Analytics
    hist_avg, hist_total, hist_roles, hist_emails = get_historical_stats()
    if hist_total > 0:
        st.markdown("## 📚 Historical Hiring Analytics")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("All-Time Avg Score", hist_avg)
        with h2:
            st.metric("Total Candidates Evaluated", hist_total)
        with h3:
            st.metric("Roles Processed", hist_roles)
        with h4:
            st.metric("Emails Sent (DB)", hist_emails)

    # Talent Intelligence Search
    st.markdown("## 🔍 Talent Intelligence Search")
    memory_query = st.text_input(
        "Search historical candidate skills",
        placeholder="e.g. LangChain, AWS, FastAPI, Python",
        label_visibility="collapsed",
    )
    if memory_query and memory_query.strip():
        rows = search_candidate_memory(memory_query.strip())
        if rows:
            st.success(
                f"Found {len(rows)} historical candidate(s) matching **{memory_query}**"
            )
            for row in rows:
                st.markdown(
                    f"""<div class='mem-card'><b>👤 {row[0]}</b>
<span style='color:var(--muted);font-size:.8rem;margin-left:8px'>— reviewed by {row[2] or 'unknown'}</span><br>
<small style='color:var(--muted)'>{row[1][:250]}…</small></div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.warning(f"No historical candidates found matching **{memory_query}**.")

    # AI Candidate Recommendations
    if st.session_state.candidates:
        recs = build_recommendation_engine(st.session_state.candidates)
        if recs:
            st.markdown("## 🧠 AI Candidate Recommendations")
            rec_cols = st.columns(len(recs))
            for col, rec in zip(rec_cols, recs):
                with col:
                    bc = "bh" if rec["score"] >= 75 else "bm"
                    st.markdown(
                        f"""<div class='rec-card'><div class='rec-rank'>#{rec['rank']}</div>
<b>{rec['name']}</b><br><span class='badge {bc}'>{rec['score']} / 100</span><br>
<small style='color:var(--muted)'>Confidence: {rec['confidence']}%</small><br>
<small style='color:var(--muted)'>{rec['summary'][:100]}…</small></div>""",
                        unsafe_allow_html=True,
                    )

    # Top candidate banner
    if st.session_state.candidates:
        top = st.session_state.candidates[0]
        top_score = clamp(top["overall_score"])
        st.info(
            f"🏆 **Top Candidate:** {top['name']} — {top_score} / 100  ·  {decision(top_score)}"
        )
        with st.expander("🏆 Why was this candidate ranked #1?"):
            with st.spinner("Analysing…"):
                result = llm_cached(
                    "top_candidate_reason",
                    f"""You are a senior hiring director.
Job Description: {st.session_state.saved_jd}
Top Candidate: {top['name']} | Summary: {top['summary']} | Score: {top_score}
Explain: Why ranked highest, biggest strengths, hiring advantages, potential risks, final recommendation.
Keep it concise and executive-level.""",
                )
            st.write(result)

    # Shortlist bar
    if st.session_state.shortlist:
        pills = "".join(
            f"<span class='shortlist-pill'>⭐ {n}</span>"
            for n in st.session_state.shortlist
        )
        st.markdown(
            f"<div style='margin-bottom:.5rem'>📋 <b>Shortlisted:</b> {pills}</div>",
            unsafe_allow_html=True,
        )
        export_data = "\n".join(
            f"Candidate: {n}\nRecruiter Notes:\n{st.session_state.get(f'notes_{n}','(none)')}\n{'-'*40}"
            for n in st.session_state.shortlist
        )
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.download_button(
                "📥 Download Shortlist",
                "\n".join(st.session_state.shortlist),
                file_name="shortlist.txt",
            )
        with sc2:
            st.download_button(
                "📥 Export + Notes", export_data, file_name="shortlist_notes.txt"
            )
        with sc3:
            if st.button("📬 Send Interview Invitations"):
                with st.spinner("Sending…"):
                    for candidate in st.session_state.shortlist:
                        time.sleep(0.3)
                        st.toast(f"✉️ Invitation sent to {candidate}")
                        log_email_to_db(
                            candidate, "invitation", st.session_state.current_user
                        )
                        log_activity(f"Invitation sent → {candidate}")
                st.success(
                    f"✅ {len(st.session_state.shortlist)} invitation(s) dispatched"
                )

    tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "🤝 Compare", "✉️ Emails & Report"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — LEADERBOARD
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        if not st.session_state.candidates:
            st.info("No candidates found. Go back and try again.")
        else:
            sf1, sf2 = st.columns([2, 1])
            with sf1:
                search_query = st.text_input(
                    "🔎 Search",
                    placeholder="Search by name or skills…",
                    label_visibility="collapsed",
                )
            with sf2:
                min_filter_score = st.slider("Min Score", 0, 100, 0, key="filter_score")

            filtered_candidates = [
                c
                for c in st.session_state.candidates
                if "Error:" not in c["name"]
                and clamp(c.get("overall_score", 0)) >= min_filter_score
                and (
                    not search_query
                    or search_query.lower() in c["name"].lower()
                    or search_query.lower() in c.get("summary", "").lower()
                )
            ]
            error_candidates = [
                c for c in st.session_state.candidates if "Error:" in c["name"]
            ]

            if not filtered_candidates and not error_candidates:
                st.info("No candidates match your search/filter criteria.")
            else:
                st.caption(
                    f"📊 Showing {len(filtered_candidates)} candidate(s) — ranked by AI score"
                )

            if filtered_candidates:
                selected_bulk = st.multiselect(
                    "⚡ Bulk Select",
                    [c["name"] for c in filtered_candidates],
                    key="bulk_select",
                )
                if selected_bulk:
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button("⭐ Bulk Shortlist Selected"):
                            added = sum(
                                1
                                for c in selected_bulk
                                if c not in st.session_state.shortlist
                                and not st.session_state.shortlist.append(c)
                            )
                            st.success(f"✅ {added} candidate(s) shortlisted")
                            log_activity(
                                f"Bulk shortlisted: {', '.join(selected_bulk)}"
                            )
                            st.rerun()
                    with bc2:
                        if st.button("📬 Bulk Queue Invites"):
                            for candidate in selected_bulk:
                                st.toast(f"✉️ Queued for {candidate}")
                                log_email_to_db(
                                    candidate,
                                    "bulk_invite",
                                    st.session_state.current_user,
                                )
                            st.success(f"✅ {len(selected_bulk)} invite(s) queued")
                            log_activity(f"Bulk invites: {', '.join(selected_bulk)}")

            for rank, cand in enumerate(
                filtered_candidates + error_candidates, start=1
            ):
                name = cand["name"]
                score = clamp(cand.get("overall_score", 0))
                is_err = "Error:" in name

                with st.container():
                    st.markdown("---")
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

                    st.progress(score / 100.0)

                    ml, dl, al, cl = st.columns(4)
                    with ml:
                        st.markdown(f"**Match:** {match_label(score)}")
                    with dl:
                        st.markdown(f"**Decision:** {decision(score)}")
                    with al:
                        st.markdown(
                            f"**Action:** <span class='action-tag'>{next_action(score)}</span>",
                            unsafe_allow_html=True,
                        )
                    with cl:
                        st.markdown(f"**Rating:** {cached_label(score)}")

                    st.markdown(
                        f"<p style='color:var(--muted);margin-top:.5rem'>{cand['summary']}</p>",
                        unsafe_allow_html=True,
                    )

                    if not is_err:
                        req_analysis = cand.get("requirement_analysis", [])
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
                                    st.markdown(
                                        " ".join(
                                            f"<span class='skill-tag skill-match'>{s}</span>"
                                            for s in matched
                                        ),
                                        unsafe_allow_html=True,
                                    )
                            with si2:
                                if missing:
                                    st.markdown("**⚠️ Key gaps:**")
                                    st.markdown(
                                        " ".join(
                                            f"<span class='skill-tag skill-missing'>{s}</span>"
                                            for s in missing
                                        ),
                                        unsafe_allow_html=True,
                                    )

                        st.markdown(
                            "<div style='height:.4rem'></div>", unsafe_allow_html=True
                        )

                        # FEATURE: Hiring Pipeline Stages
                        pip_col, tag_col, sl_col = st.columns([2, 2, 1])
                        with pip_col:
                            pipeline_stage = st.selectbox(
                                "📋 Hiring Stage",
                                [
                                    "Applied",
                                    "Screening",
                                    "Technical Interview",
                                    "Final Interview",
                                    "Offer",
                                    "Rejected",
                                ],
                                key=f"stage_{name}",
                                label_visibility="collapsed",
                            )
                            st.markdown(
                                f"<span class='stage-badge'>📋 {pipeline_stage}</span>",
                                unsafe_allow_html=True,
                            )
                        with tag_col:
                            tag = st.selectbox(
                                "🏷 Tag",
                                [
                                    "High Priority",
                                    "Needs Review",
                                    "Technical Round",
                                    "Final Interview",
                                    "Rejected",
                                ],
                                key=f"tag_{name}",
                                label_visibility="collapsed",
                            )
                            st.markdown(
                                f"<span class='tag-chip {tag_css_class(tag)}'>{tag}</span>",
                                unsafe_allow_html=True,
                            )
                        with sl_col:
                            if name not in st.session_state.shortlist:
                                if st.button("⭐ Shortlist", key=f"short_{rank}"):
                                    st.session_state.shortlist.append(name)
                                    log_activity(f"Shortlisted: {name}")
                                    st.rerun()
                            else:
                                st.markdown("⭐ **Shortlisted**")

                        # FEATURE: Candidate Tagging System (multi-select skills tags)
                        candidate_tags = st.multiselect(
                            "🏷 Candidate Tags",
                            [
                                "Strong Communication",
                                "Leadership Potential",
                                "Fast Learner",
                                "Backend Expert",
                                "AI Specialist",
                                "Needs Mentorship",
                                "Remote-Ready",
                                "Culture Fit",
                            ],
                            key=f"ctags_{name}",
                            label_visibility="collapsed",
                        )
                        if candidate_tags:
                            st.markdown(
                                "**Tags:** "
                                + ", ".join(f"`{t}`" for t in candidate_tags)
                            )

                        # Notes + Reviewer
                        note_col2, rev_col = st.columns([3, 2])
                        with note_col2:
                            note_val = st.text_area(
                                "Recruiter Notes",
                                key=f"notes_{name}",
                                placeholder="Observations, interview notes…",
                                height=68,
                                label_visibility="collapsed",
                            )
                            if st.button("💾 Save Notes", key=f"save_note_{rank}"):
                                save_note_to_db(
                                    name, note_val or "", st.session_state.current_user
                                )
                                st.toast(f"✅ Notes saved for {name}")
                                log_activity(f"Notes saved for: {name}")
                        with rev_col:
                            reviewer = st.selectbox(
                                "👥 Assign Reviewer",
                                [
                                    "Technical Lead",
                                    "Hiring Manager",
                                    "HR Team",
                                    "Senior Recruiter",
                                ],
                                key=f"reviewer_{name}",
                                label_visibility="collapsed",
                            )
                            st.caption(f"👥 Assigned: **{reviewer}**")

                        with st.expander("🧠 Why this score?"):
                            with st.spinner("Analysing…"):
                                result = llm_cached(
                                    f"explain_{name}",
                                    f"""You are a senior hiring manager.
Job Description: {st.session_state.saved_jd}
Candidate Summary: {cand.get('summary', '')}
Score: {score}/100
Give: Why this score, top strengths (bullets), weaknesses/risks, missing skills, final recommendation. Keep it concise.""",
                                )
                            st.write(result)

                        if score < 75:
                            with st.expander("❌ Why not selected?"):
                                with st.spinner("Analysing…"):
                                    result = llm_cached(
                                        f"reject_{name}",
                                        f"""Explain why this candidate may not be selected.
Candidate: {cand['summary']} | Score: {score}
Job Description: {st.session_state.saved_jd}
Give: missing skills, concerns, gaps, hiring risks. Keep it professional and concise.""",
                                    )
                                st.write(result)

                        # FEATURE: AI Risk Analysis
                        with st.expander("⚠️ AI Risk Analysis"):
                            if st.button("🔍 Analyze Hiring Risk", key=f"risk_{rank}"):
                                with st.spinner("Analysing candidate risk profile…"):
                                    risk_prompt = f"""You are an AI hiring risk analyst.

Candidate Summary:
{cand.get('summary', '')}

Score: {score}/100

Identify and structure:
- Hiring risks
- Skill gaps vs job requirements
- Retention concerns
- Communication concerns
- Leadership or culture concerns

Job Description context:
{st.session_state.saved_jd[:500]}

Keep each section concise and professional."""
                                    risk_resp = st.session_state.llm.invoke(risk_prompt)
                                    st.write(risk_resp.content)
                                    log_activity(f"Risk analysis run for: {name}")

                        with st.expander("🎤 AI Interview Questions"):
                            with st.spinner("Generating questions…"):
                                result = llm_cached(
                                    f"questions_{name}",
                                    f"""You are a senior technical recruiter.
Job Description: {st.session_state.saved_jd}
Candidate Summary: {cand.get('summary', '')}
Generate: 5 technical questions, 3 behavioral questions, 2 deep follow-up questions.
Make them highly relevant to this specific candidate and role.""",
                                )
                            st.write(result)

                        with st.expander("🎤 AI Interview Simulation"):
                            st.markdown(
                                "**Generate a full live interview session for this candidate.**"
                            )
                            if st.button(
                                "▶️ Generate Live Interview", key=f"interview_{rank}"
                            ):
                                with st.spinner("Generating interview session…"):
                                    resp = st.session_state.llm.invoke(
                                        f"""You are a senior technical interviewer.
Job Description: {st.session_state.saved_jd}
Candidate Summary: {cand.get('summary', '')}
Generate: 5 challenging technical questions, 2 behavioral questions, 1 system design question.
Include expected strong answers for each. Be realistic and challenging."""
                                    )
                                    st.write(resp.content)
                            st.markdown("---")
                            st.markdown(
                                "**📝 Evaluate a Candidate Interview Response**"
                            )
                            candidate_response = st.text_area(
                                "Paste candidate interview response here",
                                key=f"response_{rank}",
                                placeholder="Type or paste the candidate's response…",
                                height=100,
                            )
                            if st.button(
                                "🧠 Evaluate Interview Response", key=f"eval_{rank}"
                            ):
                                if candidate_response.strip():
                                    with st.spinner("Evaluating…"):
                                        eval_result = st.session_state.llm.invoke(
                                            f"""You are a hiring committee evaluator.
Evaluate this candidate interview response.
Response: {candidate_response}
Return: Communication Score (1-10), Technical Depth (1-10), Confidence Level, Red Flags, Hiring Recommendation.
Be specific and professional."""
                                        )
                                        st.write(eval_result.content)
                                        save_interview_eval_to_db(
                                            name,
                                            eval_result.content,
                                            st.session_state.current_user,
                                        )
                                        log_activity(f"Interview evaluated for: {name}")
                                        st.toast(
                                            f"✅ Interview evaluation saved for {name}"
                                        )
                                else:
                                    st.warning("Paste a candidate response first.")

                        with st.expander("📊 Full XAI Requirement Analysis"):
                            if not req_analysis:
                                st.info("No requirement data available.")
                            for r in req_analysis:
                                if r["match_status"]:
                                    st.markdown(
                                        f"<div class='xai xai-y'>✅ <b>{r['requirement']}</b><br><small><i>Evidence: \"{r['evidence']}\"</i></small></div>",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(
                                        f"<div class='xai xai-n'>❌ <b>{r['requirement']}</b><br><small><i>Reason: {r['evidence']}</i></small></div>",
                                        unsafe_allow_html=True,
                                    )

                        if st.button(
                            "🎯 Generate Structured Interview Questions",
                            key=f"iq_{rank}",
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
                                log_activity(
                                    f'Chat query about {name}: "{prompt[:40]}"'
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
                        st.markdown(f"**Rating:** {cached_label(s)}")
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
                if st.button("🤖 AI Compare Selected Candidates"):
                    compare_data = "\n\n".join(
                        f"Candidate: {sel}\nScore: {lookup[sel]['overall_score']}\nSummary:\n{lookup[sel]['summary']}"
                        for sel in selected
                    )
                    with st.spinner("Comparing candidates…"):
                        resp = st.session_state.llm.invoke(
                            f"""Compare these candidates. Job Description: {st.session_state.saved_jd}
Candidates: {compare_data}
Give: strongest candidate, each strength and weakness, hiring recommendation, final ranking. Be concise."""
                        )
                    st.markdown("### 🤖 AI Comparison Report")
                    st.write(resp.content)
                    log_activity(f"AI comparison: {', '.join(selected)}")
            elif len(selected) == 1:
                st.info("Select at least one more candidate.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — EMAILS & REPORT
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
                log_activity("Email drafts generated")
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
                        st.info("All candidates were invited.")

        st.markdown("---")
        st.markdown("## 📄 AI Hiring Report")
        if st.button("📥 Generate Hiring Report", use_container_width=True):
            report_data = "\n\n".join(
                f"Candidate: {c['name']}\nScore: {c['overall_score']}\nSummary:\n{c['summary']}"
                for c in st.session_state.candidates
                if "Error:" not in c["name"]
            )
            role_line = (
                f"Role: {st.session_state.job_name}\n\n"
                if st.session_state.job_name
                else ""
            )
            with st.spinner("Generating executive report…"):
                resp = st.session_state.llm.invoke(
                    f"""You are a senior hiring consultant.
{role_line}Job Description: {st.session_state.saved_jd}
Candidate Data: {report_data}
Generate a professional hiring report: top candidates ranked, key strengths, biggest skill gaps, overall recommendation, final shortlist.
Keep it executive-style and concise."""
                )
                st.session_state["final_hiring_report"] = resp.content
            log_activity("Hiring report generated")
        if "final_hiring_report" in st.session_state:
            st.text_area(
                "Generated Hiring Report",
                st.session_state["final_hiring_report"],
                height=420,
            )
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "📥 Download Report (.txt)",
                    st.session_state["final_hiring_report"],
                    file_name="hireiq_report.txt",
                )
            with dl2:
                if REPORTLAB_OK:
                    try:
                        pdf_path = generate_pdf_report(
                            st.session_state["final_hiring_report"]
                        )
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "📄 Download PDF Report",
                                f,
                                file_name="hireiq_report.pdf",
                                mime="application/pdf",
                            )
                    except Exception as e:
                        st.warning(f"PDF export failed: {e}")
                else:
                    st.info("Install `reportlab` to enable PDF export.")

    # ════════════════════════════════════════════════════════════════════════
    # ACTIVITY LOG
    # ════════════════════════════════════════════════════════════════════════
    if st.session_state.activity_log:
        st.markdown("---")
        st.markdown("## 📈 Recruiter Activity Timeline")
        for activity in reversed(st.session_state.activity_log[-15:]):
            st.markdown(
                f"<div class='activity-item'>{activity}</div>", unsafe_allow_html=True
            )
