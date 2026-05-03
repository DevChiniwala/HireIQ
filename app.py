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
# ★ DAY 16 FEATURE 1 — ROLE-BASED ACCESS CONTROL
#   Admin     → full platform access
#   Manager   → analytics + results + JD optimizer, no admin panel
#   Recruiter → upload + results + collab only
# ════════════════════════════════════════════════════════════════════════════
RECRUITER_ACCOUNTS = {
    "admin": {
        "password": "hireiq",
        "role": "Admin",
        "name": "Admin Recruiter",
        "permissions": [
            "upload",
            "results",
            "analytics",
            "admin",
            "jd_optimizer",
            "collab",
        ],
    },
    "hr1": {
        "password": "hr2026",
        "role": "Recruiter",
        "name": "HR Recruiter 1",
        "permissions": ["upload", "results", "collab"],
    },
    "hiring": {
        "password": "hire2026",
        "role": "Manager",
        "name": "Hiring Manager",
        "permissions": ["upload", "results", "analytics", "jd_optimizer", "collab"],
    },
}


def has_perm(perm: str) -> bool:
    return perm in st.session_state.get("current_user_permissions", [])


# ════════════════════════════════════════════════════════════════════════════
# SESSION SECURITY
# ════════════════════════════════════════════════════════════════════════════
def generate_session_token(username: str) -> str:
    payload = f"{username}:{datetime.now().date()}:hireiq_secret"
    return hashlib.sha256(payload.encode()).hexdigest()[:16].upper()


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
    cur.execute(
        """CREATE TABLE IF NOT EXISTS scheduled_interviews (
        candidate TEXT, interview_date TEXT, interview_time TEXT, recruiter TEXT, ts TEXT)"""
    )
    cur.execute("""CREATE TABLE IF NOT EXISTS bookmarks (
        candidate TEXT, recruiter TEXT, ts TEXT)""")
    # ★ DAY 16 — team collaboration comments
    cur.execute("""CREATE TABLE IF NOT EXISTS collab_comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate TEXT, comment TEXT, recruiter TEXT, role TEXT, ts TEXT)""")
    conn.commit()
    return conn


def save_candidates_to_db(candidates, job_name="", recruiter=""):
    conn = init_db()
    cur = conn.cursor()
    ts = str(datetime.now())
    for c in candidates:
        if "Error:" not in c["name"]:
            cur.execute(
                "INSERT INTO candidates (job_name,name,score,summary,recruiter,ts) VALUES (?,?,?,?,?,?)",
                (job_name, c["name"], c["overall_score"], c["summary"], recruiter, ts),
            )
            cur.execute(
                "INSERT INTO candidate_memory (candidate,skills,recruiter,ts) VALUES (?,?,?,?)",
                (c["name"], c["summary"][:500], recruiter, ts),
            )
    conn.commit()
    conn.close()


def save_note_to_db(candidate_name, note, recruiter=""):
    conn = init_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notes (candidate,note,recruiter,ts) VALUES (?,?,?,?)",
        (candidate_name, note, recruiter, str(datetime.now())),
    )
    conn.commit()
    conn.close()


def log_email_to_db(candidate, email_type, recruiter=""):
    conn = init_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO email_log (candidate,email_type,recruiter,ts) VALUES (?,?,?,?)",
        (candidate, email_type, recruiter, str(datetime.now())),
    )
    conn.commit()
    conn.close()


def save_interview_eval_to_db(candidate, eval_text, recruiter=""):
    conn = init_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO interview_evaluations (candidate,score_text,recruiter,ts) VALUES (?,?,?,?)",
        (candidate, eval_text, recruiter, str(datetime.now())),
    )
    conn.commit()
    conn.close()


def save_scheduled_interview_to_db(
    candidate, interview_date, interview_time, recruiter=""
):
    conn = init_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO scheduled_interviews (candidate,interview_date,interview_time,recruiter,ts) VALUES (?,?,?,?,?)",
        (
            candidate,
            str(interview_date),
            str(interview_time),
            recruiter,
            str(datetime.now()),
        ),
    )
    conn.commit()
    conn.close()


def save_bookmark_to_db(candidate, recruiter=""):
    conn = init_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO bookmarks (candidate,recruiter,ts) VALUES (?,?,?)",
        (candidate, recruiter, str(datetime.now())),
    )
    conn.commit()
    conn.close()


# ★ DAY 16 — collab helpers
def save_collab_comment(candidate, comment, recruiter, role):
    conn = init_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO collab_comments (candidate,comment,recruiter,role,ts) VALUES (?,?,?,?,?)",
        (candidate, comment, recruiter, role, str(datetime.now())),
    )
    conn.commit()
    conn.close()


def get_collab_comments(candidate):
    try:
        conn = init_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT comment,recruiter,role,ts FROM collab_comments WHERE candidate=? ORDER BY ts DESC LIMIT 20",
            (candidate,),
        )
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def get_historical_stats():
    try:
        conn = init_db()
        cur = conn.cursor()
        cur.execute("SELECT AVG(score) FROM candidates")
        avg = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM candidates")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(DISTINCT job_name) FROM candidates WHERE job_name!=''"
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
            "SELECT candidate,skills,recruiter,ts FROM candidate_memory WHERE skills LIKE ? LIMIT 10",
            (f"%{query}%",),
        )
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


# ★ DAY 16 — Admin DB overview
def get_admin_db_stats():
    try:
        conn = init_db()
        cur = conn.cursor()
        stats = {}
        for t in [
            "candidates",
            "notes",
            "email_log",
            "scheduled_interviews",
            "bookmarks",
            "collab_comments",
            "interview_evaluations",
        ]:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            stats[t] = cur.fetchone()[0]
        conn.close()
        return stats
    except Exception:
        return {}


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
:root{--bg:#0D1117;--card:#161B22;--border:#30363D;--text:#E2E8F0;--muted:#94A3B8;--accent:#007BFF;--glow:rgba(0,123,255,0.25);}
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
.role-admin{background:rgba(220,53,69,.15)!important;color:#dc3545!important;border-color:#dc3545!important;}
.role-manager{background:rgba(156,39,176,.15)!important;color:#9c27b0!important;border-color:#9c27b0!important;}
.token-badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:.72rem;font-weight:600;background:rgba(40,167,69,.1);color:#28a745;border:1px solid #28a745;font-family:monospace;}
.rec-card{background:#161B22;border:1px solid #30363D;border-radius:12px;padding:1.2rem;margin-bottom:.75rem;}
.rec-rank{font-size:2rem;font-weight:700;color:#007BFF;}
.mem-card{background:#0d1117;border:1px solid #30363D;border-radius:8px;padding:.8rem 1rem;margin-bottom:.5rem;}
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
.bookmark-pill{display:inline-block;background:rgba(255,193,7,.12);color:#ffc107;border:1px solid #ffc107;border-radius:20px;padding:3px 12px;font-size:.82rem;font-weight:600;margin:2px;}
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
.scheduler-card{background:rgba(0,123,255,.04);border:1px solid rgba(0,123,255,.18);border-radius:10px;padding:1rem 1.2rem;margin:.6rem 0;}
.trend-card{background:#161B22;border:1px solid #30363D;border-radius:10px;padding:1rem 1.2rem;margin:.4rem 0;text-align:center;}
.trend-label{font-size:.8rem;color:var(--muted);margin-bottom:.2rem;}
.trend-value{font-size:1.6rem;font-weight:700;}
.tv-high{color:#28a745;}.tv-mid{color:#ffc107;}.tv-low{color:#dc3545;}
/* ★ DAY 16 */
.kanban-col{background:#161B22;border:1px solid #30363D;border-radius:12px;padding:1rem;min-height:180px;}
.kanban-header{font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.7rem;padding-bottom:.35rem;border-bottom:2px solid var(--border);}
.kanban-card{background:#0d1117;border:1px solid #30363D;border-radius:8px;padding:.55rem .85rem;margin-bottom:.45rem;font-size:.83rem;}
.kanban-card:hover{border-color:var(--accent);}
.jd-tip{background:rgba(0,123,255,.06);border:1px solid rgba(0,123,255,.2);border-radius:8px;padding:.6rem .9rem;margin:.4rem 0;font-size:.86rem;}
.collab-comment{background:#0d1117;border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:.5rem .9rem;margin-bottom:.5rem;font-size:.85rem;}
.collab-meta{font-size:.75rem;color:var(--muted);margin-bottom:.2rem;}
.admin-stat{background:#161B22;border:1px solid #30363D;border-radius:10px;padding:1rem;text-align:center;}
.admin-stat-val{font-size:2rem;font-weight:700;color:#007BFF;}
.admin-stat-lbl{font-size:.78rem;color:var(--muted);}
.cal-card{background:#161B22;border:1px solid #30363D;border-radius:10px;padding:.8rem 1rem;margin-bottom:.5rem;display:flex;align-items:center;gap:1rem;}
.cal-date{background:var(--accent);color:#fff;border-radius:8px;padding:.4rem .7rem;font-weight:700;font-size:.88rem;white-space:nowrap;}
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
_DEFAULTS = {
    "step": "upload",
    "candidates": [],
    "key_requirements": [],
    "chat_histories": {},
    "rag_retrievers": {},
    "saved_jd": "",
    "saved_files": [],
    "generated_emails": {},
    "shortlist": [],
    "bookmarks": [],
    "scheduled_interviews": {},
    "kanban_stages": {},
    "job_name": "",
    "activity_log": [],
    "authenticated": False,
    "current_user": "",
    "current_user_role": "",
    "current_user_name": "",
    "current_user_permissions": [],
    "session_token": "",
    "jd_analysis": None,
}
if "step" not in st.session_state:
    st.session_state.update(_DEFAULTS)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = type(v)() if isinstance(v, (list, dict)) else v

KANBAN_STAGES = [
    "Applied",
    "Screening",
    "Technical Interview",
    "Final Interview",
    "Offer",
    "Rejected",
]


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


def role_badge_cls(role: str) -> str:
    return {"Admin": "role-admin", "Manager": "role-manager"}.get(role, "")


def kanban_color(stage: str) -> str:
    return {
        "Applied": "#94A3B8",
        "Screening": "#007BFF",
        "Technical Interview": "#ffc107",
        "Final Interview": "#9c27b0",
        "Offer": "#28a745",
        "Rejected": "#dc3545",
    }.get(stage, "#94A3B8")


def save_session_data():
    return json.dumps(
        {
            "timestamp": str(datetime.now()),
            "job_name": st.session_state.get("job_name", ""),
            "recruiter": st.session_state.get("current_user_name", ""),
            "job_description": st.session_state.saved_jd,
            "candidates": st.session_state.candidates,
            "shortlist": st.session_state.shortlist,
            "bookmarks": st.session_state.bookmarks,
            "scheduled_interviews": st.session_state.scheduled_interviews,
        },
        indent=2,
    )


def build_hiring_summary(scores, strong_matches, avg_score) -> str:
    bm = st.session_state.get("bookmarks", [])
    sl = st.session_state.get("shortlist", [])
    sc = st.session_state.get("scheduled_interviews", {})
    h = len([s for s in scores if s >= 80])
    m = len([s for s in scores if 60 <= s < 80])
    l = len([s for s in scores if s < 60])
    lines = [
        "=" * 52,
        "        HireIQ — Hiring Session Summary",
        "=" * 52,
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Role       : {st.session_state.get('job_name','N/A')}",
        f"Recruiter  : {st.session_state.get('current_user_name','N/A')}",
        "",
        "── Candidate Pool ──────────────────────────",
        f"  Total Reviewed  : {len(scores)}",
        f"  Average Score   : {avg_score} / 100",
        f"  Strong Matches  : {strong_matches}",
        "",
        "── Score Distribution ──────────────────────",
        f"  High  (≥ 80)  : {h}",
        f"  Mid   (60-79) : {m}",
        f"  Low   (< 60)  : {l}",
        "",
        "── Shortlisted ──────────────────────────────",
    ]
    lines += [f"  ⭐  {n}" for n in sl] if sl else ["  (none)"]
    lines += ["", "── Bookmarked ───────────────────────────────"]
    lines += [f"  🔖  {n}" for n in bm] if bm else ["  (none)"]
    lines += ["", "── Scheduled Interviews ─────────────────────"]
    lines += (
        [f"  📅  {c}  →  {i['date']} at {i['time']}" for c, i in sc.items()]
        if sc
        else ["  (none)"]
    )
    lines += [
        "",
        "=" * 52,
        "   Powered by HireIQ · AI-Powered Hiring Intelligence",
        "=" * 52,
    ]
    return "\n".join(lines)


def generate_pdf_report(text: str) -> str:
    path = "/tmp/hireiq_report.pdf"
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    doc.build([Paragraph(text.replace("\n", "<br/>"), styles["BodyText"])])
    return path


def build_recommendation_engine(candidates):
    valid = [c for c in candidates if "Error:" not in c["name"]]
    top3 = sorted(valid, key=lambda x: x["overall_score"], reverse=True)[:3]
    return [
        {
            "rank": i + 1,
            "name": c["name"],
            "score": clamp(c["overall_score"]),
            "confidence": max(60, min(98, clamp(c["overall_score"]) - i * 3)),
            "summary": c["summary"],
        }
        for i, c in enumerate(top3)
    ]


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

        # init kanban
        for c in st.session_state.candidates:
            if (
                "Error:" not in c["name"]
                and c["name"] not in st.session_state.kanban_stages
            ):
                st.session_state.kanban_stages[c["name"]] = "Applied"

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
# SIDEBAR
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
    role = st.session_state.current_user_role
    rb = role_badge_cls(role)
    st.sidebar.markdown(f"👤 **{st.session_state.current_user_name}**")
    st.sidebar.markdown(
        f"🔑 Role: <span class='role-badge {rb}'>{role}</span>", unsafe_allow_html=True
    )
    perm_icons = {
        "upload": "📤",
        "results": "📊",
        "analytics": "📈",
        "admin": "🔐",
        "jd_optimizer": "✍️",
        "collab": "💬",
    }
    perms_html = " ".join(
        f"<span style='font-size:.75rem'>{perm_icons.get(p,p)}</span>"
        for p in st.session_state.get("current_user_permissions", [])
    )
    st.sidebar.markdown(f"**Access:** {perms_html}", unsafe_allow_html=True)
    if st.session_state.get("session_token"):
        st.sidebar.markdown(
            f"🔒 Token: <span class='token-badge'>{st.session_state.session_token}</span>",
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")

    st.sidebar.markdown("## 🔔 Notifications")
    notifs = [
        (
            f"{len(st.session_state.candidates)} candidate(s) in current session"
            if st.session_state.candidates
            else "No candidates loaded yet"
        ),
        (
            f"{len(st.session_state.shortlist)} candidate(s) shortlisted"
            if st.session_state.shortlist
            else "Shortlist is empty"
        ),
        (
            f"🔖 {len(st.session_state.bookmarks)} bookmarked"
            if st.session_state.bookmarks
            else "No bookmarks yet"
        ),
        (
            f"📅 {len(st.session_state.scheduled_interviews)} interview(s) scheduled"
            if st.session_state.scheduled_interviews
            else "No interviews scheduled"
        ),
        "🤖 AI caching active — responses instant after first load",
    ]
    if st.session_state.candidates:
        ts = clamp(st.session_state.candidates[0].get("overall_score", 0))
        if ts >= 90:
            notifs.insert(0, f"🔥 Top candidate scored {ts}/100!")
        elif ts >= 75:
            notifs.insert(0, f"✅ Strong top candidate — {ts}/100")
    for n in notifs:
        st.sidebar.markdown(
            f"<div class='notif-item'>{n}</div>", unsafe_allow_html=True
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Quick Stats**")
    ha, ht, hr, he = get_historical_stats()
    st.sidebar.metric("Total Evaluated", ht)
    st.sidebar.metric("Emails Sent", he)

    if st.session_state.bookmarks:
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🔖 Bookmarked Candidates")
        for bm in st.session_state.bookmarks:
            st.sidebar.markdown(
                f"<span class='bookmark-pill'>🔖 {bm}</span>", unsafe_allow_html=True
            )

    if st.session_state.candidates:
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ⚡ My Productivity")
        tr = len(st.session_state.candidates)
        sc2 = len(st.session_state.shortlist)
        st.sidebar.metric("Profiles Reviewed", tr)
        st.sidebar.metric("Candidates Shortlisted", sc2)
        if tr:
            st.sidebar.metric("Shortlisting Efficiency", f"{round((sc2/tr)*100,1)}%")


# ════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    st.markdown(
        """<div class="hiq-header" style="margin-top:3rem">
      <h1>HireIQ</h1><p>AI-Powered Hiring Intelligence Platform</p></div>""",
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
                        "current_user_permissions": account["permissions"],
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
    """<div class="hiq-header">
  <h1>HireIQ</h1>
  <p>AI-Powered Hiring Intelligence &nbsp;·&nbsp; Screen Smarter. Hire Faster. Explain Every Decision.</p>
</div>""",
    unsafe_allow_html=True,
)

hdr1, hdr2 = st.columns([8, 1])
with hdr1:
    role = st.session_state.current_user_role
    rb = role_badge_cls(role)
    st.markdown(
        f"⚡ AI caching · 🗄 SQLite · 👤 **{st.session_state.current_user_name}** "
        f"<span class='role-badge {rb}'>{role}</span> &nbsp; "
        f"<span class='token-badge'>🔒 {st.session_state.session_token}</span>",
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

        # ════════════════════════════════════════════════════════════════════
        # ★ DAY 16 — FEATURE 2: AI JD OPTIMIZER (Manager / Admin only)
        # ════════════════════════════════════════════════════════════════════
        if has_perm("jd_optimizer") and st.session_state.saved_jd.strip():
            with st.expander("✍️ AI Job Description Optimizer", expanded=False):
                st.caption("Analyze and improve your JD before screening candidates.")
                if st.button("🔍 Analyze JD Quality", use_container_width=True):
                    with st.spinner("Analyzing job description…"):
                        jd_prompt = f"""You are an expert HR consultant and job description specialist.
Analyze this Job Description and return a JSON object with these exact keys:
{{
  "overall_score": <integer 0-100>,
  "clarity_score": <integer 0-100>,
  "inclusivity_score": <integer 0-100>,
  "specificity_score": <integer 0-100>,
  "issues": [<list of 3-5 specific problems found>],
  "improvements": [<list of 3-5 actionable improvements>],
  "rewritten_summary": "<an improved 3-4 sentence version of the JD opening>",
  "missing_sections": [<list of important missing sections, e.g. salary range, benefits>]
}}
Job Description:
{st.session_state.saved_jd[:2000]}
Return ONLY the JSON, no explanation."""
                        try:
                            resp = st.session_state.llm.invoke(jd_prompt)
                            raw = (
                                resp.content.strip()
                                .replace("```json", "")
                                .replace("```", "")
                            )
                            st.session_state.jd_analysis = json.loads(raw)
                            log_activity("JD quality analysis run")
                        except Exception as e:
                            st.error(f"JD analysis failed: {e}")

                if st.session_state.get("jd_analysis"):
                    jd = st.session_state.jd_analysis
                    overall = jd.get("overall_score", 0)
                    color = (
                        "#28a745"
                        if overall >= 70
                        else ("#ffc107" if overall >= 50 else "#dc3545")
                    )
                    st.markdown(
                        f"#### JD Quality Score: <span style='color:{color};font-size:1.4rem;font-weight:700'>{overall}/100</span>",
                        unsafe_allow_html=True,
                    )
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Clarity", f"{jd.get('clarity_score',0)}/100")
                    sc2.metric("Inclusivity", f"{jd.get('inclusivity_score',0)}/100")
                    sc3.metric("Specificity", f"{jd.get('specificity_score',0)}/100")
                    if jd.get("issues"):
                        st.markdown("**⚠️ Issues Found:**")
                        for issue in jd["issues"]:
                            st.markdown(
                                f"<div class='jd-tip'>🔸 {issue}</div>",
                                unsafe_allow_html=True,
                            )
                    if jd.get("improvements"):
                        st.markdown("**💡 Suggested Improvements:**")
                        for imp in jd["improvements"]:
                            st.markdown(
                                f"<div class='jd-tip'>✅ {imp}</div>",
                                unsafe_allow_html=True,
                            )
                    if jd.get("missing_sections"):
                        st.markdown(
                            "**📋 Missing Sections:** "
                            + ", ".join(f"`{s}`" for s in jd["missing_sections"])
                        )
                    if jd.get("rewritten_summary"):
                        st.markdown("**✍️ AI-Improved Opening:**")
                        st.info(jd["rewritten_summary"])
        elif not has_perm("jd_optimizer"):
            st.caption("🔒 JD Optimizer — Manager / Admin only")

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
        rl = f" — **{st.session_state.job_name}**" if st.session_state.job_name else ""
        st.success(
            f"✅ Analysis complete{rl} — **{len(st.session_state.candidates)}** candidate(s) ranked."
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

    scores = [
        c["overall_score"]
        for c in st.session_state.candidates
        if "Error:" not in c["name"]
    ]

    if scores:
        avg_score = round(sum(scores) / len(scores), 1)
        strong_matches = len([s for s in scores if s >= 75])
        hire_ready = len([s for s in scores if s >= 80])
        high_scores = hire_ready
        mid_scores = len([s for s in scores if 60 <= s < 80])
        low_scores = len([s for s in scores if s < 60])
        strong_ratio = round((high_scores / len(scores)) * 100, 1) if scores else 0

        # Analytics — gated by permission
        if has_perm("analytics"):
            st.markdown("## 📊 Executive Hiring Insights")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Candidates", len(scores))
            m2.metric("Average Score", avg_score)
            m3.metric("Strong Matches", strong_matches)
            m4.metric("Hire-Ready", hire_ready)

            st.markdown("## 📈 Hiring Trend Insights")
            ht1, ht2, ht3, ht4 = st.columns(4)
            tv = (
                "tv-high"
                if strong_ratio >= 60
                else ("tv-mid" if strong_ratio >= 30 else "tv-low")
            )
            with ht1:
                st.markdown(
                    f"<div class='trend-card'><div class='trend-label'>High Match (≥ 80)</div><div class='trend-value tv-high'>{high_scores}</div></div>",
                    unsafe_allow_html=True,
                )
            with ht2:
                st.markdown(
                    f"<div class='trend-card'><div class='trend-label'>Mid Match (60–79)</div><div class='trend-value tv-mid'>{mid_scores}</div></div>",
                    unsafe_allow_html=True,
                )
            with ht3:
                st.markdown(
                    f"<div class='trend-card'><div class='trend-label'>Low Match (&lt; 60)</div><div class='trend-value tv-low'>{low_scores}</div></div>",
                    unsafe_allow_html=True,
                )
            with ht4:
                st.markdown(
                    f"<div class='trend-card'><div class='trend-label'>High-Quality Ratio</div><div class='trend-value {tv}'>{strong_ratio}%</div></div>",
                    unsafe_allow_html=True,
                )

            if strong_ratio < 30:
                st.warning(
                    "⚠️ Low high-quality candidate density. Consider expanding sourcing channels."
                )
            elif strong_ratio < 60:
                st.info(
                    "ℹ️ Moderate talent pipeline. A few strong candidates are present."
                )
            else:
                st.success(
                    "✅ Healthy talent pipeline detected. Strong candidate pool available."
                )

            st.markdown("### 📈 Score Distribution")
            st.bar_chart(
                {
                    c["name"]: c["overall_score"]
                    for c in st.session_state.candidates
                    if "Error:" not in c["name"]
                }
            )

            st.markdown("## 🧠 AI Hiring Recommendation Engine")
            st.caption(
                "Generate a strategic hiring recommendation based on all candidate data in this session."
            )
            if st.button(
                "🧠 Generate Hiring Recommendations", use_container_width=True
            ):
                cand_block = "\n".join(
                    f"- {c['name']}: Score {clamp(c['overall_score'])}/100 — {c['summary'][:120]}"
                    for c in st.session_state.candidates
                    if "Error:" not in c["name"]
                )
                rec_prompt = f"""You are a senior hiring strategist with 20+ years of experience.
Job Role: {st.session_state.get('job_name','Not specified')}
Job Description: {st.session_state.saved_jd[:600]}
Candidate Pool ({len(scores)} candidates, avg: {avg_score}/100):
{cand_block}
Distribution — High(≥80):{high_scores} Mid(60-79):{mid_scores} Low(<60):{low_scores}
Provide concise executive-level strategy:
1. 📊 Pipeline health assessment  2. 🏆 Top 2-3 candidates to prioritise
3. ⚠️ Key bottlenecks / risks   4. 🎯 Recommended interview strategy
5. 🔄 Sourcing recommendations   6. 📅 Suggested hiring timeline
Keep it structured, professional and actionable."""
                with st.spinner("🧠 Generating strategic recommendations…"):
                    rec = st.session_state.llm.invoke(rec_prompt)
                    st.session_state["ai_hiring_recommendations"] = rec.content
                    log_activity("AI hiring recommendations generated")
            if st.session_state.get("ai_hiring_recommendations"):
                st.markdown("---")
                st.markdown("### 📋 Strategic Recommendations")
                st.write(st.session_state["ai_hiring_recommendations"])
                st.download_button(
                    "📥 Download Recommendations",
                    st.session_state["ai_hiring_recommendations"],
                    file_name="hireiq_recommendations.txt",
                    use_container_width=True,
                )

            st.markdown("## 🔮 Hiring Forecast Intelligence")
            pred = max(1, int(len(scores) * 0.25))
            tth = max(7, 30 - int(strong_ratio / 5))
            ph = (
                "🟢 Strong"
                if strong_ratio >= 60
                else ("🟡 Moderate" if strong_ratio >= 30 else "🔴 Weak")
            )
            fp1, fp2, fp3 = st.columns(3)
            fp1.metric("Predicted Successful Hires", pred)
            fp2.metric("Est. Time to Hire (days)", tth)
            fp3.metric("Pipeline Health", ph)
            if pred < 2:
                st.warning(
                    "⚠️ Pipeline may require broader sourcing or relaxed criteria."
                )
            else:
                st.success(f"✅ Healthy projected pipeline — {pred} hire(s) predicted.")

            st.markdown("## ⚡ Recruiter Productivity")
            rp1, rp2, rp3, rp4 = st.columns(4)
            rp1.metric("Profiles Reviewed", len(st.session_state.candidates))
            rp2.metric("Shortlisted", len(st.session_state.shortlist))
            rp3.metric("Bookmarked", len(st.session_state.bookmarks))
            rp4.metric(
                "Interviews Scheduled", len(st.session_state.scheduled_interviews)
            )
            if st.session_state.candidates:
                eff = round(
                    (len(st.session_state.shortlist) / len(st.session_state.candidates))
                    * 100,
                    1,
                )
                st.metric("Shortlisting Efficiency", f"{eff}%")

            st.markdown("## 📁 Export Hiring Session Summary")
            ex1, ex2 = st.columns(2)
            with ex1:
                st.download_button(
                    "📁 Export Hiring Summary (.txt)",
                    build_hiring_summary(scores, strong_matches, avg_score),
                    file_name=f"hireiq_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    use_container_width=True,
                )
            with ex2:
                st.download_button(
                    "💾 Export Full Session (.json)",
                    save_session_data(),
                    file_name=f"hireiq_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    use_container_width=True,
                )
        else:
            st.markdown("## 📊 Session Overview")
            m1, m2 = st.columns(2)
            m1.metric("Candidates Reviewed", len(scores))
            m2.metric("Your Shortlist", len(st.session_state.shortlist))
            st.caption("🔒 Full analytics dashboard — Manager / Admin only.")

    # ════════════════════════════════════════════════════════════════════════
    # ★ DAY 16 — FEATURE 3: INTERVIEW CALENDAR VIEW
    # ════════════════════════════════════════════════════════════════════════
    if st.session_state.scheduled_interviews:
        st.markdown("## 📅 Interview Calendar")
        sorted_iv = sorted(
            st.session_state.scheduled_interviews.items(), key=lambda x: x[1]["date"]
        )
        for cname, info in sorted_iv:
            cand = next(
                (c for c in st.session_state.candidates if c["name"] == cname), {}
            )
            s = clamp(cand.get("overall_score", 0))
            st.markdown(
                f"<div class='cal-card'><div class='cal-date'>📅 {info['date']}</div>"
                f"<div style='flex:1'><b>{cname}</b> &nbsp; {badge(s)}"
                f"<div style='color:var(--muted);font-size:.8rem'>⏰ {info['time']} &nbsp;·&nbsp; "
                f"👤 {info.get('recruiter','N/A')}</div></div></div>",
                unsafe_allow_html=True,
            )

    # Historical analytics
    ha, ht, hr2, he = get_historical_stats()
    if ht > 0 and has_perm("analytics"):
        st.markdown("## 📚 Historical Hiring Analytics")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("All-Time Avg Score", ha)
        h2.metric("Total Candidates Evaluated", ht)
        h3.metric("Roles Processed", hr2)
        h4.metric("Emails Sent (DB)", he)

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
                    f"<div class='mem-card'><b>👤 {row[0]}</b>"
                    f"<span style='color:var(--muted);font-size:.8rem;margin-left:8px'>— reviewed by {row[2] or 'unknown'}</span><br>"
                    f"<small style='color:var(--muted)'>{row[1][:250]}…</small></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning(f"No historical candidates found matching **{memory_query}**.")

    if st.session_state.candidates:
        recs = build_recommendation_engine(st.session_state.candidates)
        if recs:
            st.markdown("## 🧠 AI Candidate Recommendations")
            rc = st.columns(len(recs))
            for col, rec in zip(rc, recs):
                bc = "bh" if rec["score"] >= 75 else "bm"
                with col:
                    st.markdown(
                        f"<div class='rec-card'><div class='rec-rank'>#{rec['rank']}</div>"
                        f"<b>{rec['name']}</b><br><span class='badge {bc}'>{rec['score']} / 100</span><br>"
                        f"<small style='color:var(--muted)'>Confidence: {rec['confidence']}%</small><br>"
                        f"<small style='color:var(--muted)'>{rec['summary'][:100]}…</small></div>",
                        unsafe_allow_html=True,
                    )

    if st.session_state.candidates:
        top = st.session_state.candidates[0]
        ts = clamp(top["overall_score"])
        st.info(f"🏆 **Top Candidate:** {top['name']} — {ts} / 100  ·  {decision(ts)}")
        with st.expander("🏆 Why was this candidate ranked #1?"):
            with st.spinner("Analysing…"):
                result = llm_cached(
                    "top_candidate_reason",
                    f"""You are a senior hiring director.
Job Description: {st.session_state.saved_jd}
Top Candidate: {top['name']} | Summary: {top['summary']} | Score: {ts}
Explain: Why ranked highest, biggest strengths, hiring advantages, potential risks, final recommendation.
Keep it concise and executive-level.""",
                )
            st.write(result)

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

    # ════════════════════════════════════════════════════════════════════════
    # TABS — Leaderboard | Kanban | Compare | Emails | Admin
    # ════════════════════════════════════════════════════════════════════════
    tab_list = ["🏆 Leaderboard", "🗂 Kanban Board", "🤝 Compare", "✉️ Emails & Report"]
    if has_perm("admin"):
        tab_list.append("🔐 Admin Panel")
    tabs = st.tabs(tab_list)
    tab1, tab_kanban, tab2, tab3 = tabs[0], tabs[1], tabs[2], tabs[3]
    tab_admin = tabs[4] if has_perm("admin") else None

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
                    ml.markdown(f"**Match:** {match_label(score)}")
                    dl.markdown(f"**Decision:** {decision(score)}")
                    al.markdown(
                        f"**Action:** <span class='action-tag'>{next_action(score)}</span>",
                        unsafe_allow_html=True,
                    )
                    cl.markdown(f"**Rating:** {cached_label(score)}")
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

                        pip_col, tag_col, sl_col, bm_col = st.columns([2, 2, 1, 1])
                        with pip_col:
                            pipeline_stage = st.selectbox(
                                "📋 Hiring Stage",
                                KANBAN_STAGES,
                                key=f"stage_{name}",
                                label_visibility="collapsed",
                            )
                            st.session_state.kanban_stages[name] = pipeline_stage
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
                        with bm_col:
                            if name not in st.session_state.bookmarks:
                                if st.button("🔖 Bookmark", key=f"bm_{rank}"):
                                    st.session_state.bookmarks.append(name)
                                    save_bookmark_to_db(
                                        name, st.session_state.current_user
                                    )
                                    log_activity(f"Bookmarked: {name}")
                                    st.toast(f"🔖 {name} bookmarked")
                                    st.rerun()
                            else:
                                st.markdown("🔖 **Bookmarked**")
                                if st.button("✖ Remove", key=f"bm_rm_{rank}"):
                                    st.session_state.bookmarks.remove(name)
                                    log_activity(f"Bookmark removed: {name}")
                                    st.rerun()

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

                        st.markdown("### 📝 Recruiter Notes")
                        note_col2, rev_col = st.columns([3, 2])
                        with note_col2:
                            note_val = st.text_area(
                                "Recruiter Notes",
                                key=f"notes_{name}",
                                placeholder="Add interview observations, feedback, red flags or highlights…",
                                height=80,
                                label_visibility="collapsed",
                            )
                            if st.button("💾 Save Notes", key=f"save_note_{rank}"):
                                if note_val and note_val.strip():
                                    save_note_to_db(
                                        name, note_val, st.session_state.current_user
                                    )
                                    st.success(
                                        f"✅ Recruiter note saved for **{name}**"
                                    )
                                    log_activity(f"Notes saved for: {name}")
                                else:
                                    st.warning("Write a note before saving.")
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

                        st.markdown("### 📅 Interview Scheduler")
                        st.markdown(
                            "<div class='scheduler-card'>", unsafe_allow_html=True
                        )
                        sch1, sch2, sch3 = st.columns([2, 2, 1])
                        with sch1:
                            interview_date = st.date_input(
                                "Interview Date",
                                key=f"date_{name}",
                                label_visibility="collapsed",
                            )
                        with sch2:
                            interview_time = st.time_input(
                                "Interview Time",
                                key=f"time_{name}",
                                label_visibility="collapsed",
                            )
                        with sch3:
                            if st.button("📅 Schedule", key=f"schedule_{name}"):
                                st.session_state.scheduled_interviews[name] = {
                                    "date": str(interview_date),
                                    "time": str(interview_time),
                                    "recruiter": st.session_state.current_user_name,
                                }
                                save_scheduled_interview_to_db(
                                    name,
                                    interview_date,
                                    interview_time,
                                    st.session_state.current_user,
                                )
                                log_activity(
                                    f"Interview scheduled for {name} on {interview_date} at {interview_time}"
                                )
                                st.success(
                                    f"✅ Interview scheduled for **{name}**  \n📅 {interview_date.strftime('%A, %B %d, %Y')} at {interview_time.strftime('%I:%M %p')}"
                                )
                        if name in st.session_state.scheduled_interviews:
                            info = st.session_state.scheduled_interviews[name]
                            st.info(
                                f"🗓 Scheduled: **{info['date']}** at **{info['time']}** by {info.get('recruiter','N/A')}"
                            )
                        st.markdown("</div>", unsafe_allow_html=True)

                        # ════════════════════════════════════════════════════
                        # ★ DAY 16 — FEATURE 4: MULTI-RECRUITER COLLABORATION
                        # ════════════════════════════════════════════════════
                        if has_perm("collab"):
                            with st.expander("💬 Team Collaboration Comments"):
                                collab_input = st.text_area(
                                    "Your comment",
                                    key=f"collab_input_{rank}",
                                    placeholder="e.g. Great culture fit. Strong system design. Recommend fast-tracking.",
                                    height=68,
                                    label_visibility="collapsed",
                                )
                                if st.button(
                                    "💬 Post Comment", key=f"collab_post_{rank}"
                                ):
                                    if collab_input.strip():
                                        save_collab_comment(
                                            name,
                                            collab_input.strip(),
                                            st.session_state.current_user_name,
                                            st.session_state.current_user_role,
                                        )
                                        log_activity(f"Team comment posted for: {name}")
                                        st.toast("✅ Comment posted to team")
                                        st.rerun()
                                    else:
                                        st.warning("Write a comment first.")
                                comments = get_collab_comments(name)
                                if comments:
                                    st.markdown("**📋 Team Comments:**")
                                    for comment, rec_name, rec_role, ts_str in comments:
                                        rb2 = role_badge_cls(rec_role)
                                        st.markdown(
                                            f"<div class='collab-comment'><div class='collab-meta'>"
                                            f"<span class='role-badge {rb2}'>{rec_role}</span> &nbsp; "
                                            f"<b>{rec_name}</b> &nbsp;·&nbsp; {ts_str[:16]}</div>{comment}</div>",
                                            unsafe_allow_html=True,
                                        )
                                else:
                                    st.caption("No team comments yet. Be the first.")

                        with st.expander("🧠 Why this score?"):
                            with st.spinner("Analysing…"):
                                result = llm_cached(
                                    f"explain_{name}",
                                    f"""You are a senior hiring manager.
Job Description: {st.session_state.saved_jd}
Candidate Summary: {cand.get('summary','')}
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

                        with st.expander("⚠️ AI Risk Analysis"):
                            if st.button("🔍 Analyze Hiring Risk", key=f"risk_{rank}"):
                                with st.spinner("Analysing candidate risk profile…"):
                                    rr = st.session_state.llm.invoke(
                                        f"""You are an AI hiring risk analyst.
Candidate Summary: {cand.get('summary','')} | Score: {score}/100
Identify: Hiring risks, Skill gaps, Retention concerns, Communication concerns, Culture concerns.
Job Description context: {st.session_state.saved_jd[:500]} Keep concise and professional."""
                                    )
                                    st.write(rr.content)
                                    log_activity(f"Risk analysis run for: {name}")

                        with st.expander("🎤 AI Interview Questions"):
                            with st.spinner("Generating questions…"):
                                result = llm_cached(
                                    f"questions_{name}",
                                    f"""You are a senior technical recruiter.
Job Description: {st.session_state.saved_jd}
Candidate Summary: {cand.get('summary','')}
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
Candidate Summary: {cand.get('summary','')}
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
                                        er = st.session_state.llm.invoke(
                                            f"""You are a hiring committee evaluator.
Evaluate this candidate interview response: {candidate_response}
Return: Communication Score (1-10), Technical Depth (1-10), Confidence Level, Red Flags, Hiring Recommendation."""
                                        )
                                        st.write(er.content)
                                        save_interview_eval_to_db(
                                            name,
                                            er.content,
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
                                cls = "xai-y" if r["match_status"] else "xai-n"
                                icon = "✅" if r["match_status"] else "❌"
                                label = "Evidence" if r["match_status"] else "Reason"
                                st.markdown(
                                    f"<div class='xai {cls}'>{icon} <b>{r['requirement']}</b><br><small><i>{label}: \"{r['evidence']}\"</i></small></div>",
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
                                st.markdown(
                                    f"<div class='bubble {msg['role']}'>{msg['content']}</div>",
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
    # ★ DAY 16 — FEATURE 5: KANBAN PIPELINE BOARD
    # ════════════════════════════════════════════════════════════════════════
    with tab_kanban:
        st.markdown("### 🗂 Candidate Pipeline Board")
        st.caption(
            "Visual hiring pipeline — stages sync with the Leaderboard stage selector."
        )
        if not st.session_state.candidates:
            st.info("Run an analysis first to see candidates on the board.")
        else:
            valid_c = [
                c for c in st.session_state.candidates if "Error:" not in c["name"]
            ]
            stage_groups = {s: [] for s in KANBAN_STAGES}
            for c in valid_c:
                sg = st.session_state.kanban_stages.get(c["name"], "Applied")
                stage_groups[sg].append(c)

            kcols = st.columns(len(KANBAN_STAGES))
            for col, stage in zip(kcols, KANBAN_STAGES):
                color = kanban_color(stage)
                members = stage_groups[stage]
                with col:
                    st.markdown(
                        f"<div class='kanban-col'>"
                        f"<div class='kanban-header' style='color:{color};border-color:{color}'>"
                        f"{stage} <span style='font-weight:400;color:var(--muted)'>({len(members)})</span></div>",
                        unsafe_allow_html=True,
                    )
                    for c in members:
                        s = clamp(c["overall_score"])
                        bc_cls = "bh" if s >= 75 else ("bm" if s >= 50 else "bl")
                        st.markdown(
                            f"<div class='kanban-card'><b>{c['name']}</b><br>"
                            f"<span class='badge {bc_cls}' style='font-size:.72rem;padding:2px 8px'>{s}/100</span></div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**⚡ Quick Stage Mover**")
            mv1, mv2, mv3 = st.columns([3, 3, 1])
            with mv1:
                mc = st.selectbox(
                    "Candidate",
                    [c["name"] for c in valid_c],
                    key="kanban_move_cand",
                    label_visibility="collapsed",
                )
            with mv2:
                ms = st.selectbox(
                    "New Stage",
                    KANBAN_STAGES,
                    key="kanban_move_stage",
                    label_visibility="collapsed",
                )
            with mv3:
                if st.button("Move →", use_container_width=True):
                    st.session_state.kanban_stages[mc] = ms
                    log_activity(f"Kanban: {mc} → {ms}")
                    st.toast(f"✅ {mc} moved to {ms}")
                    st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # TAB — COMPARE
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
    # TAB — EMAILS & REPORT
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
                ic, rc2 = st.columns(2)
                with ic:
                    st.markdown("#### ✅ Invitations")
                    for em in st.session_state.generated_emails.get("invitations", []):
                        with st.expander(f"To: {em['name']}", expanded=True):
                            st.code(em["email_body"], language=None)
                    if not st.session_state.generated_emails.get("invitations"):
                        st.info("No candidates met the score threshold.")
                with rc2:
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
    # ★ DAY 16 — FEATURE 6: ADMIN CONTROL PANEL
    # ════════════════════════════════════════════════════════════════════════
    if has_perm("admin") and tab_admin is not None:
        with tab_admin:
            st.markdown("### 🔐 Admin Control Panel")
            st.caption("Full platform oversight — only visible to Admin role.")

            st.markdown("#### 📊 Platform Database Stats")
            db_stats = get_admin_db_stats()
            stat_items = [
                ("candidates", "👥 Total Candidates"),
                ("notes", "📝 Recruiter Notes"),
                ("email_log", "✉️ Emails Logged"),
                ("scheduled_interviews", "📅 Scheduled Interviews"),
                ("bookmarks", "🔖 Bookmarks"),
                ("collab_comments", "💬 Team Comments"),
                ("interview_evaluations", "🎤 Interview Evals"),
            ]
            cols7 = st.columns(4)
            for idx, (table, label) in enumerate(stat_items):
                with cols7[idx % 4]:
                    st.markdown(
                        f"<div class='admin-stat'><div class='admin-stat-val'>{db_stats.get(table,0)}</div>"
                        f"<div class='admin-stat-lbl'>{label}</div></div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            st.markdown("#### 👥 Registered Recruiter Accounts")
            for uname, udata in RECRUITER_ACCOUNTS.items():
                rb2 = role_badge_cls(udata["role"])
                st.markdown(
                    f"<div class='mem-card'><b>{udata['name']}</b> "
                    f"<span class='role-badge {rb2}'>{udata['role']}</span><br>"
                    f"<small style='color:var(--muted)'>Login: <code>{uname}</code> &nbsp;·&nbsp; "
                    f"Permissions: {' · '.join(udata['permissions'])}</small></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("#### ⚙️ Session Management")
            adm1, adm2 = st.columns(2)
            with adm1:
                if st.button("🗑 Clear Activity Log", use_container_width=True):
                    st.session_state.activity_log = []
                    st.success("✅ Activity log cleared.")
            with adm2:
                if st.button("♻️ Reset Kanban Board", use_container_width=True):
                    for k in st.session_state.kanban_stages:
                        st.session_state.kanban_stages[k] = "Applied"
                    st.success("✅ Kanban board reset to Applied.")

            st.markdown("---")
            st.markdown("#### 🔒 Role Permission Matrix")
            st.table(
                {
                    "Feature": [
                        "Upload & Screen",
                        "Results & Leaderboard",
                        "Analytics Dashboard",
                        "JD Optimizer",
                        "Team Collaboration",
                        "Admin Panel",
                    ],
                    "Recruiter": ["✅", "✅", "❌", "❌", "✅", "❌"],
                    "Manager": ["✅", "✅", "✅", "✅", "✅", "❌"],
                    "Admin": ["✅", "✅", "✅", "✅", "✅", "✅"],
                }
            )

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
