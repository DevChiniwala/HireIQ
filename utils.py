"""
utils.py — HireIQ core AI logic

Bug fixes applied vs original:
 - repair_and_parse_json: f-string prompt was passed directly to ChatPromptTemplate;
   any { } in the broken JSON caused a KeyError. Fixed by escaping braces before
   template rendering and using a simple HumanMessage call instead.
 - score_candidate_explainable: LLM sometimes returns overall_score as a string
   ("85") instead of int (85). Added explicit int() coercion + retry fallback.
 - FastEmbedEmbeddings: updated import path to avoid deprecation warning.
 - All LLM calls that take plain-text prompts with no template variables now go
   through a safe wrapper that escapes { and } so LangChain does not try to
   interpolate them as template slots.
"""

import json
import re
from typing import List, Dict, Any, Optional

import PyPDF2
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# FIX: updated import path — old path is deprecated in newer langchain-community
try:
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
except ImportError:
    from langchain_community.embeddings import FastEmbedEmbeddings


# ─── Utilities ────────────────────────────────────────────────────────────────

def clean_llm_output(text: str) -> str:
    """Strip markdown code fences from raw LLM text."""
    text = text.strip()
    # Remove ```json or ``` wrappers
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_call_llm_plain(llm: BaseChatModel, prompt: str) -> str:
    """
    Call the LLM with a plain string prompt (no template variables).
    Uses HumanMessage directly to bypass ChatPromptTemplate entirely,
    which avoids the { } interpolation crash when prompt contains JSON.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content if hasattr(response, "content") else str(response)


def call_llm(
    llm: BaseChatModel,
    prompt_template: str,
    input_data: Dict[str, Any],
    response_model=None,
) -> Any:
    """
    Invoke the LLM via a LangChain prompt template.
    Use only when input_data keys match the {placeholders} in prompt_template.
    For plain prompts with no variables, use safe_call_llm_plain() instead.
    """
    chain = ChatPromptTemplate.from_template(prompt_template)
    if response_model:
        structured_llm = llm.with_structured_output(response_model)
        chain = chain | structured_llm
    else:
        chain = chain | llm
    response = chain.invoke(input_data)
    return response


def repair_and_parse_json(llm: BaseChatModel, broken_json_string: str) -> Optional[Dict]:
    """
    Ask the LLM to fix a broken JSON string.
    FIX: uses safe_call_llm_plain() so that { } inside broken_json_string
         do NOT get misinterpreted as LangChain template variables.
    """
    prompt = (
        "The following string is broken JSON. Fix ALL syntax errors "
        "(unescaped newlines, trailing commas, wrong types, etc.) "
        "and return ONLY the valid JSON object — no markdown, no explanation.\n\n"
        f"Broken JSON:\n{broken_json_string}"
    )
    try:
        raw = safe_call_llm_plain(llm, prompt)
        return json.loads(clean_llm_output(raw))
    except Exception as e:
        print(f"JSON repair failed: {e}")
        return None


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class RequirementMatch(BaseModel):
    requirement: str
    match_status: bool
    evidence: str


class ExplainableCandidateScore(BaseModel):
    name: str
    overall_score: int
    summary: str
    requirement_analysis: List[RequirementMatch]


class KeyRequirements(BaseModel):
    key_requirements: List[str]


class InterviewQuestions(BaseModel):
    behavioral: List[str]
    technical: List[str]


# ─── Core Functions ───────────────────────────────────────────────────────────

def extract_key_requirements(job_description: str, llm: BaseChatModel) -> List[str]:
    """Extract 5–7 critical, specific requirements from a job description."""
    prompt = """You are an expert HR analyst. Read the job description below and extract
the 5 to 7 most critical, specific, and measurable requirements.
Focus on: years of experience with named technologies, mandatory certifications,
and quantifiable skills. Avoid vague or generic requirements.

Job Description:
{jd}"""
    response = call_llm(llm, prompt, {"jd": job_description}, response_model=KeyRequirements)
    return response.key_requirements


def score_candidate_explainable(
    job_description: str,
    resume_text: str,
    weighted_requirements: Dict,
    llm: BaseChatModel,
) -> ExplainableCandidateScore:
    """
    Score a candidate with full XAI justification.

    FIX 1: overall_score coercion — LLM sometimes returns "85" (str) instead of 85 (int).
            We forcibly cast to int before Pydantic validation.
    FIX 2: retry with repair_and_parse_json on first JSONDecodeError before giving up.
    """
    prompt = """TASK: Evaluate the candidate resume against the job description and weighted requirements.
Return ONLY a single valid JSON object — no markdown fences, no preamble, no explanation.

CANDIDATE NAME: Extract it from the resume text and put it in the "name" field.

SCORING RULES:
- Start at 100 points.
- Deduct for each requirement NOT directly evidenced in the resume:
    Normal   → -5 points
    Important → -15 points
    Critical  → -25 points
- overall_score MUST be an integer (e.g. 82, not "82").
- If any requirement marked knockout=true is missing, set overall_score to 0.

EXACT JSON SCHEMA (fill every field):
{{
  "name": "<string extracted from resume>",
  "overall_score": <integer 0-100>,
  "summary": "<2-3 sentence critical analysis of fit, naming specific gaps>",
  "requirement_analysis": [
    {{
      "requirement": "<requirement text>",
      "match_status": <true|false>,
      "evidence": "<direct quote from resume OR 'No direct evidence found in the resume.'>"
    }}
  ]
}}

DATA:
WEIGHTED REQUIREMENTS: {weights}
JOB DESCRIPTION: {jd}
RESUME TEXT: {resume}"""

    input_data = {
        "weights": json.dumps(weighted_requirements, indent=2),
        "jd": job_description,
        "resume": resume_text[:6000],
    }

    raw_response = call_llm(llm, prompt, input_data, response_model=None)
    raw_str = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    cleaned = clean_llm_output(raw_str)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        print("Initial JSON parse failed — attempting LLM repair…")
        parsed = repair_and_parse_json(llm, cleaned)
        if parsed is None:
            raise ValueError("Could not parse or repair the AI scoring response.")

    # FIX: coerce overall_score to int regardless of what the LLM returned
    if "overall_score" in parsed:
        try:
            parsed["overall_score"] = int(parsed["overall_score"])
        except (TypeError, ValueError):
            parsed["overall_score"] = 0

    # Clamp to valid range
    parsed["overall_score"] = max(0, min(parsed["overall_score"], 100))

    return ExplainableCandidateScore(**parsed)


def generate_interview_questions(
    candidate_name: str,
    candidate_summary: str,
    job_description: str,
    llm: BaseChatModel,
) -> InterviewQuestions:
    """Generate tailored behavioral and technical interview questions."""
    # Build prompt as a plain string (contains no LangChain template variables)
    prompt = (
        f"Generate targeted interview questions for the candidate below.\n\n"
        f"Candidate Name: {candidate_name}\n"
        f"AI Summary: {candidate_summary}\n"
        f"Job Description (first 1500 chars): {job_description[:1500]}\n\n"
        "Return ONLY a valid JSON object with exactly two keys:\n"
        '  "behavioral": [list of 3-4 behavioral questions tailored to their gaps]\n'
        '  "technical":  [list of 2-3 technical questions based on the JD]\n'
        "No markdown, no extra text."
    )
    try:
        # FIX: use safe_call_llm_plain — candidate data may contain { } characters
        raw = safe_call_llm_plain(llm, prompt)
        cleaned = clean_llm_output(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = repair_and_parse_json(llm, cleaned)
            if not parsed:
                raise ValueError("JSON repair failed.")
        return InterviewQuestions(**parsed)
    except Exception as e:
        print(f"Could not generate interview questions for {candidate_name}: {e}")
        return InterviewQuestions(
            behavioral=["Could not generate questions — please try again."],
            technical=["Could not generate questions — please try again."],
        )


def generate_email_templates(
    ranked_candidates: list,
    job_description: dict,
    num_to_invite: int,
    min_score: int,
    interview_datetime: str,
    llm: BaseChatModel,
) -> dict:
    """Generate personalised invitation and rejection emails."""
    invitations, rejections = [], []
    job_title = job_description.get("title", "the position")

    candidates_to_invite = [
        c for c in ranked_candidates if c.get("overall_score", 0) >= min_score
    ][:num_to_invite]
    invited_names = {c["name"] for c in candidates_to_invite}

    for candidate in ranked_candidates:
        candidate_name = candidate.get("name", "Candidate")
        if "Error:" in candidate_name:
            continue

        if candidate_name in invited_names:
            prompt = (
                f"You are a warm, professional HR manager at a fast-growing company.\n"
                f"Write a concise, enthusiastic interview invitation email to {candidate_name} "
                f"for the role of {job_title}.\n"
                f"Invite them for a 1-hour virtual interview on {interview_datetime}.\n"
                f"Ask them to confirm availability by replying to this email.\n"
                f"Keep it under 150 words. Use a professional yet friendly tone."
            )
            email_type = "invitation"
        else:
            prompt = (
                f"You are a respectful HR manager.\n"
                f"Write a brief, empathetic rejection email to {candidate_name} "
                f"for the role of {job_title}.\n"
                f"Thank them sincerely for their time and encourage them to apply again in future.\n"
                f"Keep it under 100 words. Do not mention scores or reasons."
            )
            email_type = "rejection"

        try:
            # FIX: safe_call_llm_plain — names/titles may contain { } characters
            email_body = safe_call_llm_plain(llm, prompt)
            entry = {"name": candidate_name, "email_body": email_body}
            if email_type == "invitation":
                invitations.append(entry)
            else:
                rejections.append(entry)
        except Exception as e:
            print(f"Email generation failed for {candidate_name}: {e}")

    return {"invitations": invitations, "rejections": rejections}


def extract_pdf_text(file_object: Any) -> str:
    """Extract all text from an in-memory PDF file object."""
    try:
        pdf_reader = PyPDF2.PdfReader(file_object)
        return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


def create_candidate_rag_retriever(resume_text: str, filename: str):
    """Build an in-memory FAISS RAG retriever for a single candidate's resume."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.create_documents([resume_text], metadatas=[{"source": filename}])
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def ask_rag_question(retriever, question: str, llm: BaseChatModel) -> str:
    """Answer a question grounded strictly in the retrieved resume context."""
    template = (
        "You are a helpful assistant analysing a candidate's resume.\n"
        "Answer the question based ONLY on the context provided below.\n"
        "If the answer is not in the context, say 'I could not find that information in this resume.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    response = chain.invoke({"input": question})
    return response["answer"]
