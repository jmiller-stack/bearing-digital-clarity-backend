import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "clarity_prototype.db"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4-5"


SECTION_CONFIG = {
    "SOAP": [
        {"key": "subjective", "short": "S", "title": "Subjective", "clipboard": "SUBJECTIVE"},
        {"key": "objective", "short": "O", "title": "Objective", "clipboard": "OBJECTIVE"},
        {"key": "assessment", "short": "A", "title": "Assessment", "clipboard": "ASSESSMENT"},
        {"key": "plan", "short": "P", "title": "Plan", "clipboard": "PLAN"},
    ],
    "DAP": [
        {"key": "data", "short": "D", "title": "Data", "clipboard": "DATA"},
        {"key": "assessment", "short": "A", "title": "Assessment", "clipboard": "ASSESSMENT"},
        {"key": "plan", "short": "P", "title": "Plan", "clipboard": "PLAN"},
    ],
    "BIRP": [
        {"key": "behavior", "short": "B", "title": "Behavior", "clipboard": "BEHAVIOR"},
        {"key": "intervention", "short": "I", "title": "Intervention", "clipboard": "INTERVENTION"},
        {"key": "response", "short": "R", "title": "Response", "clipboard": "RESPONSE"},
        {"key": "plan", "short": "P", "title": "Plan", "clipboard": "PLAN"},
    ],
}


SYSTEM_PROMPTS = {
    "SOAP": """You are a clinical documentation assistant working with a licensed therapist (LCSW-C). Generate a SOAP progress note from the structured session data provided by the therapist.

RULES:
- Write in professional clinical language appropriate for a medical record.
- Use third person: "Client reports...", "Client denies...", "Client was observed to...".
- Include specific details from the therapist's input. Do not generalize or fabricate.
- Include the ICD-10 code for the diagnosis if you can identify it.
- Reference treatment goals if provided.
- Note session continuity by referencing session number when clinically relevant.

SECTION GUIDELINES:
- Subjective: client self-report, symptom changes, homework completion, reported mood/affect, significant quotes or statements, and denial of SI/HI when risk level is low or none.
- Objective: therapist observations mapped from the structured data including affect, engagement, eye contact, appearance, speech, thought process, and any skills or behaviors observed in session.
- Assessment: clinical interpretation of the session, exact reported progress level, treatment response, diagnostic consistency, emerging themes, and risk assessment.
- Plan: next session focus, homework assigned, interventions to continue or introduce, frequency, and next appointment if noted.

OUTPUT FORMAT:
Return only a valid JSON object with exactly four keys: "subjective", "objective", "assessment", and "plan".
Each value must be a string of 3-6 sentences of professional clinical prose.
No markdown. No code fences. No extra text outside the JSON.

IMPORTANT:
This is a draft for clinician review and approval.""",
    "DAP": """You are a clinical documentation assistant working with a licensed therapist (LCSW-C). Generate a DAP progress note from the structured session data provided by the therapist.

RULES:
- Write in professional clinical language appropriate for a medical record.
- Use third person and stay specific to the provided details.
- Do not fabricate facts or add information that was not supplied.
- Include the ICD-10 code for the diagnosis if you can identify it.
- Reference treatment goals if provided and note session continuity when clinically relevant.

SECTION GUIDELINES:
- Data: combine client report, therapist observations, interventions used, and response to interventions into a cohesive factual account of the session.
- Assessment: provide clinical interpretation, exact reported progress level, diagnostic consistency, treatment response, and risk assessment.
- Plan: outline next session focus, homework, interventions to continue or introduce, frequency, and next appointment if provided.

OUTPUT FORMAT:
Return only a valid JSON object with exactly three keys: "data", "assessment", and "plan".
Each value must be a string of 3-6 sentences of professional clinical prose.
No markdown. No code fences. No extra text outside the JSON.

IMPORTANT:
This is a draft for clinician review and approval.""",
    "BIRP": """You are a clinical documentation assistant working with a licensed therapist (LCSW-C). Generate a BIRP progress note from the structured session data provided by the therapist.

RULES:
- Write in professional clinical language appropriate for a medical record.
- Use third person and remain grounded in the therapist's structured input.
- Do not fabricate facts or add unsupported clinical detail.
- Include the ICD-10 code for the diagnosis if you can identify it.
- Reference treatment goals if provided and note session continuity when clinically relevant.

SECTION GUIDELINES:
- Behavior: summarize the client's presentation, symptoms, self-report, and observable behavior.
- Intervention: describe what the therapist did in session, using the selected interventions and description.
- Response: capture how the client responded to the interventions, treatment engagement, progress, and risk considerations.
- Plan: cover next session focus, homework, follow-up, and next appointment details if available.

OUTPUT FORMAT:
Return only a valid JSON object with exactly four keys: "behavior", "intervention", "response", and "plan".
Each value must be a string of 3-6 sentences of professional clinical prose.
No markdown. No code fences. No extra text outside the JSON.

IMPORTANT:
This is a draft for clinician review and approval.""",
}


class NoteRequest(BaseModel):
    client_name: str
    session_number: int | None = None
    session_date: str | None = None
    duration_minutes: int | None = None
    session_type: str | None = None
    note_format: str
    primary_diagnosis: str | None = None
    treatment_modality: str | None = None
    treatment_goals: str = ""
    client_report: str
    interventions_checked: list[str] = Field(default_factory=list)
    interventions_other: str = ""
    interventions_description: str
    affect: str
    engagement: str
    eye_contact: str
    appearance: str
    speech: str
    thought_process: str
    additional_observations: str = ""
    client_response: str
    progress: str | None = None
    risk_level: str | None = None
    risk_details: str = ""
    plan_next_session: str
    homework: str = ""
    next_appointment: str = ""


class EditRequest(BaseModel):
    edits: dict[str, str]


class CopyRequest(BaseModel):
    final_output: str


class FeedbackRequest(BaseModel):
    feedback_rating: str
    feedback_notes: str = ""


app = FastAPI(title="Clarity Prototype API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS note_generations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              client_name TEXT NOT NULL,
              session_number INTEGER,
              session_date TEXT,
              duration_minutes INTEGER,
              session_type TEXT,
              note_format TEXT NOT NULL,
              primary_diagnosis TEXT,
              treatment_modality TEXT,
              input_payload TEXT NOT NULL,
              ai_output TEXT NOT NULL,
              ai_model TEXT DEFAULT 'anthropic/claude-sonnet-4-5',
              generation_time_ms INTEGER,
              edits TEXT DEFAULT '{}',
              final_output TEXT,
              copied_at TIMESTAMP,
              feedback_rating TEXT,
              feedback_notes TEXT
            );
            """
        )
        connection.commit()


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate-note")
async def generate_note(request: NoteRequest) -> dict[str, Any]:
    note_format = request.note_format.upper()
    if note_format not in SECTION_CONFIG:
        raise HTTPException(status_code=400, detail="Unsupported note format.")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured.")

    payload = request.model_dump()
    started_at = time.perf_counter()
    raw_output = await call_openrouter(note_format, payload, api_key)
    generation_time_ms = int((time.perf_counter() - started_at) * 1000)

    sections = parse_generated_sections(note_format, raw_output)
    metadata = build_metadata(payload)
    response_payload = {
        "note_id": save_generated_note(payload, note_format, sections, generation_time_ms),
        "format": note_format,
        "metadata": metadata,
        "sections": build_section_state(note_format, sections, {}),
    }
    return response_payload


@app.post("/api/notes/{note_id}/edits")
def save_note_edits(note_id: int, request: EditRequest) -> dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, note_format, input_payload, ai_output, edits FROM note_generations WHERE id = ?",
            (note_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Note not found.")

        note_format = row["note_format"]
        current_edits = parse_json_field(row["edits"], default={})
        current_edits.update({key: value.strip() for key, value in request.edits.items() if value.strip()})

        final_output = build_plaintext_output(
            build_metadata(parse_json_field(row["input_payload"], default={})),
            build_section_state(note_format, parse_json_field(row["ai_output"], default={}), current_edits),
        )
        connection.execute(
            "UPDATE note_generations SET edits = ?, final_output = ? WHERE id = ?",
            (json.dumps(current_edits), final_output, note_id),
        )
        connection.commit()

    return {"note_id": note_id, "edits": current_edits}


@app.post("/api/notes/{note_id}/copied")
def mark_note_copied(note_id: int, request: CopyRequest) -> dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute("SELECT id FROM note_generations WHERE id = ?", (note_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Note not found.")

        connection.execute(
            """
            UPDATE note_generations
            SET final_output = ?, copied_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (request.final_output, note_id),
        )
        connection.commit()

    return {"note_id": note_id, "copied": True}


@app.post("/api/notes/{note_id}/feedback")
def save_feedback(note_id: int, request: FeedbackRequest) -> dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute("SELECT id FROM note_generations WHERE id = ?", (note_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Note not found.")

        connection.execute(
            """
            UPDATE note_generations
            SET feedback_rating = ?, feedback_notes = ?
            WHERE id = ?
            """,
            (request.feedback_rating, request.feedback_notes.strip(), note_id),
        )
        connection.commit()

    return {"note_id": note_id, "feedback_rating": request.feedback_rating}


@app.get("/api/notes")
def list_notes() -> dict[str, list[dict[str, Any]]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, client_name, session_number, session_date, duration_minutes,
                   session_type, note_format, primary_diagnosis, treatment_modality,
                   generation_time_ms, copied_at, feedback_rating, edits
            FROM note_generations
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()

    notes = []
    for row in rows:
        edits = parse_json_field(row["edits"], default={})
        notes.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "client_name": row["client_name"],
                "session_number": row["session_number"],
                "session_date": row["session_date"],
                "duration_minutes": row["duration_minutes"],
                "session_type": row["session_type"],
                "note_format": row["note_format"],
                "primary_diagnosis": row["primary_diagnosis"],
                "treatment_modality": row["treatment_modality"],
                "generation_time_ms": row["generation_time_ms"],
                "copied_at": row["copied_at"],
                "feedback_rating": row["feedback_rating"],
                "edited_section_count": len(edits),
            }
        )

    return {"notes": notes}


@app.get("/api/notes/{note_id}")
def get_note(note_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM note_generations WHERE id = ?", (note_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Note not found.")

    input_payload = parse_json_field(row["input_payload"], default={})
    ai_output = parse_json_field(row["ai_output"], default={})
    edits = parse_json_field(row["edits"], default={})
    metadata = build_metadata(input_payload)
    sections = build_section_state(row["note_format"], ai_output, edits)

    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "format": row["note_format"],
        "metadata": metadata,
        "input_payload": input_payload,
        "sections": sections,
        "analysis": {
            "ai_model": row["ai_model"],
            "generation_time_ms": row["generation_time_ms"],
            "edited_sections": sorted(edits.keys()),
            "edited_section_count": len(edits),
            "copied": bool(row["copied_at"]),
            "copied_at": row["copied_at"],
            "feedback_rating": row["feedback_rating"],
            "feedback_notes": row["feedback_notes"],
        },
        "final_output": row["final_output"] or build_plaintext_output(metadata, sections),
    }


async def call_openrouter(note_format: str, payload: dict[str, Any], api_key: str) -> str:
    system_prompt = SYSTEM_PROMPTS[note_format]
    user_prompt = build_user_prompt(payload)
    request_body = {
        "model": OPENROUTER_MODEL,
        "max_tokens": 2000,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(OPENROUTER_URL, headers=headers, json=request_body)

    if response.status_code >= 400:
        detail = response.text
        try:
            parsed = response.json()
            detail = parsed.get("error", {}).get("message") or parsed.get("message") or detail
        except json.JSONDecodeError:
            pass
        raise HTTPException(status_code=502, detail=f"OpenRouter request failed: {detail}")

    body = response.json()
    content = body.get("choices", [{}])[0].get("message", {}).get("content")
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str) else item.get("text", "")
            for item in content
        ).strip()
    if isinstance(content, str):
        return content.strip()

    raise HTTPException(status_code=502, detail="OpenRouter returned no note content.")


def build_user_prompt(payload: dict[str, Any]) -> str:
    goals = payload.get("treatment_goals") or "Not specified"
    additional = payload.get("additional_observations") or "None noted"
    homework = payload.get("homework") or "None assigned"
    next_appointment = payload.get("next_appointment") or "To be scheduled"
    risk_details = payload.get("risk_details")
    interventions = ", ".join(payload.get("interventions_checked") or [])

    lines = [
        "SESSION INFORMATION:",
        f"- Client: {payload.get('client_name', '')}",
        (
            f"- Session #{payload.get('session_number') or ''} | {payload.get('session_date') or ''} | "
            f"{payload.get('duration_minutes') or ''} min | {payload.get('session_type') or ''}"
        ),
        f"- Diagnosis: {payload.get('primary_diagnosis') or 'Not specified'}",
        f"- Treatment Modality: {payload.get('treatment_modality') or 'Not specified'}",
        f"- Current Treatment Goals: {goals}",
        "",
        "CLIENT REPORT:",
        payload.get("client_report") or "",
        "",
        "INTERVENTIONS USED:",
        f"Techniques: {interventions or 'Not specified'}",
        f"Description: {payload.get('interventions_description') or 'Not specified'}",
        "",
        "THERAPIST OBSERVATIONS:",
        f"- Affect: {payload.get('affect') or 'Not specified'}",
        f"- Engagement: {payload.get('engagement') or 'Not specified'}",
        f"- Eye Contact: {payload.get('eye_contact') or 'Not specified'}",
        f"- Appearance: {payload.get('appearance') or 'Not specified'}",
        f"- Speech: {payload.get('speech') or 'Not specified'}",
        f"- Thought Process: {payload.get('thought_process') or 'Not specified'}",
        f"- Additional: {additional}",
        "",
        "CLIENT RESPONSE TO SESSION:",
        payload.get("client_response") or "",
        "",
        f"PROGRESS TOWARD GOALS: {payload.get('progress') or 'Not specified'}",
        "",
        f"RISK ASSESSMENT: {payload.get('risk_level') or 'Not specified'}",
    ]
    if risk_details:
        lines.append(f"RISK DETAILS: {risk_details}")
    lines.extend(
        [
            "",
            "PLAN:",
            f"- Next session focus: {payload.get('plan_next_session') or 'Not specified'}",
            f"- Homework: {homework}",
            f"- Next appointment: {next_appointment}",
            "",
            f"Generate a {payload.get('note_format', 'SOAP')} progress note.",
        ]
    )
    return "\n".join(lines)


def parse_generated_sections(note_format: str, raw_output: str) -> dict[str, str]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=502, detail=f"Model output was not valid JSON: {error}") from error

    required_keys = [section["key"] for section in SECTION_CONFIG[note_format]]
    missing = [key for key in required_keys if not isinstance(parsed.get(key), str) or not parsed.get(key).strip()]
    if missing:
        raise HTTPException(
            status_code=502,
            detail=f"Model output was missing required sections: {', '.join(missing)}",
        )

    return {key: parsed[key].strip() for key in required_keys}


def build_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "client_name": payload.get("client_name"),
        "session_number": payload.get("session_number"),
        "session_date": payload.get("session_date"),
        "duration_minutes": payload.get("duration_minutes"),
        "session_type": payload.get("session_type"),
        "diagnosis": payload.get("primary_diagnosis"),
        "modality": payload.get("treatment_modality"),
    }


def build_section_state(
    note_format: str, generated_sections: dict[str, str], edits: dict[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            **section,
            "content": (edits.get(section["key"]) or generated_sections.get(section["key"], "")).strip(),
            "edited": section["key"] in edits,
        }
        for section in SECTION_CONFIG[note_format]
    ]


def save_generated_note(
    payload: dict[str, Any], note_format: str, sections: dict[str, str], generation_time_ms: int
) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO note_generations (
                client_name,
                session_number,
                session_date,
                duration_minutes,
                session_type,
                note_format,
                primary_diagnosis,
                treatment_modality,
                input_payload,
                ai_output,
                generation_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("client_name"),
                payload.get("session_number"),
                payload.get("session_date"),
                payload.get("duration_minutes"),
                payload.get("session_type"),
                note_format,
                payload.get("primary_diagnosis"),
                payload.get("treatment_modality"),
                json.dumps(payload),
                json.dumps(sections),
                generation_time_ms,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def parse_json_field(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def build_plaintext_output(metadata: dict[str, Any], sections: list[dict[str, Any]]) -> str:
    lines = [
        "PROGRESS NOTE - DRAFT",
        f"Client: {metadata.get('client_name') or ''}",
        (
            f"Session #{metadata.get('session_number') or ''} | {metadata.get('session_date') or ''} | "
            f"{metadata.get('duration_minutes') or ''} minutes | {metadata.get('session_type') or ''}"
        ),
        f"Diagnosis: {metadata.get('diagnosis') or ''}",
        f"Modality: {metadata.get('modality') or ''}",
        "",
    ]
    for section in sections:
        lines.extend([f"{section['clipboard']}:", section["content"], ""])
    lines.extend(["---", "Generated by Clarity | Draft for clinician review and approval"])
    return "\n".join(lines)
