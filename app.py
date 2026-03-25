import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from authlib.integrations.starlette_client import OAuth
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from jose import JWTError, jwt as jose_jwt
from pydantic import BaseModel, Field
import bcrypt as _bcrypt_lib
from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import NullPool
from starlette.middleware.sessions import SessionMiddleware


load_dotenv()

# ---------------------------------------------------------------------------
# SQLAlchemy — PostgreSQL (prod) or SQLite (local dev) for auth tables
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL:
    # Railway provides postgres:// but SQLAlchemy needs postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    _engine = create_engine(DATABASE_URL, poolclass=NullPool)
else:
    import warnings
    warnings.warn(
        "DATABASE_URL is not set. Falling back to local SQLite for auth (dev only). "
        "Set DATABASE_URL to a PostgreSQL connection string in production.",
        stacklevel=1,
    )
    _sqlite_auth_path = str(Path(__file__).resolve().parent / "auth.db")
    _engine = create_engine(
        f"sqlite:///{_sqlite_auth_path}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )

Base = declarative_base()
SessionLocal = sessionmaker(bind=_engine)


class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    encrypted_email = Column(Text, nullable=True)
    email_lookup_hash = Column(String(64), unique=True, nullable=True, index=True)
    hashed_password = Column(Text, nullable=True)
    display_name = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    sso_accounts = relationship("SSOAccount", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("ActiveSession", back_populates="user", cascade="all, delete-orphan")


class SSOAccount(Base):
    __tablename__ = "sso_accounts"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    provider = Column(String(32), nullable=False)
    provider_id = Column(String(128), nullable=False)
    user = relationship("User", back_populates="sso_accounts")
    __table_args__ = (Index("ix_sso_provider_provider_id", "provider", "provider_id", unique=True),)


class ActiveSession(Base):
    __tablename__ = "active_sessions"
    token_hash = Column(String(64), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    user = relationship("User", back_populates="sessions")


class NoteGeneration(Base):
    __tablename__ = "note_generations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    user_id = Column(String(36), nullable=True)
    client_name = Column(Text, nullable=True)
    session_number = Column(Integer, nullable=True)
    session_date = Column(String(32), nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    session_type = Column(String(64), nullable=True)
    note_format = Column(String(16), nullable=False)
    note_template_name = Column(String(128), nullable=True)
    section_config_json = Column(Text, nullable=True)
    primary_diagnosis = Column(Text, nullable=True)
    treatment_modality = Column(String(128), nullable=True)
    input_payload = Column(Text, nullable=False, default="{}")
    ai_output = Column(Text, nullable=False, default="{}")
    ai_model = Column(String(64), default="anthropic/claude-sonnet-4-5")
    generation_time_ms = Column(Integer, nullable=True)
    edits = Column(Text, default="{}")
    final_output = Column(Text, nullable=True)
    copied_at = Column(DateTime(timezone=True), nullable=True)
    feedback_rating = Column(String(16), nullable=True)
    feedback_notes = Column(Text, nullable=True)


class NoteTemplate(Base):
    __tablename__ = "templates"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), nullable=True)
    name = Column(String(128), nullable=False)
    sections_json = Column(Text, nullable=False)
    is_builtin = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class NoteAnalytic(Base):
    __tablename__ = "note_analytics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    note_id = Column(Integer, nullable=True)
    purged_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    note_format = Column(String(16), nullable=True)
    note_template_name = Column(String(128), nullable=True)
    session_type = Column(String(64), nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    treatment_modality = Column(String(128), nullable=True)
    diagnosis_count = Column(Integer, nullable=True)
    section_count = Column(Integer, nullable=True)
    generation_time_ms = Column(Integer, nullable=True)
    ai_output_char_count = Column(Integer, nullable=True)
    final_output_char_count = Column(Integer, nullable=True)
    was_edited = Column(Integer, nullable=True)
    edited_section_count = Column(Integer, nullable=True)
    was_copied = Column(Integer, nullable=True)
    feedback_rating = Column(String(16), nullable=True)
    risk_level = Column(String(128), nullable=True)
    field_fill_json = Column(Text, nullable=True)
    interventions_count = Column(Integer, nullable=True)


def _hash_password(password: str) -> str:
    """SHA-256 pre-hash then bcrypt — avoids bcrypt 72-byte limit."""
    digest = hashlib.sha256(password.encode()).hexdigest().encode()
    return _bcrypt_lib.hashpw(digest, _bcrypt_lib.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    digest = hashlib.sha256(password.encode()).hexdigest().encode()
    return _bcrypt_lib.checkpw(digest, hashed.encode())


def _create_auth_tables() -> None:
    Base.metadata.create_all(_engine)
    # Migrate: add hashed_password column if missing (for existing deployments)
    with _engine.connect() as conn:
        if DATABASE_URL:  # PostgreSQL
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='users' AND column_name='hashed_password'"
            ))
            if result.fetchone() is None:
                conn.execute(text("ALTER TABLE users ADD COLUMN hashed_password TEXT"))
                conn.commit()
        else:  # SQLite
            result = conn.execute(text("PRAGMA table_info(users)"))
            cols = {row[1] for row in result.fetchall()}
            if "hashed_password" not in cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN hashed_password TEXT"))
                conn.commit()
    # Migrate: add display_name column if missing
    with _engine.connect() as conn:
        if DATABASE_URL:  # PostgreSQL
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='users' AND column_name='display_name'"
            ))
            if result.fetchone() is None:
                conn.execute(text("ALTER TABLE users ADD COLUMN display_name TEXT"))
                conn.commit()
        else:  # SQLite
            result = conn.execute(text("PRAGMA table_info(users)"))
            cols = {row[1] for row in result.fetchall()}
            if "display_name" not in cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN display_name TEXT"))
                conn.commit()


# ---------------------------------------------------------------------------
# Auth config
# ---------------------------------------------------------------------------

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://clarity.bearingdigital.com")
JWT_SECRET = os.getenv("JWT_SECRET", "")
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET environment variable must be set. "
        "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
BACKEND_URL = os.getenv("BACKEND_URL", "https://web-production-d6d7d.up.railway.app")

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY") or os.getenv("DB_ENCRYPTION_KEY", "")
HMAC_SECRET = os.getenv("HMAC_SECRET", ENCRYPTION_KEY)

# In-memory short-code store: {code: (user_id, expires_at)}
auth_codes: dict[str, tuple[str, datetime]] = {}
_auth_codes_lock = __import__("threading").Lock()


def _issue_auth_code(user_id: str) -> str:
    """Generate a single-use auth code and prune expired entries."""
    code = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=60)
    now = datetime.now(timezone.utc)
    with _auth_codes_lock:
        expired = [k for k, v in auth_codes.items() if v[1] <= now]
        for k in expired:
            del auth_codes[k]
        auth_codes[code] = (user_id, expires_at)
    return code


def _redeem_auth_code(code: str) -> str:
    """Validate and consume a single-use auth code. Raises 400 if invalid/expired."""
    now = datetime.now(timezone.utc)
    with _auth_codes_lock:
        entry = auth_codes.pop(code, None)
    if entry is None:
        raise HTTPException(status_code=400, detail="Invalid or expired auth code.")
    user_id, expires_at = entry
    if now > expires_at:
        raise HTTPException(status_code=400, detail="Auth code has expired.")
    return user_id

oauth = OAuth()
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def email_lookup_hash(email: str) -> str:
    """Deterministic HMAC-SHA256 for email lookups — same input always same output."""
    if not HMAC_SECRET:
        raise RuntimeError(
            "HMAC_SECRET environment variable must be set. "
            "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    secret = HMAC_SECRET.encode()
    return hmac.new(secret, email.lower().strip().encode(), hashlib.sha256).hexdigest()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


BASE_DIR = Path(__file__).resolve().parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4-5"
OPENROUTER_AUDIO_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
OPENAI_AUDIO_URL = "https://api.openai.com/v1/audio/transcriptions"
HCC_TEMPLATE_NAME = "HCC SOAP Note"
SUPPORTED_AUDIO_EXTENSIONS = {".webm", ".mp4", ".m4a", ".mp3", ".wav", ".mpeg"}
VOICE_EXTRACTABLE_KEYS = {
    "client_report",
    "interventions_checked",
    "interventions_description",
    "affect",
    "engagement",
    "eye_contact",
    "appearance",
    "speech",
    "thought_process",
    "additional_observations",
    "client_response",
    "progress",
    "risk_level",
    "risk_details",
    "plan_next_session",
    "homework",
    "next_appointment",
    "treatment_goals",
    "primary_diagnosis",
}
VOICE_ALLOWED_INTERVENTIONS = {
    "CBT",
    "TF-CBT",
    "DBT skills",
    "EFT",
    "Gottman Method",
    "Mindfulness/grounding",
    "Psychoeducation",
    "Motivational Interviewing",
    "Safety planning",
    "Mind/Body Connection",
    "ACT",
    "Communication skills training",
    "Other",
}
VOICE_ENUM_FIELDS = {
    "affect": {
        "Depressed",
        "Anxious",
        "Flat",
        "Constricted",
        "Euthymic",
        "Bright",
        "Labile",
        "Irritable",
        "Tearful",
    },
    "engagement": {
        "Fully engaged",
        "Mostly engaged",
        "Partially engaged",
        "Minimally engaged",
        "Resistant",
    },
    "eye_contact": {"Consistent", "Intermittent", "Avoidant", "N/A-Telehealth"},
    "appearance": {"Well-groomed", "Casually dressed", "Disheveled", "Unremarkable", "N/A-Telehealth"},
    "speech": {
        "Normal rate & volume",
        "Pressured",
        "Slow",
        "Soft",
        "Monotone",
        "Unremarkable",
    },
    "thought_process": {
        "Logical & goal-directed",
        "Tangential",
        "Circumstantial",
        "Racing",
        "Perseverative",
        "Unremarkable",
    },
    "progress": {
        "Significant progress",
        "Some progress",
        "No notable change",
        "Regression",
        "Too early to assess",
    },
    "risk_level": {
        "No risk indicators",
        "Low risk — no imminent concern",
        "Moderate risk — discussed safety plan",
        "High risk — safety plan activated / crisis protocol",
    },
}
VOICE_EXTRACTION_SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Given a therapist verbal session "
    "summary, extract structured data to populate a therapy progress note form. "
    "Return ONLY a valid JSON object with these keys (omit keys where information "
    "is not mentioned): client_report, interventions_checked (array of strings "
    "matching: CBT, TF-CBT, DBT skills, EFT, Gottman Method, Mindfulness/grounding, "
    "Psychoeducation, Motivational Interviewing, Safety planning, Mind/Body "
    "Connection, ACT, Communication skills training, Other), "
    "interventions_description, affect (one of: Depressed, Anxious, Flat, "
    "Constricted, Euthymic, Bright, Labile, Irritable, Tearful), engagement "
    "(one of: Fully engaged, Mostly engaged, Partially engaged, Minimally engaged, "
    "Resistant), eye_contact (one of: Consistent, Intermittent, Avoidant, "
    "N/A-Telehealth), appearance (one of: Well-groomed, Casually dressed, "
    "Disheveled, Unremarkable, N/A-Telehealth), speech (one of: Normal rate & "
    "volume, Pressured, Slow, Soft, Monotone, Unremarkable), thought_process "
    "(one of: Logical & goal-directed, Tangential, Circumstantial, Racing, "
    "Perseverative, Unremarkable), additional_observations, client_response, "
    "progress (one of: Significant progress, Some progress, No notable change, "
    "Regression, Too early to assess), risk_level (one of: No risk indicators, "
    "Low risk — no imminent concern, Moderate risk — discussed safety plan, High "
    "risk — safety plan activated / crisis protocol), risk_details, "
    "plan_next_session, homework, next_appointment, treatment_goals, "
    "primary_diagnosis (array of strings)."
)

# ---------------------------------------------------------------------------
# Database encryption — AES-256-GCM, application-layer column encryption
# ---------------------------------------------------------------------------


def _load_cipher() -> AESGCM:
    raw = os.getenv("DB_ENCRYPTION_KEY", "")
    if not raw:
        raise RuntimeError(
            "DB_ENCRYPTION_KEY is not set. "
            "Set this environment variable to a 32-byte base64url-encoded key. "
            "Generate one with: "
            "python -c \"import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())\""
        )
    try:
        key_bytes = base64.urlsafe_b64decode(raw + "==")
    except Exception as exc:
        raise ValueError(f"DB_ENCRYPTION_KEY could not be base64-decoded: {exc}") from exc
    if len(key_bytes) != 32:
        raise ValueError(
            f"DB_ENCRYPTION_KEY must decode to exactly 32 bytes for AES-256 "
            f"(got {len(key_bytes)} bytes)."
        )
    return AESGCM(key_bytes)


_CIPHER: AESGCM = _load_cipher()


def encrypt_field(plaintext: str | None) -> str | None:
    if plaintext is None:
        return None
    nonce = os.urandom(12)
    ciphertext = _CIPHER.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_field(stored: str | None) -> str | None:
    if stored is None:
        return None
    try:
        raw = base64.b64decode(stored)
        nonce, ciphertext = raw[:12], raw[12:]
        return _CIPHER.decrypt(nonce, ciphertext, None).decode("utf-8")
    except Exception:
        return stored


SECTION_CONFIG = {
    "SOAP": [
        {"key": "subjective", "short": "S", "title": "Subjective", "clipboard": "SUBJECTIVE"},
        {"key": "objective", "short": "O", "title": "Objective", "clipboard": "OBJECTIVE"},
        {"key": "assessment", "short": "A", "title": "Clinical Summary (Draft)", "clipboard": "CLINICAL SUMMARY"},
        {"key": "plan", "short": "P", "title": "Plan", "clipboard": "PLAN"},
    ],
    "DAP": [
        {"key": "data", "short": "D", "title": "Data", "clipboard": "DATA"},
        {"key": "assessment", "short": "A", "title": "Clinical Summary (Draft)", "clipboard": "CLINICAL SUMMARY"},
        {"key": "plan", "short": "P", "title": "Plan", "clipboard": "PLAN"},
    ],
    "BIRP": [
        {"key": "behavior", "short": "B", "title": "Behavior", "clipboard": "BEHAVIOR"},
        {"key": "intervention", "short": "I", "title": "Intervention", "clipboard": "INTERVENTION"},
        {"key": "response", "short": "R", "title": "Response", "clipboard": "RESPONSE"},
        {"key": "plan", "short": "P", "title": "Plan", "clipboard": "PLAN"},
    ],
}

HCC_SECTION_CONFIG = [
    {
        "key": "subjective_complaint_presenting_problem",
        "short": "1",
        "title": "Subjective Complaint / Presenting Problem",
        "clipboard": "SUBJECTIVE COMPLAINT / PRESENTING PROBLEM",
    },
    {"key": "objective", "short": "2", "title": "Objective", "clipboard": "OBJECTIVE"},
    {
        "key": "provider_assessment",
        "short": "3",
        "title": "Provider Assessment",
        "clipboard": "PROVIDER ASSESSMENT",
    },
    {
        "key": "clinical_interventions_used",
        "short": "4",
        "title": "Clinical Interventions used",
        "clipboard": "CLINICAL INTERVENTIONS USED",
    },
    {
        "key": "clients_response",
        "short": "5",
        "title": "Clients Response",
        "clipboard": "CLIENTS RESPONSE",
    },
    {
        "key": "progress_regression_toward_goals",
        "short": "6",
        "title": "Progress/Regression Toward Goals",
        "clipboard": "PROGRESS/REGRESSION TOWARD GOALS",
    },
    {
        "key": "insight_and_treatment_recommendations",
        "short": "7",
        "title": "Insight and Treatment Recommendations",
        "clipboard": "INSIGHT AND TREATMENT RECOMMENDATIONS",
    },
    {
        "key": "risk_check",
        "short": "8",
        "title": "Risk Check",
        "clipboard": "RISK CHECK",
    },
    {
        "key": "plan_recommendations_homework",
        "short": "9",
        "title": "Plan / Recommendations / Homework",
        "clipboard": "PLAN / RECOMMENDATIONS / HOMEWORK",
    },
]

BUILTIN_TEMPLATES = [
    {
        "name": HCC_TEMPLATE_NAME,
        "sections_json": HCC_SECTION_CONFIG,
        "is_builtin": True,
    }
]

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
- Clinical Summary (Draft): document the clinician's summary of session progress, treatment response, emerging themes, and risk considerations as reported by the clinician. Do not provide independent diagnostic conclusions.
- Plan: next session focus, homework assigned, interventions to continue or introduce, frequency, and next appointment if noted.
""",
    "DAP": """You are a clinical documentation assistant working with a licensed therapist (LCSW-C). Generate a DAP progress note from the structured session data provided by the therapist.

RULES:
- Write in professional clinical language appropriate for a medical record.
- Use third person and stay specific to the provided details.
- Do not fabricate facts or add information that was not supplied.
- Include the ICD-10 code for the diagnosis if you can identify it.
- Reference treatment goals if provided and note session continuity when clinically relevant.

SECTION GUIDELINES:
- Data: combine client report, therapist observations, interventions used, and response to interventions into a cohesive factual account of the session.
- Clinical Summary (Draft): document the clinician's summary of session progress, treatment response, diagnostic considerations, and risk assessment as reported by the clinician. Do not provide independent diagnostic conclusions.
- Plan: outline next session focus, homework, interventions to continue or introduce, frequency, and next appointment if provided.
""",
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
""",
}


class NoteRequest(BaseModel):
    client_name: str
    session_number: int | None = None
    session_date: str | None = None
    duration_minutes: int | None = None
    session_type: str | None = None
    note_format: str
    note_template_id: int | str | None = None
    note_template_name: str | None = None
    primary_diagnosis: list[str] | str | None = None
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


class CorrectDataRequest(BaseModel):
    note_id: int
    field: str
    value: str


class TemplateCreateRequest(BaseModel):
    name: str
    sections_json: list[dict[str, Any]] | list[str]


class ProfileUpdateRequest(BaseModel):
    display_name: str


class TakehomeRequest(BaseModel):
    note_text: str


app = FastAPI(title="Clarity Prototype API")
_raw_origins = os.getenv("ALLOWED_ORIGINS", "https://clarity.bearingdigital.com")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-User-Id"],
)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", JWT_SECRET))

scheduler = BackgroundScheduler()


def purge_expired_session_data() -> None:
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=30)
        expired_notes = (
            db.query(NoteGeneration)
            .filter(
                NoteGeneration.created_at < cutoff,
                NoteGeneration.input_payload != "{}",
            )
            .all()
        )
        for note in expired_notes:
            input_payload = parse_json_field(decrypt_field(note.input_payload), default={})
            if isinstance(input_payload, dict) and input_payload:
                db.add(NoteAnalytic(**_extract_analytics_row(note, input_payload)))
            note.input_payload = "{}"
            note.ai_output = "{}"
            note.client_name = "[purged]"
            note.primary_diagnosis = None
            note.final_output = None

        expired_sessions = db.query(ActiveSession).filter(ActiveSession.expires_at < now).all()
        expired_user_ids = list({session.user_id for session in expired_sessions})
        if expired_user_ids:
            (
                db.query(NoteGeneration)
                .filter(NoteGeneration.user_id.in_(expired_user_ids))
                .update({NoteGeneration.user_id: None}, synchronize_session=False)
            )
        db.query(ActiveSession).filter(ActiveSession.expires_at < now).delete(synchronize_session=False)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def seed_builtin_templates() -> None:
    db = SessionLocal()
    try:
        for template in BUILTIN_TEMPLATES:
            existing = db.query(NoteTemplate).filter_by(name=template["name"], is_builtin=1).first()
            if existing is None:
                db.add(
                    NoteTemplate(
                        user_id=None,
                        name=template["name"],
                        sections_json=json.dumps(template["sections_json"]),
                        is_builtin=1,
                    )
                )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.on_event("startup")
def startup_event() -> None:
    _create_auth_tables()
    seed_builtin_templates()
    scheduler.add_job(purge_expired_session_data, "interval", minutes=30, id="session_purge")
    scheduler.start()


@app.on_event("shutdown")
def shutdown_event() -> None:
    scheduler.shutdown(wait=False)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


BAA_PATH = BASE_DIR / "baa.docx"


@app.get("/legal/baa")
def download_baa() -> FileResponse:
    if not BAA_PATH.exists():
        raise HTTPException(status_code=404, detail="BAA document not found.")
    return FileResponse(
        path=str(BAA_PATH),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="Clarity-HIPAA-BAA.docx",
    )


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def get_current_user(
    request: Request,
) -> str:
    """Extract user_id from Bearer token only. No fallback."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        if token:
            token_hash = hash_token(token)
            db = SessionLocal()
            try:
                now = datetime.now(timezone.utc)
                session_obj = (
                    db.query(ActiveSession)
                    .filter(
                        ActiveSession.token_hash == token_hash,
                        ActiveSession.expires_at > now,
                    )
                    .first()
                )
                if session_obj is not None:
                    return str(session_obj.user_id)
            finally:
                db.close()
    raise HTTPException(status_code=401, detail="Authentication required.")


# ---------------------------------------------------------------------------
# OAuth / Auth endpoints
# ---------------------------------------------------------------------------


@app.get("/auth/login")
async def auth_login(request: Request) -> RedirectResponse:
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google OAuth is not configured.")
    redirect_uri = f"{BACKEND_URL}/auth/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def auth_callback(request: Request) -> RedirectResponse:
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google OAuth is not configured.")
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"OAuth error: {exc}") from exc

    userinfo = token.get("userinfo") or {}
    email = (userinfo.get("email") or "").strip().lower()
    provider_id = str(userinfo.get("sub") or "")
    google_name = (userinfo.get("name") or "").strip()

    if not email or not provider_id:
        raise HTTPException(status_code=400, detail="Could not retrieve user info from Google.")

    lookup_hash = email_lookup_hash(email)

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email_lookup_hash == lookup_hash).first()
        if user is None:
            user = User(
                id=str(uuid.uuid4()),
                encrypted_email=encrypt_field(email),
                email_lookup_hash=lookup_hash,
                display_name=google_name or None,
                created_at=datetime.now(timezone.utc),
            )
            db.add(user)
            db.flush()
        elif google_name and not user.display_name:
            user.display_name = google_name

        sso = (
            db.query(SSOAccount)
            .filter(SSOAccount.provider == "google", SSOAccount.provider_id == provider_id)
            .first()
        )
        if sso is None:
            sso = SSOAccount(
                id=str(uuid.uuid4()),
                user_id=user.id,
                provider="google",
                provider_id=provider_id,
            )
            db.add(sso)

        db.commit()
        user_id = user.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    short_code = _issue_auth_code(user_id)
    return RedirectResponse(url=f"{FRONTEND_URL}/?code={short_code}")


class AuthExchangeRequest(BaseModel):
    code: str


@app.post("/auth/exchange")
def auth_exchange(request: AuthExchangeRequest) -> dict[str, Any]:
    code = request.code.strip()
    user_id = _redeem_auth_code(code)

    now = datetime.now(timezone.utc)
    session_token = secrets.token_urlsafe(48)
    token_hash = hash_token(session_token)
    session_expires = now + timedelta(hours=24)

    db = SessionLocal()
    try:
        session_obj = ActiveSession(
            token_hash=token_hash,
            user_id=user_id,
            expires_at=session_expires,
        )
        db.add(session_obj)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return {"token": session_token, "expires_in": 86400}


@app.post("/auth/logout")
def auth_logout(request: Request) -> dict[str, str]:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        if token:
            token_hash = hash_token(token)
            db = SessionLocal()
            try:
                db.query(ActiveSession).filter(ActiveSession.token_hash == token_hash).delete()
                db.commit()
            except Exception:
                db.rollback()
            finally:
                db.close()
    return {"status": "logged out"}


@app.get("/auth/me")
def auth_me(user_id: str = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found.")
        raw_email = decrypt_field(user.encrypted_email) or ""
        masked = ""
        if raw_email:
            parts = raw_email.split("@")
            if len(parts) == 2:
                local = parts[0]
                masked = f"{local[:2]}{'*' * max(1, len(local) - 2)}@{parts[1]}"
            else:
                masked = raw_email[:3] + "***"
        return {
            "id": user.id,
            "email_masked": masked,
            "display_name": user.display_name or "",
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Email/password auth endpoints
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str | None = None


@app.post("/auth/register")
def auth_register(payload: RegisterRequest) -> dict[str, Any]:
    email = payload.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email address.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    lookup_hash = email_lookup_hash(email)
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email_lookup_hash == lookup_hash).first()
        if existing is not None:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        user = User(
            id=str(uuid.uuid4()),
            encrypted_email=encrypt_field(email),
            email_lookup_hash=lookup_hash,
            hashed_password=_hash_password(payload.password),
            display_name=payload.name.strip() if payload.name else None,
            created_at=datetime.now(timezone.utc),
        )
        db.add(user)
        db.commit()
        user_id = user.id
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    short_code = _issue_auth_code(user_id)
    return {"code": short_code}


class EmailLoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/login")
def auth_login_password(payload: EmailLoginRequest) -> dict[str, Any]:
    email = payload.email.strip().lower()
    lookup_hash = email_lookup_hash(email)
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email_lookup_hash == lookup_hash).first()
        if user is None or not user.hashed_password:
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        if not _verify_password(payload.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        user_id = user.id
    except HTTPException:
        raise
    finally:
        db.close()

    short_code = _issue_auth_code(user_id)
    return {"code": short_code}


@app.post("/auth/profile")
def update_profile(request: ProfileUpdateRequest, user_id: str = Depends(get_current_user)) -> dict[str, Any]:
    name = request.display_name.strip()
    if len(name) > 100:
        raise HTTPException(status_code=400, detail="Display name is too long (max 100 characters).")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found.")
        user.display_name = name or None
        db.commit()
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return {"display_name": name}


_CORRECTABLE_FIELDS = frozenset({"client_name", "primary_diagnosis"})


def slugify_key(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return base or "section"


def normalize_template_sections(raw_sections: list[dict[str, Any]] | list[str]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    used_keys: set[str] = set()

    for index, item in enumerate(raw_sections, start=1):
        if isinstance(item, str):
            title = item.strip()
            key = ""
            clipboard = ""
            short = ""
        else:
            title = str(item.get("title") or "").strip()
            key = str(item.get("key") or "").strip()
            clipboard = str(item.get("clipboard") or "").strip()
            short = str(item.get("short") or "").strip()

        if not title:
            raise HTTPException(status_code=400, detail="Each template section must include a title.")

        next_key = key or slugify_key(title)
        while next_key in used_keys:
            next_key = f"{next_key}_{index}"
        used_keys.add(next_key)

        normalized.append(
            {
                "key": next_key,
                "short": short or str(index),
                "title": title,
                "clipboard": clipboard or title.upper(),
            }
        )

    if not normalized:
        raise HTTPException(status_code=400, detail="Template must contain at least one section.")

    return normalized


def normalize_diagnoses(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    raw_values = value if isinstance(value, list) else [value]
    normalized = []
    for item in raw_values:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def serialize_diagnoses(value: list[str] | str | None) -> str | None:
    diagnoses = normalize_diagnoses(value)
    if not diagnoses:
        return None
    return json.dumps(diagnoses)


def parse_stored_diagnoses(value: str | None) -> list[str]:
    decrypted = decrypt_field(value)
    if not decrypted:
        return []
    parsed = parse_json_field(decrypted, default=None)
    if isinstance(parsed, list):
        return normalize_diagnoses(parsed)
    return normalize_diagnoses(decrypted)


def diagnosis_display(value: list[str] | str | None) -> str:
    diagnoses = normalize_diagnoses(value)
    return ", ".join(diagnoses) if diagnoses else "Not specified"


def parse_json_field(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def serialize_template(template: NoteTemplate) -> dict[str, Any]:
    return {
        "id": template.id,
        "user_id": template.user_id,
        "name": template.name,
        "sections_json": normalize_template_sections(parse_json_field(template.sections_json, default=[])),
        "is_builtin": bool(template.is_builtin),
        "created_at": template.created_at,
    }


def get_template_for_user(
    db: Any, user_id: str, template_id: int | None, template_name: str | None
) -> dict[str, Any] | None:
    template: NoteTemplate | None = None

    if template_id:
        template = (
            db.query(NoteTemplate)
            .filter(
                NoteTemplate.id == template_id,
                (NoteTemplate.is_builtin == 1) | (NoteTemplate.user_id == user_id),
            )
            .first()
        )
    elif template_name:
        template = (
            db.query(NoteTemplate)
            .filter(
                NoteTemplate.name == template_name,
                (NoteTemplate.is_builtin == 1) | (NoteTemplate.user_id == user_id),
            )
            .order_by(NoteTemplate.is_builtin.desc(), NoteTemplate.id.desc())
            .first()
        )

    if template is None:
        return None

    return serialize_template(template)


def resolve_section_config(
    note_format: str,
    stored_section_config: str | None = None,
    template: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if template:
        return normalize_template_sections(template["sections_json"])

    parsed_config = parse_json_field(stored_section_config, default=None)
    if isinstance(parsed_config, list) and parsed_config:
        return normalize_template_sections(parsed_config)

    return SECTION_CONFIG[note_format]


def build_output_format_instruction(section_config: list[dict[str, str]]) -> str:
    keys = [section["key"] for section in section_config]
    return "\n".join(
        [
            "OUTPUT FORMAT:",
            f'Return only a valid JSON object with exactly these keys: {", ".join(keys)}.',
            "Each value must be a string of 2-6 sentences of professional clinical prose.",
            "No markdown. No code fences. No extra text outside the JSON.",
            "",
            "IMPORTANT:",
            "This is a draft for clinician review and approval.",
        ]
    )


def build_system_prompt(
    note_format: str, section_config: list[dict[str, str]], template_name: str | None
) -> str:
    if template_name == HCC_TEMPLATE_NAME:
        return "\n".join(
            [
                "You are a clinical documentation assistant working with a licensed therapist (LCSW-C).",
                "Generate an HCC SOAP note from the structured session data provided by the therapist.",
                "",
                "RULES:",
                "- Write in professional clinical language appropriate for a medical record.",
                "- Use third person and stay grounded in the therapist's input.",
                "- Do not fabricate information or independent diagnoses.",
                "- Include all listed diagnoses when relevant and include ICD-10 codes if identifiable.",
                "- Use the therapist-selected interventions and intervention description for Clinical Interventions used.",
                "- Use the provided risk level and any risk details for Risk Check.",
                "",
                "SECTION GUIDELINES:",
                "- Subjective Complaint / Presenting Problem: summarize client-reported concerns, symptoms, updates, and significant quotes.",
                "- Objective: summarize direct therapist observations including affect, engagement, eye contact, appearance, speech, thought process, and notable in-session behavior.",
                "- Provider Assessment: document clinician assessment of themes, progress, barriers, and clinical formulation grounded in the session input.",
                "- Clinical Interventions used: describe the exact interventions used during the session, drawing from the checked intervention grid and the narrative intervention description.",
                "- Clients Response: capture the client's response to the interventions and overall session.",
                "- Progress/Regression Toward Goals: summarize progress status relative to stated goals.",
                "- Insight and Treatment Recommendations: document insight, recommendations, and clinically relevant next-step guidance.",
                "- Risk Check: summarize the stated risk level and any supporting risk details without adding unsupported risk content.",
                "- Plan / Recommendations / Homework: include next session focus, recommendations, homework, and next appointment details when provided.",
                "",
                build_output_format_instruction(section_config),
            ]
        )

    if template_name:
        section_lines = [f"- {section['title']}: write content appropriate to that section heading." for section in section_config]
        return "\n".join(
            [
                "You are a clinical documentation assistant working with a licensed therapist (LCSW-C).",
                f"Generate a clinical progress note using the custom template '{template_name}'.",
                "",
                "RULES:",
                "- Write in professional clinical language appropriate for a medical record.",
                "- Use third person and stay specific to the provided details.",
                "- Do not fabricate facts or add unsupported content.",
                "- Include all listed diagnoses when clinically relevant and include ICD-10 codes if identifiable.",
                "",
                "SECTION GUIDELINES:",
                *section_lines,
                "",
                build_output_format_instruction(section_config),
            ]
        )

    return "\n".join([SYSTEM_PROMPTS[note_format], "", build_output_format_instruction(section_config)])


def normalize_note_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["note_format"] = str(payload.get("note_format") or "SOAP").upper()
    template_id = payload.get("note_template_id")
    if template_id in ("", None):
        normalized["note_template_id"] = None
    else:
        try:
            normalized["note_template_id"] = int(template_id)
        except (TypeError, ValueError):
            normalized["note_template_id"] = None
    normalized["primary_diagnosis"] = normalize_diagnoses(payload.get("primary_diagnosis"))
    normalized["note_template_name"] = (payload.get("note_template_name") or "").strip() or None
    normalized["interventions_checked"] = [
        str(item).strip()
        for item in (payload.get("interventions_checked") or [])
        if str(item).strip()
    ]
    return normalized


def build_user_prompt(payload: dict[str, Any], note_format: str, template_name: str | None) -> str:
    goals = payload.get("treatment_goals") or "Not specified"
    additional = payload.get("additional_observations") or "None noted"
    homework = payload.get("homework") or "None assigned"
    next_appointment = payload.get("next_appointment") or "To be scheduled"
    risk_details = payload.get("risk_details")
    interventions = ", ".join(payload.get("interventions_checked") or [])
    template_label = template_name or f"{note_format} Standard"

    lines = [
        "SESSION INFORMATION:",
        f"- Note format: {note_format}",
        f"- Note template: {template_label}",
        f"- Client: {payload.get('client_name', '')}",
        (
            f"- Session #{payload.get('session_number') or ''} | {payload.get('session_date') or ''} | "
            f"{payload.get('duration_minutes') or ''} min | {payload.get('session_type') or ''}"
        ),
        f"- Diagnoses: {diagnosis_display(payload.get('primary_diagnosis'))}",
        f"- Treatment Modality: {payload.get('treatment_modality') or 'Not specified'}",
        f"- Current Treatment Goals: {goals}",
        "",
        "CLIENT REPORT:",
        payload.get("client_report") or "",
        "",
        "INTERVENTIONS USED:",
        f"Checked interventions: {interventions or 'Not specified'}",
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
            f"Generate a {template_label} progress note.",
        ]
    )
    return "\n".join(lines)


def parse_generated_sections(section_config: list[dict[str, str]], raw_output: str) -> dict[str, str]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=502, detail=f"Model output was not valid JSON: {error}") from error

    required_keys = [section["key"] for section in section_config]
    missing = [key for key in required_keys if not isinstance(parsed.get(key), str) or not parsed.get(key).strip()]
    if missing:
        raise HTTPException(
            status_code=502,
            detail=f"Model output was missing required sections: {', '.join(missing)}",
        )

    return {key: parsed[key].strip() for key in required_keys}


def build_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    diagnoses = normalize_diagnoses(payload.get("primary_diagnosis"))
    return {
        "client_name": payload.get("client_name"),
        "session_number": payload.get("session_number"),
        "session_date": payload.get("session_date"),
        "duration_minutes": payload.get("duration_minutes"),
        "session_type": payload.get("session_type"),
        "diagnosis": diagnosis_display(diagnoses),
        "diagnoses": diagnoses,
        "modality": payload.get("treatment_modality"),
        "template_name": payload.get("note_template_name"),
    }


def build_section_state(
    section_config: list[dict[str, str]], generated_sections: dict[str, str], edits: dict[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            **section,
            "content": (edits.get(section["key"]) or generated_sections.get(section["key"], "")).strip(),
            "edited": section["key"] in edits,
        }
        for section in section_config
    ]


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
    ]
    if metadata.get("template_name"):
        lines.append(f"Template: {metadata['template_name']}")
    lines.append("")
    for section in sections:
        lines.extend([f"{section['clipboard']}:", section["content"], ""])
    lines.extend(["---", "Generated by Clarity | Draft for clinician review and approval"])
    return "\n".join(lines)


def fetch_note_for_user(db: Any, note_id: int, user_id: str) -> NoteGeneration:
    note = (
        db.query(NoteGeneration)
        .filter(NoteGeneration.id == note_id, NoteGeneration.user_id == user_id)
        .first()
    )
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found.")
    return note


def save_generated_note(
    payload: dict[str, Any],
    note_format: str,
    section_config: list[dict[str, str]],
    sections: dict[str, str],
    generation_time_ms: int,
    user_id: str,
) -> int:
    db = SessionLocal()
    try:
        note = NoteGeneration(
            user_id=user_id,
            client_name=encrypt_field(payload.get("client_name")),
            session_number=payload.get("session_number"),
            session_date=payload.get("session_date"),
            duration_minutes=payload.get("duration_minutes"),
            session_type=payload.get("session_type"),
            note_format=note_format,
            note_template_name=payload.get("note_template_name"),
            section_config_json=json.dumps(section_config),
            primary_diagnosis=encrypt_field(serialize_diagnoses(payload.get("primary_diagnosis"))),
            treatment_modality=payload.get("treatment_modality"),
            input_payload=encrypt_field(json.dumps(payload)),
            ai_output=encrypt_field(json.dumps(sections)),
            generation_time_ms=generation_time_ms,
        )
        db.add(note)
        db.commit()
        db.refresh(note)
        return int(note.id)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def call_openrouter(
    note_format: str,
    payload: dict[str, Any],
    section_config: list[dict[str, str]],
    template_name: str | None,
    api_key: str,
) -> str:
    system_prompt = build_system_prompt(note_format, section_config, template_name)
    user_prompt = build_user_prompt(payload, note_format, template_name)
    return await call_openrouter_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        api_key=api_key,
        max_tokens=2400,
        temperature=0.3,
    )


async def call_openrouter_chat(
    messages: list[dict[str, str]],
    api_key: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    request_body = {
        "model": OPENROUTER_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
        "provider": {"data_collection": "deny"},
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
        return "".join(item if isinstance(item, str) else item.get("text", "") for item in content).strip()
    if isinstance(content, str):
        return content.strip()

    raise HTTPException(status_code=502, detail="OpenRouter returned no note content.")


def validate_audio_upload(audio: UploadFile) -> None:
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix and suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio file type.")
    if audio.content_type and not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be audio.")


async def transcribe_audio_upload(audio: UploadFile) -> str:
    validate_audio_upload(audio)
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio upload was empty.")

    filename = audio.filename or f"session-summary{Path(audio.filename or '.webm').suffix or '.webm'}"
    content_type = audio.content_type or "application/octet-stream"
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    files = {"file": (filename, audio_bytes, content_type)}
    data = {"model": "whisper-1"}

    if openai_key:
        headers = {"Authorization": f"Bearer {openai_key}"}
        url = OPENAI_AUDIO_URL
    elif openrouter_key:
        headers = {"Authorization": f"Bearer {openrouter_key}"}
        url = OPENROUTER_AUDIO_URL
    else:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY or OPENROUTER_API_KEY is required for transcription.",
        )

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(url, headers=headers, data=data, files=files)

    if response.status_code >= 400:
        detail = response.text
        try:
            parsed = response.json()
            detail = parsed.get("error", {}).get("message") or parsed.get("message") or detail
        except json.JSONDecodeError:
            pass
        raise HTTPException(status_code=502, detail=f"Transcription request failed: {detail}")

    body = response.json()
    transcript = str(body.get("text") or "").strip()
    if not transcript:
        raise HTTPException(status_code=502, detail="Transcription returned no text.")

    return transcript


def parse_json_object(raw_output: str) -> dict[str, Any]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=502, detail=f"Model output was not valid JSON: {error}") from error

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="Model output was not a JSON object.")

    return parsed


def normalize_extracted_fields(raw_fields: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    for key, value in raw_fields.items():
        if key not in VOICE_EXTRACTABLE_KEYS:
            continue

        if key == "interventions_checked":
            if not isinstance(value, list):
                continue
            cleaned_list = []
            for item in value:
                text = str(item).strip()
                if text and text in VOICE_ALLOWED_INTERVENTIONS and text not in cleaned_list:
                    cleaned_list.append(text)
            if cleaned_list:
                normalized[key] = cleaned_list
            continue

        if key == "primary_diagnosis":
            diagnoses = normalize_diagnoses(value if isinstance(value, list) else [value])
            if diagnoses:
                normalized[key] = diagnoses
            continue

        text = str(value).strip()
        if not text:
            continue

        allowed_values = VOICE_ENUM_FIELDS.get(key)
        if allowed_values and text not in allowed_values:
            continue

        normalized[key] = text

    return normalized


async def extract_session_fields(
    transcript: str,
    note_format: str,
    note_template_name: str | None,
    api_key: str,
) -> dict[str, Any]:
    user_prompt = "\n".join(
        [
            f"Note format: {note_format or 'SOAP'}",
            f"Note template name: {note_template_name or 'Default (SOAP/DAP/BIRP)'}",
            "",
            "Therapist session summary transcript:",
            transcript,
        ]
    )
    raw_output = await call_openrouter_chat(
        [
            {"role": "system", "content": VOICE_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        api_key=api_key,
        max_tokens=1400,
        temperature=0.1,
    )
    return normalize_extracted_fields(parse_json_object(raw_output))


@app.get("/api/templates")
def list_templates(user_id: str = Depends(get_current_user)) -> dict[str, list[dict[str, Any]]]:
    db = SessionLocal()
    try:
        templates = (
            db.query(NoteTemplate)
            .filter((NoteTemplate.is_builtin == 1) | (NoteTemplate.user_id == user_id))
            .order_by(NoteTemplate.is_builtin.desc(), NoteTemplate.created_at.asc(), NoteTemplate.id.asc())
            .all()
        )
        return {"templates": [serialize_template(template) for template in templates]}
    finally:
        db.close()


@app.post("/api/templates")
def create_template(
    request: TemplateCreateRequest, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Template name is required.")

    sections = normalize_template_sections(request.sections_json)

    db = SessionLocal()
    try:
        template = NoteTemplate(
            user_id=user_id,
            name=name,
            sections_json=json.dumps(sections),
            is_builtin=0,
        )
        db.add(template)
        db.commit()
        db.refresh(template)
        return serialize_template(template)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.post("/api/transcribe-and-extract")
async def transcribe_and_extract(
    audio: UploadFile = File(...),
    note_format: str = Form(default="SOAP"),
    note_template_name: str = Form(default=""),
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is required for field extraction.")

    transcript = await transcribe_audio_upload(audio)
    extracted_fields = await extract_session_fields(
        transcript,
        note_format=(note_format or "SOAP").upper(),
        note_template_name=note_template_name.strip() or None,
        api_key=openrouter_key,
    )

    return {
        "transcript": transcript,
        "extracted_fields": extracted_fields,
        "fields_populated": len(extracted_fields),
    }


@app.get("/api/user/data")
def get_user_data(user_id: str = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        notes = (
            db.query(NoteGeneration)
            .filter(NoteGeneration.user_id == user_id)
            .order_by(NoteGeneration.created_at.desc())
            .all()
        )

        records = []
        for note in notes:
            input_payload = parse_json_field(decrypt_field(note.input_payload), default={})
            if isinstance(input_payload, dict):
                input_payload["primary_diagnosis"] = normalize_diagnoses(input_payload.get("primary_diagnosis"))
            records.append(
                {
                    "id": note.id,
                    "created_at": note.created_at,
                    "user_id": note.user_id,
                    "client_name": decrypt_field(note.client_name),
                    "session_number": note.session_number,
                    "session_date": note.session_date,
                    "duration_minutes": note.duration_minutes,
                    "session_type": note.session_type,
                    "note_format": note.note_format,
                    "note_template_name": note.note_template_name,
                    "primary_diagnosis": parse_stored_diagnoses(note.primary_diagnosis),
                    "treatment_modality": note.treatment_modality,
                    "input_payload": input_payload,
                    "ai_output": parse_json_field(decrypt_field(note.ai_output), default={}),
                    "ai_model": note.ai_model,
                    "generation_time_ms": note.generation_time_ms,
                    "edits": parse_json_field(decrypt_field(note.edits), default={}),
                    "final_output": decrypt_field(note.final_output),
                    "copied_at": note.copied_at,
                    "feedback_rating": note.feedback_rating,
                    "feedback_notes": decrypt_field(note.feedback_notes),
                }
            )

        return {
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "service": "Clarity by Bearing Digital",
            "record_count": len(records),
            "records": records,
        }
    finally:
        db.close()


@app.put("/api/user/data/correct")
def correct_user_data(
    request: CorrectDataRequest, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    if request.field not in _CORRECTABLE_FIELDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Field '{request.field}' cannot be corrected via this endpoint. "
                f"Allowed fields: {sorted(_CORRECTABLE_FIELDS)}"
            ),
        )

    db = SessionLocal()
    try:
        note = fetch_note_for_user(db, request.note_id, user_id)
        if request.field == "primary_diagnosis":
            setattr(note, request.field, encrypt_field(json.dumps([request.value.strip()])))
        else:
            setattr(note, request.field, encrypt_field(request.value.strip()))
        note.user_id = note.user_id or user_id
        db.commit()
        return {"note_id": request.note_id, "field": request.field, "corrected": True}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.delete("/api/user/data")
def delete_user_data(user_id: str = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        deleted_count = db.query(NoteGeneration).filter(NoteGeneration.user_id == user_id).count()
        (
            db.query(NoteGeneration)
            .filter(NoteGeneration.user_id == user_id)
            .update({NoteGeneration.user_id: None}, synchronize_session=False)
        )
        (
            db.query(NoteGeneration)
            .filter(NoteGeneration.user_id.is_(None), NoteGeneration.client_name == "[purged]")
            .delete(synchronize_session=False)
        )
        db.query(NoteTemplate).filter(NoteTemplate.user_id == user_id).delete(synchronize_session=False)
        db.query(ActiveSession).filter(ActiveSession.user_id == user_id).delete(synchronize_session=False)
        db.query(SSOAccount).filter(SSOAccount.user_id == user_id).delete(synchronize_session=False)
        db.query(User).filter(User.id == user_id).delete(synchronize_session=False)
        db.commit()
        return {"deleted": True, "records_deleted": deleted_count}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.post("/api/generate-note")
async def generate_note(
    request: NoteRequest, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    payload = normalize_note_payload(request.model_dump())
    note_format = payload["note_format"]
    if note_format not in SECTION_CONFIG:
        raise HTTPException(status_code=400, detail="Unsupported note format.")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured.")

    db = SessionLocal()
    try:
        template = get_template_for_user(
            db,
            user_id,
            payload.get("note_template_id"),
            payload.get("note_template_name"),
        )
    finally:
        db.close()

    if template:
        payload["note_template_name"] = template["name"]

    section_config = resolve_section_config(note_format, template=template)
    started_at = time.perf_counter()
    raw_output = await call_openrouter(note_format, payload, section_config, payload.get("note_template_name"), api_key)
    generation_time_ms = int((time.perf_counter() - started_at) * 1000)

    sections = parse_generated_sections(section_config, raw_output)
    metadata = build_metadata(payload)
    response_payload = {
        "note_id": save_generated_note(payload, note_format, section_config, sections, generation_time_ms, user_id),
        "format": note_format,
        "template_name": payload.get("note_template_name"),
        "metadata": metadata,
        "sections": build_section_state(section_config, sections, {}),
    }
    return response_payload


@app.post("/api/notes/{note_id}/edits")
def save_note_edits(
    note_id: int, request: EditRequest, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    db = SessionLocal()
    try:
        note = fetch_note_for_user(db, note_id, user_id)
        section_config = resolve_section_config(note.note_format, note.section_config_json)
        current_edits = parse_json_field(decrypt_field(note.edits), default={})
        current_edits.update({key: value.strip() for key, value in request.edits.items() if value.strip()})

        input_payload = parse_json_field(decrypt_field(note.input_payload), default={})
        if isinstance(input_payload, dict):
            input_payload["primary_diagnosis"] = normalize_diagnoses(input_payload.get("primary_diagnosis"))
            input_payload["note_template_name"] = note.note_template_name

        final_output = build_plaintext_output(
            build_metadata(input_payload),
            build_section_state(
                section_config,
                parse_json_field(decrypt_field(note.ai_output), default={}),
                current_edits,
            ),
        )
        note.edits = encrypt_field(json.dumps(current_edits))
        note.final_output = encrypt_field(final_output)
        note.user_id = note.user_id or user_id
        db.commit()
        return {"note_id": note_id, "edits": current_edits}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.post("/api/notes/{note_id}/copied")
def mark_note_copied(
    note_id: int, request: CopyRequest, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    db = SessionLocal()
    try:
        note = fetch_note_for_user(db, note_id, user_id)
        note.final_output = encrypt_field(request.final_output)
        note.copied_at = datetime.now(timezone.utc)
        note.user_id = note.user_id or user_id
        db.commit()
        return {"note_id": note_id, "copied": True}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Analytics helpers — no PHI stored
# ---------------------------------------------------------------------------

ANALYTICS_TRACKED_FIELDS = [
    "treatment_goals",
    "interventions_description",
    "additional_observations",
    "client_response",
    "progress",
    "risk_level",
    "risk_details",
    "plan_next_session",
    "homework",
    "next_appointment",
    "client_report",
    "affect",
    "engagement",
    "appearance",
    "speech",
    "thought_process",
]


def _extract_analytics_row(note: NoteGeneration, input_payload: dict[str, Any]) -> dict[str, Any]:
    """Return an anonymized analytics dict suitable for inserting into note_analytics."""
    field_fill = {
        field: 1 if input_payload.get(field) else 0
        for field in ANALYTICS_TRACKED_FIELDS
    }
    edits = parse_json_field(decrypt_field(note.edits), default={})
    final_output = decrypt_field(note.final_output)
    ai_output_raw = decrypt_field(note.ai_output)
    diagnoses = normalize_diagnoses(input_payload.get("primary_diagnosis"))
    interventions_checked = input_payload.get("interventions_checked") or []
    section_config = resolve_section_config(note.note_format, note.section_config_json)
    return {
        "note_id": note.id,
        "note_format": note.note_format,
        "note_template_name": note.note_template_name,
        "session_type": note.session_type,
        "duration_minutes": note.duration_minutes,
        "treatment_modality": note.treatment_modality,
        "diagnosis_count": len(diagnoses),
        "section_count": len(section_config),
        "generation_time_ms": note.generation_time_ms,
        "ai_output_char_count": len(ai_output_raw) if ai_output_raw else 0,
        "final_output_char_count": len(final_output) if final_output else 0,
        "was_edited": 1 if edits else 0,
        "edited_section_count": len(edits) if isinstance(edits, dict) else 0,
        "was_copied": 1 if note.copied_at else 0,
        "feedback_rating": note.feedback_rating,
        "risk_level": input_payload.get("risk_level"),
        "field_fill_json": json.dumps(field_fill),
        "interventions_count": len(interventions_checked),
    }


@app.delete("/api/notes/{note_id}/purge")
def purge_note_session(note_id: int, user_id: str = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        note = fetch_note_for_user(db, note_id, user_id)
        input_payload = parse_json_field(decrypt_field(note.input_payload), default={})
        if isinstance(input_payload, dict) and input_payload:
            db.add(NoteAnalytic(**_extract_analytics_row(note, input_payload)))
        note.input_payload = "{}"
        note.ai_output = "{}"
        note.client_name = "[purged]"
        note.primary_diagnosis = None
        note.final_output = None
        note.user_id = note.user_id or user_id
        db.commit()
        return {"note_id": note_id, "purged": True}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.post("/api/notes/{note_id}/feedback")
def save_feedback(
    note_id: int, request: FeedbackRequest, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    db = SessionLocal()
    try:
        note = fetch_note_for_user(db, note_id, user_id)
        note.feedback_rating = request.feedback_rating
        note.feedback_notes = encrypt_field(request.feedback_notes.strip())
        note.user_id = note.user_id or user_id
        db.commit()
        return {"note_id": note_id, "feedback_rating": request.feedback_rating}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.get("/api/notes")
def list_notes(user_id: str = Depends(get_current_user)) -> dict[str, list[dict[str, Any]]]:
    db = SessionLocal()
    try:
        note_rows = (
            db.query(NoteGeneration)
            .filter(NoteGeneration.user_id == user_id)
            .order_by(NoteGeneration.created_at.desc(), NoteGeneration.id.desc())
            .all()
        )

        notes = []
        for note in note_rows:
            edits = parse_json_field(decrypt_field(note.edits), default={})
            notes.append(
                {
                    "id": note.id,
                    "created_at": note.created_at,
                    "user_id": note.user_id,
                    "client_name": decrypt_field(note.client_name),
                    "session_number": note.session_number,
                    "session_date": note.session_date,
                    "duration_minutes": note.duration_minutes,
                    "session_type": note.session_type,
                    "note_format": note.note_format,
                    "note_template_name": note.note_template_name,
                    "primary_diagnosis": parse_stored_diagnoses(note.primary_diagnosis),
                    "treatment_modality": note.treatment_modality,
                    "generation_time_ms": note.generation_time_ms,
                    "copied_at": note.copied_at,
                    "feedback_rating": note.feedback_rating,
                    "edited_section_count": len(edits),
                }
            )

        return {"notes": notes}
    finally:
        db.close()


@app.get("/api/notes/{note_id}")
def get_note(note_id: int, user_id: str = Depends(get_current_user)) -> dict[str, Any]:
    db = SessionLocal()
    try:
        note = fetch_note_for_user(db, note_id, user_id)

        input_payload = parse_json_field(decrypt_field(note.input_payload), default={})
        if isinstance(input_payload, dict):
            input_payload["primary_diagnosis"] = normalize_diagnoses(input_payload.get("primary_diagnosis"))
            input_payload["note_template_name"] = note.note_template_name
        ai_output = parse_json_field(decrypt_field(note.ai_output), default={})
        edits = parse_json_field(decrypt_field(note.edits), default={})
        section_config = resolve_section_config(note.note_format, note.section_config_json)
        metadata = build_metadata(input_payload)
        sections = build_section_state(section_config, ai_output, edits)
        final_output = decrypt_field(note.final_output)

        return {
            "id": note.id,
            "created_at": note.created_at,
            "format": note.note_format,
            "template_name": note.note_template_name,
            "metadata": metadata,
            "input_payload": input_payload,
            "sections": sections,
            "analysis": {
                "ai_model": note.ai_model,
                "generation_time_ms": note.generation_time_ms,
                "edited_sections": sorted(edits.keys()),
                "edited_section_count": len(edits),
                "copied": bool(note.copied_at),
                "copied_at": note.copied_at,
                "feedback_rating": note.feedback_rating,
                "feedback_notes": decrypt_field(note.feedback_notes),
            },
            "final_output": final_output or build_plaintext_output(metadata, sections),
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Analytics summary — no auth required, no PHI returned
# ---------------------------------------------------------------------------

@app.get("/api/analytics/summary")
def get_analytics_summary() -> dict[str, Any]:
    """Aggregate anonymized stats from note_analytics. No PHI. No auth required."""
    db = SessionLocal()
    try:
        total = db.execute(text("SELECT COUNT(*) FROM note_analytics")).scalar() or 0

        by_format = {
            (row[0] or "unknown"): row[1]
            for row in db.execute(
                text("SELECT note_format, COUNT(*) FROM note_analytics GROUP BY note_format")
            ).all()
        }
        by_template = {
            (row[0] or "standard"): row[1]
            for row in db.execute(
                text("SELECT note_template_name, COUNT(*) FROM note_analytics GROUP BY note_template_name")
            ).all()
        }

        avgs = db.execute(
            text(
                """
                SELECT
                    AVG(generation_time_ms),
                    AVG(ai_output_char_count),
                    AVG(final_output_char_count),
                    AVG(was_copied) * 100.0,
                    AVG(was_edited) * 100.0,
                    AVG(edited_section_count),
                    AVG(diagnosis_count),
                    AVG(interventions_count)
                FROM note_analytics
                """
            )
        ).first()

        def top3(col: str) -> list[dict[str, Any]]:
            rows = db.execute(
                text(
                    f"SELECT {col}, COUNT(*) AS n FROM note_analytics "
                    f"WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY n DESC LIMIT 3"
                )
            ).all()
            return [{"value": row[0], "count": row[1]} for row in rows]

        risk_dist = {
            (row[0] or "not_specified"): row[1]
            for row in db.execute(
                text("SELECT risk_level, COUNT(*) FROM note_analytics GROUP BY risk_level")
            ).all()
        }

        fill_rows = db.execute(
            text("SELECT field_fill_json FROM note_analytics WHERE field_fill_json IS NOT NULL")
        ).all()
        field_fill_rates: dict[str, float] = {}
        if fill_rows:
            totals: dict[str, int] = {}
            counts: dict[str, int] = {}
            for (field_fill_json,) in fill_rows:
                try:
                    field_data = json.loads(field_fill_json)
                    for key, value in field_data.items():
                        totals[key] = totals.get(key, 0) + int(value)
                        counts[key] = counts.get(key, 0) + 1
                except Exception:
                    pass
            field_fill_rates = {
                key: round(totals[key] / counts[key] * 100, 1)
                for key in totals
                if counts[key] > 0
            }

        def r1(value: Any) -> Any:
            return round(value, 1) if value is not None else None

        return {
            "total_notes_purged": int(total),
            "by_format": by_format,
            "by_template": by_template,
            "avg_generation_time_ms": r1(avgs[0]) if avgs else None,
            "avg_ai_output_chars": r1(avgs[1]) if avgs else None,
            "avg_final_output_chars": r1(avgs[2]) if avgs else None,
            "copy_rate_pct": r1(avgs[3]) if avgs else None,
            "edit_rate_pct": r1(avgs[4]) if avgs else None,
            "avg_edited_sections": r1(avgs[5]) if avgs else None,
            "avg_diagnosis_count": r1(avgs[6]) if avgs else None,
            "avg_interventions_count": r1(avgs[7]) if avgs else None,
            "top_session_types": top3("session_type"),
            "top_treatment_modalities": top3("treatment_modality"),
            "top_durations": top3("duration_minutes"),
            "risk_level_distribution": risk_dist,
            "field_fill_rates_pct": field_fill_rates,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Take-home summary generator
# ---------------------------------------------------------------------------

TAKEHOME_SYSTEM_PROMPT = """You are a compassionate therapy support assistant. Transform clinical session notes into warm, accessible take-home summaries for clients.

RULES:
- Write in second person ("Today we worked on..." / "In our session today...")
- Use plain, everyday language — absolutely no clinical jargon
- Keep it encouraging and strengths-focused
- Length: 1–2 short paragraphs (80–150 words total)
- Do NOT include diagnostic codes, risk details, or clinical labels
- Transform clinical language into natural, human terms

Examples of transformations:
- "Utilized cognitive restructuring to address maladaptive thought patterns" → "Today we practiced catching 'thought traps' and replacing them with more balanced perspectives."
- "Client demonstrated improved behavioral activation" → "You showed real progress in reconnecting with activities that bring you energy."
- "Psychoeducation regarding anxiety response cycle" → "We talked about how anxiety works in your body and mind, and why it feels the way it does."
- "Safety planning was reviewed" → "We checked in on your safety plan and made sure you feel supported."
"""


@app.post("/api/generate-takehome")
async def generate_takehome(
    request: TakehomeRequest, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured.")

    note_text = request.note_text.strip()
    if not note_text:
        raise HTTPException(status_code=400, detail="Note text is required.")

    user_prompt = f"Transform this clinical session note into a warm client take-home summary:\n\n{note_text}"

    summary = await call_openrouter_chat(
        [
            {"role": "system", "content": TAKEHOME_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        api_key=api_key,
        max_tokens=400,
        temperature=0.5,
    )

    return {"summary": summary}
