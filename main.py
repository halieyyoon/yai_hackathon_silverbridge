"""
Silver Bridge (실버브릿지) - Twilio Voice + OpenAI backend.

One phone call now drives the full welfare-application workflow:

  1. Intake call (Twilio /voice/* or browser /test-call):
     the caller gives their name and birth date, is looked up in the
     municipality citizen database, then states their difficulty. Both
     interfaces share the same state machine -- conversation/intake.py.

  2. Once the call ends, approval/pipeline.py runs synchronously: fetch
     candidate welfare programs (government_api), have OpenAI recommend up
     to two of them (recommendation), generate application documents
     (application_generator), and open the case for officer review
     (approval.store), best-effort emailing the officer.

  3. An officer reviews and approves/rejects each generated application on
     a *separate* dashboard page, /officer -- the citizen's own page never
     sees recommendations, application documents, or approval controls.

  4. A second, short phone call (again shared between Twilio and the
     browser simulator) tells the citizen the outcome -- see
     conversation/notification.py and /api/notification-call/start. The
     citizen page polls a reduced-detail status endpoint and offers to
     "take the call" once the officer has decided.

/test-call (citizen) and /officer (municipality staff) are two intentionally
separate pages hitting the same backend -- neither has real authentication,
so this is a UI/UX separation for the demo, not an access-control boundary.
Both are local/ngrok development tools that let you exercise the whole
pipeline without a programmable Twilio number.

All credentials are loaded from environment variables (see .env.example).
Nothing sensitive is hardcoded.
"""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import Gather, VoiceResponse

from approval.store import Case, WorkflowStatus, decide_application, get_case, list_cases
from conversation.intake import GREETING, clear_session, process_intake_turn, start_session
from conversation.notification import build_notification_message

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("silver_bridge")

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Configuration (all from environment -- see .env.example)
# --------------------------------------------------------------------------

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
VALIDATE_TWILIO_SIGNATURE = os.getenv("VALIDATE_TWILIO_SIGNATURE", "true").lower() == "true"

REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in real values."
        )
    yield


app = FastAPI(title="Silver Bridge", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN) if TWILIO_AUTH_TOKEN else None

TTS_VOICE = "Polly.Seoyeon"  # Amazon Polly Korean voice, supported by Twilio <Say>


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """Last-resort safety net so a bug never leaks a stack trace to a caller
    (Twilio or the browser). Specific routes should catch what they can
    handle meaningfully before this ever fires."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "서버에 문제가 발생했어요. 잠시 후 다시 시도해 주세요."})


def twiml_response(vr: VoiceResponse) -> Response:
    return Response(content=str(vr), media_type="application/xml")


def build_request_url(request: Request) -> str:
    """Reconstruct the exact URL Twilio called, for signature validation.

    Twilio signs requests using the URL configured in the console (the
    public ngrok/production URL), not the internal URL Uvicorn sees. Setting
    PUBLIC_BASE_URL makes validation deterministic; otherwise we fall back to
    the incoming request, honoring X-Forwarded-Proto if a proxy set it.
    """
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}{request.url.path}"
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{scheme}://{request.url.netloc}{request.url.path}"


async def verify_twilio_signature(request: Request) -> Dict[str, str]:
    """Dependency: parses the Twilio form body and validates its signature."""
    form = await request.form()
    params = dict(form)

    if VALIDATE_TWILIO_SIGNATURE:
        if twilio_validator is None:
            raise HTTPException(status_code=500, detail="Twilio auth token not configured")
        signature = request.headers.get("X-Twilio-Signature", "")
        url = build_request_url(request)
        if not twilio_validator.validate(url, params, signature):
            logger.warning("Rejected request with invalid Twilio signature for %s", url)
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    return params


def _speech_gather() -> Gather:
    return Gather(
        input="speech",
        action="/voice/process-speech",
        method="POST",
        language="ko-KR",
        speech_timeout="auto",
        timeout=6,
    )


# --------------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------------


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "silver-bridge"}


# --------------------------------------------------------------------------
# Twilio voice routes -- the real-phone-call interface, sharing
# conversation.intake with the browser simulator below.
# --------------------------------------------------------------------------


@app.post("/voice/incoming")
async def voice_incoming(form: Dict[str, str] = Depends(verify_twilio_signature)):
    call_sid = form.get("CallSid", "")
    start_session(call_sid)  # fresh intake state for this call

    vr = VoiceResponse()
    gather = _speech_gather()
    gather.say(GREETING, voice=TTS_VOICE, language="ko-KR")
    vr.append(gather)

    # Reached only if Gather finishes with no result and no action fallback
    vr.say("죄송합니다, 음성이 확인되지 않았어요. 다시 전화 주시면 도와드릴게요.", voice=TTS_VOICE, language="ko-KR")
    vr.hangup()
    return twiml_response(vr)


@app.post("/voice/process-speech")
async def voice_process_speech(form: Dict[str, str] = Depends(verify_twilio_signature)):
    call_sid = form.get("CallSid", "")
    speech_result = form.get("SpeechResult") or ""
    vr = VoiceResponse()

    result = await process_intake_turn(call_sid, speech_result)
    vr.say(result.reply, voice=TTS_VOICE, language="ko-KR")

    if result.ended:
        vr.hangup()
    else:
        vr.append(_speech_gather())
        vr.say("다음에 또 편하게 전화 주세요. 안녕히 계세요.", voice=TTS_VOICE, language="ko-KR")
        vr.hangup()

    return twiml_response(vr)


@app.post("/voice/status")
async def voice_status(form: Dict[str, str] = Depends(verify_twilio_signature)):
    """Optional: configure this as the phone number's "Call status changes"
    webhook in the Twilio console so abandoned calls free memory promptly."""
    call_status = form.get("CallStatus", "")
    if call_status in {"completed", "busy", "failed", "no-answer", "canceled"}:
        clear_session(form.get("CallSid", ""))
    return Response(status_code=204)


# --------------------------------------------------------------------------
# Browser voice simulator
#
# Local development only: lets you test the real intake engine (real
# microphone input, real Korean speech recognition, real OpenAI replies,
# real speech synthesis) without owning a Twilio programmable number, and
# drive the rest of the workflow (recommendations, applications, officer
# approval, the result call) from the same page. No Twilio signature
# validation applies here since Twilio never calls these routes -- do not
# expose this page or these endpoints on the public internet without adding
# your own authentication first.
# --------------------------------------------------------------------------


class SessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Browser-generated UUID for this call")


class ConversationRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., description="Final speech-to-text transcript for this turn; may be blank")


class ConversationResponse(BaseModel):
    reply: str
    ended: bool
    reason: Optional[str] = None
    case_id: Optional[str] = None


class StartResponse(BaseModel):
    greeting: str
    ended: bool = False


class EndResponse(BaseModel):
    ended: bool = True


class RecommendationOut(BaseModel):
    program_id: str
    program_name: str
    agency: str
    why_eligible: str
    expected_benefit: str
    elderly_friendly_explanation: str
    reasons: List[str]


class ApplicationOut(BaseModel):
    application_id: str
    program_id: str
    program_name: str
    status: str
    document_text: str
    created_at: str


class CaseSummaryOut(BaseModel):
    case_id: str
    citizen_name: str
    status: str
    decision: Optional[str] = None
    created_at: str
    updated_at: str


class CaseDetailOut(CaseSummaryOut):
    citizen: Dict[str, Any]
    difficulty_summary: str
    recommendations: List[RecommendationOut]
    applications: List[ApplicationOut]
    error_reason: Optional[str] = None


class DecisionRequest(BaseModel):
    decision: str = Field(..., pattern="^(approved|rejected)$")


class NotificationCallRequest(BaseModel):
    case_id: str = Field(..., min_length=1)


class NotificationCallResponse(BaseModel):
    greeting: str
    ended: bool = True


class SimpleStatusResponse(BaseModel):
    """Citizen-facing view of a case: just enough to drive a 3-stage
    progress indicator. Deliberately excludes citizen profile,
    recommendations, and application documents -- those are officer-only,
    see CaseDetailOut."""

    case_id: str
    stage: str
    decision_ready: bool


_SIMPLE_STAGE_BY_STATUS = {
    WorkflowStatus.CALL_COMPLETE: "상담 완료",
    WorkflowStatus.RECOMMENDED: "상담 완료",
    WorkflowStatus.APPLICATIONS_GENERATED: "상담 완료",
    WorkflowStatus.PENDING_OFFICER_REVIEW: "승인 검토 중",
    WorkflowStatus.APPROVED: "승인 완료",
    WorkflowStatus.REJECTED: "지원 불가",
}


def _simple_stage(status: str) -> str:
    return _SIMPLE_STAGE_BY_STATUS.get(status, "확인 필요")


def _case_to_summary(case: Case) -> CaseSummaryOut:
    return CaseSummaryOut(
        case_id=case.case_id,
        citizen_name=case.citizen.get("name", ""),
        status=case.status,
        decision=case.decision,
        created_at=case.created_at,
        updated_at=case.updated_at,
    )


def _case_to_detail(case: Case) -> CaseDetailOut:
    return CaseDetailOut(
        **_case_to_summary(case).model_dump(),
        citizen=case.citizen,
        difficulty_summary=case.difficulty_summary,
        recommendations=[RecommendationOut(**asdict(rec)) for rec in case.recommendations],
        applications=[
            ApplicationOut(
                application_id=app.application_id,
                program_id=app.program_id,
                program_name=app.program_name,
                status=app.status,
                document_text=app.document_text,
                created_at=app.created_at,
            )
            for app in case.applications
        ],
        error_reason=case.error_reason,
    )


@app.get("/test-call", response_class=HTMLResponse)
async def test_call_page():
    """Citizen phone interface: call controls, transcript, and a simple
    3-stage progress indicator only. No recommendations, application
    documents, or approval controls -- see /officer for those."""
    html_path = BASE_DIR / "templates" / "test_call.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/officer", response_class=HTMLResponse)
async def officer_dashboard_page():
    """Municipality officer dashboard: full case detail, recommendation
    reasons, generated application documents, and approve/reject controls.
    No phone/call UI at all -- that's the citizen page's job."""
    html_path = BASE_DIR / "templates" / "officer_dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/conversation/start", response_model=StartResponse)
async def start_browser_session(payload: SessionRequest):
    session_id = payload.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id는 비어 있을 수 없어요.")

    start_session(session_id)  # idempotent: also handles a duplicate/restarted session_id
    return StartResponse(greeting=GREETING, ended=False)


@app.post("/api/conversation", response_model=ConversationResponse)
async def continue_browser_session(payload: ConversationRequest):
    session_id = payload.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id는 비어 있을 수 없어요.")

    result = await process_intake_turn(session_id, payload.message)
    return ConversationResponse(reply=result.reply, ended=result.ended, reason=result.reason, case_id=result.case_id)


@app.post("/api/conversation/end", response_model=EndResponse)
async def end_browser_session(payload: SessionRequest):
    session_id = payload.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id는 비어 있을 수 없어요.")

    clear_session(session_id)
    return EndResponse(ended=True)


@app.get("/api/cases", response_model=List[CaseSummaryOut])
async def list_cases_route():
    return [_case_to_summary(case) for case in list_cases()]


@app.get("/api/cases/{case_id}", response_model=CaseDetailOut)
async def get_case_route(case_id: str):
    case = get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="사례를 찾을 수 없어요.")
    return _case_to_detail(case)


@app.post("/api/cases/{case_id}/applications/{application_id}/decide", response_model=CaseDetailOut)
async def decide_application_route(case_id: str, application_id: str, payload: DecisionRequest):
    if get_case(case_id) is None:
        raise HTTPException(status_code=404, detail="사례를 찾을 수 없어요.")
    try:
        case = decide_application(case_id, application_id, payload.decision)
    except KeyError:
        raise HTTPException(status_code=404, detail="신청서를 찾을 수 없어요.")
    return _case_to_detail(case)


@app.get("/api/cases/{case_id}/simple-status", response_model=SimpleStatusResponse)
async def get_case_simple_status(case_id: str):
    """Citizen-safe polling endpoint -- the /test-call page uses this (never
    the full /api/cases/{case_id}) to drive its 3-stage progress display."""
    case = get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="사례를 찾을 수 없어요.")
    return SimpleStatusResponse(
        case_id=case.case_id,
        stage=_simple_stage(case.status),
        decision_ready=case.decision is not None,
    )


@app.post("/api/notification-call/start", response_model=NotificationCallResponse)
async def start_notification_call(payload: NotificationCallRequest):
    case_id = payload.case_id.strip()
    case = get_case(case_id) if case_id else None
    if case is None:
        raise HTTPException(status_code=404, detail="사례를 찾을 수 없어요.")

    message = build_notification_message(case)
    return NotificationCallResponse(greeting=message, ended=True)
