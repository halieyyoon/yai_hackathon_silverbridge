"""
Smoke tests for Silver Bridge. OpenAI's recommendation call is mocked --
these never make a real API call, so they're safe and free to run
repeatedly. The welfare-program lookup runs in its real "mock" catalog
mode (no external HTTP either).
"""

import pytest
from fastapi.testclient import TestClient

import approval.pipeline as pipeline
import main
from recommendation.recommender import Recommendation


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(autouse=True)
def mock_recommendations(monkeypatch):
    async def fake_recommend_programs(citizen, difficulty_summary, candidates):
        if not candidates:
            return []
        top = candidates[0]
        return [
            Recommendation(
                program_id=top.program_id,
                program_name=top.name,
                agency=top.agency,
                why_eligible="테스트용 적합 이유입니다.",
                expected_benefit=top.benefit,
                elderly_friendly_explanation="테스트용 쉬운 설명입니다.",
                reasons=["65세 이상", "저소득"],
            )
        ]

    monkeypatch.setattr(pipeline, "recommend_programs", fake_recommend_programs)


def run_intake_call(client, session_id, name_utterance, birth_date_utterance, difficulty_utterance):
    client.post("/api/conversation/start", json={"session_id": session_id})
    client.post("/api/conversation", json={"session_id": session_id, "message": name_utterance})
    client.post("/api/conversation", json={"session_id": session_id, "message": birth_date_utterance})
    return client.post("/api/conversation", json={"session_id": session_id, "message": difficulty_utterance})


def test_health_check(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_test_call_page_serves_html(client):
    resp = client.get("/test-call")
    assert resp.status_code == 200
    assert "통화 시작" in resp.text


def test_citizen_page_has_no_officer_controls(client):
    # The citizen page must never render approval buttons or "view
    # application document" controls -- those belong on /officer only.
    resp = client.get("/test-call")
    assert resp.status_code == 200
    for officer_only_text in ["승인</button>", "반려</button>", "신청서 내용 보기", "case-action"]:
        assert officer_only_text not in resp.text


def test_officer_page_serves_html(client):
    resp = client.get("/officer")
    assert resp.status_code == 200
    assert "공무원" in resp.text
    assert "사례 새로고침" in resp.text


def test_officer_page_has_no_call_controls(client):
    resp = client.get("/officer")
    for citizen_only_text in ["통화 시작", "통화 종료", "마이크"]:
        assert citizen_only_text not in resp.text


def test_static_assets_load(client):
    assert client.get("/static/test_call.js").status_code == 200
    assert client.get("/static/test_call.css").status_code == 200
    assert client.get("/static/officer_dashboard.js").status_code == 200
    assert client.get("/static/officer_dashboard.css").status_code == 200


def test_twilio_routes_still_exist_and_require_signature(client):
    # Unsigned request must be rejected (403), not 404 -- confirms the route
    # is still registered and still signature-protected.
    assert client.post("/voice/incoming", data={"CallSid": "CAtest"}).status_code == 403
    assert client.post(
        "/voice/process-speech", data={"CallSid": "CAtest", "SpeechResult": "김영수입니다"}
    ).status_code == 403
    assert client.post(
        "/voice/status", data={"CallSid": "CAtest", "CallStatus": "completed"}
    ).status_code == 403


def test_no_credentials_leak_to_responses(client):
    texts = [client.get("/test-call").text, client.get("/officer").text]
    resp = run_intake_call(client, "leak-check", "김영수입니다", "1952년 3월 14일입니다", "월세가 부담돼요")
    texts.append(resp.text)
    for text in texts:
        for secret_marker in ["OPENAI_API_KEY", "TWILIO_AUTH_TOKEN", "sk-proj", "SMTP_PASSWORD"]:
            assert secret_marker not in text


def test_intake_flow_identifies_citizen_and_creates_pending_case(client):
    session_id = "intake-flow-1"

    start = client.post("/api/conversation/start", json={"session_id": session_id})
    assert start.status_code == 200
    assert start.json()["greeting"]

    turn1 = client.post("/api/conversation", json={"session_id": session_id, "message": "김영수입니다"})
    assert turn1.status_code == 200
    assert turn1.json()["ended"] is False

    turn2 = client.post("/api/conversation", json={"session_id": session_id, "message": "1952년 3월 14일입니다"})
    assert turn2.status_code == 200
    assert turn2.json()["ended"] is False

    turn3 = client.post("/api/conversation", json={"session_id": session_id, "message": "월세가 너무 부담돼요"})
    assert turn3.status_code == 200
    body = turn3.json()
    assert body["ended"] is True
    case_id = body["case_id"]
    assert case_id

    detail = client.get(f"/api/cases/{case_id}")
    assert detail.status_code == 200
    case = detail.json()
    assert case["citizen_name"] == "김영수"
    assert case["status"] == "주무관 검토 중"
    assert 1 <= len(case["applications"]) <= 2
    assert 1 <= len(case["recommendations"]) <= 2
    for application in case["applications"]:
        assert application["status"] == "pending"

    listing = client.get("/api/cases")
    assert listing.status_code == 200
    assert any(c["case_id"] == case_id for c in listing.json())


def test_citizen_not_found_ends_call_without_creating_case(client):
    # Deliberately stops after the birth-date turn -- that's the turn where
    # the "not found" branch fires, so a 3rd turn would just start a new
    # session rather than exercising this path.
    session_id = "unknown-citizen"
    client.post("/api/conversation/start", json={"session_id": session_id})
    client.post("/api/conversation", json={"session_id": session_id, "message": "없는사람입니다"})
    resp = client.post("/api/conversation", json={"session_id": session_id, "message": "1990년 1월 1일입니다"})
    body = resp.json()
    assert body["ended"] is True
    assert body["reason"] == "citizen_not_found"
    assert body["case_id"] is None


def test_goodbye_keyword_ends_intake_immediately(client):
    session_id = "goodbye-during-intake"
    client.post("/api/conversation/start", json={"session_id": session_id})
    resp = client.post("/api/conversation", json={"session_id": session_id, "message": "괜찮아요 됐어요"})
    body = resp.json()
    assert body["ended"] is True
    assert body["reason"] == "goodbye"


def test_empty_message_is_handled_gracefully_not_as_error(client):
    session_id = "empty-message-session"
    client.post("/api/conversation/start", json={"session_id": session_id})
    resp = client.post("/api/conversation", json={"session_id": session_id, "message": "   "})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ended"] is False
    assert body["reply"]


def test_officer_approve_sets_case_approved_and_enables_notification_call(client):
    resp = run_intake_call(client, "approve-flow", "박순자입니다", "1948년 11월 2일입니다", "병원비가 너무 많이 나와요")
    case_id = resp.json()["case_id"]
    case = client.get(f"/api/cases/{case_id}").json()
    application_id = case["applications"][0]["application_id"]

    decide = client.post(
        f"/api/cases/{case_id}/applications/{application_id}/decide",
        json={"decision": "approved"},
    )
    assert decide.status_code == 200
    updated = decide.json()
    assert updated["status"] == "승인 완료"
    assert updated["decision"] == "approved"

    notify = client.post("/api/notification-call/start", json={"case_id": case_id})
    assert notify.status_code == 200
    assert notify.json()["greeting"] == "지원 대상으로 선정되셨습니다. 등록된 계좌를 확인해 주세요."
    assert notify.json()["ended"] is True


def test_officer_reject_all_applications_marks_case_rejected(client):
    resp = run_intake_call(client, "reject-flow", "이만호입니다", "1945년 7월 22일입니다", "난방비가 너무 부담돼요")
    case_id = resp.json()["case_id"]
    case = client.get(f"/api/cases/{case_id}").json()

    for application in case["applications"]:
        decide = client.post(
            f"/api/cases/{case_id}/applications/{application['application_id']}/decide",
            json={"decision": "rejected"},
        )
        assert decide.status_code == 200

    final = client.get(f"/api/cases/{case_id}").json()
    assert final["status"] == "지원 불가"
    assert final["decision"] == "rejected"

    notify = client.post("/api/notification-call/start", json={"case_id": case_id})
    assert notify.json()["greeting"] == (
        "아쉽게도 이번에는 지원 대상에 선정되지 않았습니다. 추가 지원 가능한 제도를 다시 찾아드릴 수 있습니다."
    )


def test_decide_unknown_application_returns_404(client):
    resp = run_intake_call(client, "decide-404", "김영수입니다", "1952년 3월 14일입니다", "생활비가 힘들어요")
    case_id = resp.json()["case_id"]
    decide = client.post(
        f"/api/cases/{case_id}/applications/APP-DOES-NOT-EXIST/decide",
        json={"decision": "approved"},
    )
    assert decide.status_code == 404


def test_notification_call_unknown_case_returns_404(client):
    resp = client.post("/api/notification-call/start", json={"case_id": "CASE-DOES-NOT-EXIST"})
    assert resp.status_code == 404


def test_conversation_missing_session_id_is_validation_error(client):
    resp = client.post("/api/conversation", json={"message": "안녕하세요"})
    assert resp.status_code == 422


def test_conversation_response_never_includes_citizen_or_case_detail(client):
    # The /api/conversation response (what the citizen page consumes) must
    # only ever be {reply, ended, reason, case_id} -- no citizen profile,
    # recommendations, or application documents riding along.
    resp = run_intake_call(client, "citizen-payload-shape", "김영수입니다", "1952년 3월 14일입니다", "생활비가 힘들어요")
    assert set(resp.json().keys()) == {"reply", "ended", "reason", "case_id"}


def test_simple_status_reflects_case_lifecycle(client):
    resp = run_intake_call(client, "simple-status-flow", "박순자입니다", "1948년 11월 2일입니다", "병원비가 부담돼요")
    case_id = resp.json()["case_id"]

    pending = client.get(f"/api/cases/{case_id}/simple-status")
    assert pending.status_code == 200
    pending_body = pending.json()
    assert pending_body["stage"] == "승인 검토 중"
    assert pending_body["decision_ready"] is False
    # citizen-safe view must not leak the full case shape
    assert set(pending_body.keys()) == {"case_id", "stage", "decision_ready"}

    case = client.get(f"/api/cases/{case_id}").json()
    application_id = case["applications"][0]["application_id"]
    client.post(f"/api/cases/{case_id}/applications/{application_id}/decide", json={"decision": "approved"})

    approved = client.get(f"/api/cases/{case_id}/simple-status").json()
    assert approved["stage"] == "승인 완료"
    assert approved["decision_ready"] is True


def test_simple_status_unknown_case_returns_404(client):
    resp = client.get("/api/cases/CASE-DOES-NOT-EXIST/simple-status")
    assert resp.status_code == 404
