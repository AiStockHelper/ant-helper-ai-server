"""예측 서버: 캔들을 받아 예측/스코어링만 수행한다 (KIS·DB·계정 정보 없음)."""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import batch
from .indicators import InsufficientDataError, prepare_frame
from .predictor import predict_frame
from .schemas import (
    BatchResult,
    BatchStartRequest,
    BatchStatus,
    HealthResponse,
    PredictRequest,
    PredictResult,
    ScreenRequest,
    ScreenResponse,
)
from .screener import screen_items

logger = logging.getLogger(__name__)

app = FastAPI(title="Ant Helper Prediction Server", version="1.0.0")


@app.middleware("http")
async def require_internal_key(request: Request, call_next):
    """INTERNAL_API_KEY 환경변수가 있으면 백엔드와 공유한 X-Internal-Key 헤더가 맞아야 한다(/health 제외)."""
    expected = os.environ.get("INTERNAL_API_KEY", "").strip()
    if expected and request.url.path != "/health":
        given = request.headers.get("x-internal-key", "")
        if not hmac.compare_digest(given.encode(), expected.encode()):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)


@app.exception_handler(InsufficientDataError)
async def insufficient_data_handler(_: Request, exc: InsufficientDataError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return JSONResponse(
        status_code=422, content={"detail": "요청 형식이 올바르지 않습니다 — " + "; ".join(parts)}
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="UP")


@app.post("/v1/predict", response_model=PredictResult, response_model_by_alias=True)
def predict(req: PredictRequest) -> PredictResult:
    df = prepare_frame(req.candles)
    return predict_frame(df, req.symbol, req.name, req.currency)


@app.post("/v1/screen", response_model=ScreenResponse, response_model_by_alias=True)
def screen(req: ScreenRequest) -> ScreenResponse:
    return ScreenResponse(ranked=screen_items(req.items, req.top_n, req.min_score))


def _status(job: batch.Job) -> BatchStatus:
    return BatchStatus(
        job_id=job.id, market=job.market, status=job.status, phase=job.phase, done=job.done, total=job.total,
        count=len(job.items), trade_date=job.trade_date, error=job.error,
    )


@app.post("/v1/batch/screen", response_model=BatchStatus, response_model_by_alias=True)
def batch_start(req: BatchStartRequest) -> BatchStatus:
    """종목 목록의 일봉을 수집·저장하고 유동성 상위 종목을 전부 스코어링하는 작업을 시작한다(비동기)."""
    try:
        job = batch.start_job(
            req.market,
            [{"symbol": s.symbol, "name": s.name, "market": s.market} for s in req.symbols],
            req.top_liquid, req.min_trade_value, req.min_price,
        )
    except batch.JobBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status(job)


@app.get("/v1/batch/{job_id}", response_model=BatchStatus, response_model_by_alias=True)
def batch_status(job_id: str) -> BatchStatus:
    job = batch.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return _status(job)


@app.get("/v1/batch/{job_id}/result", response_model=BatchResult, response_model_by_alias=True)
def batch_result(job_id: str) -> BatchResult:
    job = batch.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    if job.status != "DONE":
        raise HTTPException(status_code=409, detail=f"작업이 아직 끝나지 않았습니다({job.status}).")
    return BatchResult(status=_status(job), items=job.items)