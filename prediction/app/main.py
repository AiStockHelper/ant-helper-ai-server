"""예측 서버: 캔들을 받아 예측/스코어링만 수행한다 (KIS·DB·계정 정보 없음)."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .indicators import InsufficientDataError, prepare_frame
from .predictor import predict_frame
from .schemas import (
    HealthResponse,
    PredictRequest,
    PredictResult,
    ScreenRequest,
    ScreenResponse,
)
from .screener import screen_items

logger = logging.getLogger(__name__)

app = FastAPI(title="Ant Helper Prediction Server", version="1.0.0")


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
