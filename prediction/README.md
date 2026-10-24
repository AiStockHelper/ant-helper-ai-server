# Ant Helper 예측 서버

캔들 데이터를 받아 다음 거래일 예측과 종목 스코어링만 수행하는 FastAPI 서버입니다.
KIS 호출, DB, 로그인, 계정 정보를 사용하지 않습니다(상태 없음, 내부망 전용). API 계약은 `docs/DESIGN.md` 5절입니다.

## 실행

```powershell
cd C:\Projects\ant_helper\prediction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.bat                      # http://127.0.0.1:8000
.\.venv\Scripts\python.exe -m pytest
```

## API

- `GET /health` → `{"status":"UP"}`
- `POST /v1/predict` → `PredictResult`
- `POST /v1/screen` → `{"ranked": ScreenItem[]}` (score 내림차순, `minScore`(기본 0) 미만 제외, 상위 `topN`(기본 5))

`currency`(`"KRW"` | `"USD"`, 기본 `"KRW"`)는 선택 필드이며 `/v1/predict` 본문과 `/v1/screen` 의 각 `items[]` 에 넣습니다. 응답(`PredictResult`, `ScreenItem`)에 그대로 되돌려 줍니다. 그 외 값은 422.

```json
POST /v1/predict   (국내)
{"symbol":"005930","name":"삼성전자","currency":"KRW",
 "candles":[{"date":"2026-10-23","open":70000,"high":71000,"low":69500,"close":70500,"volume":12345678}]}
→ {"symbol":"005930","name":"삼성전자","currency":"KRW","currentPrice":70500.0,"predictedPrice":70720.0,
   "predictedChangeRate":0.31,"direction":"UP","confidence":58.3,"summary":"...","signals":[...],"indicators":{...},"asOf":"..."}
```

```json
POST /v1/predict   (미국)
{"symbol":"AAPL","name":"Apple","currency":"USD",
 "candles":[{"date":"2026-10-23","open":188.9,"high":190.4,"low":188.1,"close":189.5,"volume":51234567}]}
→ {"symbol":"AAPL","currency":"USD","currentPrice":189.5,"predictedPrice":190.12,"predictedChangeRate":0.33, ...}
```

```json
POST /v1/screen
{"items":[{"symbol":"005930","name":"삼성전자","currency":"KRW","candles":[...]},
          {"symbol":"AAPL","name":"Apple","currency":"USD","candles":[...]}],"topN":5,"minScore":0}
→ {"ranked":[{"symbol":"AAPL","name":"Apple","currency":"USD","currentPrice":189.5,"score":31.2,
              "predictedChangeRate":0.33,"direction":"UP","confidence":58.3,"reasons":["..."]}]}
```

가격은 float 로 내려가며 통화별로 반올림합니다(KRW 소수 0자리, USD 소수 2자리). `predictedChangeRate` 는 %(소수 2자리). 한 요청에서 통화를 섞어도 항목별로 처리하며, 점수는 비율 기반이라 통화 간 비교가 가능하지만 UI 에서 합산하지는 마세요.

## 동작 메모

- 캔들은 서버가 날짜 오름차순으로 정렬하고, 같은 날짜가 중복되면 마지막 값을 씁니다. 정제 후 30개 미만이면 HTTP 422 `{"detail":"..."}`. 60개 이상 권장(120개 이상이면 ML 사용).
- 요청 형식 오류도 422 `{"detail":"<문자열>"}` 입니다.
- 지표·판정은 모두 비율 기반으로 가격 단위에 의존하지 않습니다. 통화는 가격 반올림, 요약 문구의 표기(`70,500원` / `$189.50`), 유동성 감점 기준(최근 5일 평균 거래대금: KRW 1억 원, USD 500만 달러 — `app/currency.py` 의 `MIN_TRADE_VALUE`)에만 쓰입니다.
- `/v1/screen` 에서 캔들이 부족한 항목은 오류 없이 결과에서 제외됩니다.
- `name` 이 없으면 응답 `name` 은 `symbol` 로 채웁니다. `asOf` 는 계산 시각(KST, `+09:00`).
- 예측은 규칙 기반 점수 + GradientBoosting(내일 수익률)을 검증 방향 적중률에 따라 가중 평균합니다(`random_state=42`, 결정적). 학습 데이터가 부족하면 기술지표만 사용합니다. 신뢰도는 최대 85 미만으로 제한되며, ML 검증이 약하거나 기술지표만 쓰면 70~72, 캔들 60개 미만이면 55 로 더 낮게 제한됩니다.
- `indicators` 키: `close, ma5, ma10, ma20, ma60, rsi14, macd, macdSignal, bbUpper, bbLower, bbPercentB, stochK, atr14, atrPct, volumeRatio, ret5d, ret20d, volatility, historyDays, ruleScore, mlWeight, tradeValue5d` (계산 불가능한 값은 생략).
