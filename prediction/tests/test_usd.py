from app.currency import MIN_TRADE_VALUE
from tests.conftest import make_candles

USD_KEYS = dict(start_price=150.0, decimals=2)


def usd_candles(**kw):
    return make_candles(**{**USD_KEYS, **kw})


def predict(client, candles, **extra):
    return client.post("/v1/predict", json={"symbol": "AAPL", "name": "Apple", "candles": candles, **extra})


def decimals_of(x):
    return len(repr(round(x, 6)).split(".")[1].rstrip("0")) if "." in repr(x) else 0


def test_krw_regression_snapshot(client):
    # 해외주식 지원 이전 구현의 실제 출력값 (고정 시드 합성 캔들)
    candles = make_candles(150, seed=5, drift=0.003, vol=0.01)
    body = predict(client, candles, name="삼성전자").json()
    assert body["currency"] == "KRW"
    assert body["currentPrice"] == 67924
    assert body["predictedPrice"] == 67999
    assert body["predictedChangeRate"] == 0.11
    assert body["direction"] == "FLAT"
    assert body["confidence"] == 46.2
    assert "67,924원" in body["summary"] and "67,999원" in body["summary"]


def test_explicit_krw_equals_default(client, candles):
    a = predict(client, candles).json()
    b = predict(client, candles, currency="KRW").json()
    a.pop("asOf"), b.pop("asOf")
    assert a == b


def test_usd_predict_keeps_decimal_prices(client):
    candles = usd_candles(n=150, seed=11, drift=0.002, vol=0.012)
    body = predict(client, candles, currency="USD").json()
    assert body["currency"] == "USD"
    assert body["currentPrice"] == candles[-1]["close"]
    assert any(not float(c["close"]).is_integer() for c in candles)
    assert decimals_of(body["currentPrice"]) <= 2 and decimals_of(body["predictedPrice"]) <= 2
    expected = body["currentPrice"] * (1 + body["predictedChangeRate"] / 100)
    assert abs(body["predictedPrice"] - expected) <= 0.011
    assert decimals_of(body["predictedChangeRate"]) <= 2
    assert "$" in body["summary"] and "원" not in body["summary"]
    assert body["confidence"] < 90


def test_usd_predict_not_sensitive_to_price_scale(client):
    base = make_candles(n=150, seed=21, start_price=50000.0, decimals=6)
    scaled = [
        {**c, **{k: c[k] / 500 for k in ("open", "high", "low", "close")}} for c in base
    ]
    a = predict(client, base, currency="USD").json()
    b = predict(client, scaled, currency="USD").json()
    assert a["direction"] == b["direction"]
    assert a["confidence"] == b["confidence"]
    assert abs(a["predictedChangeRate"] - b["predictedChangeRate"]) < 0.02
    assert [s["value"] for s in a["signals"]] == [s["value"] for s in b["signals"]]


def test_usd_short_history_fallback_and_422(client):
    ok = predict(client, usd_candles(n=35), currency="USD")
    assert ok.status_code == 200 and ok.json()["currency"] == "USD"
    bad = predict(client, usd_candles(n=20), currency="USD")
    assert bad.status_code == 422 and isinstance(bad.json()["detail"], str)


def test_invalid_currency_422(client, candles):
    r = predict(client, candles, currency="EUR")
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)


def test_usd_screen_keeps_decimals_and_currency(client):
    body = {
        "items": [
            {"symbol": "AAA", "currency": "USD", "candles": usd_candles(n=150, seed=1, drift=0.004, vol=0.008)},
            {"symbol": "BBB", "currency": "USD", "candles": usd_candles(n=150, seed=2, drift=-0.004, vol=0.008)},
            {"symbol": "005930", "candles": make_candles(n=150, seed=3)},
        ],
        "topN": 10,
        "minScore": -100,
    }
    r = client.post("/v1/screen", json=body)
    assert r.status_code == 200
    ranked = {x["symbol"]: x for x in r.json()["ranked"]}
    assert set(ranked) == {"AAA", "BBB", "005930"}
    assert ranked["AAA"]["currency"] == "USD" and ranked["005930"]["currency"] == "KRW"
    last_close = body["items"][0]["candles"][-1]["close"]
    assert ranked["AAA"]["currentPrice"] == last_close
    assert decimals_of(ranked["AAA"]["currentPrice"]) <= 2
    assert float(ranked["005930"]["currentPrice"]).is_integer()


def liquidity_flag(client, currency, base_volume, start_price):
    candles = make_candles(
        n=120, seed=9, start_price=start_price, base_volume=base_volume, decimals=2 if currency == "USD" else 0
    )
    r = client.post(
        "/v1/screen",
        json={"items": [{"symbol": "X", "currency": currency, "candles": candles}], "minScore": -100},
    )
    return any("유동성" in reason for reason in r.json()["ranked"][0]["reasons"])


def test_liquidity_threshold_is_per_currency(client):
    assert MIN_TRADE_VALUE["USD"] == 5_000_000 and MIN_TRADE_VALUE["KRW"] == 100_000_000
    # 평균 거래대금: 150달러 x 약 1.1만주 ≈ 170만 달러 (< 500만 달러) → 감점
    assert liquidity_flag(client, "USD", 10_000, 150.0)
    # 150달러 x 약 100만주 ≈ 1.7억 달러 → 감점 없음
    assert not liquidity_flag(client, "USD", 1_000_000, 150.0)
    # 거래대금 약 2,000만: USD 기준(500만 달러)은 넘고 KRW 기준(1억 원)은 못 넘는다
    assert not liquidity_flag(client, "USD", 130_000, 150.0)
    assert liquidity_flag(client, "KRW", 130_000, 150.0)
    assert not liquidity_flag(client, "KRW", 1_000_000, 50000.0)
