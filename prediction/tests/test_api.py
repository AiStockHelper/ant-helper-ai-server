import re

from tests.conftest import make_candles

PREDICT_KEYS = {
    "symbol", "name", "currency", "currentPrice", "predictedPrice", "predictedChangeRate", "direction",
    "confidence", "summary", "signals", "indicators", "asOf",
}
SCREEN_KEYS = {
    "symbol", "name", "currency", "currentPrice", "score", "predictedChangeRate", "direction",
    "confidence", "reasons",
}
HANGUL = re.compile(r"[가-힣]")


def predict(client, candles, **extra):
    return client.post("/v1/predict", json={"symbol": "005930", "name": "삼성전자", "candles": candles, **extra})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "UP"}


def test_predict_shape(client, candles):
    r = predict(client, candles)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == PREDICT_KEYS
    assert body["symbol"] == "005930" and body["name"] == "삼성전자"
    assert body["currency"] == "KRW"
    assert body["currentPrice"] == candles[-1]["close"]
    assert float(body["currentPrice"]).is_integer() and float(body["predictedPrice"]).is_integer()
    assert body["direction"] in {"UP", "DOWN", "FLAT"}
    assert 0 <= body["confidence"] < 90
    assert HANGUL.search(body["summary"])
    assert body["signals"]
    for s in body["signals"]:
        assert set(s) == {"name", "value", "tone"}
        assert s["tone"] in {"POSITIVE", "NEGATIVE", "NEUTRAL"}
        assert HANGUL.search(s["value"])
    assert body["indicators"]["rsi14"] >= 0
    assert all(isinstance(v, (int, float)) for v in body["indicators"].values())
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00$", body["asOf"])


def test_predict_change_rate_matches_price(client, candles):
    body = predict(client, candles).json()
    expected = body["currentPrice"] * (1 + body["predictedChangeRate"] / 100)
    assert abs(body["predictedPrice"] - expected) <= 1
    if body["direction"] == "UP":
        assert body["predictedChangeRate"] > 0
    if body["direction"] == "DOWN":
        assert body["predictedChangeRate"] < 0


def test_predict_is_deterministic(client, candles):
    a, b = predict(client, candles).json(), predict(client, candles).json()
    a.pop("asOf"), b.pop("asOf")
    assert a == b


def test_predict_sorts_candles_and_drops_duplicates(client, candles):
    expected = predict(client, candles).json()
    shuffled = list(reversed(candles)) + [candles[10]]
    got = predict(client, shuffled).json()
    expected.pop("asOf"), got.pop("asOf")
    assert got == expected


def test_predict_name_optional(client, candles):
    r = client.post("/v1/predict", json={"symbol": "000660", "candles": candles})
    assert r.status_code == 200
    assert r.json()["name"] == "000660"


def test_predict_fallback_with_short_history(client):
    r = predict(client, make_candles(n=35))
    assert r.status_code == 200
    body = r.json()
    assert body["confidence"] <= 55
    assert any(s["name"] == "머신러닝" and "기술지표만" in s["value"] for s in body["signals"])
    assert "ma60" not in body["indicators"]


def test_predict_uses_ml_with_enough_history(client):
    body = predict(client, make_candles(n=200)).json()
    ml = [s for s in body["signals"] if s["name"] == "머신러닝"]
    assert ml and "추정" in ml[0]["value"]


def test_uptrend_not_bearish_and_downtrend_not_bullish(client):
    up = predict(client, make_candles(n=120, seed=1, drift=0.006, vol=0.006)).json()
    down = predict(client, make_candles(n=120, seed=1, drift=-0.006, vol=0.006)).json()
    assert up["direction"] != "DOWN" or up["confidence"] < 60
    assert down["direction"] != "UP" or down["confidence"] < 60
    assert up["predictedChangeRate"] > down["predictedChangeRate"]


def test_confidence_never_exaggerated(client):
    for seed in range(8):
        for drift in (-0.01, 0.0, 0.01):
            body = predict(client, make_candles(n=150, seed=seed, drift=drift, vol=0.004)).json()
            assert body["confidence"] <= 85


def test_predict_insufficient_candles_422(client):
    r = predict(client, make_candles(n=29))
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str) and "30" in r.json()["detail"]


def test_predict_empty_candles_422(client):
    r = predict(client, [])
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)


def test_predict_duplicates_do_not_count(client):
    c = make_candles(n=29)
    r = predict(client, c + c)
    assert r.status_code == 422


def test_predict_bad_date_422(client, candles):
    candles[3]["date"] = "not-a-date"
    r = predict(client, candles)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)


def test_predict_schema_violation_422_string_detail(client):
    r = client.post("/v1/predict", json={"candles": []})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)


def screen_body(**extra):
    return {
        "items": [
            {"symbol": "AAA", "name": "상승주", "candles": make_candles(n=120, seed=1, drift=0.005, vol=0.006)},
            {"symbol": "BBB", "name": "하락주", "candles": make_candles(n=120, seed=2, drift=-0.005, vol=0.006)},
            {"symbol": "CCC", "name": "횡보주", "candles": make_candles(n=120, seed=3)},
            {"symbol": "DDD", "candles": make_candles(n=10, seed=4)},
        ],
        **extra,
    }


def test_screen_ranked_descending_and_defaults(client):
    r = client.post("/v1/screen", json=screen_body(topN=10, minScore=-100))
    assert r.status_code == 200
    ranked = r.json()["ranked"]
    assert [x["symbol"] for x in ranked].count("DDD") == 0  # 캔들 부족 항목은 제외
    assert len(ranked) == 3
    scores = [x["score"] for x in ranked]
    assert scores == sorted(scores, reverse=True)
    for item in ranked:
        assert set(item) == SCREEN_KEYS
        assert item["reasons"] and all(HANGUL.search(x) for x in item["reasons"])
        assert item["confidence"] < 90


def test_screen_top_n_and_min_score(client):
    full = client.post("/v1/screen", json=screen_body(topN=10, minScore=-100)).json()["ranked"]
    top1 = client.post("/v1/screen", json=screen_body(topN=1, minScore=-100)).json()["ranked"]
    assert len(top1) == 1 and top1[0]["symbol"] == full[0]["symbol"]

    threshold = full[0]["score"] + 0.1
    none = client.post("/v1/screen", json=screen_body(topN=10, minScore=threshold)).json()["ranked"]
    assert none == []

    default = client.post("/v1/screen", json=screen_body()).json()["ranked"]
    assert all(x["score"] >= 0 for x in default)
    assert len(default) <= 5


def test_screen_empty_items(client):
    r = client.post("/v1/screen", json={"items": []})
    assert r.status_code == 200
    assert r.json() == {"ranked": []}


def test_screen_invalid_top_n_422(client):
    r = client.post("/v1/screen", json={"items": [], "topN": 0})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)


def test_screen_missing_items_422(client):
    r = client.post("/v1/screen", json={})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)
