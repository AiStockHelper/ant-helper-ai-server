import time
from datetime import datetime
from zoneinfo import ZoneInfo

from app import batch, marketdata

from .conftest import make_candles


def fake_rows(symbols, n=120, base_volume=1_000_000, start_price=50000.0):
    rows = []
    for i, sym in enumerate(symbols):
        for c in make_candles(n=n, seed=100 + i, start_price=start_price, base_volume=base_volume):
            rows.append((sym, c["date"], c["open"], c["high"], c["low"], c["close"], c["volume"]))
    return rows


def wait_done(job, timeout=60):
    end = time.time() + timeout
    while job.status == "RUNNING" and time.time() < end:
        time.sleep(0.05)
    return job


def test_candle_store_upsert_and_load(tmp_path):
    store = marketdata.CandleStore(tmp_path / "c.sqlite")
    rows = fake_rows(["AAA"], n=50)
    assert store.upsert("KRX", rows) == 50
    assert store.upsert("KRX", rows) == 50  # 같은 키는 덮어쓴다
    assert store.last_dates("KRX")["AAA"] == rows[-1][1]
    assert len(store.load("KRX", ["AAA"], limit=30)["AAA"]) == 30
    assert store.load("US", ["AAA"])["AAA"] == []


def test_select_liquid_applies_value_price_and_freshness_filters():
    def bars(close, vol, n=80, last="2026-04-30"):
        from datetime import date, timedelta
        end = date.fromisoformat(last)
        return [((end - timedelta(days=n - 1 - i)).isoformat(), close, close, close, close, vol) for i in range(n)]

    candles = {
        "BIG": bars(10000, 1_000_000),      # 거래대금 1e10
        "MID": bars(10000, 200_000),        # 2e9
        "THIN": bars(10000, 1_000),         # 1e7 → 제외
        "PENNY": bars(500, 50_000_000),     # 가격 하한 미달
        "STALE": bars(10000, 1_000_000, last="2026-03-01"),
        "SHORT": bars(10000, 1_000_000, n=40),
    }
    picked = batch.select_liquid(candles, top_n=10, min_trade_value=1e9, min_price=1000)
    assert [p[0] for p in picked] == ["BIG", "MID"]
    assert [p[0] for p in batch.select_liquid(candles, top_n=1, min_trade_value=1e9, min_price=1000)] == ["BIG"]


def test_open_session_bar_is_dropped_only_while_market_is_open():
    seoul = ZoneInfo("Asia/Seoul")
    rows = [("A", "2026-11-03", 1, 1, 1, 1, 1), ("A", "2026-11-04", 1, 1, 1, 1, 1)]
    during = datetime(2026, 11, 4, 11, 0, tzinfo=seoul)
    after = datetime(2026, 11, 4, 16, 30, tzinfo=seoul)
    assert [r[1] for r in marketdata.drop_open_session_bar(rows, "KRX", during)] == ["2026-11-03"]
    assert len(marketdata.drop_open_session_bar(rows, "KRX", after)) == 2

    ny = ZoneInfo("America/New_York")
    us_rows = [("A", "2026-11-04", 1, 1, 1, 1, 1)]
    assert marketdata.drop_open_session_bar(us_rows, "US", datetime(2026, 11, 4, 12, 0, tzinfo=ny)) == []
    assert len(marketdata.drop_open_session_bar(us_rows, "US", datetime(2026, 11, 4, 17, 0, tzinfo=ny))) == 1


def test_batch_job_collects_filters_and_scores(tmp_path):
    store = marketdata.CandleStore(tmp_path / "c.sqlite")
    syms = [f"S{i:03d}" for i in range(30)]

    def fetcher(market, symbols, last_dates, progress):
        progress(len(symbols), len(symbols))
        return fake_rows(symbols)

    symbols = [{"symbol": s, "name": f"종목{s}", "market": "KRX"} for s in syms]
    job = batch.start_job("KRX", symbols, top_liquid=25, store=store, fetcher=fetcher, workers=1)
    wait_done(job)
    assert job.status == "DONE", job.error
    assert 0 < len(job.items) <= 25
    first = job.items[0]
    assert first["market"] == "KRX" and first["currency"] == "KRW"
    assert first["direction"] in ("UP", "DOWN", "FLAT")
    assert [i["score"] for i in job.items] == sorted((i["score"] for i in job.items), reverse=True)
    assert job.trade_date == job.items[0]["tradeDate"]


def test_batch_fails_cleanly_without_data(tmp_path):
    store = marketdata.CandleStore(tmp_path / "c.sqlite")
    job = batch.start_job("KRX", [{"symbol": "X1", "name": "x", "market": "KRX"}], top_liquid=10, store=store,
                          fetcher=lambda *a, **k: [], workers=1)
    wait_done(job)
    assert job.status == "FAILED"
    assert "조건" in job.error


def test_batch_api_returns_409_while_running_and_404_for_unknown(client, monkeypatch, tmp_path):
    release = {"go": False}

    def slow_fetcher(market, symbols, last_dates, progress):
        while not release["go"]:
            time.sleep(0.02)
        return fake_rows(symbols)

    monkeypatch.setattr(marketdata, "CandleStore", lambda *a, **k: _Store(tmp_path))
    monkeypatch.setattr(marketdata, "fetch", slow_fetcher)
    body = {"market": "US", "symbols": [{"symbol": "AAA", "name": "A", "market": "NASDAQ"}], "topLiquid": 10,
            "minTradeValue": 0, "minPrice": 0}
    first = client.post("/v1/batch/screen", json=body)
    assert first.status_code == 200, first.text
    job_id = first.json()["jobId"]
    assert client.post("/v1/batch/screen", json=body).status_code == 409
    assert client.get(f"/v1/batch/{job_id}/result").status_code == 409
    assert client.get("/v1/batch/nope").status_code == 404
    release["go"] = True
    for _ in range(300):
        if client.get(f"/v1/batch/{job_id}").json()["status"] != "RUNNING":
            break
        time.sleep(0.05)
    status = client.get(f"/v1/batch/{job_id}").json()
    assert status["status"] in ("DONE", "FAILED")


class _Store(marketdata.CandleStore):
    def __init__(self, tmp_path):
        super().__init__(tmp_path / "api.sqlite")
