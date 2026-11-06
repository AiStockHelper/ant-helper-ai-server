import pandas as pd

from app import factors


def _item(vol, rsi, ret5, ret20=0.0, ret1=0.0, px20=0.0, mom60=0.0, tv=1e10, direction="UP", rate=0.5, conf=60.0):
    return {"symbol": "X", "facts": {"vol20": vol, "rsi": rsi, "ret5": ret5, "ret20": ret20, "ret1": ret1, "pxMa20": px20, "mom60": mom60},
            "avgTradeValue": tv, "direction": direction, "predictedChangeRate": rate, "confidence": conf}


def _cohort():
    # 변동성·RSI 가 골고루 퍼진 30종목
    return [_item(vol=0.8 + i * 0.1, rsi=30 + i * 1.5, ret5=-5 + i * 0.4, tv=1e10 - i) for i in range(30)]


def test_calm_oversold_stock_outranks_hot_volatile_one():
    items = _cohort()
    calm = _item(vol=0.8, rsi=33, ret5=-4)
    hot = _item(vol=4.0, rsi=72, ret5=9, ret1=3)
    items += [calm, hot]
    factors.apply_ranking(items)
    assert calm["score"] > 20
    assert hot["score"] < 0
    assert calm["score"] > hot["score"]


def test_falling_knife_and_chasing_are_penalised():
    base = _item(vol=1.0, rsi=35, ret5=-6)
    knife = _item(vol=1.0, rsi=35, ret5=-6, ret20=-30)
    chase = _item(vol=1.0, rsi=35, ret5=-6, ret1=15)
    factors.apply_ranking(_cohort() + [base, knife, chase])
    assert knife["score"] < base["score"] - 40
    assert chase["score"] < base["score"] - 20
    assert any("떨어지는 칼날" in r for r in knife["reasons"])
    assert any("추격" in r for r in chase["reasons"])


def test_illiquid_names_are_never_recommended():
    items = _cohort()
    small = _item(vol=0.5, rsi=30, ret5=-6, tv=1.0)
    items.append(small)
    factors.apply_ranking(items, universe_n=30)
    assert small["score"] <= 0
    assert any("추천 대상에서 제외" in r for r in small["reasons"])


def test_model_direction_is_shown_but_does_not_move_the_score():
    a = _item(vol=1.0, rsi=40, ret5=-3, direction="UP", rate=3.0, conf=80)
    b = _item(vol=1.0, rsi=40, ret5=-3, direction="DOWN", rate=-3.0, conf=80)
    factors.apply_ranking(_cohort() + [a, b])
    assert a["score"] == b["score"]
    assert any("참고: 단기 모델" in r for r in a["reasons"])


def test_missing_facts_are_neutral_and_reasons_are_korean():
    sparse = {"symbol": "S", "facts": {}, "avgTradeValue": 1e10}
    factors.apply_ranking(_cohort() + [sparse])
    assert -5 <= sparse["score"] <= 5
    assert sparse["reasons"]


def test_compute_facts_from_indicator_frame():
    from app.indicators import add_indicators
    n = 120
    close = pd.Series([100 + i * 0.2 for i in range(n)])
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=n), "open": close, "high": close * 1.01,
                       "low": close * 0.99, "close": close, "volume": [1e6] * n})
    facts = factors.compute_facts(add_indicators(df))
    for key in ("ret5", "ret20", "vol20", "rsi", "mom60", "distHigh60", "pxMa20", "pxMa60", "align"):
        assert key in facts, key
    assert facts["align"] == 1.0
    assert facts["distHigh60"] == 0.0