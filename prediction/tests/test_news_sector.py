"""뉴스→섹터 백테스트의 순수 계산부 확인."""
import numpy as np
import pandas as pd

import tools.news_sector_backtest as N


def test_parse_day_groups_items_by_category():
    text = "[Business and economy]\n- Oil rises: Brent up 3%.\n- Bank reports profit.\n[Sports]\n- Team wins."
    d = N.parse_day(text)
    assert d["Business and economy"] == ["Oil rises: Brent up 3%.", "Bank reports profit."] and d["Sports"] == ["Team wins."]


def test_spread_keeps_everything_under_budget_and_samples_evenly_over_budget():
    items = [(f"01-{i:02d}", "x" * 100) for i in range(1, 31)]
    assert spread_all(items, 10_000) == items
    kept = N.spread(items, 1000)
    assert 0 < len(kept) < len(items) and kept[0] == items[0] and kept[-1] == items[-1]      # 한 달 전체에 고르게
    assert sum(len(t) + 8 for _, t in kept) <= 1400


def spread_all(items, budget):
    return N.spread(items, budget)


def test_hold_return_uses_first_open_and_last_close_of_the_next_month():
    idx = pd.bdate_range("2020-04-01", "2020-04-30")
    px = pd.DataFrame({"open": np.linspace(100, 110, len(idx)), "close": np.linspace(101, 120, len(idx))}, index=idx)
    r = N.hold_return(px, pd.Period("2020-03", freq="M"))             # 3월 말에 결정 → 4월 보유
    assert abs(r - (120 / 100 - 1)) < 1e-9
    assert N.hold_return(px, pd.Period("2020-04", freq="M")) is None   # 5월 데이터가 없으면 계산하지 않는다


def test_available_requires_listing_before_the_decision_date():
    meta = pd.DataFrame({"name": ["A", "B", "BM"], "first_date": pd.to_datetime(["2015-06-01", "2018-06-19", "2015-06-01"])}, index=["A", "B", "BM"])
    assert list(N.available(meta, pd.Period("2018-06", freq="M"), "BM").index) == ["A"]       # B 는 상장 11일 뒤라 아직 제외
    assert list(N.available(meta, pd.Period("2018-07", freq="M"), "BM").index) == ["A", "B"]
