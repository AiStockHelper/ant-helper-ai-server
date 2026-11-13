"""뉴스 AI 스킬 백테스트 엔진: 스킬의 규칙(검증·익절·손절·보유일·손실 한도)을 그대로 따르는지 확인한다."""
import numpy as np
import pandas as pd

from tools.news_skill_backtest import Panel, Params, metrics, name_matches, random_decisions, signal_quality, simulate

DATES = pd.bdate_range("2026-03-02", periods=16)          # 월 3/2 ~ 화 3/23
NAMES = {"AAA": "에이에이에이", "BBB": "비비비", "CCC": "씨씨씨"}


def panel(rows: dict, krx: bool = True, liquid=True, refs=()):
    """rows: {symbol: [(o,h,l,c), ...]} (날짜 수만큼). 모든 종목이 같은 날짜 수를 가진다."""
    syms = list(rows)
    n = len(DATES)
    arr = {k: np.array([[rows[s][d][i] for s in syms] for d in range(n)], dtype=float) for i, k in enumerate("ohlc")}
    liq = np.full((n, len(syms)), liquid) if not isinstance(liquid, np.ndarray) else liquid
    return Panel(DATES, syms, arr["o"], arr["h"], arr["l"], arr["c"], np.full((n, len(syms)), 1e6), names=dict(NAMES), liquid=liq,
                 ref_etfs=set(refs), krx=krx)


def flat(price=100.0, n=16):
    return [(price, price, price, price)] * n


def buy(symbol, name, conviction=5):
    return {"symbol": symbol, "name": name, "conviction": conviction, "reason": "r"}


def day(i):
    return DATES[i]


NOCOST = dict(commission=0.0, sell_tax=0.0, sell_tax_etf=0.0, slippage=0.0, loss_limit=None)


def prm(**kw):
    return Params(**{**dict(budget=300_000.0, max_positions=3, chase=0.08), **NOCOST, **kw})


def test_take_profit_hits_at_the_limit_price_and_the_cash_is_booked():
    rows = {"AAA": flat()}
    rows["AAA"][3] = (100, 109, 99, 105)                       # 시가 매수 후 같은 날 +8% 도달
    sim = simulate(panel(rows), {day(3): {"buys": [buy("AAA", "에이에이에이")]}}, prm(), 2, 8)
    assert [(t.reason, round(t.net_pct, 2)) for t in sim.trades] == [("익절", 8.0)]
    assert sim.trades[0].qty == 1000                            # 시드 30만원 ÷ 3 = 10만원 → 100원짜리 1000주
    assert abs(sim.equity[8] - 300_000 - 8_000) < 1e-6


def test_stop_loss_and_the_conservative_same_bar_rule():
    rows = {"AAA": flat()}
    rows["AAA"][3] = (100, 109, 94, 100)                       # 같은 날 손절선(95)과 익절선(108) 모두 닿는다
    d = {day(3): {"buys": [buy("AAA", "에이에이에이")]}}
    assert [t.reason for t in simulate(panel(rows), d, prm(), 2, 8).trades] == ["손절"]
    assert [t.reason for t in simulate(panel(rows), d, prm(same_bar="target"), 2, 8).trades] == ["익절"]
    rows["AAA"][3] = (100, 101, 94, 96)
    t = simulate(panel(rows), d, prm(), 2, 8).trades[0]
    assert t.reason == "손절" and round(t.net_pct, 2) == -5.0


def test_a_position_is_closed_at_the_close_after_the_max_hold_days():
    rows = {"AAA": flat()}
    for k in range(4, 16):
        rows["AAA"][k] = (100, 101, 99, 101)
    sim = simulate(panel(rows), {day(3): {"buys": [buy("AAA", "에이에이에이")]}}, prm(max_hold_days=10), 2, 14)
    t = sim.trades[0]
    assert t.reason == "보유일 초과" and t.close_idx - t.open_idx >= 7          # 달력 10일 ≈ 거래일 8일째
    assert round(t.net_pct, 2) == 1.0


def test_the_stock_is_not_chased_when_it_gaps_up_by_the_limit():
    rows = {"AAA": flat(), "BBB": flat()}
    rows["AAA"][3] = (109, 110, 108, 109)                      # 전일 종가 100 → 시가 109 (+9%)
    rows["BBB"][3] = (107, 108, 106, 107)                      # +7%: 통과
    d = {day(3): {"buys": [buy("AAA", "에이에이에이"), buy("BBB", "비비비")]}}
    sim = simulate(panel(rows), d, prm(), 2, 8)
    assert sim.rejects == {"급등 추격 금지": 1} and [t.sym for t in sim.trades] == ["BBB"]
    off = simulate(panel(rows), d, prm(chase=0), 2, 8)
    assert sorted(t.sym for t in off.trades) == ["AAA", "BBB"]


def test_korean_names_must_match_but_us_names_and_reference_etfs_are_not_checked():
    rows = {"AAA": flat(), "BBB": flat()}
    d = {day(3): {"buys": [buy("AAA", "엉뚱한회사"), buy("BBB", "")]}}
    sim = simulate(panel(rows, refs={"BBB"}), d, prm(), 2, 5)
    assert sim.rejects == {"종목명 불일치": 1} and sim.accepted == 1               # BBB 는 산업 ETF 참고 목록이라 이름 검사 없음
    us = simulate(panel(rows, krx=False), d, prm(), 2, 5)
    assert us.rejects == {} and us.accepted == 2


def test_stocks_outside_the_liquid_set_and_unknown_symbols_are_rejected_but_reference_etfs_pass():
    rows = {"AAA": flat(), "BBB": flat()}
    liq = np.array([[True, False]] * len(DATES))
    d = {day(3): {"buys": [buy("AAA", "에이에이에이"), buy("BBB", "비비비"), buy("ZZZ", "없는회사")]}}
    sim = simulate(panel(rows, liquid=liq), d, prm(), 2, 5)
    assert sim.rejects == {"거래대금 상위 아님": 1, "시세 자료 없음": 1} and sim.accepted == 1
    etf = simulate(panel(rows, liquid=liq, refs={"BBB"}), d, prm(), 2, 5)
    assert etf.accepted == 2                                                         # 참고 ETF 는 거래대금 검사 예외


def test_only_convictions_at_or_above_the_threshold_fill_the_free_slots_by_conviction_order():
    rows = {s: flat() for s in ("AAA", "BBB", "CCC")}
    d = {day(3): {"buys": [buy("AAA", "에이에이에이", 3), buy("BBB", "비비비", 4), buy("CCC", "씨씨씨", 5)]}}
    sim = simulate(panel(rows), d, prm(max_positions=1), 2, 5)
    assert [t.sym for t in sim.trades] == ["CCC"]                                    # 자리 1개: 확신 5 가 먼저, 3 은 기준 미달
    three = simulate(panel(rows), d, prm(max_positions=3), 2, 5)
    assert sorted(t.sym for t in three.trades) == ["BBB", "CCC"]


def test_position_size_is_capped_by_budget_over_max_positions_and_buys_whole_shares_only():
    rows = {"AAA": flat(150_000)}
    d = {day(3): {"buys": [buy("AAA", "에이에이에이")]}}
    sim = simulate(panel(rows), d, prm(budget=300_000.0), 2, 5)
    assert sim.accepted == 0 and sim.rejects == {"1주도 못 삼": 1}                  # 종목당 한도 10만원으로는 15만원짜리 1주도 못 산다
    one = simulate(panel(rows), d, prm(budget=300_000.0, max_positions=1), 2, 5)
    assert one.accepted == 1 and one.trades[0].qty == 2                              # 한도 30만원이면 2주(정수 주식만)


def test_a_held_stock_the_ai_says_to_avoid_is_sold_at_the_next_open():
    rows = {"AAA": flat()}
    rows["AAA"][5] = (97, 98, 96, 97)
    d = {day(3): {"buys": [buy("AAA", "에이에이에이")]}, day(5): {"buys": [], "avoid": ["AAA"]}}
    sim = simulate(panel(rows), d, prm(), 2, 8)
    assert [(t.reason, round(t.net_pct, 2)) for t in sim.trades] == [("AI 회피", -3.0)]
    assert [t.reason for t in simulate(panel(rows), d, prm(ai_exits=False), 2, 8).trades] != ["AI 회피"]


def test_the_cumulative_loss_limit_liquidates_everything_and_stops_all_further_buying():
    rows = {"AAA": flat()}
    rows["AAA"][4] = (100, 100, 70, 70)                                              # 급락: 손절선 이하 → 손절 -5% 로 체결
    for k in range(5, 16):
        rows["AAA"][k] = (70, 70, 70, 70)
    d = {day(3): {"buys": [buy("AAA", "에이에이에이")]}, day(8): {"buys": [buy("AAA", "에이에이에이")]}}
    sim = simulate(panel(rows), d, prm(max_positions=1, loss_limit=0.01, stop_loss=0.50), 2, 12)
    assert sim.killed_at is not None
    assert [t.reason for t in sim.trades] == ["손실 한도"]                            # 손절선(50%)보다 먼저 누적 손실 한도(1%)에 닿았다
    assert sim.accepted == 1                                                         # 정지 뒤의 제안(8일)은 무시
    assert metrics(sim, 2, 12, 300_000.0)["정지"] == "예"


def test_costs_make_a_flat_round_trip_a_loss():
    rows = {"AAA": flat()}
    d = {day(3): {"buys": [buy("AAA", "에이에이에이")]}}
    for k in range(4, 16):
        rows["AAA"][k] = (100, 100, 100, 100)
    costs = Params(budget=300_000.0, max_positions=3, commission=0.00015, sell_tax=0.002, slippage=0.0007, loss_limit=None)
    t = simulate(panel(rows), d, costs, 2, 14).trades[0]
    assert t.net_pct < -0.2                                                          # 거래세 0.2% + 수수료 + 슬리피지 양쪽
    etf_costs = Params(budget=300_000.0, max_positions=3, commission=0.00015, sell_tax=0.002, sell_tax_etf=0.0, slippage=0.0007, loss_limit=None)
    e = simulate(panel(rows, refs={"AAA"}), d, etf_costs, 2, 14).trades[0]
    assert e.net_pct > t.net_pct + 0.15                                              # 산업 ETF 는 매도세가 없다


def test_signal_quality_compares_each_accepted_proposal_with_the_same_day_universe():
    rows = {"AAA": flat(), "BBB": flat(), "CCC": flat()}
    for k in range(3, 8):
        rows["AAA"][k] = (100, 106, 100, 106)            # 제안 종목은 오른다
        rows["BBB"][k] = (100, 100, 99, 99)              # 나머지는 약간 내린다
        rows["CCC"][k] = (100, 100, 99, 99)
    d = {day(3): {"buys": [buy("AAA", "에이에이에이")]}}
    sq = signal_quality(panel(rows), d, 4, 2, 10, 0.08)
    row = sq.loc["당일 종가"]
    assert row["제안 수"] == 1 and abs(row["평균수익%"] - 6.0) < 1e-9 and row["초과%p"] > 4


def test_random_control_proposes_the_same_number_of_liquid_names_each_day():
    rows = {s: flat() for s in ("AAA", "BBB", "CCC")}
    p = panel(rows)
    d = {day(3): {"buys": [buy("AAA", "에이에이에이")]}, day(4): {"buys": [buy("AAA", "에이에이에이", 3)]}}   # 4일은 기준 미달 → 무작위도 제안 없음
    rd = random_decisions(p, d, 4, seed=1)
    assert set(rd) == {day(3)} and 1 <= len(rd[day(3)]["buys"]) <= 3


def test_name_matching_rules_mirror_the_skill_validator():
    assert name_matches("삼성전자", "삼성전자") and name_matches("SK 하이닉스", "SK하이닉스") and name_matches("하이닉스", "SK하이닉스")
    assert not name_matches("삼성", "삼성전자") and not name_matches("", "삼성전자") and not name_matches("카카오", None)
