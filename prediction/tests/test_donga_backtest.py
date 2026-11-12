"""주간동아 방식 백테스트 엔진: 합성 데이터로 체결·청산·비용 계산을 검증한다(결과 숫자를 믿으려면 엔진이 맞아야 한다)."""
import numpy as np
import pandas as pd

from tools.donga_strategy_backtest import Costs, ceil_tick, floor_tick, simulate, tick_of

DATES = pd.date_range("2026-01-05", periods=6, freq="B")


def panel(prices0, prices1):
    """prices: 날짜별 (open, high, low, close). 종목 0·1 을 (날짜, 종목) 배열로 만든다."""
    o = np.array([[a[0], b[0]] for a, b in zip(prices0, prices1)], dtype=float)
    h = np.array([[a[1], b[1]] for a, b in zip(prices0, prices1)], dtype=float)
    l = np.array([[a[2], b[2]] for a, b in zip(prices0, prices1)], dtype=float)
    c = np.array([[a[3], b[3]] for a, b in zip(prices0, prices1)], dtype=float)
    return o, h, l, c


FLAT = (10000, 10000, 10000, 10000)


def test_tick_sizes_and_rounding():
    assert tick_of(np.array([1500.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0, 800000.0])).tolist() == [1, 5, 10, 50, 100, 500, 1000]
    assert ceil_tick(np.array([10200.0, 10201.0, 31001.0])).tolist() == [10200.0, 10210.0, 31050.0]
    assert floor_tick(np.array([9500.0, 9507.0, 31049.0])).tolist() == [9500.0, 9500.0, 31000.0]


def test_cheapest_first_one_share_per_pass_and_limit_exit_with_loser_kept():
    # 종목0: 3일째 장중 10,300 까지 올라 +2%(10,200) 지정가 체결. 종목1: 9,000 으로 떨어져 계속 보유.
    p0 = [FLAT, FLAT, (10000, 10300, 10000, 10100), FLAT, FLAT, FLAT]
    p1 = [FLAT, FLAT, (9000, 9000, 9000, 9000), (9000, 9000, 9000, 9000), (9000, 9000, 9000, 9000), (9000, 9000, 9000, 9000)]
    o, h, l, c = panel(p0, p1)
    r = simulate(DATES, o, h, l, c, {0: np.array([0, 1])}, first_day=1, capital=100_000, costs=Costs.none())
    # 10,000원짜리 두 종목을 번갈아 5주씩(100,000원) 산 뒤, 종목0 5주를 10,200 에 판다 → 현금 51,000, 종목1 5주 평가 45,000
    assert r.trades_closed == 1
    assert r.win_rate == 1.0
    assert r.open_positions == 1
    assert r.open_underwater_value_share == 1.0
    assert abs(r.equity[-1] - 96_000) < 1e-6
    assert abs(r.final_net - 96_000) < 1e-6


def test_gap_up_fills_at_the_open_not_the_limit():
    p0 = [FLAT, FLAT, (10500, 10600, 10400, 10500), FLAT, FLAT, FLAT]
    p1 = [FLAT] * 6
    o, h, l, c = panel(p0, p1)
    r = simulate(DATES, o, h, l, c, {0: np.array([0])}, first_day=1, capital=50_000, costs=Costs.none())
    assert r.trades_closed == 1
    assert abs(r.equity[2] - (50_000 + 5 * (10_500 - 10_000))) < 1e-6   # 5주 × (시가 10,500 - 매수가 10,000)


def test_no_exit_on_the_buy_day_unless_same_day_exit_is_allowed():
    p0 = [FLAT, (10000, 10300, 10000, 10100), FLAT, FLAT, FLAT, FLAT]
    p1 = [FLAT] * 6
    o, h, l, c = panel(p0, p1)
    conservative = simulate(DATES, o, h, l, c, {0: np.array([0])}, first_day=1, capital=10_000, costs=Costs.none())
    optimistic = simulate(DATES, o, h, l, c, {0: np.array([0])}, first_day=1, capital=10_000, costs=Costs.none(), same_day_exit=True)
    assert conservative.trades_closed == 0
    assert optimistic.trades_closed == 1
    assert abs(optimistic.equity[1] - 10_200) < 1e-6


def test_costs_are_charged_on_both_sides_and_on_the_final_liquidation():
    p0 = [FLAT, FLAT, (10000, 10300, 10000, 10100), FLAT, FLAT, FLAT]
    p1 = [FLAT] * 6
    o, h, l, c = panel(p0, p1)
    costs = Costs(commission=0.001, sell_tax=0.002, slippage=0.0)
    r = simulate(DATES, o, h, l, c, {0: np.array([0])}, first_day=1, capital=10_010, costs=costs)
    # 매수: 10,000 + 수수료 10 = 10,010(1주). 매도: 10,200 × (1 - 0.001 - 0.002) = 10,169.4
    assert r.trades_closed == 1
    assert abs(r.final_net - 10_169.4) < 1e-6


def test_stop_loss_and_expiry_close_the_loser_at_a_loss():
    p0 = [FLAT, FLAT, (10000, 10000, 9400, 9500), FLAT, FLAT, FLAT]
    p1 = [FLAT] * 6
    o, h, l, c = panel(p0, p1)
    r = simulate(DATES, o, h, l, c, {0: np.array([0])}, first_day=1, capital=10_000, costs=Costs.none(), stop=0.05)
    assert r.trades_closed == 1
    assert r.win_rate == 0.0
    assert abs(r.equity[-1] - 9_500) < 1e-6      # 손절선 9,500 에 체결(호가 단위 내림)

    held = simulate(DATES, o, h, l, c, {0: np.array([0])}, first_day=1, capital=10_000, costs=Costs.none(), max_hold=2)
    assert held.trades_closed == 1               # 2일 보유 뒤 종가에 정리


def test_nothing_is_bought_without_enough_cash():
    o, h, l, c = panel([FLAT] * 6, [FLAT] * 6)
    r = simulate(DATES, o, h, l, c, {0: np.array([0, 1])}, first_day=1, capital=5_000, costs=Costs.none())
    assert r.open_positions == 0 and r.trades_closed == 0
    assert abs(r.equity[-1] - 5_000) < 1e-6
