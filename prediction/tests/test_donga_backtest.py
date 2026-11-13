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


DATES9 = pd.date_range("2026-01-05", periods=9, freq="B")


def test_expiry_sells_at_the_close_of_the_nth_business_day_after_the_buy_day():
    # 신호 0일 → 1일 시가 매수. max_hold=3 이면 매수일로부터 3영업일째인 4일 종가에 정리한다(그 뒤 하락은 맞지 않는다).
    drift = [FLAT, FLAT, FLAT, FLAT, (10000, 10000, 9900, 9900)] + [(9000, 9000, 9000, 9000)] * 4
    o, h, l, c = panel(drift, [FLAT] * 9)
    r = simulate(DATES9, o, h, l, c, {0: np.array([0])}, first_day=1, capital=10_000, costs=Costs.none(), max_hold=3)
    assert r.exits_expire == 1 and r.trades_closed == 1
    assert abs(r.equity[3] - 10_000) < 1e-6       # 3일째까지는 보유 중(평가 10,000)
    assert abs(r.equity[4] - 9_900) < 1e-6        # 4일 종가 9,900 에 정리
    assert abs(r.equity[8] - 9_900) < 1e-6        # 이후 하락(9,000)에 노출되지 않는다
    held = simulate(DATES9, o, h, l, c, {0: np.array([0])}, first_day=1, capital=10_000, costs=Costs.none())
    assert abs(held.equity[8] - 9_000) < 1e-6      # 기간 정리가 없으면 계속 들고 있다


def test_symmetric_stop_and_take_profit_are_counted_by_reason():
    p0 = [FLAT, FLAT, (10000, 10000, 9700, 9800), FLAT, FLAT, FLAT]                    # 3일째 -2%(9,800) 손절
    p1 = [FLAT, FLAT, (10000, 10300, 10000, 10100), FLAT, FLAT, FLAT]                  # 3일째 +2%(10,200) 익절
    o, h, l, c = panel(p0, p1)
    r = simulate(DATES, o, h, l, c, {0: np.array([0, 1])}, first_day=1, capital=20_000, costs=Costs.none(), stop=0.02)
    assert (r.exits_take_profit, r.exits_stop, r.exits_expire) == (1, 1, 0)
    assert abs(r.equity[-1] - (20_000 - 200 + 200)) < 1e-6        # 손절 -200, 익절 +200


def test_net_return_per_trade_includes_both_commissions_and_the_sell_tax():
    p0 = [FLAT, FLAT, (10000, 10300, 10000, 10100), FLAT, FLAT, FLAT]
    o, h, l, c = panel(p0, [FLAT] * 6)
    r = simulate(DATES, o, h, l, c, {0: np.array([0])}, first_day=1, capital=10_010, costs=Costs(0.001, 0.002, 0.0))
    expected = (10_200 * (1 - 0.003) - 10_000 * 1.001) / (10_000 * 1.001)
    assert abs(r.avg_trade_net - expected) < 1e-9


def test_skip_held_does_not_add_to_a_position_that_is_already_held():
    # 종목0·1 을 1일에 2주씩 산 뒤, 종목1 이 3일에 익절돼 현금이 생긴다. 4일에 종목0 신호가 오면 skip_held 일 땐 사지 않는다.
    p0 = [FLAT] * 9
    p1 = [FLAT, FLAT, FLAT, (10000, 10300, 10000, 10100), FLAT, FLAT, FLAT, FLAT, FLAT]
    o, h, l, c = panel(p0, p1)
    picks = {0: np.array([0, 1]), 3: np.array([0])}
    add = simulate(DATES9, o, h, l, c, picks, first_day=1, capital=40_000, costs=Costs.none(), skip_held=False)
    skip = simulate(DATES9, o, h, l, c, picks, first_day=1, capital=40_000, costs=Costs.none(), skip_held=True)
    assert add.avg_invested_share > skip.avg_invested_share


def test_cash_accounting_balances_across_tp_stop_expiry_and_open_positions():
    # 최종 현금 = 자본 + 확정 손익 - 아직 들고 있는 종목의 원가(수수료 포함). 어떤 청산이 섞여도 장부가 맞아야 한다.
    tp = [FLAT, FLAT, (10000, 10300, 10000, 10100)] + [FLAT] * 6                                   # 익절
    st = [FLAT, FLAT, (10000, 10000, 9700, 9800)] + [(9800, 9800, 9800, 9800)] * 6                 # 손절
    ex = [FLAT] * 3 + [(10000, 10000, 9900, 9900)] + [(9900, 9900, 9900, 9900)] * 5                # 기간만료(5일 중 변화 없음)
    hold = [FLAT] * 9                                                                              # 끝까지 보유
    o = np.array([[a[0], b[0], c_[0], d_[0]] for a, b, c_, d_ in zip(tp, st, ex, hold)], dtype=float)
    h = np.array([[a[1], b[1], c_[1], d_[1]] for a, b, c_, d_ in zip(tp, st, ex, hold)], dtype=float)
    l = np.array([[a[2], b[2], c_[2], d_[2]] for a, b, c_, d_ in zip(tp, st, ex, hold)], dtype=float)
    c = np.array([[a[3], b[3], c_[3], d_[3]] for a, b, c_, d_ in zip(tp, st, ex, hold)], dtype=float)
    costs = Costs(0.0005, 0.002, 0.001)
    r = simulate(DATES9, o, h, l, c, {0: np.array([0, 1, 2, 3])}, first_day=1, capital=400_000, costs=costs, stop=0.02, max_hold=2)
    assert r.exits_take_profit >= 1 and r.exits_stop >= 1
    assert abs(r.final_cash - (400_000 + r.closed_pnl - r.open_cost)) < 1e-6


def test_equal_allocation_buys_the_same_value_per_name_in_signal_order():
    p0 = [FLAT] * 6
    p1 = [(20000, 20000, 20000, 20000)] * 6
    o, h, l, c = panel(p0, p1)
    r = simulate(DATES, o, h, l, c, {0: np.array([1, 0])}, first_day=1, capital=100_000, costs=Costs.none(), alloc="equal", max_positions=2)
    # 슬롯 50,000원: 종목1(20,000원) 2주=40,000, 종목0(10,000원) 5주=50,000
    assert r.open_positions == 2
    assert abs(r.equity[-1] - 100_000) < 1e-6
    assert abs(r.final_cash - 10_000) < 1e-6
    one_slot = simulate(DATES, o, h, l, c, {0: np.array([1, 0])}, first_day=1, capital=100_000, costs=Costs.none(), alloc="equal", max_positions=1)
    assert one_slot.open_positions == 1                                                          # 자리가 하나뿐이면 신호 1순위만


def _both_barriers_day():
    # 매수 뒤 둘째 날: 시가 10,000 → 고가 10,300(+3%, 익절선 10,200 이상) 과 저가 9,700(-3%, 손절선 9,800 이하)이 같은 날 모두 닿는다.
    p0 = [FLAT, FLAT, (10000, 10300, 9700, 10000)] + [FLAT] * 3
    return panel(p0, [FLAT] * 6)


def test_same_bar_assumption_decides_which_barrier_is_hit_first():
    o, h, l, c = _both_barriers_day()
    kw = dict(first_day=1, capital=10_000, costs=Costs.none(), stop=0.02)
    stop_first = simulate(DATES, o, h, l, c, {0: np.array([0])}, same_bar="stop", **kw)
    target_first = simulate(DATES, o, h, l, c, {0: np.array([0])}, same_bar="target", **kw)
    assert (stop_first.exits_stop, stop_first.exits_take_profit) == (1, 0)
    assert abs(stop_first.equity[-1] - 9_800) < 1e-6
    assert (target_first.exits_stop, target_first.exits_take_profit) == (0, 1)
    assert abs(target_first.equity[-1] - 10_200) < 1e-6


def test_random_same_bar_hits_both_outcomes_across_many_positions():
    o, h, l, c = _both_barriers_day()
    # 같은 모양의 종목 60개: 동전 던지기면 익절·손절이 모두 나오고 합이 60 이다
    n = 60
    o2, h2, l2, c2 = (np.tile(x[:, :1], (1, n)) for x in (o, h, l, c))
    r = simulate(DATES, o2, h2, l2, c2, {0: np.arange(n)}, first_day=1, capital=600_000, costs=Costs.none(), stop=0.02, same_bar="random")
    assert r.exits_take_profit + r.exits_stop == n
    assert 10 < r.exits_take_profit < 50


def test_a_gap_beyond_a_barrier_wins_regardless_of_the_same_bar_assumption():
    p0 = [FLAT, FLAT, (9600, 10300, 9500, 10000)] + [FLAT] * 3           # 시가가 이미 손절선(9,800) 아래로 갭하락
    o, h, l, c = panel(p0, [FLAT] * 6)
    r = simulate(DATES, o, h, l, c, {0: np.array([0])}, first_day=1, capital=10_000, costs=Costs.none(), stop=0.02, same_bar="target")
    assert r.exits_stop == 1 and r.exits_take_profit == 0
    assert abs(r.equity[-1] - 9_600) < 1e-6                              # 손절선이 아니라 시가에 체결된다


def test_adv_cap_limits_the_amount_bought_per_name():
    o, h, l, c = panel([FLAT] * 6, [FLAT] * 6)
    adv = np.full((6, 2), 100_000.0)                                      # 하루 평균 거래대금 10만원
    r = simulate(DATES, o, h, l, c, {0: np.array([0])}, first_day=1, capital=1_000_000, costs=Costs.none(), alloc="equal", max_positions=1,
                 adv=adv, adv_cap=0.5)                                    # 한 종목 최대 5만원 = 5주
    assert abs(r.final_cash - 950_000) < 1e-6
    free = simulate(DATES, o, h, l, c, {0: np.array([0])}, first_day=1, capital=1_000_000, costs=Costs.none(), alloc="equal", max_positions=1)
    assert abs(free.final_cash - 0) < 1e-6                                # 한도가 없으면 100주(전액)


def test_oracle_pick_order_does_not_leak_which_names_will_rise():
    from tools.donga_strategy_backtest import oracle_picks, next_day_return
    rng = np.random.default_rng(0)
    c = np.cumprod(1 + rng.normal(0, 0.02, size=(40, 400)), axis=0) * 10_000
    v = np.full_like(c, 1e6)
    picks = oracle_picks(c, v, range(1, 38), 0.15, 0.65, seed=3)
    fwd = next_day_return(c)
    head_up = np.mean([np.mean(fwd[d][idx[: max(1, len(idx) // 2)]] > 0) for d, idx in picks.items()])
    tail_up = np.mean([np.mean(fwd[d][idx[len(idx) // 2:]] > 0) for d, idx in picks.items()])
    assert abs(head_up - tail_up) < 0.08                                  # 앞쪽 절반과 뒤쪽 절반의 상승 비율이 비슷해야 한다
    overall = np.mean([np.mean(fwd[d][idx] > 0) for d, idx in picks.items()])
    assert 0.55 < overall < 0.75                                          # 전체 적중률은 약 65%
