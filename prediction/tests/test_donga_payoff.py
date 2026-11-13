"""거래 손익 라벨이 simulate() 의 청산 규칙과 같은 방식으로 계산되는지 확인한다."""
import numpy as np

from tools.donga_payoff_backtest import edge, payoff_labels, picks_from_scores, walk_forward_scores
from tools.donga_strategy_backtest import build_features


def _paths():
    D, S = 7, 5
    o = np.full((D, S), 100.0)
    h = np.full((D, S), 100.0)
    l = np.full((D, S), 100.0)
    c = np.full((D, S), 100.0)
    h[1, 4] = 105.0                         # E: 매수일(1일) 장중 +5% 는 청산 사유가 아니다
    h[2, 0] = 103.0                         # A: 2일에 익절선(102) 도달
    l[2, 1] = 97.0                          # B: 2일에 손절선(98) 도달
    c[3, 2] = 99.0                          # C: 못 닿고 3일(=매수 후 hold 일째) 종가 99
    o[2, 3], l[2, 3] = 95.0, 94.0           # D: 손절선 아래로 갭하락 → 시가에 체결
    return o, h, l, c


def test_labels_follow_take_profit_stop_expiry_and_gap_rules():
    o, h, l, c = _paths()
    y = payoff_labels(o, h, l, c, tp=0.02, stop=0.02, hold=2)
    assert np.allclose(y[0], [0.02, -0.02, -0.01, -0.05, 0.0])
    assert np.isnan(y[-3:]).all() and np.isfinite(y[:4]).any()          # 라벨이 확정되지 않은 마지막 날들은 NaN
    no_stop = payoff_labels(o, h, l, c, tp=0.02, stop=None, hold=2)
    assert np.allclose(no_stop[0], [0.02, 0.0, -0.01, 0.0, 0.0])        # 손절이 없으면 B·D 는 3일 종가(100)로 정리


def test_same_bar_tie_goes_to_the_gap_side_and_otherwise_to_a_coin_flip():
    D, S = 7, 400
    o = np.full((D, S), 100.0)
    h = np.full((D, S), 100.0)
    l = np.full((D, S), 100.0)
    c = np.full((D, S), 100.0)
    h[2], l[2] = 103.0, 97.0                                            # 둘 다 닿는다
    y = payoff_labels(o, h, l, c, tp=0.02, stop=0.02, hold=2)[0]
    share_tp = np.mean(np.isclose(y, 0.02))
    assert 0.35 < share_tp < 0.65 and np.isclose(share_tp + np.mean(np.isclose(y, -0.02)), 1.0)
    o2 = o.copy()
    o2[2] = 96.0                                                         # 시가가 이미 손절선 아래 → 손절이 먼저
    y2 = payoff_labels(o2, h, l, c, tp=0.02, stop=0.02, hold=2)[0]
    assert np.allclose(y2, 96.0 / 100.0 - 1)


def test_walk_forward_scores_use_only_resolved_labels_and_rank_by_expected_payoff():
    rng = np.random.default_rng(3)
    D, S = 420, 80
    c = np.cumprod(1 + rng.normal(0, 0.02, size=(D, S)), axis=0) * 100
    o, h, l = c * 0.999, c * 1.012, c * 0.988
    v = np.full_like(c, 1e6)
    feat = build_features(o, h, l, c, v)
    y = payoff_labels(o, h, l, c, 0.02, None, 5)
    scores = walk_forward_scores(feat, y, c, v, range(300, 400), 200, 25, 5)
    assert scores and all(len(i) == len(p) for i, p in scores.values())
    pk = picks_from_scores(scores, 0.1)
    assert all(len(x) == max(1, round(len(scores[d][0]) * 0.1)) for d, x in pk.items())
    e = edge(scores, y, [0.1])
    assert "차이%p" in e.columns and len(e) == 1
