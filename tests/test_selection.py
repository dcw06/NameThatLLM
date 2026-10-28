"""Nested selection keeps the declared tie order even when float rounding splits a tie."""

from ntl.selection import first_best, snap_tie


def test_ties_split_by_float_rounding_keep_the_first_option():
    # The 18-class head's fold 3: 902/2160 at both b = 0 and b = 1/4, one ulp apart as float means.
    scores = {0.0: 0.4175925925925925, 0.25: 0.41759259259259257, 0.5: 0.41}
    assert scores[0.25] > scores[0.0]
    assert first_best((0.0, 0.25, 0.5), scores.__getitem__) == 0.0
    # QLoRA control2, fold 3: three settings at exactly 137/360, in (duration, rate) order.
    means = {(1e-05, 0.25): 0.38055555555555554, (3e-05, 0.25): 0.3805555555555556,
             (1e-05, 1): 0.38055555555555554, (3e-05, 2): 0.37}
    assert first_best(sorted(means, key=lambda setting: (setting[1], setting[0])), means.__getitem__) == (1e-05, 0.25)


def test_a_clear_best_wins_and_the_order_given_is_the_tie_order():
    assert first_best((0.0, 0.25, 0.5), {0.0: 0.40, 0.25: 0.41, 0.5: 0.41}.__getitem__) == 0.25
    assert first_best((0.5, 0.25), {0.25: 0.41, 0.5: 0.41}.__getitem__) == 0.5
    assert first_best(iter((0.0, 0.25)), {0.0: 0.3, 0.25: 0.4}.__getitem__) == 0.25  # any iterable


def test_differences_within_float_rounding_of_zero_snap_to_zero():
    assert 0.1 + 0.2 - 0.3 != 0 and snap_tie(0.1 + 0.2 - 0.3) == 0.0
    assert snap_tie(-5.551115123125783e-17) == 0.0 and snap_tie(-1e-12) == 0.0
    assert snap_tie(2.5e-6) == 2.5e-6 and snap_tie(-1 / 2160) == -1 / 2160
