"""Nested selection: the best option by mean inner score, under a declared tie order."""

# Inner balanced accuracies are rationals with small denominators, so distinct option
# scores differ by far more than this, while float means of equal scores can differ in
# the last bit: an exact tie at 902/2160 once came out one ulp apart.
TIE_TOLERANCE = 1e-12


def first_best(options, score):
    """The first option, in tie order, whose score is the best up to float rounding."""
    options = list(options)
    best = max(score(option) for option in options)
    return next(option for option in options if score(option) >= best - TIE_TOLERANCE)


def snap_tie(difference: float) -> float:
    """`difference`, or exactly zero when it is within float rounding of zero: two balanced
    accuracies that tie exactly can differ by about 1e-17 of either sign."""
    return 0.0 if abs(difference) <= TIE_TOLERANCE else float(difference)
