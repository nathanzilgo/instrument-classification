import math


def pow2db(value):
    SILENCE_CUTOFF = 1e-10
    DB_SILENCE_CUTOFF = -100
    return (
        DB_SILENCE_CUTOFF
        if value < SILENCE_CUTOFF
        else 10.0 * math.log10(value)
    )


def squeeze_range(x, x1, x2):
    return 0.5 + 0.5 * math.tanh(-1.0 + 2.0 * (x - x1) / (x2 - x1))
