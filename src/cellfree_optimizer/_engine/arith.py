"""Arith."""

from fractions import Fraction as F
import math
import functools
from .bounds import outward_rate

GRID = 2**32


def f(v):
    return F.from_float(float(v))


def pair(q):
    return [str(q.numerator), str(q.denominator)]


def unpack(v):
    return F(int(v[0]), int(v[1]))


def root(q, up=False):
    r = math.isqrt(q.numerator * GRID * GRID // q.denominator)
    if up and F(r, GRID) ** 2 < q:
        r += 1
    return F(r, GRID)


def scale(values):
    s = 1
    v = max(map(abs, values), default=F(1))
    while v > s:
        s *= 2
    return s


def lower_int(v):
    return v.numerator * GRID // v.denominator


def upper_int(v):
    return -lower_int(-v)


@functools.lru_cache(None)
def ln2():
    z = F(1, 3)
    value = 2 * sum((z ** (2 * j + 1) / (2 * j + 1) for j in range(48)))
    tail = 2 * z**97 / (97 * (1 - z * z))
    return (value, value + tail)


def tangent(B, p0, rho):
    low, up = ln2()
    r = f(rho)
    alpha_up = r * B / (1 + B * p0) / low
    alpha_low = r * B / (1 + B * p0) / up
    value = 1 + B * p0
    exponent = 0
    while value >= 2:
        value /= 2
        exponent += 1
    beta_up = r * exponent + f(outward_rate(value - 1, rho)) - alpha_low * p0
    assert beta_up >= 0
    return (alpha_up, beta_up)
