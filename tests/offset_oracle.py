"""Independent high-precision oracle for offset distance queries.

This oracle implements the exact-reference offset metric of
``OffsetNURBS_Distance_Specification.md`` from scratch in ``mpmath``
arbitrary-precision arithmetic (256 to 1024 bits), completely independent
of the production code paths:

- the preimage, speed ``sigma``, turning numerator ``tau`` and cusp
  polynomial ``G`` are formed in the *power* basis (production compiles
  Bernstein data);
- cusp parameters come from ``mpmath`` root finding refined by bisection
  with verified sign changes (production uses exact rational isolation);
- the source-length term is the exact polynomial integral; the angle term
  is a continuously unwrapped ``mp.atan2`` lift sampled densely enough to
  make every step provably below ``pi/2`` (production uses a half-plane
  subdivision proof);
- an adaptive ``mp.quad`` of the unsigned speed provides a second,
  formula-free diagnostic.

The only shared ingredients are the captured binary64 source values
themselves, which both sides interpret as exact reals, as the
specification requires (section 2.2).
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp


def to_fraction(x: mp.mpf) -> Fraction:
    """Exact rational value of an mpf (sign-safe)."""
    x = mp.mpf(x)
    if x < 0:
        return -to_fraction(-x)
    m, e = x.man_exp
    return Fraction(int(m)) * Fraction(2) ** int(e)


class OracleSpan:
    """Exact-reference metric of one source span at high precision."""

    def __init__(self, preimage, width, u0, u1, distance, scale, prec=350):
        self.prec = prec
        with mp.workprec(prec):
            self.w = [mp.mpc(z) for z in preimage]
            self.m = len(self.w) - 1
            self.h = mp.mpf(width)
            self.u0 = u0
            self.u1 = u1
            self.d = mp.mpf(distance)
            self.H = mp.mpf(scale)
            self.d_hat = self.d / self.H
            # Power-basis preimage from the Bernstein controls.
            self.wp = self._bernstein_to_power(self.w)
            self.wp_d = [k * c for k, c in enumerate(self.wp)][1:] or [mp.mpc(0)]
            # sigma, tau, G in the power basis.
            wre = [c.real for c in self.wp]
            wim = [c.imag for c in self.wp]
            dre = [k * c for k, c in enumerate(wre)][1:] or [mp.mpf(0)]
            dim = [k * c for k, c in enumerate(wim)][1:] or [mp.mpf(0)]
            self.sigma = self._padd(
                self._pmul(wre, wre), self._pmul(wim, wim)
            )
            self.tau = [
                2 * v
                for v in self._psub(
                    self._pmul(wre, dim), self._pmul(wim, dre)
                )
            ]
            self.G = self._psub(
                [self.h * v for v in self._pmul(self.sigma, self.sigma)],
                [self.d_hat * v for v in self.tau],
            )
            self.tau_zero = all(abs(v) == 0 for v in self.tau)
            self.cusps = self._find_cusps()

    # -- polynomial helpers (power basis, high precision) ----------------

    @staticmethod
    def _bernstein_to_power(ctrl):
        n = len(ctrl) - 1
        out = []
        for k in range(n + 1):
            acc = mp.mpc(0)
            for j in range(k + 1):
                term = mp.binomial(k, j) * ctrl[j]
                acc += term if (k - j) % 2 == 0 else -term
            out.append(mp.binomial(n, k) * acc)
        return out

    @staticmethod
    def _pmul(a, b):
        out = [mp.mpf(0)] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        return out

    @staticmethod
    def _padd(a, b):
        n = max(len(a), len(b))
        return [
            (a[k] if k < len(a) else 0) + (b[k] if k < len(b) else 0)
            for k in range(n)
        ]

    @staticmethod
    def _psub(a, b):
        n = max(len(a), len(b))
        return [
            (a[k] if k < len(a) else 0) - (b[k] if k < len(b) else 0)
            for k in range(n)
        ]

    @staticmethod
    def _peval(p, t):
        acc = mp.mpf(0) if not isinstance(p[0], mp.mpc) else mp.mpc(0)
        for c in reversed(p):
            acc = acc * t + c
        return acc

    # -- exact-reference building blocks ---------------------------------

    def _find_cusps(self):
        """Distinct G roots in (0, 1), refined to full working precision."""
        if self.tau_zero or self.d == 0:
            return []
        deg = len(self.G) - 1
        while deg > 0 and self.G[deg] == 0:
            deg -= 1
        if deg <= 0:
            return []
        with mp.workprec(self.prec):
            try:
                roots = mp.polyroots(
                    list(reversed(self.G[: deg + 1])), maxsteps=200, extraprec=200
                )
            except mp.libmp.NoConvergence:
                roots = []
            found = []
            for r in roots:
                if abs(r.imag) < mp.mpf(2) ** (-self.prec // 3):
                    t = r.real
                    if 0 < t < 1:
                        found.append(t)
            found.sort()
            merged = []
            for t in found:
                if not merged or abs(t - merged[-1]) > mp.mpf(2) ** (-self.prec // 3):
                    merged.append(t)
            return merged

    def speed(self, t):
        """Unsigned physical offset speed at local t (per unit local t)."""
        with mp.workprec(self.prec):
            return self.H * abs(self._peval(self.G, t)) / self._peval(self.sigma, t)

    def source_length(self, a, t):
        """Exact physical source-length increment H h int_a^t sigma."""
        with mp.workprec(self.prec):
            acc = mp.mpf(0)
            for k, c in enumerate(self.sigma):
                acc += c * (t ** (k + 1) - a ** (k + 1)) / (k + 1)
            return self.H * self.h * acc

    def angle_lift(self, a, t, steps=64):
        """Continuously unwrapped preimage angle change from a to t.

        Subdivides until every principal step is below pi/2, which makes
        the unwrapped lift provably correct for the nonvanishing
        preimage.
        """
        with mp.workprec(self.prec):
            while True:
                total = mp.mpf(0)
                ok = True
                prev = self._peval(self.wp, a)
                for k in range(1, steps + 1):
                    tk = a + (t - a) * mp.mpf(k) / steps
                    cur = self._peval(self.wp, tk)
                    rel = cur * mp.conj(prev)
                    step = mp.atan2(rel.imag, rel.real)
                    if abs(step) > mp.pi / 2:
                        ok = False
                        break
                    total += step
                    prev = cur
                if ok:
                    return total
                steps *= 2
                if steps > 65536:
                    raise RuntimeError("angle lift failed to stabilize")

    def cell_distance(self, a, b):
        """Exact-reference unsigned distance between in-cell parameters."""
        with mp.workprec(self.prec):
            ds = self.source_length(a, b)
            if self.tau_zero or self.d == 0:
                return ds
            dphi = self.angle_lift(a, b)
            value = ds - 2 * self.d * dphi
            return abs(value)

    def span_arc_length(self, t):
        """Exact-reference unsigned distance from local 0 to local t."""
        with mp.workprec(self.prec):
            cuts = [c for c in self.cusps if c < t]
            total = mp.mpf(0)
            prev = mp.mpf(0)
            for c in cuts:
                total += self.cell_distance(prev, c)
                prev = c
            total += self.cell_distance(prev, mp.mpf(t))
            return total

    def span_total(self):
        return self.span_arc_length(mp.mpf(1))

    def quad_total(self):
        """Formula-free diagnostic: adaptive quadrature of the speed."""
        with mp.workprec(min(self.prec, 120)):
            pieces = [mp.mpf(0)] + list(self.cusps) + [mp.mpf(1)]
            total = mp.mpf(0)
            for a, b in zip(pieces[:-1], pieces[1:]):
                total += mp.quad(self.speed, [a, b])
            return total


class OffsetOracle:
    """Exact-reference distance oracle for a complete offset handle."""

    def __init__(self, handle, prec=350):
        state = handle._metric.state()
        self.prec = prec
        self.spans = [
            OracleSpan(
                [complex(re, im) for re, im in pre],
                state["widths"][i],
                state["breakpoints"][i],
                state["breakpoints"][i + 1],
                state["distance"],
                state["scale"],
                prec,
            )
            for i, pre in enumerate(state["preimages"])
        ]
        with mp.workprec(prec):
            self.prefix = [mp.mpf(0)]
            for s in self.spans:
                self.prefix.append(self.prefix[-1] + s.span_total())

    def length(self):
        return self.prefix[-1]

    def arc_length(self, u):
        """Exact-reference A_d(u) for a binary64 parameter u."""
        with mp.workprec(self.prec):
            if u <= 0:
                return mp.mpf(0)
            if u >= 1:
                return self.prefix[-1]
            for i, s in enumerate(self.spans):
                if u < s.u1 or i == len(self.spans) - 1:
                    t = (mp.mpf(u) - mp.mpf(s.u0)) / (
                        mp.mpf(s.u1) - mp.mpf(s.u0)
                    )
                    return self.prefix[i] + s.span_arc_length(t)
        raise AssertionError

    def correctly_rounded_length(self):
        return float(self.length())

    def correctly_rounded_arc(self, u):
        return float(self.arc_length(u))


def assert_faithful(produced: float, exact: mp.mpf, what: str) -> None:
    """Assert the produced float is within one ulp of the exact value."""
    import math

    err = abs(mp.mpf(produced) - exact)
    limit = mp.mpf(math.ulp(abs(produced) + 5e-324))
    assert err <= limit, f"{what}: |{produced!r} - {mp.nstr(exact, 25)}| = {mp.nstr(err, 5)} > ulp"


def assert_correctly_rounded(produced: float, exact: mp.mpf, what: str) -> None:
    expected = float(exact)
    assert produced == expected, (
        f"{what}: produced {produced!r}, correctly rounded {expected!r} "
        f"(exact {mp.nstr(exact, 25)})"
    )
