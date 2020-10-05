from numpy import pi, exp
from scipy.special import gamma


def b(n):
    """ Get the b_n normalization constant for the sersic profile. From Graham and Driver."""
    return 2 * n - (1 / 3) + (4 / (405 * n)) + (46 / (25515 * (n ** 2)))


def i_at_r50(mag, n=2, r_50=2, m_0=27):
    """ Get the intensity at the half-light radius """
    b_n = b(n)
    l_tot = 10 ** ((mag - m_0) / -2.5)
    denom = (r_50 ** 2) * 2 * pi * n * exp(b_n) * gamma(2 * n) / (b_n ** (2 * n))
    i_e = l_tot / denom

    return i_e


