from __future__ import annotations

import math
import numpy as np


def sph2cart(r: np.ndarray, azi: np.ndarray, ele: np.ndarray) -> np.ndarray:
    r"""Spherical to Cartesian coordinate.

    Args:
        r: (n,), radius
        azi: (n,), azimuth, [0, 2π)
        ele: (n,) ,elevation, [-π/2, π/2]. (ele = π/2 - colatidue)

    Outputs:
        out: (n, 3), Cartesian coordinate
    """
    x = r * np.cos(ele) * np.cos(azi)
    y = r * np.cos(ele) * np.sin(azi)
    z = r * np.sin(ele)
    out = np.stack([x, y, z], axis=-1)
    return out


def cart2sph(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Cartesian to spherical coordinate.

    Args:
        v: (n, 3), Cartesian coordinate

    Returns:
        r: (n,), radius
        azi: (n,), azimuth, [0, 2π)
        ele: (n,), elevation, [-π/2, π/2]. (ele = π/2 - colatidue)
    """
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azi = np.arctan2(y, x)
    ele = np.arcsin(z / r)
    return r, azi, ele


def fractional_delay_filter(delayed_samples: int) -> np.ndarray:
    r"""Fractional delay with Whittaker–Shannon interpolation formula. 
    Ref: https://tomroelandts.com/articles/how-to-create-a-fractional-delay-filter

    Args:
        x: (n,), input signal
        delay_samples: float >= 0.

    Outputs:
        out: (n,), delayed signal
    """

    d_integer = math.floor(delayed_samples)
    d_fraction = delayed_samples % 1

    N = 99     # Filter length.
    n = np.arange(N)

    # Compute sinc filter.
    center = (N - 1) // 2
    h = np.sinc(n - center - d_fraction)

    # Multiply sinc filter by window.
    h *= np.blackman(N)
    
    # Normalize to get unity gain.
    h /= np.sum(h)

    # Combined filter.
    new_len = np.abs(d_integer) * 2 + N
    new_h = np.zeros(new_len)

    bgn = (new_len - 1) // 2 + d_integer - (N - 1) // 2
    end = (new_len - 1) // 2 + d_integer + (N - 1) // 2
    new_h[bgn : end + 1] = h

    return new_h