# Licensed under the GPLv3 - see LICENSE
import numpy as np
import astropy.units as u


def lincover(a, n):
    """Cover the range spanned by a in n points.

    Create an array that exactly covers the range spanned by ``a``, i.e.,
    ``a.min()`` is at the lower border of the first pixel and ``a.max()``
    is at the upper border of the last pixel.

    a : array
        Holding the values the range of which should be convered.
    n : int
        Number of points to use.

    Returns
    -------
    out : array
        Linearly increasing values.
    """
    start, stop = a.min(), a.max()
    step = (stop - start) / n
    return np.linspace(start + step/2, stop - step/2, n)


def remap_time(ds, t_map, new_t):
    """Remap DS(t, f) to new(t_map[t], f).

    Parameters
    ----------
    ds : array
        Dynamic spectrum that is to be remapped in time.
    t_map : array
        Holding the new times each old time (or each pixel) should map to.
    new_t : array or int
        Time array for the output.  The frequency array is assumed to be
        unchanged. Should be monotonously increasing. Input that covers
        the range in ``tmap`` can be created with ``lincover(t_map, n)``.

    Returns
    -------
    out, weight : array
        Summed fractional values and fractions

    See Also
    --------
    lincover : to create a linspace that exactly covers the range of an array

    """
    # For the whole map, find the pixel just before where it should go.
    ipix = np.clip(np.searchsorted(new_t, t_map), 1, len(new_t) - 1) - 1
    # Calculate the fractional pixel target positions.
    pix = u.Quantity((t_map - new_t[ipix]) / (new_t[ipix+1] - new_t[ipix]),
                     u.one, copy=False).value + ipix
    # Use these to calculate where the pixel boundaries would land.
    dpix = np.diff(pix, axis=0)
    bounds_l = pix - 0.5 * np.concatenate([dpix[:1], dpix])
    bounds_u = pix + 0.5 * np.concatenate([dpix, dpix[-1:]])
    # Ensure the lower bound is always below the upper bound.
    bounds_l, bounds_u = (np.minimum(bounds_l, bounds_u),
                          np.maximum(bounds_l, bounds_u))
    # Create output and weight arrays.
    out = np.zeros_like(ds, shape=(len(new_t),)+ds.shape[1:])
    weight = np.zeros((len(new_t),)+ds.shape[1:])
    # As well as a fake frequency pixel axis (needed for the case that t_map
    # depends on the frequency axis as well).
    if t_map.ndim == 2:
        f = np.broadcast_to(np.arange(ds.shape[1]), ds.shape)
    else:
        f = None

    # Find the range in pixels each input pixel will fall into.
    pix_start = np.maximum(np.round(bounds_l).astype(int), 0)
    pix_stop = np.minimum(np.round(bounds_u).astype(int) + 1, len(out))
    # Loop over the series of pixels inputs should go in to.
    for ipix in range((pix_stop - pix_start).max()):
        # Nominal output pixel index for each input pixel (for some this
        # will be beyond the actual needed range, but those will get weight 0).
        indx = pix_start + ipix
        # Calculate fraction of the output pixel covered by the input pixel.
        # Example: bounds_l, u = 0.9, 1.7, indx = 0:  0.5 - (-0.5) = 0.
        #                        0.9, 1.7, indx = 1:  0.5 - (-0.1) = 0.6
        #                        0.9, 1.7, indx = 2: -0.3 - (-0.5) = 0.2
        #                        0.9, 1.7, indx = 3: -0.5 - (-0.5) = 0.
        w = (np.clip(bounds_u-indx, -0.5, 0.5)
             - np.clip(bounds_l-indx, -0.5, 0.5))
        # Only care about pixels with a fraction > 0 that fall inside output.
        ok = (w > 0.) & (indx < len(out))
        # If locations vary with frequency, we need to pass in arrays for both
        # time and frequency.
        wok = w[ok]
        if ok.ndim == 2:
            indices = indx[ok], f[ok]
        else:
            indices = indx[ok]
            if ds.ndim == 2:
                wok = wok[:, np.newaxis]
        # Add fraction of each input pixel to output and track fractions added.
        np.add.at(out, indices, wok * ds[ok])
        np.add.at(weight, indices, wok)

    return out, weight
