from astropy.units import Quantity


def axis_extent(*args):
    """Treat the arguments as axis arrays of an image, and get their extent.

    Just the first and last element of each array, but properly taking into
    account that the limits of the image are half a pixel before and after.

    Parameters
    ----------
    args : array
        Arrays with coordinates of the images.  Assumed to be contiguous,
        and have only one dimension with non-unity shape.

    Returns
    -------
    extent : list of int
        Suitable for passing into matplotlib's ``imshow``.
    """
    result = []
    for a in args:
        x = Quantity(a).squeeze().value
        result.extend([x[0] - (x[1]-x[0])/2, x[-1] + (x[-1]-x[-2])/2])
    return result
