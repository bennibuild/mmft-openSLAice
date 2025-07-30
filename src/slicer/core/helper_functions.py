
def round_TOL(value: float, tol: float) -> float:
    """
    Round a value to the nearest multiple of a given tolerance rounded to 4 decimal places for numerical stability.
    Parameters
    ----------
    value : float
        The value to be rounded.
    tol : float
        The tolerance to round to.
    Returns
    -------
    float
        The rounded value.
    Raises
    ------
    ValueError
        If the tolerance is not positive and non-zero.
    """
    if tol <= 0:
        raise ValueError("Tolerance must be positive and non-zero.")
    return round(round(value / tol) * tol, 4)

