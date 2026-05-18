"""Default parameter sets for impulse models."""


def default_two_gamma_impulse_glover_hrf() -> dict[str, float]:
    """Glover default parameters for the difference of two gamma distributions impulse model.

    Default parameters for a hemodynamic response function (HRF) impulse model following Glover et al. (1999) [1]_.

    Returns
    -------
    dict
        Dictionary with default values for each parameter of :class:`prfmodel.impulse.TwoGammaImpulse`.

    Notes
    -----
    The default values are adapted from
    `nilearn <https://github.com/nilearn/nilearn/blob/62934704a/nilearn/glm/first_level/hemodynamic_models.py>`_.

    References
    ----------
    .. [1] Glover, G. H. (1999). Deconvolution of impulse response in event-related BOLD fMRI. *NeuroImage*, 9(4),
        416-429. https://doi.org/10.1006/nimg.1998.0419

    Examples
    --------
    >>> default_two_gamma_impulse_glover_hrf()
    {'delay': 6.0, 'dispersion': 0.9, 'undershoot': 12.0, 'u_dispersion': 0.9, 'ratio': 0.48}

    """
    return {
        "delay": 6.0,
        "dispersion": 0.9,
        "undershoot": 12.0,
        "u_dispersion": 0.9,
        "ratio": 0.48,
    }


def default_two_gamma_impulse_spm_hrf() -> dict[str, float]:
    """SPM default parameters for the difference of two gamma distributions impulse model.

    Default parameters for a hemodynamic response function (HRF) impulse model following SPM default values [1]_.

    Returns
    -------
    dict
        Dictionary with default values for each parameter of :class:`prfmodel.impulse.TwoGammaImpulse`.

    Notes
    -----
    The default values are adapted from
    `nilearn <https://github.com/nilearn/nilearn/blob/62934704a/nilearn/glm/first_level/hemodynamic_models.py>`_.

    References
    ----------
    .. [1] https://www.fil.ion.ucl.ac.uk/spm/

    Examples
    --------
    >>> default_two_gamma_impulse_spm_hrf()
    {'delay': 6.0, 'dispersion': 1.0, 'undershoot': 16.0, 'u_dispersion': 1.0, 'ratio': 0.167}

    """
    return {
        "delay": 6.0,
        "dispersion": 1.0,
        "undershoot": 16.0,
        "u_dispersion": 1.0,
        "ratio": 0.167,
    }


def _fetch_default(name: str) -> dict[str, float]:
    match name:
        case "glover_hrf":
            return default_two_gamma_impulse_glover_hrf()
        case "spm_hrf":
            return default_two_gamma_impulse_spm_hrf()
        case other:
            msg = f"Default name {other} not supported"
            raise ValueError(msg)
