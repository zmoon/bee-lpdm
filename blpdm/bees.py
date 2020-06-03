"""
Bee flight model

Based on the Lévy flight random walk model as implemented in
- J.D. Fuentes et al. / Atmospheric Environment (2016) 

"""

import numpy as np


def get_step_length(*, l_0=1.0, mu=2.0):
    """
    Draw a step length `l` from the distribution
      p(l) = (l/l_0)^-mu


    Parameters
    ----------
    l_0 : float
        the minimum step size
    mu : distribution shape parameter
        =3 => Brownian motion
        =2 => "super diffusive Lévy walk"

    """
    r = np.random.rand()  # draw from [0, 1]
    l = l_0 * r**(1/-mu)

    return l

