"""
Bee flight model

Based on the Lévy flight random walk model as implemented in
- J.D. Fuentes et al. / Atmospheric Environment (2016) 

"""

import numpy as np
from scipy.stats import rv_continuous


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
    if mu < 1 or mu > 3:
        raise ValueError(f"`mu` should be in [1, 3] but is {mu!r}")

    u = np.random.rand()  # draw from [0, 1] (uniform)
    l = l_0 * u**(1/(1-mu))
    # ^ note 1-mu not mu, this comes from the inverse of the CDF
    # ? does this work for the general case, or only mu=2?
    # does seem to work well in the l_0=1, mu in [2, 2.5] cases

    return l



# scipy.stats.powerlaw doesn't let its param `a` be negative, so can't use it

class step_length_dist_gen(rv_continuous):
    """Step length distribution class
    using the scipy.stats machinery
    """

    def _pdf(self, x, l_0, mu):
        return (x/l_0)**(-mu)

    def _cdf(self, x, l_0, mu):
        # https://www.wolframalpha.com/input/?i=anti-derivative+%28l%2Fc%29%5E-mu+dl
        # https://www.wolframalpha.com/input/?i=anti-derivative+%28l%2Fc%29%5E-mu+dl%2C+from+l%3Dc+to+x
        # return l_0**mu * x**(1-mu) / (1-mu)
        # return x * (x/l_0)**(-mu) / (1-mu)
        
        c = -l_0 / (1-mu)
        # if x >= l_0:
        #     return x * (x/l_0)**(-mu) / (1-mu) + c
        # else:
        #     return 0
        F = x * (x/l_0)**(-mu) / (1-mu) + c
        # F[x < l_0] = 0
        return F

        # return (l_0 - x*(x/l_0)**(-mu)) / (mu-1)

    # these forms seem to only work in a few special cases.
    # if we don't provide _ppf it will numerically estimate it from _cdf
    # def _ppf(self, q, l_0, mu):
    #     # return (q*(mu-1) - l_0 + (1/l_0)**-mu)**(1/(1-mu))
    #     # return (q*(mu-1) - l_0 + l_0**mu)**(1/(1-mu))

    #     x = (q*(mu-1) - l_0 + l_0**mu)**(1/(1-mu))
    #     # x[x < l_0] = np.nan
    #     return x

    #     # return ( (q-1)*(1-mu) / l_0**mu )**(1/(1-mu))

    def _argcheck(self, *args):
        # below is the default
        # --------------------
        # cond = 1
        # print(args)
        # for arg in args:
        #     cond = np.logical_and(cond, (np.asarray(arg) > 0))
        # return cond

        l_0, mu = args  # unpack

        # l_0 (minimum step length) must be positive
        cond_l_0 = np.asarray(l_0) > 0

        # shape parameter mu is supposed to be in range [1, 3]
        # not sure why it can't just be positive tho...
        arr_mu = np.asarray(mu)
        cond_mu = arr_mu >= 1 and arr_mu <= 3

        return np.logical_and(cond_l_0, cond_mu)


    def _get_support(self, *args, **kwargs):
        # need to return support as 2-tuple
        return args[0], np.inf


# step_length_dist = step_length_dist_gen(a=1.0, name="step_length_dist")
step_length_dist = step_length_dist_gen(name="step_length_dist")


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # test the step length dist
    mu = 2.0
    l_0 = 1.0
    n_steps = 20000
    x_stop = 10  # in the plots
    steps = [get_step_length(l_0=l_0, mu=mu) for _ in range(n_steps)]
    fig, [ax, ax2] = plt.subplots(1, 2)
    bins = np.arange(0, x_stop, 0.1)
    ax.hist(steps, bins=bins, density=True, histtype='stepfilled', alpha=0.7, 
        label="orig"
    )

    # compare to the scipy.stats version
    # x = np.linspace(l_0, x_stop, 200)
    x = np.linspace(0, x_stop, 200)
    dist = step_length_dist(mu=mu, l_0=l_0)
    steps2 = dist.rvs(n_steps)
    ax.hist(steps2, bins=bins, density=True, histtype='stepfilled', alpha=0.3, color="green",
        label="using .rvs"
    )
    ax.plot(x, dist.pdf(x), label="pdf - analytical")

    # ax2.plot()
    bins = np.linspace(0, 50, 200)  # like integrating from 0->inf
    ax2.plot(x, dist.cdf(x), label="cdf - analytical")
    # ax2.hist(steps2[steps2 > l_0], 
    ax2.hist(steps2, 
        bins=bins, density=True, cumulative=True, histtype='stepfilled', alpha=0.3, color="green",
        label="using .rvs"
    )

    ax.set_xlim(xmin=0)
    ax.legend()
    ax2.set_xlim(xmin=0, xmax=x_stop*1.5)
    ax2.legend()

    fig.tight_layout()

    plt.show()

