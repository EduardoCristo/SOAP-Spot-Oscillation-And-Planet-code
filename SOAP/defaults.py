import numpy as np

from .classes import ActiveRegion, Planet, Star, solarCCF
from .units import U

_default_psi = np.linspace(0, 1, 501)

# the Sun...
_default_STAR = Star(
    prot=25.05,
    incl=90.0,
    diffrotB=0.0,
    diffrotC=0.0,
    u1=0.29,
    u2=0.34,
    start_psi=0.0,
    radius=1.0,
    mass=1.0,
    teff=5778,
)


_default_CCF = solarCCF(vrot=_default_STAR.vrot)
_default_CCF_active_region = solarCCF(vrot=_default_STAR.vrot, active_region=True)

_default_PLANET = Planet(
    P=4.0,
    a=8.76,
    Rp=0.1,
    Mp=1.0,
    e=0.0,
    w=90.0,
    ip=90.0,
    lbda=0.0,
    t0=0.0,
)

Earth = Planet(
    P=365.25,
    t0=0.0,
    e=0.0167,
    w=90.0,
    ip=90.0,
    lbda=0.0,
    a=1 * U.au,
    Rp=1 * U.R_earth,
    Mp=1.0,
)

# the one default active region, as in config.cfg
_default_ACTIVE_REGIONS = []

active_region = ActiveRegion(
    check=True,
    lon=180.0,
    lat=30.0,
    size=0.1,
    active_region_type=0,
    temp_diff=663,  # in K, this is the default value
)

# active_region.calc_maps(_default_STAR, 300, 20)
_default_ACTIVE_REGIONS.append(active_region)


# a number of random activity regions
_default_many_ACTIVE_REGIONS = []

for _ in range(5):
    AR = ActiveRegion.random()
    # AR.calc_maps(_default_STAR, 300, 20)
    _default_many_ACTIVE_REGIONS.append(AR)
