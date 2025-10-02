import os
import numpy as np
from .units import ms
from matplotlib.colors import LinearSegmentedColormap
from astropy.constants import M_earth, M_sun
#colormaps
transgrad = LinearSegmentedColormap.from_list('transmission_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%209491=008080-FFF2F2-440154
    (0.000, (1.000, 0.000, 0.000)),  # Bright red (#FF0000) at the low end
    (0.500, (1.000, 1.000, 1.000)),  # Bright gray (white) at the middle
    (1.000, (0.000, 0.000, 0.545))))

# Define color gradients
star_gradient = LinearSegmentedColormap.from_list(
    "star_gradient",
    [
        (0.000, (0.000, 0.169, 0.867)),
        (0.050, (0.000, 0.169, 0.867)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.950, (0.867, 0.000, 0.031)),
        (1.000, (0.867, 0.000, 0.031)),
    ],
)

star_gradient_flux = LinearSegmentedColormap.from_list(
    "star_gradient",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%209491=0:FFFFFF-20:786500-25:806B00-30:887200-35:907900-40:988000-45:A08700-50:A88E00-55:B09500-60:B99C00-65:C1A300-70:CAAA00-75:D2B100-80:DBB900-85:E4C000-90:EDC700-95:F6CE00-100:FFD600
        (0.000, (0.184, 0.157, 0.000)),
        (0.063, (0.227, 0.192, 0.000)),
        (0.125, (0.271, 0.231, 0.000)),
        (0.188, (0.318, 0.271, 0.000)),
        (0.250, (0.365, 0.310, 0.000)),
        (0.313, (0.412, 0.349, 0.000)),
        (0.375, (0.463, 0.392, 0.000)),
        (0.438, (0.510, 0.435, 0.000)),
        (0.500, (0.561, 0.478, 0.000)),
        (0.563, (0.612, 0.522, 0.000)),
        (0.625, (0.667, 0.565, 0.000)),
        (0.688, (0.718, 0.608, 0.000)),
        (0.750, (0.773, 0.655, 0.000)),
        (0.813, (0.827, 0.698, 0.000)),
        (0.875, (0.886, 0.745, 0.000)),
        (0.938, (0.941, 0.792, 0.000)),
        (1.000, (1.000, 0.839, 0.000)),
    ),
)

planet_gradient_2 = LinearSegmentedColormap.from_list(
    "planet_gradient_2",
    [
        (0.000, (1.000, 0.365, 0.051)),
        (0.119, (0.961, 0.510, 0.000)),
        (0.238, (0.918, 0.620, 0.000)),
        (0.357, (0.867, 0.714, 0.000)),
        (0.476, (0.804, 0.800, 0.243)),
        (0.607, (0.686, 0.816, 0.310)),
        (0.738, (0.553, 0.824, 0.408)),
        (0.869, (0.404, 0.827, 0.506)),
        (1.000, (0.224, 0.824, 0.600)),
    ],
)

c = 299792458.0 * ms
sqrt2pi = np.sqrt(2.0 * np.pi)

def compute_planet_doppler_shift(sim, psi, absorption_spec):
    from.fast_starspot import doppler_shift
    """
    Compute Doppler-shifted absorption spectra due to planet's orbital motion.

    Parameters:
    - sim: simulation object containing planet and star parameters.
    - psi_planet: orbital phase array of the planet.
    - absorption_spec: array (or list) of absorption spectra to shift.
    - M_sun: solar mass constant (in appropriate mass units).
    - M_earth: Earth mass constant (in appropriate mass units).

    Returns:
    - absorpt_prest: array of Doppler-shifted absorption spectra.
    """
    psi_planet=(psi*sim.star.prot/sim.planet.P).value
    Kp = (sim.planet.K) * (M_sun * sim.star.mass) / (sim.planet.Mp * M_earth)
    pshift = (Kp * np.sin(2 * np.pi * psi_planet)).value
    absorpt_prest = np.array([
        doppler_shift(sim.pixel.wave, absorption_spec[x], -1000 * pshift[x])
        for x in range(len(pshift))
    ])
    return absorpt_prest

def transit_durations(sim):
    """
    Calculate transit duration and ingress/egress duration for a planet transiting a star.
    Supports both circular and eccentric orbits.

    Parameters:
    P  : float
        Orbital period (same time units as desired output)
    a  : float
        Semi-major axis in units of stellar radii (a/R_star)
    Rp : float
        Planet radius in units of stellar radii (R_p/R_star)
    ip : float
        Orbital inclination in degrees (90° is edge-on)
    e  : float, optional
        Orbital eccentricity (default 0 for circular orbits)
    w  : float, optional
        Argument of periastron in degrees (default 90° means periastron at transit)

    Returns:
    T14 : float
        Total transit duration (first to last contact), in same units as P
    tau : float
        Ingress or egress duration
    
    Notes:
    - The formulas follow the analytic approximations described by Seager & Mallén-Ornelas (2003)
      for circular orbits, extended for eccentricity following Kipping (2010):
      https://arxiv.org/abs/1004.3819
    - The eccentricity correction is applied through the planet's distance at transit
      and the velocity factor.

    References:
    Seager & Mallén-Ornelas (2003), The Astrophysical Journal 585, 1038
    Kipping (2010), MNRAS, 407, 301, https://arxiv.org/abs/1004.3819
    """

    a=sim.planet.a.value
    P=1
    ip=sim.planet.ip.value
    Rp=sim.planet.Rp.value
    e=sim.planet.e
    w=sim.planet.w.value

    # Convert degrees to radians for angle calculations
    i = np.radians(ip)
    w_rad = np.radians(w)

    # Computes the star-planet distance ratio at mid-transit (normalized by a)
    r_tp = (1 - e**2) / (1 + e * np.sin(w_rad))

    # Impact parameter b (normalized by stellar radius)
    b = a * r_tp * np.cos(i)

    # Velocity correction factor accounting for orbital speed at transit
    velocity_factor = np.sqrt(1 - e**2) / (1 + e * np.sin(w_rad))

    # Calculate total transit duration T14 using modified formula:
    # arcsin argument: chord length over the projected orbital distance factor including eccentricity
    arg_total = np.sqrt((1 + Rp)**2 - b**2) / (a * r_tp * np.sin(i))
    # Clip argument to valid range [-1,1] to avoid NaNs
    arg_total = np.clip(arg_total, -1, 1)
    T14 = (P / np.pi) * velocity_factor * np.arcsin(arg_total)

    # Calculate full transit duration T23 (between second and third contacts)
    arg_full = np.sqrt((1 - Rp)**2 - b**2) / (a * r_tp * np.sin(i))
    arg_full = np.clip(arg_full, -1, 1)
    T23 = (P / np.pi) * velocity_factor * np.arcsin(arg_full)

    # Ingress or egress duration is half the difference
    #tau = 0.5 * (T14 - T23)

    return T14, T23

def read_rdb(fname):
    d = np.loadtxt(fname, skiprows=2)
    keys = ["vrad", "CCF", "CCF_spot", "CCF_plage"]
    data = {key: d[:, i] for i, key in enumerate(keys)}
    return data


def download_goettingen_solar_atlas():
    import requests

    url = "http://www.astro.physik.uni-goettingen.de/research/solar-lib/"
    print(f"Downloading IAG solar atlas from {url}")

    μ = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1]

    for μi in μ:
        url = "http://www.astro.physik.uni-goettingen.de/"
        path = "research/solar-lib/data/"
        filename = f"solarspectrum_mu{μi:.2f}.fits"
        here = os.path.dirname(__file__)
        folder = os.path.join(here, "../data/IAGatlas")
        os.makedirs(folder, exist_ok=True)
        file = os.path.join(folder, filename)
        print(filename)
        if not os.path.exists(file):
            with requests.get(url + path + filename, stream=True) as r:
                r.raise_for_status()
                with open(file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
