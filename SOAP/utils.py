import os

import numpy as np

from .units import ms

c = 299792458.0 * ms
sqrt2pi = np.sqrt(2.0 * np.pi)


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
