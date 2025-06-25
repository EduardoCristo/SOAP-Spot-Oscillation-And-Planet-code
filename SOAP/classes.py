import os
import uuid
from glob import glob
from pathlib import Path

import numpy as np
from airvacuumvald import vacuum_to_air
from astropy.io import fits
from expecto import get_spectrum
from scipy.interpolate import RegularGridInterpolator

from .fast_starspot import doppler_shift
from .gaussian import ip_convolution
from .units import U, has_unit, kms, maybe_quantity_input, unit_arange
from .utils import c, read_rdb, sqrt2pi


def set_object_attributes(object, attrs):
    """
    Set (possibly multiple) attributes of a given object, taking care of each
    attribute's units, if it has them.
    """
    for k, v in attrs.items():
        try:
            old = getattr(object, k)
            if has_unit(old):
                setattr(object, k, v << old.unit)
            else:
                setattr(object, k, v)
        except AttributeError:
            print(f'attribute "{k}" does not exist')


class Star:
    """This object holds information about the star

    Parameters
    ----------
    prot : float
        Stellar rotation period [days]. If differential rotation is on, this is
        the rotation period at the equator.
    incl : float
        Inclination of the rotational axis [degrees]
    diffrotB, diffrotC : float
        Coefficients for the latitudinal differential rotation
            w(lat) = w_eq - diffrotB * sin^2(lat) - diffrotC * sin^4(lat)
        (NOTE the minus signs!)
    cb1: float
        Absolute value of the convective blueshift (m/s)
        V_CB=cb1*r*cos(theta)
    u1, u2 : float
        Linear and quadratic coefficients of the quadratic limb-darkening law
    radius : float
        Stellar radius [Rsun]
    mass : float
        Stellar mass [Msun]
    teff : float
        Effective temperature of the star [K]
    start_psi : float
        Starting phase [in units of rotation period]
    """

    @maybe_quantity_input(
        prot=U.day, incl=U.deg, teff=U.K, radius=U.R_sun, mass=U.M_sun
    )
    def __init__(
        self,
        prot=25,
        incl=90,
        diffrotB=0,
        diffrotC=0,
        cb1=0,
        u1=0.29,
        u2=0.34,
        radius=1,
        mass=1,
        teff=5778,
        start_psi=0,
    ):

        self.prot = prot
        self.incl = incl
        self.diffrotB, self.diffrotC = diffrotB, diffrotC
        self.cb1 = cb1
        self.u1, self.u2 = u1, u2
        self.start_psi = start_psi
        self.radius = radius
        self.mass = mass
        self.teff = teff
        # default units for vrot
        self._vrot_units = kms

    def set_vrot_units(self, units):
        self._vrot_units = units

    def __setattr__(self, name, value):
        try:
            old = getattr(self, name)
            if has_unit(old):
                super().__setattr__(name, value << old.unit)
            else:
                super().__setattr__(name, value)
        except AttributeError:
            super().__setattr__(name, value)
        # try to update self.vrot
        try:
            _ = self.vrot
        except AttributeError:
            pass

    @property
    def vrot(self):
        """Rotation velocity of the star at the equator"""
        vrot = (2 * np.pi * self.radius) / (self.prot)
        vrot <<= self._vrot_units
        try:  # update CCFs vrot
            self._ccf.vrot = vrot
            self._ccf_active_region.vrot = vrot
        except AttributeError:
            pass
        return vrot

    def __repr__(self):
        pars = (
            f"prot={self.prot}; incl={self.incl}; radius={self.radius}; "
            f"teff={self.teff:.0f}"
        )
        return "SOAP.Star(%s)" % pars

    def set(self, **kwargs):
        set_object_attributes(self, kwargs)


class ActiveRegion:
    """An active region (spot or plage)

    Parameters
    ----------
    lon, lat : float
        Longitude and latitude of the active region [degree]
    size : float
        The size of the active region, in units of stellar radius. To get a size
        S1 in area of the visible hemisphere, provide sqrt(2*S1).
        Example: for 0.1% of the visible hemisphere, provide 0.045
    active_region_type : int
        Either 0 for a spot or 1 for a plage. By default, a spot.
    temp_diff : float, default 663K for spots and -250K for plage
        Temperature difference between active region and the surface [K]
    check : boolean
        Turn this active region on (check=True) or off (check=False)
    """

    _temperature_differences = {"spot": 663 * U.K, "plage": 250 * U.K}
    _default_temp_diff = True

    @maybe_quantity_input(lon=U.deg, lat=U.deg, temp_diff=U.K)
    def __init__(
        self, lon, lat, size, active_region_type=0, temp_diff=None, check=True
    ):
        self.lon = lon
        self.lat = lat
        self.size = size
        self.check = check

        if active_region_type in (0, 1):
            self.active_region_type = active_region_type
        elif active_region_type in ("spot", "plage"):
            self.active_region_type = {"spot": 0, "plage": 1}[active_region_type]
        else:
            raise ValueError('active_region_type should be 0/1 or "spot"/"plage"')

        if temp_diff is None:
            self.temp_diff = self._temperature_differences[self.type]
        else:
            self._default_temp_diff = False
            self.temp_diff = temp_diff

    def __repr__(self):
        pars = f"{self.type}; lon={self.lon:.2f}; lat={self.lat:.2f};"

        if isinstance(self.size, float):
            pars += " size=%.2f" % self.size
        elif isinstance(self.size, np.ndarray):
            pars += " size=array"
        elif callable(self.size):
            pars += " size=f(t)"

        return "SOAP.ActReg(%s)" % pars

    @classmethod
    def random(cls, ARtype="spot"):
        AR = cls(
            check=1,
            lon=np.random.uniform(0, 360),
            lat=np.random.normal(0, 30),
            size=np.random.uniform(0, 0.2),
            active_region_type=ARtype,
        )
        return AR

    @property
    def size_area_visible_hemisphere(self):
        """Size of the active region, in area of the visible hemisphere"""
        # S1 = area_AR / area_visible_hemisphere
        #    = pi*(size*Rstar)**2/(2*pi*Rstar**2) = size**2/2
        try:
            return self.size**2 / 2.0
        except:
            return np.nan

    @size_area_visible_hemisphere.setter
    def size_area_visible_hemisphere(self, value):
        self.size = np.sqrt(2 * value)

    @property
    def type(self):
        """Active region type, 'spot' or 'plage'"""
        if self.active_region_type == 0:
            return "spot"
        elif self.active_region_type == 1:
            return "plage"

    @type.setter
    def type(self, type):
        if isinstance(type, str):
            type = type.lower()
            if type == "spot":
                self.active_region_type = 0
            elif type == "plage":
                self.active_region_type = 1

        elif isinstance(type, int):
            msg = "ARs can only be of type 0 (spot) or 1 (plage)"
            assert type in (0, 1), msg
            self.active_region_type = type
        else:
            raise ValueError('Provide type=0/1 or "spot"/"plage"')

        if self._default_temp_diff:
            self.temp_diff = self._temperature_differences[self.type]

    def set(self, **kwargs):
        set_object_attributes(self, kwargs)

    def get_size(self, t):
        return triangular(t, Amax=self.Amax, tmax=self.tmax, lifetime=self.lifetime)


class Ring:
    """Ring of the planet

    Parameters
    ----------
    fi : float
        Ring inner radius (must be >= 1)
    fe : float
        Ring outer radius (must be >= 1 and >= fi)
    ir : float
        Projected inclination of ring wrt to the skyplane [degrees].
        90 for an edge-on ring, 0 for face on
    theta : float
        Projected ring tilt [degrees]. 90 means that the image of the ring in
        perpendicular to the orbit in the plane of the sky.
    """

    @maybe_quantity_input(ir=U.deg, theta=U.deg)
    def __init__(self, fi, fe, ir, theta):
        if fe < fi:
            raise ValueError("Outer radius must be larger than inner radius")

        self.fi = fi
        self.fe = fe
        self.ir = ir
        self.theta = theta

    def __repr__(self):
        if self.fi == self.fe or (self.fe <= 1.0 and self.fi <= 1.0):
            return "no ring"

        pars = f"fi={self.fi}; fe={self.fe}; ir={self.ir}; theta={self.theta}"
        return f"SOAP.Ring({pars})"

    def set(self, **kwargs):
        for k, v in kwargs.items():
            try:
                getattr(self, k)
                setattr(self, k, v)
            except AttributeError:
                print(f'attribute "{k}" does not exist')


class Planet:
    """A planet (which may have rings)

    Args:
        P (float):
            Orbital period [day]
        a (float):
            Semi-major axis [stellar radius]
        Rp (float):
            Planet radius [stellar radius]
        Mp (float, optional):
            Planet mass [Earth mass]
        e (float, optional):
            Eccentricity of the orbit. Default is 0
        w (float, optional):
            Argument of periastron [degree]. w=0 means periastron is on the yz
            plane, w=90 means it is in front of the observer. Default is 90
        ip (float, optional):
            Inclination of the orbital plane [degree]. Default is 90
        lbda (float):
            Projected spin-orbit misalignment angle [degree]. Default is 0
        t0 (float):
            Time of periastron passage [day]. Default is 0
    """

    @maybe_quantity_input(
        P=U.day,
        a=U.solRad,
        Rp=U.solRad,
        Mp=U.earthMass,
        w=U.deg,
        ip=U.deg,
        lbda=U.deg,
        t0=U.day,
    )
    def __init__(self, P, a, Rp, Mp=None, e=0.0, w=90.0, ip=90.0, lbda=0.0, t0=0.0):
        self.P = P
        self.t0 = t0
        self.e = e
        self.w = w
        self.ip = ip
        self.lbda = lbda
        self.a = a << U.solRad
        self.Rp = Rp
        self.Mp = Mp

        self.ring = None

    def add_ring(self, fi=1.0, fe=1.0, ir=0.0, theta=0.0):
        """
        Add a ring to this planet.

        Parameters
        ----------
        fi : float, optional [default: 1.0]
            Ring inner radius
        fe : float, optional [default: 1.0]
            Ring outer radius
        ir : float, optional [default: 0.0]
            Projected inclination of ring wrt to the skyplane [degrees].
            90 for an edge-on ring, 0 for face on
        theta : float, optional [default: 0.0]
            Projected ring tilt [degrees]. 90 means that the image of the ring in
            perpendicular to the orbit in the plane of the sky.
        """
        # text = input('do you really wanna put a ring on it? ')
        # if text == 'yes':
        #     import webbrowser
        #     webbrowser.open('https://youtu.be/4m1EFMoRFvY', new=2)
        # if self.Rp == 0.0:
        #     print('Adding ring to non-existent planet. Set planet.Rp > 0')
        # if fi == fe:
        #     print('Ring has inner radius = outer radius, will not have any effect')
        self.ring = Ring(fi, fe, ir, theta)

    def remove_ring(self):
        self.ring = None

    @property
    def has_ring(self):
        if self.ring is not None:
            if self.ring.fe > self.ring.fi and self.ring.fe > 1.0:
                return True
        return False

    @maybe_quantity_input(stellar_mass=U.M_sun)
    def semi_amplitude(self, stellar_mass=1.0, unit=kms):
        if self.Mp is None:
            raise ValueError("no planet mass defined")
        from astropy.constants import G

        f = (2 * np.pi * G / self.P) ** (1 / 3)
        Mratio = self.Mp / (self.Mp + stellar_mass) ** (2 / 3)
        self.K = f * Mratio / np.sqrt((1 - self.e**2))
        self.K <<= unit
        return self.K

    @maybe_quantity_input(stellar_mass=U.M_sun)
    def rv_curve(self, time, stellar_mass=1.0, unit=kms):
        if self.Mp is None:
            raise ValueError("no planet mass defined")

        from .keplerian import keplerian

        # calculate semi-amplitude
        self.semi_amplitude(stellar_mass)

        rv_kep = keplerian(
            time,
            self.P.value,
            self.K.value,
            self.e,
            np.deg2rad(self.w.value),
            self.t0.value,
            0.0,
        )
        return rv_kep * unit

    def __repr__(self):
        if self.Rp == 0:
            return "no planet (to enable, set planet.Rp > 0)"

        pars = (
            f"P={self.P}; t0={self.t0}; e={self.e}; w={self.w}; "
            f"ip={self.ip}; lbda={self.lbda}; a={self.a}, Rp={self.Rp}"
        )

        if self.ring:  # and self.ring.fi != self.ring.fe:
            ring_pars = f"fi={self.ring.fi}, fe={self.ring.fe}, ir={self.ring.ir}, theta={self.ring.theta}"
            return f"SOAP.Planet({pars})\n\t\tring({ring_pars})"
        else:
            return f"SOAP.Planet({pars})"

    def __setattr__(self, name, value):
        try:
            old = getattr(self, name)
            if has_unit(old):
                super().__setattr__(name, value << old.unit)
            else:
                super().__setattr__(name, value)
        except AttributeError:
            super().__setattr__(name, value)

    def set(self, **kwargs):
        set_object_attributes(self, kwargs)


class CCF:
    # is this a CCF?
    _ccf: bool = True

    @maybe_quantity_input(rv=kms)
    def __init__(self, rv, intensity, normalize=True, convolved=False):
        """
        Arguments
        ---------
        rv : array
            RV array where the CCF is defined [km/s or astropy unit]
        intensity : array
            CCF array
        normalize : bool
            Whether to normalize the CCF by its median
        convolved : bool
            Whether the CCF is already convolved with the instrumental profile
        """
        self.rv = rv
        if normalize:
            self.intensity = intensity / np.median(intensity)
        else:
            self.intensity = intensity
        self.convolved = convolved
        self.n = len(self.rv)  # Number of points of the CCF
        self.step = self.rv[1] - self.rv[0]
        #! is this correct? it assumes a symmetric CCF?
        self.width = np.max(np.abs(self.rv))
        self._rv_units = rv.unit
        self._vrot = 0.0 * self._rv_units

    @property
    def vrot(self):
        """Rotation velocity of the star to which the CCF is associated"""
        return self._vrot

    @vrot.setter
    def vrot(self, value):
        self._vrot = value

    @property
    def n_v(self):
        """
        Total number of RV bins in the CCF after considering a stellar rotation
        velocity equal to self.vrot.
        """
        # The CCF is assumed to have a rotational velocity equal to 0 because it
        # is taken in the stellar disk center. To consider rotation when
        # simulating the emerging spectrum on the limb of the star (velocity
        # different from 0), we have to extrapolate outside of the range
        # [-self.width, self.width]. This is done by calculating self.n_v which
        # is the number of additional RV bins required to consider a stellar
        # rotation of star.vrot

        # self.n_v must by an odd integer so that we have as many values on the
        # positive side of the CCF as on the negative one (because 0 is present)
        r = np.round(((1.1 * self.vrot) / self.step) * 2)
        if r % 2 == 0:
            return int(self.n + r)
        else:
            return int(self.n + r + 1)

    @property
    def v_interval(self):
        # self.v_interval gives the difference in radial velocity
        # between the minimum and maximum values of the CCF,
        # once the extrapolation is done
        # self.n_v gives the number of points of the extrapolated CCF.
        # (self.n_v-1) gives the number of intervals,
        # which is then multiplied by the sampling of the CCF
        # self.step to give self.v_interval
        return self.step * (self.n_v - 1) / 2.0

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(self.rv, self.intensity, label=str(self), **kwargs)
        ax.set(ylabel="CCF", xlabel=f"RV [{self._rv_units}]")


class solarCCF(CCF):
    """Solar CCF obtained with the FTS spectrograph"""

    @maybe_quantity_input(vrot=kms)
    def __init__(self, vrot, active_region=False):
        """
        Arguments
        ---------
        vrot : float
            Rotation velocity of the star (used to select a wider CCF window)
        active_region : bool, default False
            Get the CCF for the active region instead of the quiet Sun
        """

        this_dir = os.path.dirname(__file__)
        if vrot < 10 * kms:
            f = "CCF_solar_spectrum_G2_FTS_reso_not_evenly_sampled_in_freq.rdb"
            file = os.path.join(this_dir, "solarCCFs", f)
        else:
            f = "CCF_solar_spectrum_G2_FTS_reso_not_evenly_sampled_in_freq_extra_large_low_resolution.rdb"
            file = os.path.join(this_dir, "solarCCFs", f)

        data_solar_ccf = read_rdb(file)
        rv_ccf = data_solar_ccf["vrad"].copy()
        if active_region:
            intensity_ccf = data_solar_ccf["CCF_spot"].copy()
        else:
            intensity_ccf = data_solar_ccf["CCF"].copy()
        self.active_region = active_region

        super().__init__(rv_ccf, intensity_ccf, convolved=False)
        self.vrot = vrot

    def __repr__(self):
        which = "sunspot" if self.active_region else "quiet Sun"
        return f"solarCCF(FTS, {which})"


class gaussianCCF(CCF):
    """A Gaussian CCF defined by a depth [0-1] and a FWHM in km/s"""

    @maybe_quantity_input(fwhm=kms, RV=kms, window=kms, step=kms)
    def __init__(
        self, depth=0.56, fwhm=2.5, RV=0.0, window=20, step=0.1, convolved=True
    ):
        """
        Args:
            depth (float):
                Amplitude of the CCF, between 0 and 1
            fwhm (float):
                Full width at half maximum of the CCF [km/s]
            RV (float):
                Radial velocity where the CCF is centered [km/s]
            window (float):
                The CCF is defined between -window and +window [km/s]
            step (float):
                Radial velocity step of the CCF [km/s]
            convolved (bool, optional):
                Whether the CCF is already convolved with the instrumental
                profile.
        """
        self._depth = depth
        self._fwhm = fwhm
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        rv_ccf = unit_arange(RV - window, RV + window + step, step)
        intensity_ccf = -depth * np.exp(-((rv_ccf - RV) ** 2) / (2 * sigma**2)) + 1
        super().__init__(rv_ccf, intensity_ccf, convolved=convolved)

    def __repr__(self):
        # which = 'sunspot' if self.active_region else 'quiet Sun'
        pars = f"depth={self._depth:.3f}, fwhm={self._fwhm:.3f}"
        return f"gaussianCCF({pars})"

from numba import float64, int32
from numba.experimental import jitclass

spectrum_attr_spec = [("wave", float64[:]), ("flux_arr", float64[:]), ("n", int32)]


@jitclass(spectrum_attr_spec)
class SpectrumNumba:
    def __init__(self, wave, flux):
        self.n = wave.size
        self.wave = wave
        self.flux_arr = flux

    def flux(self, μ: float = 0.0):
        return self.flux_arr


from numba import objmode, typed, typeof, types

spectrum2d_attr_spec = [
    ("wave", float64[:]),
    ("flux2d", float64[:, :]),
    ("n", int32),
    # ('interp', RegularGridInterpolator),
]


@jitclass(spectrum2d_attr_spec)
class SpectrumNumbaInterpolated:
    def __init__(self, wave, flux2d):
        self.n = wave.size
        self.wave = wave
        self.flux2d = flux2d

    # def interpolate(self):
    #     with objmode():
    #     return interp

    def flux(self, mu=0.0):
        # don't do extrapolations, use the μ=0.2 spectrum
        if mu < 0.2:
            return self.flux2d[:, 0]
        # do interpolation
        μ = np.array(
            [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1.0]
        )
        xi = np.column_stack((np.full(self.wave.size, mu), self.wave))
        with objmode(y="float64[:]"):
            interp = RegularGridInterpolator((μ, self.wave), self.flux2d.T)
            y = interp(xi)
        return y


class Spectrum:
    # is this a CCF?
    _ccf: bool = False

    def __init__(self, wave: np.ndarray, flux: np.ndarray, step=0.5) -> None:
        self.wave = wave
        self.flux = flux
        self._rv_units = kms
        self.step = step
        self._vrot = 0.0

    @property
    def n(self):
        return self.wave.size

    @property
    def vrot(self):
        """Rotation velocity of the star to which the CCF is associated"""
        return self._vrot

    @property
    def n_v(self):
        """
        Total number of RV bins in the CCF after considering a stellar rotation
        velocity equal to self.vrot.
        """
        # The CCF is assumed to have a rotational velocity equal to 0 because it
        # is taken in the stellar disk center. To consider rotation when
        # simulating the emerging spectrum on the limb of the star (velocity
        # different from 0), we have to extrapolate outside of the range
        # [-self.width, self.width]. This is done by calculating self.n_v which
        # is the number of additional RV bins required to consider a stellar
        # rotation of star.vrot

        # self.n_v must by an odd integer so that we have as many values on the
        # positive side of the CCF as on the negative one (because 0 is present)
        r = np.round(((1.1 * self.vrot) / self.step) * 2)
        if r % 2 == 0:
            return int(self.n + r)
        else:
            return int(self.n + r + 1)

    @property
    def v_interval(self):
        # self.v_interval gives the difference in radial velocity
        # between the minimum and maximum values of the CCF,
        # once the extrapolation is done
        # self.n_v gives the number of points of the extrapolated CCF.
        # (self.n_v-1) gives the number of intervals,
        # which is then multiplied by the sampling of the CCF
        # self.step to give self.v_interval
        return self.step * (self.n_v - 1) / 2.0

    def __call__(self, *args, **kwargs):
        return self.flux

    def plot(self, ax=None, thin=100, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if self.flux.ndim == 1:
            kwargs.setdefault("color", "k")

        if self.flux.shape[0] > 50000:
            ax.plot(self.wave[::thin], self.flux[::thin], **kwargs)
        else:
            ax.plot(self.wave, self.flux, **kwargs)
        ax.set(xlabel=r"wavelength [$\AA$]", ylabel="flux")
        return fig, ax

    def doppler_shift(self, rv: float, inplace=False):
        shifted = doppler_shift(self.wave, self.flux, rv)
        if inplace:
            self.flux = shifted
        else:
            return shifted

    def interpolate_to(self, wave, inplace=False, **kwargs):
        # interpolate this spectrum to wavelengths `wave`
        interpolated = np.interp(wave, self.wave, self.flux)
        if inplace:
            self.flux = interpolated
            self.wave = wave
        else:
            return interpolated

    def convolve_to(self, R, inplace=False):
        if self.flux.ndim == 2:
            flux = np.zeros_like(self.flux)
            for i in range(self.flux.shape[1]):
                _, f = ip_convolution(self.wave, self.flux[:, i], R)
                flux[:, i] = f
        else:
            _, flux = ip_convolution(self.wave, self.flux, R)

        if inplace:
            self.flux = flux
        else:
            return flux

    def resample(self, every, inplace=False):
        wave = self.wave[::every]
        flux = self.flux[::every]
        if inplace:
            self.wave = wave
            self.flux = flux
        else:
            return wave, flux

    def to_numba(self):
        return SpectrumNumba(self.wave.astype(float), self.flux.astype(float))

    @classmethod
    def random(cls, min_wave: float = 6000.0, max_wave: float = 6010.0):
        # define high-level parameters, especially including spectrograph parameters
        R = 135000  # resolution
        SNR = 100.0  # s/n ratio in the continuum
        continuum_ivar = SNR**2  # inverse variance of the noise in the continuum
        sigma_x = 1.0 / R  # LSF sigma in x units
        Delta_x = 1.0 / (3.0 * R)  # pixel spacing
        x_min = np.log(min_wave)  # minimum ln wavelength
        x_max = np.log(max_wave)  # maximum ln wavelength
        lines_per_x = (
            2.0e4  # mean density (Poisson rate) of lines per unit ln wavelength
        )
        ew_max_x = 3.0e-5  # maximum equivalent width in x units
        ew_power = 5.0  # power parameter in EW maker
        # set up the true line list for the true spectral model
        x_margin = 1.0e6 / c.value  # hoping no velocities are bigger than 1000 km/s
        x_range = (
            x_max - x_min + 2.0 * x_margin
        )  # make lines in a bigger x range than the data range
        nlines = np.random.poisson(
            x_range * lines_per_x
        )  # set the total number of lines
        line_xs = (x_min - x_margin) + x_range * np.random.uniform(size=nlines)
        # give those lines equivalent widths from a power-law distribution
        line_ews = ew_max_x * np.random.uniform(size=nlines) ** ew_power

        xs = np.arange(x_min, x_max, Delta_x)
        true_doppler = 0.0
        ys, y_ivars = Spectrum._noisy_true_spectrum(
            xs, true_doppler, continuum_ivar, line_xs, line_ews, sigma_x
        )
        # y_ivars_empirical = Spectrum._ivar(ys, continuum_ivar)
        return cls(np.exp(xs), ys)

    @staticmethod
    def _oned_gaussian(dxs, sigma):
        return np.exp(-0.5 * dxs**2 / sigma**2) / (sqrt2pi * sigma)

    @staticmethod
    def _true_spectrum(xs, doppler, lxs, ews, sigma):
        g = Spectrum._oned_gaussian(xs[:, None] - doppler - lxs[None, :], sigma)
        return np.exp(-1.0 * np.sum(ews[None, :] * g, axis=1))

    @staticmethod
    def _ivar(ys, continuum_ivar):
        return continuum_ivar / ys

    @staticmethod
    def _noisy_true_spectrum(xs, doppler, continuum_ivar, lxs, ews, sigma):
        ys_true = Spectrum._true_spectrum(xs, doppler, lxs, ews, sigma)
        y_ivars = Spectrum._ivar(ys_true, continuum_ivar)
        ys = ys_true + np.random.normal(size=xs.shape) / np.sqrt(y_ivars)
        return ys, y_ivars


class solarFTS(Spectrum):

    def __init__(
        self, spot=False, wave_range=(3500, 7500), resolution=None, air_wave=True
    ) -> None:
        here = os.path.dirname(__file__)
        if spot:
            file = os.path.join(here, "../data/sunspot_FTS.npy")
        else:
            file = os.path.join(here, "../data/quiet_sun_FTS.npy")

        wave, flux = np.load(file)

        if air_wave:
            wave = vacuum_to_air(wave)

        mask = (wave > wave_range[0]) & (wave < wave_range[1])
        wave = wave[mask]
        flux = flux[mask]

        if resolution is not None:
            from scipy.ndimage.filters import gaussian_filter1d

            # resolution R = λ / Δλ => Δλ = λ / R
            inst_profile_sig = np.median(wave) / resolution
            # inst_profile_sig = inst_profile_FWHM / (2 * np.sqrt(2 * np.log(2)))
            flux = gaussian_filter1d(flux, inst_profile_sig)

        super().__init__(wave, flux)
        self.wave_range = wave_range

class Spec_mu(Spectrum):
    def __init__(self,
                 mu_array,
                 wavelength,
                 spectra,
                 wave_range=(4198, 7998),
                 air_wave=True
                 ):
        self.μ = mu_array
        wave=wavelength
        flux=spectra

        # wave cuts
        mask = (wave > wave_range[0]) & (wave < wave_range[1])
        wave = wave[mask]
        flux = flux[mask]

        # remove NaNs
        nan_mask = np.zeros_like(wave, dtype=bool)
        for i,f in enumerate(flux.T):
            #flux.T[i]=f/np.max(flux.T[-1])
            flux.T[i]=f/np.max(flux.T[i])

        for f in flux.T:
            nan_mask = np.logical_or(nan_mask, np.isnan(f))
        wave = wave[~nan_mask]
        flux = flux[~nan_mask, :]

        # conversions
        wave = wave.astype(float)
        flux = flux.astype(float)

        super().__init__(wave, flux)
        self.wave_range = wave_range
        self.interpolate()

    def interpolate(self):
        self.interp = RegularGridInterpolator((self.μ, self.wave), self.flux.T)

    def __call__(self, mu: float):
        xi = np.c_[np.full(self.wave.size, mu), self.wave]
        return self.interp(xi)

    def to_numba(self):
        return SpectrumNumbaInterpolated(
            self.wave.astype(float), self.flux.astype(float)
        )

class solarIAGatlas(Spectrum):
    def __init__(self, wave_range=(4198, 7998), air_wave=True):
        here = os.path.dirname(__file__)
        files = sorted(glob(os.path.join(here, "../data/IAGatlas/*.fits")))

        # need to download the atlas
        if len(files) < 14:
            from .utils import download_goettingen_solar_atlas

            download_goettingen_solar_atlas()

        self.μ = np.array([float(f.split("mu")[1][:-5]) for f in files])
        self.atlas_files = files

        wave_file = os.path.join(here, "../data/IAGatlas/wave.npy")
        if os.path.exists(wave_file):
            wave = np.load(wave_file)
        else:
            wave = fits.getdata(files[0])[0]
            np.save(wave_file, wave)

        flux_file = os.path.join(here, "../data/IAGatlas/flux.npy")
        if os.path.exists(flux_file):
            flux = np.load(flux_file)
        else:
            flux = np.array([fits.getdata(f)[1] for f in files]).T
            np.save(flux_file, flux)

        if air_wave:
            wave = vacuum_to_air(wave)

        # wave cuts
        mask = (wave > wave_range[0]) & (wave < wave_range[1])
        wave = wave[mask]
        flux = flux[mask]

        # remove NaNs
        nan_mask = np.zeros_like(wave, dtype=bool)
        for f in flux.T:
            nan_mask = np.logical_or(nan_mask, np.isnan(f))
        wave = wave[~nan_mask]
        flux = flux[~nan_mask, :]

        # conversions
        wave = wave.astype(float)
        flux = flux.astype(float)

        super().__init__(wave, flux)
        self.wave_range = wave_range
        self.interpolate()

    def interpolate(self):
        self.interp = RegularGridInterpolator((self.μ, self.wave), self.flux.T)

    def __call__(self, mu: float):
        xi = np.c_[np.full(self.wave.size, mu), self.wave]
        return self.interp(xi)

    def to_numba(self):
        return SpectrumNumbaInterpolated(
            self.wave.astype(float), self.flux.astype(float)
        )


class PHOENIX(Spectrum):
    # get wave and flux from PHOENIX

    # mask = (wave > wave_range[0]) & (wave < wave_range[1])
    # wave = wave[mask]
    # flux = flux[mask]
    def __init__(
        self,
        spot=False,
        wave_range=(3500, 7500),
        resolution=None,
        teff=5780,
        logg=4.4,
        Z=0.012,
        St_alpha=0,
        contrast=0,
        normalize=False,
        cache=True,
    ) -> None:
        if cache == True:
            read_dict = None
            directory = Path().resolve()
            matching_files = [
                f.name
                for f in directory.iterdir()
                if f.is_file()
                and f.name.startswith("spectrum")
                and f.name.endswith("npz")
            ]
            prop_dict = {
                "spot": spot,
                "wave_range": wave_range,
                "resolution": resolution,
                "teff": teff,
                "logg": logg,
                "Z": Z,
                "St_alpha": St_alpha,
                "contrast": contrast,
                "normalize": normalize,
                "cache": cache,
            }
            if len(matching_files) == 0:
                pass
            else:
                for saved_spec in matching_files:
                    data = np.load(saved_spec, allow_pickle=True)
                    flux_read = data["flux"]
                    wave_read = data["wavelength"]
                    read_dict = data["prop_dict"]
                    if prop_dict == read_dict:
                        flux = flux_read
                        wave = wave_read
                        super().__init__(wave, flux)
                        return None
                    else:
                        pass
        else:
            pass
        if spot:
            spectrum = get_spectrum(
                T_eff=teff + contrast, log_g=logg, Z=Z, alpha=St_alpha
            )
        else:
            spectrum = get_spectrum(T_eff=teff, log_g=logg, Z=Z, alpha=St_alpha)
        # Read the spectrum
        wave = vacuum_to_air((spectrum.wavelength.value).astype(float))
        flux = (spectrum.flux.value).astype(float)

        mask_0 = (wave > wave_range[0]) & (wave < wave_range[1])

        flux = flux[mask_0]
        wave = wave[mask_0]
        if normalize:
            n = len(flux)
            # Compute 10% of the points in the spectrum
            n_10percent = int(n * 0.1)

            # From the 10% of the points in the spectrum compute the 99 percentile in flux
            mask_thresh_1 = flux[:n_10percent] >= np.percentile(flux[:n_10percent], 99)
            mask_thresh_2 = flux[-n_10percent:] >= np.percentile(
                flux[-n_10percent:], 99
            )

            # Compute the respective points in flux and wavelength
            percent_continuum_flux = np.concatenate(
                (flux[:n_10percent][mask_thresh_1], flux[-n_10percent:][mask_thresh_2])
            )
            percent_continuum_wave = np.concatenate(
                (wave[:n_10percent][mask_thresh_1], wave[-n_10percent:][mask_thresh_2])
            )

            # Fit the points with a first degree polynomial
            slope_coefs = np.polyfit(percent_continuum_wave, percent_continuum_flux, 1)
            slope_spectrum = slope_coefs[0] * wave + slope_coefs[1]

            # Correct the continuum
            flux = flux / slope_spectrum
        else:
            None

        if resolution is not None:
            from scipy.ndimage.filters import gaussian_filter1d

            # resolution R = λ / Δλ => Δλ = λ / R
            inst_profile_sig = np.median(wave) / resolution
            # inst_profile_sig = inst_profile_FWHM / (2 * np.sqrt(2 * np.log(2)))
            flux = gaussian_filter1d(flux, inst_profile_sig)
        if cache == True:
            np.savez(
                "spectrum" + str(uuid.uuid4()) + ".npz",
                wavelength=wave,
                flux=flux,
                prop_dict=prop_dict,
            )
        else:
            pass
        super().__init__(wave, flux)


from functools import partial

from better_uniform import buniform as uniform
from scipy.stats import lognorm, norm, truncnorm


class distribution_sum:
    def __init__(self, *dists, weights=None):
        self.dists = dists
        self.n = len(dists)

        if weights is not None:
            if not isinstance(weights, (list, tuple, np.ndarray)):
                raise ValueError("weights should be list, tuple, or ndarray")
            if len(weights) != self.n:
                raise ValueError("Wrong number of weights")
            self.weights = weights / np.sum(weights)
        else:
            self.weights = [1.0 / self.n] * self.n

    def rvs(self, *args, **kwargs):
        choice = np.random.choice(np.arange(self.n), p=self.weights)
        return self.dists[choice].rvs(*args, **kwargs)


class ActiveRegionEvolution:
    """
    This class holds information about the evolution of active regions over the
    stellar magnitic cycle
    """

    def __init__(self, tref=0.0, isolated_spots_fraction=0.4):
        self.tref = tref

        self.isolated_spots_fraction = isolated_spots_fraction

        # LATITUDE ##

        # # option 1
        # mu, s = 15.1, 7.3
        # a, b = (-90 - mu) / s, (90 - mu) / s
        # self.latitude_distribution = truncnorm(a, b, loc=mu, scale=s)

        # from Perger+2020 https://arxiv.org/abs/2012.01862
        self.latitude_distribution = distribution_sum(uniform(-35, -5), uniform(5, 35))

        # LONGITUDE ##

        # # uniform, all longitudes
        # self.longitude_distribution = uniform(-180, 180)

        # active longitude
        self.longitude_distribution = norm(180, 60)

        self.spot_size_distributions = {
            "isolated": lognorm(s=2.14, loc=46.51),
            "groups": lognorm(s=2.49, loc=90.24),
        }

        self.decay_rate_distributions = {
            "isolated": lognorm(s=0.806, loc=2.619),
            "groups": lognorm(s=0.869, loc=3.373),
        }

    def get_lifetime(self, A0, D):
        """
        Calculate the spot lifetime, A0 / D, where A0 is the maximum spot area,
        D is the average decay rate
        """
        return A0 / D

    def generate_spots(self, N=1, time_range=(0, 25)):
        spots = []
        for _ in range(N):
            if np.random.rand() < self.isolated_spots_fraction:
                artype = "isolated"
            else:
                artype = "groups"

            tmax = np.random.uniform(*time_range)
            A0 = self.spot_size_distributions[artype].rvs()
            s = self.msh2size(A0)
            D = self.decay_rate_distributions[artype].rvs()
            lifetime = self.get_lifetime(A0, D)

            lat = self.latitude_distribution.rvs()
            lon = self.longitude_distribution.rvs()

            AR = ActiveRegion(lon, lat, s, 0)
            AR.Amax = s
            AR.tmax = tmax
            AR.lifetime = lifetime
            AR.size = (
                AR.get_size
            )  # partial(triangular, Amax=s, tmax=tmax, lifetime=lifetime)
            spots.apend(AR)

        return spots

    def build_interval_tree(self, spots):
        from intervaltree import IntervalTree

        tree = IntervalTree()
        for spot in spots:
            w1, w2 = triangular(
                None, spot.size, spot.tmax, spot.lifetime, return_visible_window=True
            )
            tree[w1:w2] = spot
        return tree

    def msh2size(self, msh):
        """Convert millionth of solar hemisphere (MSH) to linear size of spot"""
        return np.sqrt(2 * 1e-6 * msh)


def triangular(t, Amax, tmax, lifetime, factor=0.1, return_visible_window=False):
    """Linear growth and decay law for active region size

    Arguments
    ---------
    t : float or array
        Time at which to calculate the size
    Amax : float
        Maximum size of the active region
    tmax : float
        Time at which the active region has the maximum size
    lifetime : float
        Total lifetime of the active region
    factor : float, default 0.1
        The active region grows for factor*lifetime and then decays for
        (1-factor)*lifetime
    return_visible_window : bool, default False
        Just return the window when the spot is visible
    """
    if return_visible_window:
        return tmax - factor * lifetime, tmax + (1 - factor) * lifetime

    a = tmax - factor * lifetime
    b = tmax + (1 - factor) * lifetime
    c = tmax
    t = np.atleast_1d(t)

    m1 = t < a

    m2 = ~m1 & (t < c)
    f2 = 2 * (t - a) / ((b - a) * (c - a))

    m3 = ~m1 & ~m2 & (t <= b)
    f3 = 2 * (b - t) / ((b - a) * (b - c))

    pdf = m2 * f2 + m3 * f3
    return Amax * (b - a) * pdf / 2
