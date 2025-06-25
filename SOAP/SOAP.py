# Copyright (C) 2020-2022 Institute of Astrophysics and Space Sciences
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import time
from copy import copy, deepcopy

import numpy as np

from SOAP import units

from . import classes
from . import fast_starspot as stspnumba
from . import plots
from .classes import ActiveRegion
from .defaults import (
    _default_ACTIVE_REGIONS,
    _default_CCF,
    _default_CCF_active_region,
    _default_PLANET,
    _default_psi,
    _default_STAR,
)
from .fast_starspot import precompile_functions
from .gaussian import (
    _precompile_gauss,
    compute_rv,
    compute_rv_2d,
    compute_rv_fwhm,
    compute_rv_fwhm_2d,
    compute_rv_fwhm_bis,
    compute_rv_fwhm_bis_2d,
)
from .units import U, has_unit, pp1, without_units
from .visualizer import animate_visualization, visualize

DEBUG = False


def remove_units(*objs):
    return map(without_units, objs)


class output:
    """A simple object to hold SOAP outputs"""

    __slots__ = [
        "psi",
        "flux",
        "rv",
        "rv_bconv",
        "rv_flux",
        "ccf_fwhm",
        "ccf_bis",
        "ccf_depth",
        "itot_quiet",
        "itot_flux",
    ]

    def __init__(self, **kwargs):
        # set defaults
        for attr in self.__slots__:
            setattr(self, attr, None)
        # set from arguments
        for attr, val in kwargs.items():
            setattr(self, attr, val)

    def plot(self, ms=False, ax=None, fig=None, label=None):
        """Make a simple plot of the quantities available in the output"""
        import matplotlib.pyplot as plt

        def get_label(name, arr=None):
            try:
                if arr is None:
                    raise AttributeError
                unit = f"{arr.unit}".replace(" ", "")
                if unit == "":
                    raise AttributeError
                return f"{name} [{unit}]"
            except AttributeError:
                return f"{name}"

        if self.rv is None:
            if ax is None and fig is None:
                _, ax = plt.subplots(1, 1)
            elif fig:
                ax = fig.axes[0]
            ax.plot(self.psi, self.flux, label=label)
            ax.set(xlabel="rotation phase", ylabel=get_label("flux"))
            if label is not None:
                ax.legend()
            return ax

        else:
            if ax is None and fig is None:
                _, axs = plt.subplots(2, 2, constrained_layout=True)
            elif fig:
                axs = fig.axes
                assert len(axs) == 4
            else:
                assert len(ax) == 4
                axs = np.array(ax)

            axs = np.ravel(axs)

            flux = self.flux
            if self.flux.size == 1:
                flux = np.full_like(self.psi, self.flux)
            axs[0].plot(self.psi, flux)
            axs[0].set_ylabel(get_label("flux", self.flux))

            rv = self.rv.to(units.ms) if ms else self.rv
            axs[1].plot(self.psi, rv)
            axs[1].set_ylabel(get_label("RV", rv))

            if self.ccf_fwhm is not None:
                fwhm = self.ccf_fwhm.to(units.ms) if ms else self.ccf_fwhm
                axs[2].plot(self.psi, fwhm)
                axs[2].set_ylabel(get_label("FWHM", fwhm))
            else:
                axs[2].axis("off")

            if self.ccf_bis is not None:
                bis = self.ccf_bis
                axs[3].plot(self.psi, bis)
                axs[3].set_ylabel(get_label("BIS", bis))
            else:
                axs[3].axis("off")

            for ax in axs:
                ax.set_xlabel("rotation phase")

        return axs

    def change_units(self, new_units, quantity="all"):
        """
        Change units of some or all attribute.

        Parameters
        ----------
        new_units : :class:`astropy.units.Unit`
            The new units to convert the attribute(s) to.
        quantity : str, optional, default 'all'
            Which attribute for which to change units. By default, the function
            changes all attributes which can be converted to `new_units`.
        """
        if quantity == "all":  # try to change all units
            for attr in self.__slots__:
                try:
                    setattr(self, attr, getattr(self, attr).to(new_units))
                except (U.UnitConversionError, AttributeError):
                    pass

        else:  # or just try to change one
            try:
                setattr(self, quantity, getattr(self, quantity).to(new_units))
            except AttributeError:
                msg = (
                    f"'{quantity}' is not an attribute "
                    f"or cannot be converted to {new_units}"
                )
                raise U.UnitConversionError(msg) from None

    def set_units(self, units, quantity):
        """
        Set the units of a given attribute. Note that this function *resets* the
        units if they already exist. To change units use change_units()

        Parameters
        ----------
        units : :class:`astropy.units.Unit`
            The units to set the attribute to.
        quantity : str
            Which attribute to set the units.
        """
        # does the attribute exist?
        if not hasattr(self, quantity):
            raise AttributeError(f"'{quantity}' is not an attribute")

        try:
            temp = getattr(self, quantity).value * units
            setattr(self, quantity, temp)
        except AttributeError:
            temp = getattr(self, quantity) * units
            setattr(self, quantity, temp)

    # Warning! We are not saving the spectra yet
    def save_rdb(
        self,
        filename,
        prot=None,
        simulate_errors=False,
        typical_error_rv=0.8 * units.ms,
    ):
        header = "bjd\tvrad\tsvrad\tfwhm\tsfwhm\n"
        header += "---\t----\t-----\t----\t-----"
        fmt = ["%-9.5f"] + 4 * ["%-7.3f"]

        # ptp = self.rv.value.ptp()
        # rv_err = np.random.uniform(0.1 * ptp, 0.2 * ptp, size=self.rv.size)
        # ptp = self.ccf_fwhm.value.ptp()
        # fw_err = np.random.uniform(0.1 * ptp, 0.2 * ptp, size=self.rv.size)
        n = self.rv.size
        e = typical_error_rv.value
        rv_err = np.random.uniform(0.9 * e, 1.1 * e, size=n)
        fw_err = np.random.uniform(2 * 0.9 * e, 2 * 1.1 * e, size=n)

        if simulate_errors:
            rv = self.rv.value + np.random.normal(np.zeros(n), rv_err)
            fw = self.ccf_fwhm.value + np.random.normal(np.zeros(n), fw_err)
        else:
            rv = self.rv.value
            fw = self.ccf_fwhm.value

        if prot is None:
            psi = self.psi
        else:
            psi = self.psi * prot

        data = np.c_[psi, rv, rv_err, fw, fw_err]

        np.savetxt(filename, data, header=header, fmt=fmt, comments="", delimiter="\t")


class Simulation:
    """
    Create a SOAP simulation

    Args:
        star (:class:`SOAP.Star`):
            The star to be simulated. By default, this is the Sun (with
            Prot=25.05 days).
        pixel (:class:`SOAP.CCF` or :class:`SOAP.Spectrum`):
            The CCF or spectrum to attribute to each pixel in the quiet star.
            Default is the solar CCF.
        pixel_spot (:class:`SOAP.CCF` or :class:`SOAP.Spectrum`):
            The CCF or spectrum corresponding to the spot(s). Default is the
            solar spot CCF.
        active_regions (list):
            A list of :class:`SOAP.ActiveRegion` instances, with the active
            regions to be included in the simulation.
        nrho (int):
            Resolution for the active region's circumference. Default: 20
        grid (int):
            Stellar grid resolution (grid x grid). Default is 300
        inst_reso (int):
            Resolution of the spectrograph (115000 for HARPS, 0 for FTS
            resolution) Default is 115000
        wlll (double, optional):
            Observation wavelength for temperature contrast by Planck's law, in
            Angstrom. Default is 5293.4115 (mean of Kitt peak spectrum used to
            calculate solar CCF)
        resample_spectra (int, optional):
            Resample the quiet star and spot spectra by this amount. Has no
            effect if using a CCF.
        interp_strategy (str, optional):
            If using a spectrum for the pixel, how to do interpolation of
            `pixel` and `pixel_spot` to a common wavelength array. Default is
            'spot2quiet', which interpolates `pixel_spot` to `pixel`'s
            wavelength array. Another option is 'quiet2spot'.
    """

    def __init__(
        self,
        star=None,
        planet=None,
        pixel="CCF",
        pixel_spot="CCF",
        active_regions=None,
        ring=None,
        nrho=20,
        grid=300,
        inst_reso=115000,
        wlll=5293.4115,
        resample_spectra=1,
        interp_strategy="spot2quiet",
        verbose=False
    ):

        self.star = star
        self.planet = planet
        self.verbose=verbose
        # The default is the CCF, otherwhise there is a need to state the type explicitly
        if pixel == "CCF":
            self.pixel = deepcopy(_default_CCF)
        else:
            self.pixel = deepcopy(pixel)

        if pixel_spot == "CCF":
            self.pixel_spot = deepcopy(_default_CCF_active_region)
        elif pixel_spot == None:
            self.pixel_spot = None
        else:
            self.pixel_spot = deepcopy(pixel_spot)

        _possible = (classes.CCF, classes.Spectrum)
        if not issubclass(self.pixel.__class__, _possible):
            raise ValueError("`pixel` can only be a CCF or a Spectrum")
        if pixel_spot:
            if not issubclass(self.pixel_spot.__class__, _possible):
                raise ValueError("`pixel_spot` can only be a CCF or a Spectrum")

            # Test if the spectrum size of the active region and spot are the same. If not it can follow two strategies:
            # - spot2quiet: Interpolate the spot spectrum to match the quiet spectrum size
            # - quet2spot : Interpolate the quiet spectrum to match the spot spectrum size
            if not self._ccf_mode:
                if self.pixel.n != self.pixel_spot.n:
                    if interp_strategy == "spot2quiet":
                        self.pixel_spot.interpolate_to(self.pixel.wave, inplace=True)
                    elif interp_strategy == "quiet2spot":
                        self.pixel.interpolate_to(self.pixel_spot.wave, inplace=True)
                if verbose:
                    print(f"convolving spot spectra to R={inst_reso}")
                self.pixel_spot.convolve_to(inst_reso, inplace=True)
                self.pixel_spot.resample(resample_spectra, inplace=True)

                # convolve the quiet and spot spectra to the instrument resolution
                if verbose:
                    print(f"convolving quiet star to R={inst_reso}")
                self.pixel.convolve_to(inst_reso, inplace=True)
                # resample
                self.pixel.resample(resample_spectra, inplace=True)
        else:
            # convolve the quiet and spot spectra to the instrument resolution
            if verbose:
                print(f"convolving quiet star to R={inst_reso}")
            self.pixel.convolve_to(inst_reso, inplace=True)
            # resample
            self.pixel.resample(resample_spectra, inplace=True)
        self.active_regions = active_regions
        self.ring = ring

        self.nrho = nrho
        self.grid = grid
        self.inst_reso = inst_reso
        self.wlll = wlll

        if self.star is None:
            self.star = deepcopy(_default_STAR)  # Sun()

        if self.planet is None:
            self.planet = deepcopy(_default_PLANET)

        elif self.planet is False:
            self.planet = deepcopy(_default_PLANET)
            self.planet.Rp = 0.0

        self.planet.ring = self.ring
        del self.ring

        if self.active_regions is None:
            self.active_regions = deepcopy(_default_ACTIVE_REGIONS)

        self.itot_cached = False

        # connect pixels with the star
        self.star._pixel = self.pixel
        self.star._pixel_spot = self.pixel_spot

        # convert star's vrot to the same units as the pixel
        self.star.set_vrot_units(self.pixel._rv_units)
        if self._ccf_mode:
            self.pixel.vrot = self.star.vrot
            _precompile_gauss(self.pixel.rv)

    # CCF mode or spectrum mode?
    @property
    def _ccf_mode(self):
        return issubclass(self.pixel.__class__, classes.CCF)

    def get_results(self, rv, CCFs, skip_fwhm=False, skip_bis=False):
        # Initialize arrays to store the results
        RVs = np.empty(CCFs.shape[0])
        FWHMs = np.empty(CCFs.shape[0])
        BISs = np.empty(CCFs.shape[0])

        # Process each CCF sequentially
        for i, ccf in enumerate(CCFs):
            # Perform the computation for each CCF
            if skip_bis and skip_fwhm:
                RV = compute_rv(rv, ccf)
                FW, BIS = 0.0, 0.0
            elif skip_bis:
                RV, FW = compute_rv_fwhm(rv, ccf)
                BIS = 0.0
            else:
                RV, FW, BIS = compute_rv_fwhm_bis(rv, ccf)

            # Store the results
            RVs[i] = RV
            FWHMs[i] = FW
            BISs[i] = BIS

        return RVs, FWHMs, BISs

    @property
    def has_planet(self):
        if self.planet.Rp == 0:
            self.planet.Mp = 0
        return self.planet.Rp > 0.0

    @property
    def has_active_regions(self):
        counter = False
        for ar in self.active_regions:
            counter += bool(ar.size * ar.check)
        return bool(counter)

    @property
    def has_ring(self):
        return self.planet.has_ring

    def __repr__(self):
        if len(self.active_regions) == 0:
            return (
                f"Simulation(\n\tR={self.inst_reso}, grid={self.grid}\n"
                f"\t{self.star}\n"
                f"\tno active regions\n"
                f"\t{self.planet}\n)"
            )
        else:
            return (
                f"Simulation(\n\tR={self.inst_reso}, grid={self.grid}\n"
                f"\t{self.star}\n"
                f"\t{self.active_regions}\n"
                f"\t{self.planet}\n)"
            )

    def set(self, **kwargs):
        """Set (several) attributes of the simulation at once"""
        for k, v in kwargs.items():
            try:
                getattr(self, k)
                setattr(self, k, v)
            except AttributeError:
                print(f'attribute "{k}" does not exist')

    def set_pixel(self, pixel, pixel_spot=None, pixel_plage=None):
        """Set this simulation's pixel

        Args:
            pixel (SOAP.CCF or SOAP.Spectrum):
                The CCF or spectrum for each pixel in the quiet star
            pixel_spot (SOAP.CCF or SOAP.Spectrum):
                The CCF for the spots
            pixel_plage (SOAP.CCF or SOAP.Spectrum):
                The CCF for the plages

        Examples:
            sim.set_pixel( SOAP.gaussianCCF() )
            sim.set_pixel( SOAP.solarCCF(sim.star.vrot) ) # this is the default
        """
        pixel.vrot = self.star.vrot
        self.pixel = pixel

        if pixel_spot is not None:
            self.pixel_spot = copy(pixel_spot)
        if pixel_plage is not None:
            raise NotImplementedError("spots and plages use the same pixel")

        # force recalculation of itot
        self.itot_cached = False
        # connect CCFs with the star
        self.star._pixel = self.pixel
        self.star._pixel_spot = self.pixel_spot

    def plot(self, psi=None, **kwargs):
        fig, axs, ani = plots.plot_simulation(self, psi=psi, **kwargs)
        if ani is not None:
            return ani

    def visualize(
        self, output, plot_type, lim=None, ref_wave=0, plot_lims=None, show_data=True
    ):
        return visualize(self, output, plot_type, lim, ref_wave, plot_lims, show_data)

    def visualize_animation(
        self,
        output,
        plot_type,
        lim=None,
        ref_wave=0,
        plot_lims=None,
        interval=100,
        repeat=True,
    ):
        return animate_visualization(
            self, output, plot_type, lim, ref_wave, plot_lims, interval, repeat
        )

    def plot_surface(self, psi=None, fig=None, colors=("m", "b"), plot_time=None):
        plots.plot_surface(self, psi, fig, colors, plot_time)

    def add_random_active_regions(self, N=2, plage=False):
        """
        Add a given number of active regions to the simulation, randomly
        distributed in the stellar surface

        Args:
            N (int):
                Number of active regions to add
            plage (bool):
                Whether to add plages or spots
        """
        for _ in range(N):
            AR = ActiveRegion.random()
            if plage:
                AR.type = "plage"
            self.active_regions.append(AR)

    def run_itot(self, skip_rv=False, cache=True):
        """Calculate the CCF and the total flux for the quiet star"""
        if cache and self.itot_cached:
            return self.pixel_quiet, self.flux_quiet

        star = without_units(self.star)
        if DEBUG:
            t1 = time.time()
        if self._ccf_mode:
            pixel = without_units(self.pixel)
            if skip_rv:
                pixel_quiet = np.zeros(pixel.n_v)
                flux_quiet = stspnumba.itot_flux(star.u1, star.u2, self.grid)
            else:
                pixel_quiet, flux_quiet = stspnumba.itot_rv(
                    star.vrot,
                    star.incl,
                    star.u1,
                    star.u2,
                    self.star.diffrotB,
                    self.star.diffrotC,
                    self.star.cb1,
                    self.grid,
                    pixel.rv,
                    pixel.intensity,
                    pixel.v_interval,
                    pixel.n_v,
                )

        else:
            precompile_functions()
            pixel = self.pixel.to_numba()
            # pixel = without_units(self.pixel)
            flux_quiet = stspnumba.itot_flux(star.u1, star.u2, self.grid)
            pixel_quiet = stspnumba.itot_spectrum_par(
                star.vrot,
                star.incl,
                star.u1,
                star.u2,
                self.star.diffrotB,
                self.star.diffrotC,
                self.star.cb1,
                self.grid,
                pixel,
            )

        if DEBUG:
            print("finished itot, took %f sec" % (time.time() - t1))
            print("shape of pixel_quiet: %d" % pixel_quiet.shape)
            print("flux_quiet: %f" % flux_quiet)

        if cache:
            self.itot_cached = True

        self.pixel_quiet = pixel_quiet
        self.flux_quiet = flux_quiet
        return pixel_quiet, flux_quiet

    def calculate_signal(
        self,
        psi=None,
        skip_itot=True,
        skip_rv=False,
        skip_fwhm=False,
        skip_bis=False,
        renormalize_rv=True,
        save_ccf=False,
        template=None,
        itot=None,
        **kwargs,
    ):
        """
        Estimate the photometric and spectrocopic effects for this simulation,
        on a grid of stellar rotation phases `psi`.

        Parameters
        ----------
        psi : array_like [default psi=linspace(0, 1, 501)]
            Phases at which the signals will be calculated (in units of the
            stellar rotation period)
        skip_itot : bool, default False
            Only do the calculation of the quiet star once (and cache it)
        skip_rv : bool, default False
            If True, skip calculating the RV signal
        renormalize_rv : bool, default True
            Set RV when the spot is not visible to 0
        save_ccf : bool, default False
            Save the output CCFs to a file
        template : dict, default None
            User provided template dictionary with the keywords "wave" (nm) and "flux" containing the wavelength and flux arrays.

        Returns
        -------
        out : instance of `output`
            Contains psi, flux, and rv (if skip_rv=False) as attributes
        """
        # deal with the phases array
        if psi is None:
            psi = _default_psi
        psi = np.atleast_1d(psi)
        if has_unit(psi):
            psi = psi.value

        star, planet = remove_units(self.star, self.planet)
        active_regions = list(remove_units(*self.active_regions))

        added_ring = False
        if self.planet.ring is None:
            # need a dummy ring which does nothing
            self.planet.add_ring(fi=1.0, fe=1.0, ir=90, theta=0.0)
            added_ring = True
        ring = without_units(self.planet.ring)
        date = (psi + 0.0) * star.prot

        if itot:
            pixel_quiet, flux_quiet = deepcopy(itot)
            self.pixel_quiet, self.flux_quiet = pixel_quiet, flux_quiet
            self.itot_pixel_quiet, self.itot_flux_quiet = deepcopy(
                [pixel_quiet, flux_quiet]
            )
        else:
            pixel_quiet, flux_quiet = self.run_itot(skip_rv, cache=skip_itot)
            self.itot_pixel_quiet, self.itot_flux_quiet = deepcopy(
                [pixel_quiet, flux_quiet]
            )

        FLUXstar = flux_quiet
        pixel_flux = np.tile(pixel_quiet, (psi.size, 1))
        pixel_bconv = np.tile(pixel_quiet, (psi.size, 1))
        pixel_tot = np.tile(pixel_quiet, (psi.size, 1))

        t1 = time.time()
        if DEBUG:
            None
            # print("Active region pixel")
            # plt.plot(self.pixel_spot.wave, self.pixel_spot.flux )
            # plt.xlabel("Wavelength")
            # plt.ylabel("Flux")
            # plt.show()
        if self._ccf_mode:
            pixel = without_units(self.pixel)
            if self.pixel_spot:
                pixel_spot = without_units(self.pixel_spot)
        else:
            pixel = self.pixel.to_numba()
            if self.pixel_spot:
                pixel_spot = self.pixel_spot.to_numba()
        if DEBUG:
            import matplotlib.pyplot as plt

            plt.plot(pixel.rv, pixel.intensity)
            plt.plot(pixel_spot.rv, pixel_spot.intensity)
            plt.show()
        if len(active_regions) != 0:
            out = stspnumba.active_region_contributions(
                psi,
                star,
                active_regions,
                pixel,
                pixel_spot,
                self.grid,
                self.nrho,
                self.wlll,
                planet,
                ring,
                self._ccf_mode,
                skip_rv,
            )
            flux_spot = out[0]
            if DEBUG:
                try:
                    plt.plot(flux_spot)
                    plt.show()
                    plt.plot(pixel_spot_flux.T)
                    plt.show()
                except:
                    None

            pixel_spot_bconv = out[1]
            pixel_spot_flux = out[2]

            pixel_spot_tot = out[3]
            # total flux of the star affected by active regions
            FLUXstar = FLUXstar - flux_spot
            # plt.plot(FLUXstar)
            # plt.show()
            # CCF of the star affected by the flux effect of active regions
            pixel_flux = pixel_flux - pixel_spot_flux
            # CCF of the star affected by the convective blueshift effect of
            # active regions
            pixel_bconv = pixel_bconv - pixel_spot_bconv
            # CCF of the star affected by the total effect of active regions
            pixel_tot = pixel_tot - pixel_spot_tot
            if DEBUG:
                if skip_rv == False:
                    plt.close()
                    print("Effect of the spot in the spectra")
                    plt.plot(pixel_spot_tot.T / np.max(pixel_spot_tot, axis=1))
                    plt.plot(pixel_tot.T / np.max(pixel_tot, axis=1), "--")
                    plt.show()
                else:
                    None
        if DEBUG:
            t2 = time.time()
            print("finished spot_scan_npsi, took %f sec" % (t2 - t1))

        if DEBUG:
            print(self.has_planet)
        if self.has_planet:
            if not self._ccf_mode:
                
                if DEBUG:
                    import matplotlib.pyplot as plt

                    plt.plot(pixel.wave, self.pixel.flux)
                    plt.show()

                out = stspnumba.planet_scan_ndate(
                    star.vrot,
                    star.incl,
                    date,
                    date.size,
                    star.u1,
                    star.u2,
                    self.grid,
                    pixel.wave,
                    self.pixel.to_numba(),
                    self.pixel.v_interval,
                    self.pixel.n_v,
                    pixel.n,
                    planet.P,
                    planet.t0,
                    planet.e,
                    planet.w,
                    planet.ip,
                    planet.a,
                    planet.lbda,
                    planet.Rp,
                    ring.fe,
                    ring.fi,
                    ring.theta + planet.lbda,
                    ring.ir,
                    not skip_rv,
                    "spectrum",
                    self.star.diffrotB,
                    self.star.diffrotC,
                    self.star.cb1,
                )

                if DEBUG:
                    print("I have a planet!")
                    plt.plot(pixel.wave, out[0].T)
                    plt.show()

            else:
                t1 = time.time()

                # calculate the flux and CCF contribution from the planet
                out = stspnumba.planet_scan_ndate(
                    star.vrot,
                    star.incl,
                    date,
                    date.size,
                    star.u1,
                    star.u2,
                    self.grid,
                    pixel.rv,
                    pixel.intensity,
                    pixel.v_interval,
                    pixel.n_v,
                    pixel.n,
                    planet.P,
                    planet.t0,
                    planet.e,
                    planet.w,
                    planet.ip,
                    planet.a,
                    planet.lbda,
                    planet.Rp,
                    ring.fe,
                    ring.fi,
                    ring.theta + planet.lbda,
                    ring.ir,
                    not skip_rv,
                    "ccf",
                    self.star.diffrotB,
                    self.star.diffrotC,
                    self.star.cb1,
                )
                if DEBUG:
                    import matplotlib.pyplot as plt

                    plt.plot(out[0].T)
                    plt.show()

            # not skip_rv
            # theta is modified to follow ldba (i.e measured from transit chord)

            pixel_planet, FLUX_planet, self.xyzplanet = out

            t2 = time.time()
            # remove the contribution from the planet
            FLUXstar = FLUXstar - FLUX_planet
            pixel_flux = pixel_flux - pixel_planet
            pixel_bconv = pixel_bconv
            old_pixel_tot = deepcopy(pixel_tot)
            pixel_tot = pixel_tot - pixel_planet
        # normalize the flux of the star
        FLUXstar = FLUXstar / flux_quiet

        if added_ring:
            self.planet.remove_ring()
        # put units on the flux (with in-place conversion, avoiding copies)
        FLUXstar <<= pp1

        if skip_rv:
            # out.rv=None by defaults
            out = output(psi=psi, flux=FLUXstar)
            return out
        n1 = np.max(pixel_flux, axis=1)

        # normalization
        self.pixel_flux = pixel_flux = (pixel_flux.T / np.max(pixel_flux, axis=1)).T
        self.pixel_bconv = pixel_bconv = (pixel_bconv.T / np.max(pixel_bconv, axis=1)).T
        self.pixel_tot = pixel_tot = (pixel_tot.T / np.max(pixel_tot, axis=1)).T

        # self.pixel_quiet = pixel_quiet = pixel_quiet / max(pixel_quiet)

        # return pixel_flux, pixel_bconv

        if self._ccf_mode:
            out = stspnumba.clip_ccfs(
                pixel, pixel_flux, pixel_bconv, pixel_tot, pixel_quiet
            )
            pixel_flux, pixel_bconv, pixel_tot, pixel_quiet = out

            out = stspnumba.convolve_ccfs(
                pixel, self.inst_reso, pixel_quiet, pixel_flux, pixel_bconv, pixel_tot
            )
            pixel_quiet, pixel_flux, pixel_bconv, pixel_tot = out
            if DEBUG:
                import matplotlib.pyplot as plt

                print("I have a planet!")
                plt.plot(pixel.rv, pixel_flux.T)
                plt.show()

            # calculate the CCF parameters RV, depth, BIS SPAN, and FWHM, for each
            # of the contributions flux, bconv and total
            _rv = pixel.rv

            t1 = time.time()
            if DEBUG:
                import matplotlib.pyplot as plt

                plt.close()
                for k in pixel_flux:
                    plt.plot(_rv, k)
                plt.show()
            # Check this
            if skip_bis and skip_fwhm:
                rv_flux = compute_rv_2d(_rv, pixel_flux)
                fwhm_flux, span_flux = 0.0, 0.0
                rv_bconv = compute_rv_2d(_rv, pixel_bconv)
                fwhm_bconv, span_bconv = 0.0, 0.0
                rv_tot = compute_rv_2d(_rv, pixel_bconv)
                fwhm_tot, span_tot = 0.0, 0.0
            elif skip_bis:
                rv_flux, fwhm_flux = compute_rv_fwhm_2d(_rv, pixel_flux).T
                span_flux = 0.0
                rv_bconv, fwhm_bconv = compute_rv_fwhm_2d(_rv, pixel_bconv).T
                span_bconv = 0.0
                rv_tot, fwhm_tot = compute_rv_fwhm_2d(_rv, pixel_tot).T
                span_tot = 0.0
            else:
                _ = compute_rv_fwhm_bis_2d(_rv, pixel_flux).T
                rv_flux, fwhm_flux, span_flux = _
                _ = compute_rv_fwhm_bis_2d(_rv, pixel_bconv).T
                rv_bconv, fwhm_bconv, span_bconv = _
                _ = compute_rv_fwhm_bis_2d(_rv, pixel_tot).T
                rv_tot, fwhm_tot, span_tot = _

            depth_tot = None

            if DEBUG:
                print("finished map(bis), took %f sec" % (time.time() - t1))

        else:
            if self.verbose:
                print("Computing CCFs for each spectra")
            _rv= np.arange(-20, 20 + 0.5, 0.5)
            _precompile_gauss(_rv)

            t1 = time.time()
            if template:
                ccf_pixel_flux = np.array(
                    [
                        stspnumba.calculate_ccf(
                            template["wave"], template["flux"], self.pixel.wave, i
                        )[1]
                        for i in pixel_flux
                    ]
                )
            else:
                ccf_pixel_flux = np.array(
                    [
                        stspnumba.calculate_ccf(
                            self.pixel.wave, self.pixel.flux, self.pixel.wave, i
                        )[1]
                        for i in pixel_flux
                    ]
                )

            if DEBUG:
                print("I am in line 965")
                plt.plot(_rv, ccf_pixel_flux.T)
                plt.show()

            _ = self.get_results(_rv, ccf_pixel_flux, skip_fwhm, skip_bis)
            rv_flux, fwhm_flux, span_flux = _
            if template:
                ccf_pixel_bconv = np.array(
                    [
                        stspnumba.calculate_ccf(
                            template["wave"], template["flux"], self.pixel.wave, i
                        )[1]
                        for i in pixel_bconv
                    ]
                )
            else:
                ccf_pixel_bconv = np.array(
                    [
                        stspnumba.calculate_ccf(
                            self.pixel.wave, self.pixel.flux, self.pixel.wave, i
                        )[1]
                        for i in pixel_bconv
                    ]
                )

            _ = self.get_results(_rv, ccf_pixel_bconv, skip_fwhm, skip_bis)
            rv_bconv, fwhm_bconv, span_bconv = _
            if template:
                ccf_pixel_tot = np.array(
                    [
                        stspnumba.calculate_ccf(
                            template["wave"], template["flux"], self.pixel.wave, i
                        )[1]
                        for i in pixel_tot
                    ]
                )
            else:
                ccf_pixel_tot = np.array(
                    [
                        stspnumba.calculate_ccf(
                            self.pixel.wave, self.pixel.flux, self.pixel.wave, i
                        )[1]
                        for i in pixel_tot
                    ]
                )
            self.ccf=ccf_pixel_tot
            self.rv=_rv
            _ = self.get_results(_rv, ccf_pixel_tot, skip_fwhm, skip_bis)
            rv_tot, fwhm_tot, span_tot = _
            if DEBUG:
                plt.plot(rv_tot)
                plt.show()
            depth_tot = None

            if DEBUG:
                print("finished map(bis), took %f sec" % (time.time() - t1))

        if renormalize_rv:
            # find where rv_flux = rv_bconv, which corresponds to the phases
            # where the active regions are not visible
            index_equal_rv = np.where((rv_flux - rv_bconv) == 0)[0]
            if len(index_equal_rv) != 0:
                # velocity when the spot is not visible
                zero_velocity = rv_flux[index_equal_rv][0]
                # set velocity when the active region is not visible to 0
                rv_flux -= zero_velocity
                rv_bconv -= zero_velocity
                rv_tot -= zero_velocity
                if not skip_fwhm:
                    # FWHM when the spot is not visible
                    zero_fw = fwhm_flux[index_equal_rv][0]
                    # set FWHM when the active region is not visible to 0
                    fwhm_flux -= zero_fw
                    fwhm_bconv -= zero_fw
                    fwhm_tot -= zero_fw
                if not skip_bis:
                    # BIS when the spot is not visible
                    zero_bis = span_flux[index_equal_rv][0]
                    # set FWHM when the active region is not visible to 0
                    span_flux -= zero_bis
                    span_bconv -= zero_bis
                    span_tot -= zero_bis
        self.integrated_spectra = self.pixel_tot
        ###################
        if self.has_planet:

            tr_dur = (
                1.0
                / np.pi
                * np.arcsin(
                    1.0
                    / (self.planet.a).value
                    * np.sqrt(
                        (1 + (self.planet.Rp).value) ** 2.0
                        - (self.planet.a).value ** 2.0
                        * np.cos(np.radians((self.planet.ip).value)) ** 2.0
                    )
                )
            )
            planet_phases = psi * self.star.prot / self.planet.P
            phase_mask = np.logical_or(
                planet_phases < -tr_dur / 2, planet_phases > tr_dur / 2
            )
            # Note: There is not effect of the keplerian here, so everything is in the stellar rest-frame unless we have 
            # a AR
            rv_tot_out = rv_tot[phase_mask]
            psi_out = planet_phases[phase_mask]

            slope_coefs = np.polyfit(psi_out, rv_tot_out, deg=1)

            slope_rvs = slope_coefs[0] * planet_phases + slope_coefs[1]

            if DEBUG == True:
                print("Slopes coefficients out-of-transit")
                print(slope_coefs)
                print("RVs out-of-transit")
                print(rv_tot_out)
                print("RVs obtained from a linear fit from the out-of-transit")
                print(slope_rvs)
            try:
                corr_pixel_flux = np.array(
                    [
                        stspnumba.doppler_shift(pixel.wave, pixel_tot[i], -slope_rvs[i])
                        for i in range(len(pixel_tot))
                    ]
                )

                # Carefull! When we have an active region, the master out does not correspond to pflux, the user must make it manually!
                flux_weighted_spectra = np.array(
                    [
                        corr_pixel_flux[j] * FLUXstar[j]
                        for j in range(len(corr_pixel_flux))
                    ]
                )
                self.master_out_fw = np.mean(flux_weighted_spectra[phase_mask], axis=0)
                self.integrated_spectra_fw = flux_weighted_spectra

                master_out = np.mean(corr_pixel_flux[phase_mask], axis=0)
                self.pixel_trans = corr_pixel_flux / master_out
                self.integrated_spectra = corr_pixel_flux
            except:
                None

        ###################
        # put back the units (with in-place conversion, avoiding copies)
        rv_flux <<= self.pixel._rv_units
        rv_bconv <<= self.pixel._rv_units
        rv_tot <<= self.pixel._rv_units
        if not skip_fwhm:
            fwhm_flux <<= self.pixel._rv_units
            fwhm_bconv <<= self.pixel._rv_units
            fwhm_tot <<= self.pixel._rv_units
        if not skip_bis:
            span_flux <<= self.pixel._rv_units
            span_bconv <<= self.pixel._rv_units
            span_tot <<= self.pixel._rv_units

        if self.has_planet:
            rv_kep = self.planet.rv_curve(psi * star.prot, stellar_mass=star.mass)

            if not self._ccf_mode:
                # shift by the Keplerian RV
                for i in range(pixel_tot.shape[0]):
                    pixel_tot[i] = stspnumba.doppler_shift(
                        pixel.wave, pixel_tot[i], rv_kep[i].value
                    )
            else:
                # shift by the Keplerian RV
                for i in range(pixel_tot.shape[0]):
                    pixel_tot[i] = stspnumba.shift_ccf(
                        pixel.rv, pixel_tot[i], rv_kep[i].value
                    )

            # add Keplerian signal to final RVs
            rv_tot += rv_kep

        self.pixel_tot = pixel_tot

        out = output(
            psi=psi,
            flux=FLUXstar,
            rv=rv_tot,
            rv_bconv=rv_bconv,
            rv_flux=rv_flux,
            ccf_fwhm=fwhm_tot,
            ccf_bis=span_tot,
            ccf_depth=depth_tot,
            itot_quiet=self.itot_pixel_quiet,
            itot_flux=self.itot_flux_quiet,
        )
        return out

    def config_export(self, simVar="sim", show_all=False):
        """
        Return list (as string) of all variables that can easily be re-imported.
        """

        if type(simVar) is not str:
            raise TypeError("simVar must be type 'str' not {}.".format(type(simVar)))

        outputstring = ""

        # run over all of simulation's attributes
        for key1 in self.__dict__.keys():

            # skip some values
            if not show_all:
                if key1 in ["ccf", "ccf_active_region", "itot_cached", "xyzplanet"]:
                    continue

            a1 = self.__getattribute__(key1)

            # Object planet and star have second layer
            if key1 in ["planet", "star"]:

                if key1 == "planet":
                    outputstring += "\n# %s.has_planet = {}".format(self.has_planet) % (
                        simVar
                    )

                for key2 in a1.__dict__.keys():

                    # skip some values
                    if not show_all:
                        if key2 in ["diffrotB", "diffrotC", "start_psi", "rad_sun"]:
                            continue

                    a2 = a1.__getattribute__(key2)

                    # ring has third layer
                    if key2 == "ring" and a2 is not None:
                        outputstring += "\n"
                        outputstring += "\n# %s.planet.has_ring = {}".format(
                            self.planet.has_ring
                        ) % (simVar)
                        for key3 in a2.__dict__.keys():
                            a3 = a2.__getattribute__(key3)
                            outputstring += "\n%s.%s.%s.%s = {}".format(a3) % (
                                simVar,
                                key1,
                                key2,
                                key3,
                            )
                    else:
                        outputstring += "\n%s.%s.%s = {}".format(a2) % (
                            simVar,
                            key1,
                            key2,
                        )
                outputstring += "\n"

            # active regions are list
            elif key1 == "active_regions":
                outputstring += "\n# %s.has_active_regions = {}".format(
                    self.has_active_regions
                ) % (simVar)
                outputstring += "\n%s.active_regions = [" % (simVar)
                for ar in a1:
                    outputstring += (
                        'SOAP.ActiveRegion(lon=%.9g,lat=%.9g,size=%.9g,active_region_type="%s",check=%i),'
                        % (ar.lon, ar.lat, ar.size, ar.type, ar.check)
                    )
                outputstring += "]"
                outputstring += "\n"

            # all other paramters (first layer)
            else:
                outputstring += "\n%s.%s = {}".format(a1) % (simVar, key1)

        # remove first linebreaks and add linebreak at end
        while outputstring[0] == "\n":
            outputstring = outputstring[1:]
        outputstring += "\n"

        return outputstring
