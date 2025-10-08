import matplotlib.pyplot as plt
import numpy as np

from .fast_starspot import spot_init, spot_phase
from .units import without_units

try:
    import cartopy.crs as ccrs

    cartopy_available = True
except ImportError:
    cartopy_msg = "Please install cartopy to use this function\n"
    cartopy_msg += (
        "(see https://scitools.org.uk/cartopy/docs/latest/installing.html#installing)"
    )
    cartopy_available = False


def plot_simulation(sim, psi=None):
    if psi is None:
        layout = [
            ["vis", "ccf"],
            ["inv", "ar_ccf"],
        ]
    else:
        layout = [
            ["vis", "ccf"],
            ["vis", "ar_ccf"],
        ]

    with plt.rc_context({"toolbar": "None"}):
        fig, axs = plt.subplot_mosaic(layout, constrained_layout=True)

    sim.pixel.plot(ax=axs["ccf"], color="k")
    if sim.pixel_spot:
        sim.pixel_spot.plot(ax=axs["ar_ccf"], color="C1")
        axs["ar_ccf"].set_title("spot", loc="right")
    axs["ar_ccf"].sharex(axs["ccf"])
    axs["ar_ccf"].sharey(axs["ccf"])
    axs["ccf"].set_title("quiet-star", loc="right")

    def setup_axis(name):
        # remove axis and add it back with 3D projection
        ss = axs[name].get_subplotspec()
        axs[name].remove()
        axs[name] = ax = fig.add_subplot(
            ss, projection="3d", azim=0, elev=0, proj_type="ortho"
        )
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1))
        return ax

    def add_star(ax):
        # the star
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color="y", shade=False, alpha=0.2)

    def _plot_phase(φ=0.0):
        artists = []
        for i, ar in enumerate(sim.active_regions):
            ar = without_units(ar)
            if ar.type == "spot":
                color = "b"
            elif ar.type == "plage":
                color = "m"

            xyz = spot_init(ar.size, ar.lon + ax.azim, ar.lat, sim.star.incl, sim.nrho)
            if φ != 0:
                xyz = spot_phase(xyz, sim.star.incl, φ)
            xyz = xyz.T
            vis = xyz[0] > 0
            (line,) = ax.plot(xyz[0][vis], xyz[1][vis], xyz[2][vis], color=color)
            artists.append(line)
            # if vis.any():
            #     t = ax.text(xyz[0][vis][0], xyz[1][vis][0], xyz[2][vis][0], str(i))
            #     artists.append(t)
            # ax.plot_trisurf(*xyz, color=color)
        return artists

    if psi is None:  # just plot phase zero
        ax = setup_axis("vis")
        add_star(ax)
        _plot_phase()
        ax.set_title("visible hemisphere")
        ax = setup_axis("inv")
        add_star(ax)
        _plot_phase(0.5)
        ax.set_title("invisible hemisphere")
        return fig, axs, None
    else:
        from matplotlib.animation import FuncAnimation

        ax = setup_axis("vis")
        add_star(ax)

        def update(φ):
            return _plot_phase(φ)

        ani = FuncAnimation(
            fig,
            update,
            frames=psi,
            init_func=_plot_phase,
            blit=True,
            interval=50,
            cache_frame_data=True,
        )
        return fig, axs, ani


def plot_surface(sim, psi=None, fig=None, colors=("m", "b"), plot_time=None):
    """Plot the stellar surface and the active regions"""
    if not cartopy_available:
        print(cartopy_msg)
        return

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    kwargs = {"marker": "o", "mfc": "w", "markersize": 10}
    elms = [
        Line2D([0], [0], color="m", **kwargs),
        Line2D([0], [0], color="b", **kwargs),
    ]

    lats = np.linspace(-90, 90, sim.grid)
    lons = np.linspace(-270, 270, sim.grid)
    lons, lats = np.meshgrid(lons, lats)
    # 36 points for spot circunference
    phi = np.linspace(0, 2.0 * np.pi, 36)

    if psi is None:
        if fig is None:
            fig = plt.figure(constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            ax.set_title("active region distribution at psi=0\n")
        else:
            assert len(fig.axes) == 1, "Expected only one axis in `fig`"
            ax = fig.axes[0]
        axes = [ax]
    else:
        if fig is None:
            fig = plt.figure(figsize=[10, 5], constrained_layout=True)
            i = sim.star.incl.value
            ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Orthographic(0, 90 - i))
            ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Orthographic(180, -90 + i))
            ax1.set_title("visible hemisphere")
            ax2.set_title("non-visible hemisphere")
        else:
            assert len(fig.axes) == 2, "Expected two axes in `fig`"
            ax1, ax2 = fig.axes

        axes = [ax1, ax2]

    for i, ax in enumerate(axes):
        ax.set_global()
        ax.gridlines(xlocs=range(0, 361, 30), ylocs=range(-90, 91, 30))

        for ar in sim.active_regions:
            ar = without_units(ar)
            if not ar.check:
                continue
            size_in_rad = ar.size_area_visible_hemisphere * 2 * np.pi
            if isinstance(
                ar.size, np.ndarray
            ):  # Changed to accept size arrays of same dimention as the psi
                if len(ar.size) != len(psi):
                    raise Exception(
                        "Active region size array is not the same length as the psi array"
                    )
                # size_in_rad = size_in_rad[0]

            size_in_deg = np.rad2deg(size_in_rad)
            size_in_deg /= 0.68268949213708585
            if ar.type == "spot":
                color = colors[0]
            elif ar.type == "plage":
                color = colors[1]

            shift = np.array(psi) * 360
            try:
                if plot_time == None:
                    for i in range(len(psi)):
                        ax.plot(
                            ar.lon + shift[i] + size_in_deg[i] * np.cos(phi),
                            ar.lat + size_in_deg[i] * np.sin(phi),
                            color=color,
                            transform=ccrs.PlateCarree(),
                            alpha=1 - i / (1.1 * len(psi)),
                        )
                else:
                    ax.plot(
                        ar.lon
                        + shift[plot_time]
                        + size_in_deg[plot_time] * np.cos(phi),
                        ar.lat + size_in_deg[plot_time] * np.sin(phi),
                        color=color,
                        transform=ccrs.PlateCarree(),
                        alpha=1 - i / (1.1 * len(psi)),
                    )
            except:
                ax.plot(
                    ar.lon + shift + size_in_deg * np.cos(phi),
                    ar.lat + size_in_deg * np.sin(phi),
                    color=color,
                    transform=ccrs.PlateCarree(),
                )

    fig.legend(handles=elms, labels=("spots", "plages"), loc="lower center")
    return fig


def plot1(
    CCF_folder_outputs,
    rv_ccf,
    intensity_ccf,
    rv_ccf_magn_region,
    intensity_ccf_magn_region,
):
    plt.figure()
    plt.title("solar CCFs")
    plt.plot(rv_ccf, intensity_ccf, "b", lw=3, label="Quiet photosphere")
    plt.plot(
        rv_ccf_magn_region, intensity_ccf_magn_region, "r", ls="--", lw=3, label="Spot"
    )
    plt.ylim(0.4, 1.05)
    plt.xlim(-20, 20)
    plt.ylabel("Normalized flux")
    plt.xlabel("RV [km/s]")
    plt.legend(loc="best")
    plt.subplots_adjust(top=0.93, left=0.1, right=0.96, bottom=0.13)
    # plt.savefig(os.path.join(CCF_folder_outputs, 'FTS_solar_CCFs.pdf'))
    plt.show()


def plot2(
    CCF_folder_outputs,
    filename,
    PSI,
    flux,
    rv_flux,
    rv_bconv,
    rv_tot,
    span_flux,
    span_bconv,
    span_tot,
    fwhm_flux,
    fwhm_bconv,
    fwhm_tot,
    simulation,
):

    sim = simulation
    ACTIVE_REGIONS = sim.active_regions

    majorFormatter = plt.ScalarFormatter()
    majorFormatter.set_powerlimits((-8, 8))
    majorFormatter.set_useOffset(0)

    # Plot the Flux, RV, BIS SPAN and FWHM variations induced by the active regions defined in "config.cfg"
    plt.figure(1, [16, 9])
    plt.title("")
    ax = plt.subplot(511)
    ax.plot(PSI, flux, color="r")
    ax.set_ylabel("Norm. Flux", size=16)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.yaxis.set_major_formatter(majorFormatter)

    ax2 = plt.subplot(512, sharex=ax)
    ax2.plot(PSI, rv_flux * 1000, color="b", label="flux")
    ax2.plot(PSI, rv_bconv * 1000, color="g", label="conv. blue.")
    ax2.plot(PSI, rv_tot * 1000, color="r", label="tot")
    ax2.legend()
    ax2.set_ylabel("RV [m/s]", size=16)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3 = plt.subplot(513, sharex=ax)
    ax3.plot(PSI, span_flux * 1000, color="b")
    ax3.plot(PSI, span_bconv * 1000, color="g")
    ax3.plot(PSI, span_tot * 1000, color="r")
    ax3.set_ylabel("BIS span [m/s]", size=16)
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax4 = plt.subplot(514, sharex=ax)
    ax4.plot(PSI, fwhm_flux * 1000, color="b", label="flux")
    ax4.plot(PSI, fwhm_bconv * 1000, color="g", label="conv. blue.")
    ax4.plot(PSI, fwhm_tot * 1000, color="r", label="tot")
    ax4.set_ylabel("FWHM [m/s]", size=16)
    ax4.yaxis.set_major_formatter(majorFormatter)
    ax4.set_xlabel("Phase", size=16)

    plt.subplots_adjust(top=0.98, left=0.07, right=0.98, bottom=0.07, hspace=0.15)

    text1 = (
        "{0:19} = {1:<8d}, {2:23} = {3:<8d}\n".format(
            "Grid reso", sim.grid, "Circumference reso", sim.nrho
        )
        + "{0:19} = {1:<8d}, {2:23} = {3:<8d}\n".format(
            "Instr reso", sim.inst_reso, "Radius of the sun [km]", sim.star.rad_sun
        )
        + "{0:19} = {1:<8.2f}, {2:23} = {3:<8.2f}\n".format(
            "Star Rot Period [d]",
            sim.star.prot,
            "Stellar radius [Rsun]",
            sim.star.rad / sim.star.rad_sun,
        )
        + "{0:19} = {1:<8.1f}, {2:23} = {3:<8.2f}\n".format(
            "Star incli", sim.star.incl, "Stellar vsini [m/s]", sim.star.vrot
        )
        + "{0:19} = {1:<8d}, {2:23} = {3:<8d}\n".format(
            "Stellar Teff [K]",
            sim.star.teff,
            "Tdiff spot-photo [K]",
            sim.star.Temp_diff_spot,
        )
        + "{0:19} = {1:<8.3f}, {2:23} = {3:<8.3f}".format(
            "Limb-dark lin", sim.star.limba1, "Limb-dark quad", sim.star.limba2
        )
    )

    text2 = ""
    for i, ar in enumerate(ACTIVE_REGIONS):
        if ar.check == 1:
            text2 += "Act Reg {0}: {1:<6}, lon = {2:<5.1f}, lat = {3:<4.1f}, size = {4:<5.3f} [Rsun] (= {5:.2f}%)\n".format(
                str(i + 1),
                ar.active_region_type_str,
                ar.long,
                ar.lat,
                ar.size,
                ar.size**2 / 2.0 * 100,
            )

    plt.figtext(
        0.015, 0.03, text1, name="Bitstream Vera Sans Mono", size=13
    )  # name='Courier' very important because monotype font (all the characters have the same space)
    plt.figtext(
        0.485, 0.076, text2, name="Bitstream Vera Sans Mono", size=13
    )  # name='Courier' very important because monotype font (all the characters have the same space)

    # plt.savefig(os.path.join(CCF_folder_outputs, filename+'.pdf'))
    plt.show()
