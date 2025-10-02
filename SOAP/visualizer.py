from copy import deepcopy as dpcy
from SOAP.fast_starspot import doppler_shift
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon, FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.constants import M_earth, M_sun
from SOAP.fast_starspot import ld, spot_area, spot_init, spot_phase
from SOAP.utils import planet_gradient_2, star_gradient, star_gradient_flux,transgrad, transit_durations

# Set font size for plots
matplotlib.rc("font", size=16)


# Helper function to generate the planetary disk position
def disk_position(sim, xyz, l, ys, zs):
    xl, yl, zl = xyz[l]
    planet_arr = np.zeros((len(ys), len(zs)))
    y_p = np.where(
        (ys >= (yl - sim.planet.Rp.value)) & (ys <= (yl + sim.planet.Rp.value))
    )[0]
    z_p = np.where(
        (zs >= (-zl - sim.planet.Rp.value)) & (zs <= (-zl + sim.planet.Rp.value))
    )[0]

    for k in y_p:
        for j in z_p:
            if ((ys[k] - yl) ** 2 + (zs[j] + zl) ** 2 <= sim.planet.Rp.value**2) and ((ys[k] - yl) ** 2 + (zs[j] + zl) ** 2>= 0.5*sim.planet.Rp.value**2):
                planet_arr[j, k] = 0.2

    return np.ma.masked_array(planet_arr, planet_arr < 0.2)


def visualize(sim, out, plot_type, lim, ref_wave, plot_lims, show_data):
    psi = out.psi
    flux = out.flux.value
    if sim.has_planet == True:
        psi_i = out.psi * sim.star.prot / sim.planet.P
        xyz = sim.xyzplanet
        y, z = np.array(xyz).T[1:]
    else:
        psi_i = out.psi

    # grid_res = 0.0005
    grid_res = 0.0005
    Vsini = (2 * np.pi * sim.star.radius.value * 696340) / (
        sim.star.prot.value * 24 * 3600
    )
    i = np.radians(sim.star.incl.value)
    ys = np.linspace(-1.5, 1.5, int(1.0 / grid_res))
    zs = np.linspace(-1.5, 1.5, int(1.0 / grid_res))
    val, alpha = np.nan * np.zeros((len(ys), len(zs))), np.nan * np.zeros(
        (len(ys), len(zs))
    )

    ########## Compute the stellar surface velocities + flux #########################
    sin_i = np.sin(i)
    cos_i = np.cos(i)

    # Create mesh grid for ys and zs (this assumes ys and zs are 1D arrays)
    Y, Z = np.meshgrid(ys, zs)

    # Compute r2 once for the whole array
    r2 = Y**2 + Z**2

    # Mask where r2 <= 1 to avoid unnecessary calculations
    mask = r2 <= 1

    # Compute r_cos and latitude for valid points (r2 <= 1)
    r_cos = np.sqrt(1.0 - r2[mask])
    latitude = Z[mask] * sin_i + r_cos * cos_i

    # Compute delta_z
    delta_z = Vsini * (
        1.0 - sim.star.diffrotB * latitude**2 - sim.star.diffrotC * latitude**4
    )

    # Compute delta
    delta = Y[mask] * delta_z * sin_i + (sim.star.cb1 / 1000) * r_cos

    # Store delta in the appropriate positions of val
    val[mask] = delta

    # Limb darkening calculations (use the mask to avoid unnecessary calculations)
    alpha[mask] = ld(Y[mask], Z[mask], sim.star.u1, sim.star.u2)[1]
    #####################################
    if show_data == True:
        fig, axs = plt.subplots(
            1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1]}
        )
    else:
        fig, axs = plt.subplots(
            1, 2, figsize=(7, 5), gridspec_kw={"width_ratios": [1, 0]}
        )

    # Choose what to plot based on plot_type argument
    if plot_type == "rv":
        rv = out.rv.value
        data_to_plot = val
        points_to_plot = rv * 1000
        color_label = "Local RV (km/s)"
        cmap = star_gradient
        if show_data == True:
            axs[1].set_ylabel("RV [m/s]")
            axs[1].plot(psi_i, points_to_plot, "*k")
        else:
            None
    elif plot_type == "flux":
        data_to_plot = alpha
        points_to_plot = flux
        color_label = "Local normalized flux"
        cmap = star_gradient_flux
        if show_data == True:
            axs[1].set_ylabel("Normalized flux")
            axs[1].plot(psi_i, points_to_plot, "*k")

    elif plot_type == "tr":
        data_to_plot = val
        psi_planet=(psi*sim.star.prot/sim.planet.P).value
        Kp=(sim.planet.K)*(M_sun*sim.star.mass)/(sim.planet.Mp*(M_earth))
        pshift=(Kp*np.sin(2*np.pi*psi_planet)).value
        absorption_spec=(1-sim.pixel_trans)
        absorpt_prest=np.array([doppler_shift(sim.pixel.wave,absorption_spec[x],-1000*pshift[x]) for x in range(len(pshift))])
        points_to_plot = absorption_spec
        color_label = "Local RV (km/s)"
        if show_data == True:
            axs[1].set_ylabel("Absorption (%)")
            axs[1].set_xlabel(
                r"$\lambda$" + "+" + str(ref_wave) + " (" + r"$\AA$" + ")"
            )
        cmap = star_gradient
    elif plot_type == "shadow":
        data_to_plot = val
        points_to_plot = sim.master_out_fw - sim.integrated_spectra_fw
        color_label = "Local RV (km/s)"
        if show_data == True:
            axs[1].set_ylabel("Flux")
            axs[1].set_xlabel(
                r"$\lambda$" + "+" + str(ref_wave) + " (" + r"$\AA$" + ")"
            )
        cmap = star_gradient

    else:
        raise ValueError("Invalid plot_type. Choose either 'rv' or 'flux'.")

    # Left plot: Either local radial velocities or limb darkening on stellar disk
    h = axs[0].imshow(data_to_plot, cmap=cmap, extent=(-1.5, 1.5, -1.5, 1.5))
    axs[0].set_xlabel(r"$R_{\star}$")
    axs[0].set_ylabel(r"$R_{\star}$")
    axs[0].set_aspect("equal", adjustable="datalim")

    if sim.has_planet == True:
        axs[0].plot(y, z, "--k")
        # Transit duration and phase masking
        tr_dur = (
            1.0
            / np.pi
            * np.arcsin(
                1.0
                / sim.planet.a.value
                * np.sqrt(
                    (1 + sim.planet.Rp.value) ** 2
                    - (sim.planet.a.value) ** 2
                    * np.cos(np.radians(sim.planet.ip.value)) ** 2
                )
            )
        )
        planet_phases = psi * sim.star.prot.value / sim.planet.P.value
        phase_mask = np.logical_or(
            planet_phases < -tr_dur / 2, planet_phases > tr_dur / 2
        )

    else:
        planet_phases = psi_i
        phase_mask = np.logical_or(
            planet_phases >= planet_phases[0], planet_phases <= planet_phases[-1]
        )

    if sim.has_planet == True:
        # Right plot: radial velocity vs orbital phase
        cmap = planet_gradient_2
        norm = Normalize(
            vmin=min(planet_phases[~phase_mask]), vmax=max(planet_phases[~phase_mask])
        )

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Plot the planet's position during transit on both plots
        planet_arr = [
            disk_position(sim, xyz, x, ys, zs) for x in np.arange(len(psi))[~phase_mask]
        ]
        for i, k in enumerate(planet_arr):
            cmapi = ListedColormap(cmap(norm(planet_phases[~phase_mask][i])))
            axs[0].imshow(
                k,
                extent=(-1.5, 1.5, -1.5, 1.5),
                origin="upper",
                zorder=3,
                cmap=cmapi,
                alpha=0.9,
            )
            if show_data == True:
                if plot_type == "tr" or plot_type == "shadow":
                    if plot_type == "tr":
                        axs[1].plot(
                            sim.pixel.wave - ref_wave,
                            (points_to_plot[~phase_mask][i]) * 100,
                            color=cmap(norm(planet_phases[~phase_mask][i])),
                            alpha=0.9,
                        )
                    elif plot_type == "shadow":
                        axs[1].plot(
                            sim.pixel.wave - ref_wave,
                            points_to_plot[~phase_mask][i],
                            color=cmap(norm(planet_phases[~phase_mask][i])),
                            alpha=0.9,
                        )

                    if lim:
                        axs[1].set_xlim(lim[0] - ref_wave, lim[1] - ref_wave)
                    else:
                        axs[1].set_xlim(min(sim.pixel.wave), max(sim.pixel.wave))

                    if plot_lims:
                        axs[1].set_ylim(plot_lims[0], plot_lims[1])
                    else:
                        None

                else:
                    axs[1].plot(
                        psi_i[~phase_mask][i],
                        points_to_plot[~phase_mask][i],
                        "*",
                        color=cmap(norm(psi[~phase_mask][i])),
                    )
                    axs[1].set_xlim([min(psi_i), max(psi_i)])
                    axs[1].set_xlabel("Orbital Phase")

                if len(sim.active_regions) != 0:
                            for j in sim.active_regions:
                                spot = spot_init(
                                    j.size, (j.lon).value, (j.lat).value, (sim.star.incl).value, 40
                                )
                                spot_phase_position = spot_phase(
                                    spot, (sim.star.incl).value, out.psi[i] - out.psi[0]
                                )
                                polygon = Polygon(
                                    np.array([spot_phase_position[:, 1], spot_phase_position[:, 2]]).T,
                                    closed=True,
                                    facecolor="black",
                                )
                                axs[0].add_patch(polygon)
        # axs[1].set_aspect("auto")

        # Add colorbar to the left plot
        divider1 = make_axes_locatable(axs[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cb1 = fig.colorbar(h, cax=cax1, orientation="vertical")
        cb1.set_label(color_label, rotation=270, labelpad=16)
        if show_data == True:
            # Add label for the second plot, aligning it similarly to the first subplot
            divider2 = make_axes_locatable(axs[1])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cb2 = fig.colorbar(sm, cax=cax2, orientation="vertical")
            cb2.set_label("Orbital Phase", rotation=270, labelpad=16)
        else:
            None
    else:
        if show_data == True:
            axs[1].set_xlabel("Rotation Phase")
        cmap = planet_gradient_2
        norm = Normalize(vmin=psi_i[int(len(psi) / 4)], vmax=psi_i[-int(len(psi) / 4)])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if len(sim.active_regions) != 0:
            for l, frame in enumerate(psi_i):
                cmapi = cmap(norm(frame))
                for j in sim.active_regions:
                    spot = spot_init(
                        j.size, (j.lon).value, (j.lat).value, (sim.star.incl).value, 40
                    )
                    spot_phase_position = spot_phase(spot, (sim.star.incl).value, frame)
                    vis, _, _, _, _ = spot_area(spot_phase_position, sim.nrho, sim.grid)
                    if vis:
                        polygon = Polygon(
                            np.array(
                                [spot_phase_position[:, 1], spot_phase_position[:, 2]]
                            ).T,
                            closed=True,
                            facecolor=cmapi if show_data == True else "k",
                        )
                        axs[0].add_patch(polygon)
                        if show_data == True:
                            axs[1].plot(psi_i[l], points_to_plot[l], "*", color=cmapi)

        # Add colorbar to the left plot
        divider1 = make_axes_locatable(axs[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cb1 = fig.colorbar(h, cax=cax1, orientation="vertical")
        cb1.set_label(color_label, rotation=270, labelpad=16)
        if show_data == True:
            # Add label for the second plot, aligning it similarly to the first subplot
            divider2 = make_axes_locatable(axs[1])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cb2 = fig.colorbar(sm, cax=cax2, orientation="vertical")
            cb2.set_label("Rotation Phase", rotation=270, labelpad=16)
        else:
            axs[1].set_axis_off()
            None
    plt.tight_layout()

def animate_visualization(
    sim, out, plot_type, lim, ref_wave, plot_lims, interval=100, repeat=True
):
    psi = out.psi
    flux = out.flux.value
    psi_i = dpcy(out.psi) * sim.star.prot / sim.planet.P
    xyz = sim.xyzplanet
    y, z = np.array(xyz).T[1:]

    # Grid resolution
    grid_res = 0.0005
    Vsini = (2 * np.pi * sim.star.radius.value * 696340) / (
        sim.star.prot.value * 24 * 3600
    )
    i = np.radians(sim.star.incl.value)
    ys = np.linspace(-1.5, 1.5, int(max(2, 1.0 / grid_res)))
    zs = np.linspace(-1.5, 1.5, int(max(2, 1.0 / grid_res)))
    yp = np.linspace(
        -1.5, 1.5, int(max(2, 1.0 / (10 * grid_res * (sim.planet.Rp).value)))
    )
    zp = np.linspace(
        -1.5, 1.5, int(max(2, 1.0 / (10 * grid_res * (sim.planet.Rp).value)))
    )
    val, alpha = np.nan * np.zeros((len(ys), len(zs))), np.nan * np.zeros(
        (len(ys), len(zs))
    )

    ########## Compute the stellar surface velocities + flux #########################
    sin_i = np.sin(i)
    cos_i = np.cos(i)

    # Create mesh grid for ys and zs
    Y, Z = np.meshgrid(ys, zs)

    # Compute r2 once for the whole array
    r2 = Y**2 + Z**2

    # Mask where r2 <= 1
    mask = r2 <= 1

    # Compute r_cos and latitude for valid points
    r_cos = np.sqrt(1.0 - r2[mask])
    latitude = Z[mask] * sin_i + r_cos * cos_i

    # Compute delta_z
    delta_z = Vsini * (
        1.0 - sim.star.diffrotB * latitude**2 - sim.star.diffrotC * latitude**4
    )

    # Compute delta
    delta = Y[mask] * delta_z * sin_i + (sim.star.cb1 / 1000) * r_cos

    # Store delta and limb darkening
    val[mask] = delta
    alpha[mask] = ld(Y[mask], Z[mask], sim.star.u1, sim.star.u2)[1]
    #####################################

    # Setup the figure and axes: make right subplot wider and enable constrained layout
    fig, axs = plt.subplots(
        1, 2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [1, 1.35]},  # right wider than left
        constrained_layout=True
    )

    # Choose plot data based on plot_type argument
    if plot_type == "rv":
        rv = out.rv.value
        data_to_plot = val
        points_to_plot = rv * 1000
        color_label = "Local RV (km/s)"
        cmap_s = star_gradient
    elif plot_type == "flux":
        data_to_plot = alpha
        points_to_plot = flux
        color_label = "Local normalized flux"
        cmap_s = star_gradient_flux
    elif plot_type == "tr":
        data_to_plot = val
        points_to_plot = sim.pixel_trans
        color_label = "Local RV (km/s)"
        cmap_s = star_gradient
    elif plot_type == "shadow":
        data_to_plot = val
        points_to_plot = sim.master_out_fw - sim.integrated_spectra_fw
        color_label = "Local RV (km/s)"
        cmap_s = star_gradient
    else:
        raise ValueError("Invalid plot_type. Choose either 'rv' or 'flux'.")

    # Transit duration and phase masking
    tr_dur = (
        1.0
        / np.pi
        * np.arcsin(
            1.0
            / sim.planet.a.value
            * np.sqrt(
                (1 + sim.planet.Rp.value) ** 2
                - (sim.planet.a.value) ** 2
                * np.cos(np.radians(sim.planet.ip.value)) ** 2
            )
        )
    )
    planet_phases = psi_i
    phase_mask = np.logical_or(
        planet_phases < -1.1 * tr_dur / 2, planet_phases > 1.1 * tr_dur / 2
    )

    # Right plot: radial velocity vs orbital phase
    cmap = planet_gradient_2
    if np.any(~phase_mask):
        vmin_phase = np.min(planet_phases[~phase_mask])
        vmax_phase = np.max(planet_phases[~phase_mask])
    else:
        vmin_phase = -1
        vmax_phase = 1
    norm = Normalize(vmin=vmin_phase, vmax=vmax_phase)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Set up the planet's position precomputation
    sel = np.arange(len(psi))[~phase_mask]
    planet_arr = [disk_position(sim, xyz, x, yp, zp) for x in sel]

    # Left axis static setup
    axs[0].set_aspect("equal")
    axs[0].set_xlabel(r"$R_{\star}$")
    axs[0].set_ylabel(r"$R_{\star}$")
    chord_line, = axs[0].plot(y, z, "--k")

    # Establish extent/origin once
    extent = (-1.5, 1.5, -1.5, 1.5)
    origin = "upper"

    # Base star image and planet overlay
    im_star = axs[0].imshow(data_to_plot, cmap=cmap_s, extent=extent, origin=origin)
    cmapi = ListedColormap(cmap(norm(planet_phases[~phase_mask][0]))) if np.any(~phase_mask) else ListedColormap(cmap(0.5))
    im_planet = axs[0].imshow(
        planet_arr[0], extent=extent, origin=origin, zorder=3, cmap=cmapi, alpha=0.9
    )
    # Add latitude circles for 3D effect on stellar disk
    latitudes = [-60, -30, 0, 30, 60]  # degrees
    for lat in latitudes:
        y_lat, z_lat, mask_lat = latitude_circle(lat,np.pi/2- i)
        plot_visible_arcs(axs[0], y_lat, z_lat, mask_lat, color='k', linestyle='--', linewidth=0.5, alpha=0.7)
        # y_lat, z_lat = latitude_circle(lat, i)
        # mask_circle = y_lat**2 + z_lat**2 <= 1.0
        # axs[0].plot(y_lat[mask_circle], z_lat[mask_circle], color='k', linestyle='--', linewidth=0.5, alpha=0.7)
        axs[0].set_xlim(-1.5, 1.5)
        axs[0].set_ylim(-1.5, 1.5)
    
    # Longitude lines at angles (in degrees)
    longitudes = [-120, -150,-90,-60, -30, 0, 30, 60, 90, 120, 150, 180]  # degrees
    for lon in longitudes:
        y_lon, z_lon = longitude_line(lon,np.pi/2- i)
        mask_line = y_lon**2 + z_lon**2 <= 1.0
        axs[0].plot(y_lon[mask_line], z_lon[mask_line], color='k', linestyle=':', linewidth=0.5, alpha=0.7)
    

    # Active regions
    spot_polys = []
    if len(sim.active_regions) != 0:
        for j in sim.active_regions:
            spot = spot_init(j.size, (j.lon).value, (j.lat).value, (sim.star.incl).value, 40)
            spot_phase_position = spot_phase(spot, (sim.star.incl).value, out.psi[sel[0]] - out.psi[0] / 2 if len(sel) else 0.0)
            poly = Polygon(
                np.array([spot_phase_position[:, 1], spot_phase_position[:, 2]]).T,
                closed=True, facecolor="black",
            )
            axs[0].add_patch(poly)
            spot_polys.append((poly, spot))

    # Right axis: create artists once
    right_artists = []
    if plot_type in ("tr", "shadow"):
        if plot_type == "tr":
            x_right = sim.pixel.wave - ref_wave
            y0 = (1 - points_to_plot[~phase_mask][0]) * 100 if np.any(~phase_mask) else np.zeros_like(x_right)
            line_spec, = axs[1].plot(
                x_right, y0,
                color=cmap(norm(planet_phases[~phase_mask][0])) if np.any(~phase_mask) else cmap(0.5),
                alpha=0.9,
            )
            axs[1].set_ylabel("Absorption (%)")
            axs[1].set_xlabel(r"$\lambda$+" + str(ref_wave) + "(" + r"$\AA$" + ")")
        else:
            x_right = sim.pixel.wave - ref_wave
            y0 = points_to_plot[~phase_mask][0] if np.any(~phase_mask) else np.zeros_like(x_right)
            line_spec, = axs[1].plot(
                x_right, y0,
                color=cmap(norm(planet_phases[~phase_mask][0])) if np.any(~phase_mask) else cmap(0.5),
                alpha=0.9,
            )
            axs[1].set_ylabel("Flux")
            axs[1].set_xlabel(r"$\lambda$+" + str(ref_wave) + "(" + r"$\AA$" + ")")

        if lim:
            axs[1].set_xlim(lim[0] - ref_wave, lim[1] - ref_wave)
        else:
            axs[1].set_xlim(np.min(sim.pixel.wave), np.max(sim.pixel.wave))

        if plot_lims:
            axs[1].set_ylim(plot_lims[0], plot_lims[1])

        right_artists.append(line_spec)
    else:
        pts_all, = axs[1].plot(psi_i, points_to_plot, "*", color="k")
        if np.any(~phase_mask):
            idx0 = np.where(~phase_mask)[0][0]
            curr, = axs[1].plot(
                [psi_i[idx0]], [points_to_plot[idx0]],
                "*", color=cmap(norm(psi[~phase_mask][0])),
            )
        else:
            curr, = axs[1].plot([], [], "*", color=cmap(0.5))
        axs[1].set_xlabel("Orbital Phase")
        right_artists.extend([pts_all, curr])

    # Colorbars — attach to axes via fig.colorbar so constrained layout reserves space
    cb1 = fig.colorbar(im_star, ax=axs[0], orientation="vertical", fraction=0.05, pad=0.04)
    cb1.set_label(color_label, rotation=270, labelpad=16)

    cb2 = fig.colorbar(sm, ax=axs[1], orientation="vertical", fraction=0.05, pad=0.04)
    cb2.set_label("Orbital Phase", rotation=270, labelpad=16)

    # Slightly tighter global pads to better use the canvas while avoiding overlap
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.02, wspace=0.02)  # scaled pads with constrained layout.

    # Collect artists for blit/init
    artists_left = [im_star, im_planet, chord_line] + [p[0] for p in spot_polys]
    artists_right = right_artists
    all_artists = artists_left + artists_right

    def init():
        return all_artists

    def update(frame):
        im_planet.set_data(planet_arr[frame])
        cmapi = ListedColormap(cmap(norm(planet_phases[~phase_mask][frame]))) if np.any(~phase_mask) else ListedColormap(cmap(0.5))
        im_planet.set_cmap(cmapi)

        if len(spot_polys) != 0 and np.any(~phase_mask):
            for (poly, spot) in spot_polys:
                spp = spot_phase(spot, (sim.star.incl).value, out.psi[sel[frame]] - out.psi[0] / 2)
                poly.set_xy(np.array([spp[:, 1], spp[:, 2]]).T)

        if plot_type in ("tr", "shadow"):
            if plot_type == "tr":
                ydat = (1 - points_to_plot[~phase_mask][frame]) * 100
            else:
                ydat = points_to_plot[~phase_mask][frame]
            line_spec.set_ydata(ydat)
            line_spec.set_color(cmap(norm(planet_phases[~phase_mask][frame])))
        else:
            idx = np.where(~phase_mask)[0][frame] if np.any(~phase_mask) else None
            if idx is not None:
                artists_right[1].set_data([psi_i[idx]], [points_to_plot[idx]])
                artists_right[1].set_color(cmap(norm(psi[~phase_mask][frame])))

        cb2.update_normal(sm)
        return all_artists

    ani = FuncAnimation(
        fig, update, frames=len(planet_arr), init_func=init, interval=interval, repeat=repeat, blit=True
    )
    fig = plt.gcf()
    plt.close(fig)
    return ani

def latitude_circle(lat_deg, i_rad, npoints=500):
    phi = np.radians(lat_deg)
    theta = np.linspace(0, 2 * np.pi, npoints)
    
    x0 = np.cos(phi) * np.cos(theta)
    y0 = np.cos(phi) * np.sin(theta)
    z0 = np.full_like(theta, np.sin(phi))
    
    x = np.cos(i_rad) * x0 + np.sin(i_rad) * z0
    y = y0
    z = -np.sin(i_rad) * x0 + np.cos(i_rad) * z0
    
    mask = (x > 0) & (y**2 + z**2 <= 1.0)
    
    return y, z, mask

def longitude_line(lon_deg, i_rad, npoints=200):
    lam = np.radians(lon_deg)
    phi = np.linspace(-np.pi/2, np.pi/2, npoints)
    
    x0 = np.cos(phi) * np.cos(lam)
    y0 = np.cos(phi) * np.sin(lam)
    z0 = np.sin(phi)
    
    x = np.cos(i_rad) * x0 + np.sin(i_rad) * z0
    y = y0
    z = -np.sin(i_rad) * x0 + np.cos(i_rad) * z0
    
    mask = (x > 0) & (y**2 + z**2 <= 1.0)
    return y[mask], z[mask]


def plot_visible_arcs(ax, y, z, mask, **plot_kwargs):
    """Plot continuous arcs of points where mask=True on ax, avoid connecting gaps."""
    if len(y) == 0:
        return
    
    mask_int = mask.astype(int)
    diffs = np.diff(mask_int)
    
    # Start and end indices of True segments
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    
    # Handle edges if line starts or ends with True
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))
    
    # Plot each segment separately to avoid spurious lines
    for start, end in zip(starts, ends):
        ax.plot(y[start:end], z[start:end], **plot_kwargs)

def plot_absorption_map(sim,psi, absorption_spec,λ,cmap=transgrad):
    """
    Plot the absorption spectrum as a color map over wavelength and orbital phase,
    with transit duration and ingress/egress duration lines annotated.

    Parameters:
    sim_pixel_wave      : array-like
        Wavelength array (x-axis data)
    psi_planet          : array-like
        Orbital phase array (y-axis data)
    absorption_spec     : 2D array
        Absorption data to be plotted (shape matches meshgrid of sim_pixel_wave and psi_planet)
    transgrad           : matplotlib colormap
        Colormap for the absorption spectrum
    λ                   : array-like with 2 elements
        Wavelength limits [min, max] for plot x-axis
    tr_dur              : float
        Total transit duration (used for horizontal lines)
    tr_ingress_egress   : float
        Ingress/egress duration (used for horizontal dashed lines)

    Returns:
    fig, ax : matplotlib Figure and Axes
        The figure and axis containing the plot
    """
    tr_dur, tr_ingress_egress = transit_durations(sim)
    sim_pixel_wave = sim.pixel.wave
    # Convert rotation phases to planetary orbital phases
    psi_planet=(psi*sim.star.prot/sim.planet.P).value
    # Create meshgrid for plotting 2D array
    X, Y = np.meshgrid(sim_pixel_wave, psi_planet)
    fig, ax = plt.subplots(figsize=(14,7))

    # Normalize with center at zero for cyan-white-red style gradient
    norm = TwoSlopeNorm(vmin=np.min(absorption_spec), vcenter=0, vmax=np.max(absorption_spec))
    im = ax.pcolor(X, Y, absorption_spec, cmap=cmap, norm=norm)

    # Plot horizontal lines for transit durations in orbital phase
    ax.plot(λ, [-tr_dur/2, -tr_dur/2], '-k')
    ax.plot(λ, [tr_dur/2, tr_dur/2], '-k')
    ax.plot(λ, [-tr_ingress_egress/2, -tr_ingress_egress/2], '--k')
    ax.plot(λ, [tr_ingress_egress/2, tr_ingress_egress/2], '--k')

    ax.set_xlim(λ[0], λ[1])
    ax.set_xlabel("Wavelength "+"("+r"$\AA$"+")")
    ax.set_ylabel("Orbital Phase")

    # Colorbar creation and positioning next to the plot
    sm = ScalarMappable(cmap=im.get_cmap())
    sm.set_clim(np.min(absorption_spec), np.max(absorption_spec))
    sm.set_array([])  # Required for colorbar

    # Get figure and image axes position
    image_position = im.axes.get_position()

    # Define new axes for colorbar slightly to the right
    colorbar_width = image_position.width * 0.03  # small width relative to plot
    cax = fig.add_axes([image_position.x1 + 0.011, image_position.y0, 
                        colorbar_width, image_position.height])

    # Add colorbar with label
    fig.colorbar(sm, cax=cax, label="Absorption (%)")

    return fig, ax