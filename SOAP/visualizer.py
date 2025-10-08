from copy import deepcopy as dpcy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon
from SOAP.fast_starspot import ld, spot_init, spot_phase,spot_area
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
            if ((ys[k] - yl) ** 2 + (zs[j] + zl) ** 2 <= sim.planet.Rp.value**2) and ((ys[k] - yl) ** 2 + (zs[j] + zl) ** 2>= 0.8*sim.planet.Rp.value**2):
                planet_arr[j, k] = 0.1

    return np.ma.masked_array(planet_arr, planet_arr < 0.1)

def visualize(sim, out, plot_type, lim=None, ref_wave=None, plot_lims=None, show_data=True):
    # Phases and basic outputs
    psi = out.psi
    flux = out.flux.value
    if sim.has_planet:
        psi_i = dpcy(out.psi) * sim.star.prot / sim.planet.P
        xyz = sim.xyzplanet
        y, z = np.array(xyz).T[1:]
    else:
        psi_i = out.psi

    # Grid resolution, geometry, and arrays (match animate)
    grid_res = 0.0005
    Vsini = (2 * np.pi * sim.star.radius.value * 696340) / (sim.star.prot.value * 24 * 3600)
    i = np.radians(180-sim.star.incl.value)

    ys = np.linspace(-1.5, 1.5, int(max(2, 1.0 / grid_res)))
    zs = np.linspace(-1.5, 1.5, int(max(2, 1.0 / grid_res)))
    # Finer grid for planet mask like animate
    if sim.has_planet:
        yp = np.linspace(-1.5, 1.5, int(max(2, 1.0 / (10 * grid_res * sim.planet.Rp.value))))
        zp = np.linspace(-1.5, 1.5, int(max(2, 1.0 / (10 * grid_res * sim.planet.Rp.value))))
    else:
        yp = ys
        zp = zs

    val = np.nan * np.zeros((len(ys), len(zs)))
    alpha = np.nan * np.zeros((len(ys), len(zs)))

    # Stellar surface velocities + limb darkening (match animate)
    sin_i = np.sin(i)
    cos_i = np.cos(i)
    Y, Z = np.meshgrid(ys, zs)
    r2 = Y**2 + Z**2
    mask = r2 <= 1.0

    r_cos = np.sqrt(1.0 - r2[mask])
    latitude = Z[mask] * sin_i + r_cos * cos_i
    delta_z = Vsini * (1.0 - sim.star.diffrotB * latitude**2 - sim.star.diffrotC * latitude**4)
    delta = Y[mask] * delta_z * sin_i + (sim.star.cb1 / 1000) * r_cos

    val[mask] = delta
    alpha[mask] = ld(Y[mask], Z[mask], sim.star.u1, sim.star.u2)[1]

    # Figure and axes (match relative sizes to animate)
    if show_data:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.35]}, constrained_layout=True)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={"width_ratios": [1, 0.0001]}, constrained_layout=True)

    # Choose plot data based on plot_type (align semantics with animate)
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
        # Use sim.pixel_trans as in animate; plot as Absorption (%) on right
        points_to_plot = sim.pixel_trans
        color_label = "Local RV (km/s)"
        cmap_s = star_gradient
    elif plot_type == "shadow":
        data_to_plot = val
        points_to_plot = sim.master_out_fw - sim.integrated_spectra_fw
        color_label = "Local RV (km/s)"
        cmap_s = star_gradient
    else:
        raise ValueError("Invalid plot_type. Choose 'rv', 'flux', 'tr', or 'shadow'.")

    # Extent/origin consistent with animate
    extent = (-1.5, 1.5, -1.5, 1.5)
    origin = "upper"

    # Left: base star image
    im_star = axs[0].imshow(data_to_plot, cmap=cmap_s, extent=extent, origin=origin)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel(r"$R_{\star}$")
    axs[0].set_ylabel(r"$R_{\star}$")

    # Planet phases and mask, consistent with animate
    if sim.has_planet:
        tr_dur,tr_ingress_egress = transit_durations(sim)
        planet_phases = psi_i
        phase_mask = np.logical_or(planet_phases < -1.1 * tr_dur / 2, planet_phases > 1.1 * tr_dur / 2)
    else:
        planet_phases = psi_i
        # No planet: use inner half range for color normalization similar to animate's no-planet branch
        phase_mask = np.zeros_like(planet_phases, dtype=bool)

    # Draw the transit chord if planet present
    if sim.has_planet:
        chord_line, = axs[0].plot(y, z, "--k")
    else:
        None
        #chord_line, = axs[0].plot([], [], "--k")
    axs[0].set_xlim(-1.5, 1.5)
    axs[0].set_ylim(-1.5, 1.5)
    # Overplot planet position(s) with the same resolution approach as animate
    artists_planet = []
    if sim.has_planet and np.any(~phase_mask):
        sel = np.arange(len(psi))[~phase_mask]
        planet_arr = [disk_position(sim, sim.xyzplanet, x, yp, zp) for x in sel]

        cmap = planet_gradient_2
        vmin_phase = np.min(planet_phases[~phase_mask])
        vmax_phase = np.max(planet_phases[~phase_mask])
        norm = Normalize(vmin=vmin_phase, vmax=vmax_phase)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for i_frame, k in enumerate(planet_arr):
            cmapi = ListedColormap(cmap(norm(planet_phases[~phase_mask][i_frame])))
            im_planet = axs[0].imshow(k, extent=extent, origin=origin, zorder=3, cmap=cmapi, alpha=0.9)
            artists_planet.append(im_planet)

        # Spots consistent with animate: draw once per phase overlay if desired
        if len(sim.active_regions) != 0:
            # Use the first unmasked phase to position spots (static snapshot choice)
            idx0 = sel[0]
            for j in sim.active_regions:
                spot = spot_init(j.size, (j.lon).value, (j.lat).value, (sim.star.incl).value, 40)
                spot_phase_position = spot_phase(spot, (sim.star.incl).value, out.psi[idx0] - out.psi[0] / 2)
                if j.active_region_type == 0:
                    facecolor = "black"
                else:
                    facecolor = "yellow"
                polygon = Polygon(
                    np.array([spot_phase_position[:, 1], spot_phase_position[:, 2]]).T,
                    closed=True,
                    facecolor=facecolor,
                )
                axs[0].add_patch(polygon)

    # Right panel behavior consistent with animate
    cmap_phase = planet_gradient_2
    if show_data:
        if plot_type in ("tr", "shadow") and sim.has_planet and np.any(~phase_mask):
            # Set up phase-colored series
            vmin_phase = np.min(planet_phases[~phase_mask])
            vmax_phase = np.max(planet_phases[~phase_mask])
            norm = Normalize(vmin=vmin_phase, vmax=vmax_phase)

            x_right = sim.pixel.wave - ref_wave if ref_wave is not None else sim.pixel.wave * 0.0
            for i_frame, idx in enumerate(np.where(~phase_mask)[0]):
                color_i = cmap_phase(norm(planet_phases[idx]))
                if plot_type == "tr":
                    ydat = (1 - points_to_plot[idx]) * 100
                    axs[1].plot(x_right, ydat, color=color_i, alpha=0.9)
                    axs[1].set_ylabel("Absorption (%)")
                else:
                    ydat = points_to_plot[idx]
                    axs[1].plot(x_right, ydat, color=color_i, alpha=0.9)
                    axs[1].set_ylabel("Flux")
            axs[1].set_xlabel(r"$\lambda$+" + str(ref_wave) + "(" + r"$\AA$" + ")" if ref_wave is not None else r"$\lambda$")
            if lim is not None:
                axs[1].set_xlim(lim[0] - (ref_wave or 0.0), lim[1] - (ref_wave or 0.0))
            if plot_lims is not None:
                axs[1].set_ylim(plot_lims[0], plot_lims[1])
            sm2 = ScalarMappable(cmap=cmap_phase, norm=norm)
            sm2.set_array([])
            cb2 = fig.colorbar(sm2, ax=axs[1], orientation="vertical", fraction=0.05, pad=0.04)
            cb2.set_label("Orbital Phase", rotation=270, labelpad=16)
        else:
            # RV/flux time series or no-planet
            if plot_type == "rv":
                axs[1].set_ylabel("RV [m/s]")
            elif plot_type == "flux":
                axs[1].set_ylabel("Flux")
            # Scatter all points in black plus highlight in color if planet phases exist
            axs[1].plot(psi_i[phase_mask], points_to_plot[phase_mask], "o", color="k")
            axs[1].axvspan(-tr_dur/2., tr_dur/2., alpha=0.1, color='orange')
            axs[1].axvspan(-tr_ingress_egress/2., tr_ingress_egress/2., alpha=0.1, color='orange')
            if sim.has_planet and np.any(~phase_mask):
                vmin_phase = np.min(planet_phases[~phase_mask])
                vmax_phase = np.max(planet_phases[~phase_mask])
                norm = Normalize(vmin=vmin_phase, vmax=vmax_phase)
                
                # Plot all points with corresponding colors
                colors = cmap_phase(norm(psi_i[~phase_mask]))
                axs[1].scatter(psi_i[~phase_mask], points_to_plot[~phase_mask], marker="o", color=colors, zorder=5, edgecolor='k')
    
                sm2 = ScalarMappable(cmap=cmap_phase, norm=norm)
                sm2.set_array([])
                cb2 = fig.colorbar(sm2, ax=axs[1], orientation="vertical", fraction=0.05, pad=0.04)
                cb2.set_label("Orbital Phase", rotation=270, labelpad=16)
            axs[1].set_xlabel("Orbital Phase")
    else:
        axs[1].set_axis_off()

    # Left colorbar consistent with animate
    cb1 = fig.colorbar(im_star, ax=axs[0], orientation="vertical", fraction=0.05, pad=0.04)
    cb1.set_label(color_label, rotation=270, labelpad=16)

    # Optional: add the same stellar latitude/longitude guides as in animate
    latitudes = [-60, -30, 0, 30, 60]
    for lat in latitudes:
        y_lat, z_lat, mask_lat = latitude_circle(lat, np.pi/2 - i)
        plot_visible_arcs(axs[0], y_lat, z_lat, mask_lat, color='k', linestyle='--', linewidth=0.5, alpha=0.7)
    longitudes = [-120, -150, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
    for lon in longitudes:
        y_lon, z_lon = longitude_line(lon, np.pi/2 - i)
        mask_line = y_lon**2 + z_lon**2 <= 1.0
        axs[0].plot(y_lon[mask_line], z_lon[mask_line], color='k', linestyle=':', linewidth=0.5, alpha=0.7)

    # Global layout pads similar to animate
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.02, wspace=0.02)

    #plt.show()


def animate_visualization(
    sim, out, plot_type, lim, ref_wave, plot_lims, interval=100, repeat=True
):
    psi = out.psi
    flux = out.flux.value
    psi_i = dpcy(out.psi) * sim.star.prot / sim.planet.P
    if sim.has_planet:
        xyz = sim.xyzplanet
        y, z = np.array(xyz).T[1:]


    # Grid resolution
    grid_res = 1/sim.grid
    Vsini = (2 * np.pi * sim.star.radius.value * 696340) / (
        sim.star.prot.value * 24 * 3600
    )
    i = np.radians(sim.star.incl.value)
    ys = np.linspace(-1.5, 1.5, int(max(2, 1.0 / grid_res)))
    zs = np.linspace(-1.5, 1.5, int(max(2, 1.0 / grid_res)))
    if sim.has_planet:
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
        raise ValueError("Invalid plot_type. Choose 'rv', 'flux', 'tr', or 'shadow'.")


    # Transit duration and phase masking
    tr_dur, tr_ingress_egress = transit_durations(sim)
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
    if sim.has_planet:
        # Set up the planet's position precomputation
        sel = np.arange(len(psi))[~phase_mask]
        planet_arr = [disk_position(sim, xyz, x, yp, zp) for x in sel]
    else:
        sel = np.arange(len(psi))


    # Left axis static setup
    axs[0].set_aspect("equal")
    axs[0].set_xlabel(r"$R_{\star}$")
    axs[0].set_ylabel(r"$R_{\star}$")
    if sim.has_planet:
        chord_line, = axs[0].plot(y, z, "--k")


    # Establish extent/origin once
    extent = (-1.5, 1.5, -1.5, 1.5)
    origin = "upper"


    # Base star image and planet overlay
    im_star = axs[0].imshow(data_to_plot, cmap=cmap_s, extent=extent, origin=origin)
    if sim.has_planet:
        cmapi = ListedColormap(cmap(norm(planet_phases[~phase_mask][0]))) if np.any(~phase_mask) else ListedColormap(cmap(0.5))
        im_planet = axs[0].imshow(
            planet_arr[0], extent=extent, origin=origin, zorder=3, cmap=cmapi, alpha=0.9
        )
    # Add latitude circles for 3D effect on stellar disk
    latitudes = [-60, -30, 0, 30, 60]  # degrees
    for lat in latitudes:
        y_lat, z_lat, mask_lat = latitude_circle(lat,np.pi/2- i)
        plot_visible_arcs(axs[0], y_lat, z_lat, mask_lat, color='k', linestyle='--', linewidth=0.5, alpha=0.7)
        axs[0].set_xlim(-1.5, 1.5)
        axs[0].set_ylim(-1.5, 1.5)
    
    # Longitude lines at angles (in degrees)
    longitudes = [-120, -150,-90,-60, -30, 0, 30, 60, 90, 120, 150, 180]  # degrees
    for lon in longitudes:
        y_lon, z_lon = longitude_line(lon,np.pi/2- i)
        mask_line = y_lon**2 + z_lon**2 <= 1.0
        axs[0].plot(y_lon[mask_line], z_lon[mask_line], color='k', linestyle=':', linewidth=0.5, alpha=0.7)


    # Active regions - create polygons once with initial empty coords and add to axes
    spot_polys = []
    if len(sim.active_regions) != 0:
        for j in sim.active_regions:
            spot = spot_init(j.size, (j.lon).value, (j.lat).value, (sim.star.incl).value, 40)
            facecolor = "black" if j.active_region_type == 0 else "yellow"
            poly = Polygon(np.zeros((0, 2)), closed=True, facecolor=facecolor, visible=False)
            axs[0].add_patch(poly)
            spot_polys.append((poly, spot, j.active_region_type))


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
        if sim.has_planet:
            pts_all, = axs[1].plot(psi_i, points_to_plot, "o", color="k")
            axs[1].axvspan(-tr_dur / 2.0, tr_dur / 2.0, alpha=0.1, color="orange")
            axs[1].axvspan(-tr_ingress_egress / 2.0, tr_ingress_egress / 2.0, alpha=0.1, color="orange")
            idx0 = np.where(~phase_mask)[0][0]
            curr, = axs[1].plot(
                [psi_i[idx0]], [points_to_plot[idx0]], "o", color=cmap(norm(psi_i[~phase_mask][0])), markeredgecolor="k"
            )
            axs[1].set_xlabel("Orbital Phase")
            if plot_type == "rv":
                axs[1].set_ylabel("RV [m/s]")
            elif plot_type == "flux":
                axs[1].set_ylabel("Flux")
        else:
            pts_all, = axs[1].plot(psi, points_to_plot, "o", color="gray")
            curr, = axs[1].plot([psi[0]], [points_to_plot[0]], "o", color="k")
            axs[1].set_xlabel("Rotational Phase")
            if plot_type == "rv":
                axs[1].set_ylabel("RV [m/s]")
            elif plot_type == "flux":
                axs[1].set_ylabel("Flux")

        right_artists.extend([pts_all, curr])


    # Colorbars — attach to axes via fig.colorbar so constrained layout reserves space
    cb1 = fig.colorbar(im_star, ax=axs[0], orientation="vertical", fraction=0.05, pad=0.04)
    cb1.set_label(color_label, rotation=270, labelpad=16)

    cb2 = fig.colorbar(sm, ax=axs[1], orientation="vertical", fraction=0.05, pad=0.04)
    if sim.has_planet:
        cb2.set_label("Orbital Phase", rotation=270, labelpad=16)
    else:
        cb2.set_label("Rotational Phase", rotation=270, labelpad=16)


    # Slightly tighter global pads to better use the canvas while avoiding overlap
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.02, wspace=0.02)  # scaled pads with constrained layout.


    # Collect artists for blit/init
    if sim.has_planet:
        artists_left = [im_star, im_planet, chord_line] + [p[0] for p in spot_polys]
    else:
        artists_left = [im_star] + [p[0] for p in spot_polys]
    artists_right = right_artists
    all_artists = artists_left + artists_right


    def init():
        return all_artists


    def update(frame):
        if sim.has_planet:
            im_planet.set_data(planet_arr[frame])
            cmapi = ListedColormap(cmap(norm(planet_phases[~phase_mask][frame]))) if np.any(~phase_mask) else ListedColormap(cmap(0.5))
            im_planet.set_cmap(cmapi)

        if len(spot_polys) != 0 and np.any(~phase_mask):
            for (poly, spot, active_region_type) in spot_polys:
                spp = spot_phase(spot, (sim.star.incl).value, out.psi[sel[frame]] - out.psi[0] / 2)
                vis=spp.T[0]>0
                if vis.any():
                    poly.set_xy(np.array([spp[:, 1], spp[:, 2]]).T)
                    poly.set_visible(True)
                else:
                    poly.set_visible(False)


        if plot_type in ("tr", "shadow"):
            if plot_type == "tr":
                ydat = (1 - points_to_plot[~phase_mask][frame]) * 100
            else:
                ydat = points_to_plot[~phase_mask][frame]
            line_spec.set_ydata(ydat)
            line_spec.set_color(cmap(norm(planet_phases[~phase_mask][frame]))
            )
        else:
            if not sim.has_planet:
                idx = frame
                artists_right[1].set_data([psi[idx]], [points_to_plot[idx]])
                artists_right[1].set_color("k")
            else:
                idx = np.where(~phase_mask)[0][frame] if np.any(~phase_mask) else None
                if idx is not None:
                    artists_right[1].set_data([psi_i[idx]], [points_to_plot[idx]])
                    artists_right[1].set_color(cmap(norm(psi_i[~phase_mask][frame])))


        cb2.update_normal(sm)
        return all_artists

    if sim.has_planet:
        ani = FuncAnimation(
            fig, update, frames=len(planet_arr), init_func=init, interval=interval, repeat=repeat, blit=True
        )
    else:
        ani = FuncAnimation(
            fig, update, frames=len(psi), init_func=init, interval=interval, repeat=repeat, blit=True
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
    sim                 : simulation object
        The simulation object containing stellar and planetary parameters.
    psi                 : array-like
        Stellar rotation phase array.
    absorption_spec     : 2D array
        Absorption data to be plotted (shape matches meshgrid of wavelength and orbital phase).
    λ                   : array-like with 2 elements
        Wavelength limits [min, max] for plot x-axis.
    cmap                : matplotlib colormap
        Colormap for the absorption spectrum.

    Returns:
    fig, ax : matplotlib Figure and Axes
        The figure and axis containing the plot.
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