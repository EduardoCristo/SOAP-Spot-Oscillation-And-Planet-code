from copy import deepcopy as dpcy
from SOAP.fast_starspot import doppler_shift
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.constants import M_earth, M_sun
from SOAP.fast_starspot import ld, spot_area, spot_init, spot_phase

# Set font size for plots
matplotlib.rc("font", size=16)

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
    ys = np.linspace(-1.5, 1.5, int(1.0 / grid_res))
    zs = np.linspace(-1.5, 1.5, int(1.0 / grid_res))
    yp = np.linspace(-1.5, 1.5, int(1.0 / (10 * grid_res * (sim.planet.Rp).value)))
    zp = np.linspace(-1.5, 1.5, int(1.0 / (10 * grid_res * (sim.planet.Rp).value)))
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

    # Setup the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1]})

    # Choose plot data based on plot_type argument
    if plot_type == "rv":
        rv = out.rv.value
        data_to_plot = val
        points_to_plot = rv * 1000
        color_label = "Local RV (km/s)"
        # axs[1].set_ylabel("RV [m/s]")
        cmap_s = star_gradient
    elif plot_type == "flux":
        data_to_plot = alpha
        points_to_plot = flux
        color_label = "Local normalized flux"
        # axs[1].set_ylabel('Normalized flux')
        cmap_s = star_gradient_flux
    elif plot_type == "tr":
        data_to_plot = val
        points_to_plot = sim.pixel_trans
        color_label = "Local RV (km/s)"
        # axs[1].set_ylabel('Absorption (%)')
        # axs[1].set_xlabel(r"$\lambda$+" + str(ref_wave) + "(" + r"$\AA$" + ")")
        cmap_s = star_gradient
    elif plot_type == "shadow":
        data_to_plot = val
        points_to_plot = sim.master_out_fw - sim.integrated_spectra_fw
        color_label = "Local RV (km/s)"
        # axs[1].set_ylabel('Flux')
        # axs[1].set_xlabel(r"$\lambda$+" + str(ref_wave) + "(" + r"$\AA$" + ")")
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
    norm = Normalize(
        vmin=min(planet_phases[~phase_mask]), vmax=max(planet_phases[~phase_mask])
    )
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Set up the planet's position for animation
    planet_arr = [
        disk_position(sim, xyz, x, yp, zp) for x in np.arange(len(psi))[~phase_mask]
    ]
    axs[0].set_aspect("equal")  # Set aspect ratio globally for the left plot

    def update(frame):
        cmapi = ListedColormap(cmap(norm(planet_phases[~phase_mask][frame])))
        # Update the left plot with the current planet position
        axs[0].cla()  # Clear previous frame
        axs[1].cla()  # Clear previous frame
        h = axs[0].imshow(data_to_plot, cmap=cmap_s, extent=(-1.5, 1.5, -1.5, 1.5))
        axs[0].plot(y, z, "--k")
        axs[0].set_xlabel(r"$R_{\star}$")
        axs[0].set_ylabel(r"$R_{\star}$")
        # Update the planet's position
        axs[0].imshow(
            planet_arr[frame],
            extent=(-1.5, 1.5, -1.5, 1.5),
            origin="upper",
            zorder=3,
            cmap=cmapi,
            alpha=0.9,
        )

        # Add the active regions and update their position according to the rotation phase
        # Verfy a phase shift between spot_init and spot_phase position at the peggining of the simulation
        if len(sim.active_regions) != 0:
            for j in sim.active_regions:
                spot = spot_init(
                    j.size, (j.lon).value, (j.lat).value, (sim.star.incl).value, 40
                )
                spot_phase_position = spot_phase(
                    spot, (sim.star.incl).value, out.psi[frame] - out.psi[0] / 2
                )
                polygon = Polygon(
                    np.array([spot_phase_position[:, 1], spot_phase_position[:, 2]]).T,
                    closed=True,
                    facecolor="black",
                )
                axs[0].add_patch(polygon)

        if plot_type == "tr" or plot_type == "shadow":
            if plot_type == "tr":
                axs[1].plot(
                    sim.pixel.wave - ref_wave,
                    (1 - points_to_plot[~phase_mask][frame]) * 100,
                    color=cmap(norm(planet_phases[~phase_mask][frame])),
                    alpha=0.9,
                )
                axs[1].set_ylabel("Absorption (%)")
                axs[1].set_xlabel(r"$\lambda$+" + str(ref_wave) + "(" + r"$\AA$" + ")")
            elif plot_type == "shadow":
                axs[1].plot(
                    sim.pixel.wave - ref_wave,
                    points_to_plot[~phase_mask][frame],
                    color=cmap(norm(planet_phases[~phase_mask][frame])),
                    alpha=0.9,
                )
                axs[1].set_ylabel("Flux")
                axs[1].set_xlabel(r"$\lambda$+" + str(ref_wave) + "(" + r"$\AA$" + ")")

            if lim:
                axs[1].set_xlim(lim[0] - ref_wave, lim[1] - ref_wave)
            else:
                axs[1].set_xlim(min(sim.pixel.wave), max(sim.pixel.wave))

            if plot_lims:
                axs[1].set_ylim(plot_lims[0], plot_lims[1])
            else:
                None
        else:
            axs[1].plot(psi_i, points_to_plot, "*", color="k")
            axs[1].plot(
                psi_i[~phase_mask][frame],
                points_to_plot[~phase_mask][frame],
                "*",
                color=cmap(norm(psi[~phase_mask][frame])),
            )
            axs[1].set_xlabel("Orbital Phase")

        # Set colorbar and labels for the left plot
        divider1 = make_axes_locatable(axs[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cb1 = fig.colorbar(h, cax=cax1, orientation="vertical")
        cb1.set_label(color_label, rotation=270, labelpad=16)

        # Set colorbar for the right plot
        divider2 = make_axes_locatable(axs[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cb2 = fig.colorbar(sm, cax=cax2, orientation="vertical")
        cb2.set_label("Orbital Phase", rotation=270, labelpad=16)
        plt.tight_layout()

    # Create animation
    ani = FuncAnimation(
        fig, update, frames=len(planet_arr), interval=interval, repeat=repeat
    )
    return ani
