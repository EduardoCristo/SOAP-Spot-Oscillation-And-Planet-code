from concurrent.futures import ProcessPoolExecutor
import numba
import numpy as np
from numpy import cos, pi, sin
from numba import float64, int32, prange
from scipy import signal
from .utils import c

c = c.value


@numba.njit(cache=True)
def sincos(x: float64) -> np.ndarray:
    """
    Calculate the sine and cosine of a number using Numba's math module.

    Arguments
    ---------
    x : float64
        The input value in radians.
    Returns
    -------
    np.ndarray
        An array containing the sine and cosine of the input value.
    """
    return [numba.math.sin(x), numba.math.cos(x)]


@numba.njit("f8(f8, f8)", cache=True)
def planck_law(λ, T):
    """
    Calculate the Planck function for a given wavelength and temperature.

    Arguments
    ---------
    λ : float
        Wavelength in Angstroms.
    T : float
        Temperature in Kelvin.
    Returns
    -------
    float
        The value of the Planck function at the given wavelength and temperature.
    """
    h = 6.62607015e-34  # Planck constant
    k_b = 1.380649e-23  # Boltzmann constant
    return 2 * h * c**2 * 1 / λ**5 * 1 / (np.exp((h * c) / (λ * k_b * T)) - 1)



@numba.njit(cache=True, nopython=True)
def itot_flux(u1: np.float64, u2: np.float64, grid: np.int32):
    """
    Calculate the flux in each cell of the grid and integrate over the entire
    stellar disc.

    Arguments
    ---------
    u1, u2 : float
        Quadratic limb-darkening parameters
    grid : int
        Grid size
    """
    # step of the grid. grid goes from -1 to 1, therefore 2 in total
    delta_grid = 2.0 / grid
    # total stellar intensity (without spots)
    total = 0
    y_positions = np.linspace(-1.0 + delta_grid / 2.0, 1.0 - delta_grid / 2.0, grid)
    z_positions = np.linspace(-1.0 + delta_grid / 2.0, 1.0 - delta_grid / 2.0, grid)
    # Scan of each cell on the grid
    for iy,y in enumerate(y_positions):
        for iz,z in enumerate(z_positions):
            # projected radius on the sky smaller than 1,
            # which means that we are on the stellar disc
            if (y**2 + z**2) <=1:
                _, limb = ld(y, z, u1, u2)
                total+= limb # check this

    return total

@numba.njit(cache=True, nopython=True)
def doppler_shift_wave(wave: np.ndarray, rv: float):
    """
    Doppler shift the wavelength array by a given radial velocity.
    Arguments
    ---------
    wave : np.ndarray
        Wavelength array in Angstroms.
    rv : float
        Radial velocity in km/s.
    Returns
    -------
    np.ndarray
        Doppler-shifted wavelength array.
    """
    return wave * (1.0 - rv / c)


@numba.njit(cache=True, nopython=True)
def doppler_shift(wave: np.ndarray, flux: np.ndarray, rv: float):
    """
    Doppler shift a spectrum (wave, flux) by given radial velocity and return the flux in the original wavelength array.
    Arguments
    ---------
    wave : np.ndarray
        Wavelength array in Angstroms.
    flux : np.ndarray
        Spectrum flux array.
    rv : float
        Radial velocity in km/s.
    Returns
    -------
    np.ndarray
        Doppler-shifted flux array interpolated to the original wavelength array.
    """
    new_wave = doppler_shift_wave(wave, rv)
    return linear_interpolator(wave, flux, new_wave)


@numba.njit(cache=True, nopython=True)
def ld(y: float, z: float, u1: float, u2: float):
    """
    Calculate the limb-darkening intensity at position (y, z) on the stellar
    surface, using a limb-darkening law defined by u1, u2

    Args:
        y (float): y position in the stellar grid
        z (float): z position in the stellar grid
        u1 (float): Limb-darkening linear coefficient
        u2 (float): Limb-darkening quadratic coefficient
    """
    r_cos = np.sqrt(1.0 - (y * y + z * z))
    return r_cos, 1.0 - u1 * (1.0 - r_cos) - u2 * (1.0 - r_cos) * (1.0 - r_cos)


@numba.njit(cache=True, nopython=True)
def vrot(v_eq, r_cos, y, z, alphaB, alphaC, i, cb1):
    """
    Calculate the projected rotational velocity (line-of-sight) at position (y,z)
    on the stellar surface, using linear equatorial velocity and differential rotation.

    Parameters
    ----------
    v_eq : float
        Equatorial linear velocity [m/s]
    r_cos : float
        Projected distance from rotation axis (≈ R cos(latitude))
    y : float
        Sky-projected y-position (in rotation direction)
    z : float
        Projected height along rotation axis
    alphaB : float
        Relative coefficient for sin²(latitude) term
    alphaC : float
        Relative coefficient for sin⁴(latitude) term
    i : float
        Inclination angle [radians]
    cb1 : float
        Additional constant velocity term (e.g. convective blueshift) [m/s]

    Returns
    -------
    delta : float
        Line-of-sight rotational velocity at (y,z) [m/s]
    """
    # Approximate latitude using projection geometry
    latitude = z * sin(i) + r_cos * cos(i)

    # Differential linear velocity profile
    v = v_eq * (1 - alphaB * latitude**2 - alphaC * latitude**4)

    # Projected velocity along line of sight
    delta = y * v * sin(i)

    # Convective blueshift
    delta += (cb1 / 1000.0) * r_cos  # cb1 is assumed to be in m/s

    return delta

@numba.njit(parallel=True, cache=True)
def itot_spectrum_par(
    v_eq: float64,
    i: float64,
    u1: float64,
    u2: float64,
    alphaB: float64,
    alphaC: float64,
    cb1: float64,
    grid: int32,
    spec,
) -> np.ndarray:
    """
    Calculate the spectrum in each cell of the grid and integrate over the entire
    stellar disc.
    This function uses parallel processing to speed up the calculation.
    Arguments
    ---------
    v_eq : float64
        Equatorial linear velocity in km/s.
    i : float64
        Inclination angle in degrees.
    u1 : float64
        Quadratic limb-darkening coefficient.
    u2 : float64
        Quadratic limb-darkening coefficient.
    alphaB : float64
        Relative coefficient for sin²(latitude) term.
    alphaC : float64
        Relative coefficient for sin⁴(latitude) term.
    cb1 : float64
        Additional constant velocity term (e.g. convective blueshift) in m/s.
    grid : int32
        Grid size (number of cells along one dimension).
    spec : Spectrum
        Spectrum object containing wavelength and flux data.
    Returns
    -------
    np.ndarray
        Integrated spectrum over the entire stellar disc.
    """
    i = i * np.pi / 180.0  # Convert degrees to radians
    delta_grid = 2.0 / grid
    wave_size = len(spec.wave)
    y_positions = np.linspace(-1.0 + delta_grid / 2.0, 1.0 - delta_grid / 2.0, grid)
    z_positions = np.linspace(-1.0 + delta_grid / 2.0, 1.0 - delta_grid / 2.0, grid)

    # Each thread writes to its own copy
    thread_spectra = np.zeros((grid, wave_size))

    for iy in prange(grid):
        y = y_positions[iy]
        local_spectrum = np.zeros(wave_size)
        for iz in range(grid):
            z = z_positions[iz]

            if (y**2 + z**2) <= 1.0:
                r_cos, limb = ld(y, z, u1, u2)
                delta = vrot(v_eq, r_cos, y, z, alphaB, alphaC, i, cb1)
                shifted_spectrum = doppler_shift(
                    spec.wave, spec.flux(r_cos), delta * 1e3
                )
                local_spectrum += shifted_spectrum * limb

        thread_spectra[iy, :] = local_spectrum

    # Reduce the result after the parallel loop
    total_spectrum = np.sum(thread_spectra, axis=0)
    return total_spectrum


@numba.njit(nopython=True, cache=True)
def linear_interpolator(x:np.ndarray, y: np.ndarray, new_x:np.ndarray) -> np.ndarray:
    """
    Cubic Hermite spline interpolator compatible with Numba.
    
    Parameters:
    - x (ndarray): The x-coordinates of the data points (sorted, unique).
    - y (ndarray): The y-coordinates of the data points.
    - new_x (ndarray): The new x-coordinates for which to interpolate.
    
    Returns:
    - ndarray: The interpolated values corresponding to new_x.
    """
    n = len(x)
    interpolated_values = np.empty_like(new_x)
    
    # Compute finite differences (numerical derivatives)
    slopes = np.empty(n - 1)
    for i in range(n - 1):
        slopes[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    
    # Compute derivatives (Hermite conditions)
    derivatives = np.empty(n)
    derivatives[0] = slopes[0]  # Forward difference for the first point
    derivatives[-1] = slopes[-1]  # Backward difference for the last point
    for i in range(1, n - 1):
        derivatives[i] = (slopes[i - 1] + slopes[i]) / 2.0  # Central difference
    
    i = 0  # Index for interval
    for idx, new_x_val in enumerate(new_x):
        # Clamp values outside the range
        if new_x_val <= x[0]:
            interpolated_values[idx] = y[0]
            continue
        elif new_x_val >= x[-1]:
            interpolated_values[idx] = y[-1]
            continue
        
        # Find the interval [x_i, x_{i+1}]
        while i < n - 2 and x[i + 1] < new_x_val:
            i += 1
        
        # Hermite cubic interpolation
        h = x[i + 1] - x[i]
        t = (new_x_val - x[i]) / h
        t2 = t * t
        t3 = t2 * t
        
        h00 = 2 * t3 - 3 * t2 + 1  # Basis function 1
        h10 = t3 - 2 * t2 + t      # Basis function 2
        h01 = -2 * t3 + 3 * t2     # Basis function 3
        h11 = t3 - t2              # Basis function 4
        
        interpolated_values[idx] = (
            h00 * y[i] + h10 * h * derivatives[i] +
            h01 * y[i + 1] + h11 * h * derivatives[i + 1]
        )
    
    return interpolated_values



# Calculate the CCF in each cell of the grid and integrate over the entire
# stellar disc
@numba.njit(cache=True, nopython=True)
def itot_rv(
    v_eq: float64,
    i: float64,
    u1: float64,
    u2: float64,
    alphaB: float64,
    alphaC: float64,
    cb1: float64,
    grid: int32,
    rv: np.ndarray,
    ccf: np.ndarray,
    v_interval: float64,
    n_v: int32,
):
    # conversions
    i = i * pi / 180.0  # [degree] --> [radian]

    # step of the grid. grid goes from -1 to 1, therefore 2 in total
    delta_grid = 2.0 / grid
    # total stellar intensity (without spots)
    total = 0
    f_star = np.zeros(n_v)
    rv_vrot = np.linspace(-v_interval, v_interval, n_v)

    y_positions = np.linspace(-1.0 + delta_grid / 2.0, 1.0 - delta_grid / 2.0, grid)
    z_positions = np.linspace(-1.0 + delta_grid / 2.0, 1.0 - delta_grid / 2.0, grid)
    # Scan of each cell on the grid
    for iy,y in enumerate(y_positions):
        for iz,z in enumerate(z_positions):
            # projected radius on the sky smaller than 1,
            # which means that we are on the stellar disc
            if (y**2 + z**2) <=1:
                # r_cos = np.sqrt(1.0 - (y * y + z * z))
                r_cos, limb = ld(y, z, u1, u2)
                delta = vrot(v_eq, r_cos, y, z, alphaB, alphaC, i, cb1)
                f_star += linear_interpolator(rv, ccf, rv_vrot - delta) * limb
                total += limb

    return f_star, total


@numba.njit(cache=True, nopython=True)
def spot_init(
    s: float, longitude: float, latitude: float, inclination: float, nrho: int
) -> np.ndarray:
    """
    Calculate the position of the spot, initialized at the disc center.

    Args:
        s: spot radius
        longitude [degree]
        latitude  [degree]
        inclination [degree] i=0  -> pole-on (North)
                             i=90 -> equator-on
        nrho : Spot circumference resolution

    Returns:
        xyz : real position of the spot after applying rotations
    """

    # Conversions [deg] -> [rad]
    longitude = longitude * pi / 180.0
    latitude = latitude * pi / 180.0
    inclination = inclination * pi / 180.0

    # In this initial disc center position, we calculate the coordinates
    # (x,y,z) of points of the active region's circumference

    # A circular active region has a resolution given by nrho, which implies
    # that we will have a point on the disc circumference every 2*pi/(nrho-1)
    # -1 because there is (nrho-1) intervals
    rho_step = 2.0 * pi / (nrho - 1)

    rho = np.arange(-pi, pi + rho_step, rho_step)
    # rho = np.linspace(-pi, pi, nrho, endpoint=False) # check

    # x = sqrt(r^2-s^2), where r is the radius. r=1 therefore r^2=1
    # The active region is on the surface, so very close to x=1. However with
    # the curvature of the sphere, the circumference of the active region is at
    # x = sqrt(r^2-s^2)
    xyz2 = np.empty((3, nrho))
    xyz2[0] = np.sqrt(1 - s**2)
    xyz2[1] = s * np.cos(rho)  # projection of the circumference on the y axis
    xyz2[2] = s * np.sin(rho)  # projection of the circumference on the z axis

    # to account for the real projection of the spot, we rotate the star and
    # look how the coordinates of the circumference of the spot change position
    # according to latitude, longitude and inclination.
    # It consists of three rotations
    #
    # Conventions :
    # - for inclination = 0, the star rotates around z axis
    # - the line of sight is along x-axis
    # - sky plane is the yz-plane
    #
    # Let Rx(α), Ry(β), Rz(γ) be the rotation matrices around the x, y, and z
    # axis with angles α, β, γ
    # (counter-clockwise direction when looking toward the origin)
    #
    # The rotations to apply are:
    #   Ry(inclination) x Rz(longitude) x Ry(latitude)
    #
    #         |  cos(β)  0  sin(β) |                    | cos(γ) -sin(γ) 0 |
    # Ry(β) = |    0     1    0    |            Rz(γ) = | sin(γ)  cos(γ) 0 |
    #         | -sin(β)  0  cos(β) |                    |   0       0    1 |
    #
    #
    # |x'|   |  cos(b2)cos(γ)cos(b)-sin(b)sin(b2)  -sin(γ)cos(b2)  cos(b2)cos(γ)sin(b)+sin(b2)cos(b) |   |x|
    # |y'| = |              sin(γ)cos(b)               cos(γ)                   sin(γ)sin(b)         | x |y|
    # |z'|   |  -sin(b2)cos(γ)cos(b)-cos(b2)sin(b)  sin(b2)sin(γ) -sin(b2)cos(γ)sin(b)+cos(b2)cos(b) |   |z|

    b = -latitude
    g = longitude
    b2 = pi / 2.0 - inclination

    R = np.array(
        [
            [
                cos(b2) * cos(g) * cos(b) - sin(b) * sin(b2),
                -sin(g) * cos(b2),
                cos(b2) * cos(g) * sin(b) + sin(b2) * cos(b),
            ],
            [sin(g) * cos(b), cos(g), sin(g) * sin(b)],
            [
                -sin(b2) * cos(g) * cos(b) - cos(b2) * sin(b),
                sin(b2) * sin(g),
                -sin(b2) * cos(g) * sin(b) + cos(b2) * cos(b),
            ],
        ]
    )

    # xyz is the real position of the active region after rotating the star to
    # have the initial equatorial active region at the correct longitude and
    # latitude, also taking into account the stellar inclination
    xyz = np.dot(R, xyz2)
    return xyz.T


@numba.njit(cache=True, nopython=True)
def spot_phase(xyz, inclination, phase):
    """
    Calculate the xyz position of the active region circumference at phase φ
    """
    # the phase is between 0 and 1, so φ is in radian between -2pi and 0
    φ = -phase * (2 * pi)
    i = inclination * pi / 180  # [deg] --> [rad]

    # projection of the rotation axis on the xyz coordinate
    axe = [cos(i), 0, sin(i)]

    # The rotation around the axis (axe[0], axe[1], axe[2]) by an angle φ is
    # given by the following matrix
    cos_φ = cos(φ)
    sin_φ = sin(φ)
    R = np.array(
        [
            [
                (1 - cos_φ) * axe[0] * axe[0] + cos_φ,
                (1 - cos_φ) * axe[0] * axe[1] + sin_φ * axe[2],
                (1 - cos_φ) * axe[0] * axe[2] - sin_φ * axe[1],
            ],
            [
                (1 - cos_φ) * axe[1] * axe[0] - sin_φ * axe[2],
                (1 - cos_φ) * axe[1] * axe[1] + cos_φ,
                (1 - cos_φ) * axe[1] * axe[2] + sin_φ * axe[0],
            ],
            [
                (1 - cos_φ) * axe[2] * axe[0] + sin_φ * axe[1],
                (1 - cos_φ) * axe[2] * axe[1] - sin_φ * axe[0],
                (1 - cos_φ) * axe[2] * axe[2] + cos_φ,
            ],
        ]
    )

    # Note: there are some sign differences with respect to e.g. Wikipedia
    # because in this case ψ is defined negative (c.f. start of function). In
    # this case, cos(-φ) = cos(φ) --> no sign change,
    # but sin(-φ)=-sin(φ) --> sign change

    xyz2 = np.dot(xyz, R.T)
    return xyz2


@numba.njit(cache=True, nopython=True)
def spot_area(xyz, nrho, grid):
    """
    Determine a smaller yz-area of the stellar disk where the active region is
    The different cases are:
    - the active region is completely visible (all x of the circumference >=0)
    - the active region is completely invisible (all x of the circumference < 0)
    - the active region is on the disk edge and partially visible only
    """
    grid_step = 2.0 / grid  # the stellar disc goes from -1 to 1, therefore 2
    # initialize to 'opposite'-extreme values
    miny = 1
    minz = 1
    maxy = -1
    maxz = -1
    # to count how many points of the circumference are visible and invisible
    counton = 0
    countoff = 0
    # scan each point the circumference
    
    for j in range(nrho):
        x = xyz[j, 0]
        y = xyz[j, 1]
        z = xyz[j, 2]

        if x >= 0.0:
            counton += 1

            if y < miny:
                miny = y
            if y > maxy:
                maxy = y

            if z < minz:
                minz = z
            if z > maxz:
                maxz = z
        else:
            countoff = 1

    # if there are both visible and invisible points
    if counton > 0 and countoff > 0:
        #  --> active region is on the edge
        # in this situation there are cases where the yz-area define above is
        # actually smaller than the real area of the active region on the
        # stellar disk.
        # The minima/maxima are over/under-estimated if the active region is on
        # one of the axis (y or z). Because if on the y axis, the minimum
        # (or maximum) won't be on the circumference of the active region.
        # Same for z axis

        # active region on the z-axis because one pois on the positive side
        # of z, and the other on the negative side of z
        if miny * maxy < 0:
            if minz < 0:
                minz = -1  # active region on the bottom-z axis (z<0)
            else:
                maxz = 1  # active region on the top-z axis (z>=0)

        # active region on the y-axis because one pois on the positive side
        # of y, and the other on the negative side of z
        if minz * maxz < 0:
            if miny < 0:
                miny = -1  # active region on the left hand-y axis (y<0)
            else:
                maxy = 1  # active region on the right hand-y axis (y>=0)

    if counton == 0:
        visible = False
    else:
        visible = True

    # find the indices of miny, minz,... on the grid
    iminy = int(np.floor((1 + miny) / grid_step))
    iminz = int(np.floor((1 + minz) / grid_step))

    imaxy = int(np.ceil((1 + maxy) / grid_step))
    imaxz = int(np.ceil((1 + maxz) / grid_step))

    return visible, iminy, iminz, imaxy, imaxz


@numba.njit(cache=True, nopython=True)
def spot_inverse_rotation(
    xyz: np.ndarray, longitude: float, latitude: float, inclination: float, phase: float
):
    """
    Relocate a point (x,y,z) to the 'initial' configuration, i.e. when the
    active region is on the disc center. This consists in rotating the point,
    according to latitude, longitude, inclination and phase, but in the reverse
    order.

    Conventions:
    - when inclination=0 the star rotates around z axis
    - line of sight is along x-axis
    - sky plane = yz-plane
    """

    g2 = (-1.0 + phase) * (2.0 * pi)  # inverse phase ([0-1] -> [rad])
    i = inclination * pi / 180.0
    b = latitude * pi / 180.0
    g = -longitude * pi / 180.0
    b2 = -(pi / 2.0 - i)

    R = [
        [
            (1 - cos(g2)) * cos(i) * cos(i) + cos(g2),
            sin(g2) * sin(i),
            (1 - cos(g2)) * cos(i) * sin(i),
        ],
        [-sin(g2) * sin(i), cos(g2), sin(g2) * cos(i)],
        [
            (1 - cos(g2)) * sin(i) * cos(i),
            -sin(g2) * cos(i),
            (1 - cos(g2)) * sin(i) * sin(i) + cos(g2),
        ],
    ]

    R2 = [
        [
            cos(b) * cos(g) * cos(b2) - sin(b2) * sin(b),
            -sin(g) * cos(b),
            cos(b) * cos(g) * sin(b2) + sin(b) * cos(b2),
        ],
        [sin(g) * cos(b2), cos(g), sin(g) * sin(b2)],
        [
            -sin(b) * cos(g) * cos(b2) - cos(b) * sin(b2),
            sin(b) * sin(g),
            -sin(b) * cos(g) * sin(b2) + cos(b) * cos(b2),
        ],
    ]

    R3 = [
        [
            R2[0][0] * R[0][0] + R2[0][1] * R[1][0] + R2[0][2] * R[2][0],
            R2[0][0] * R[0][1] + R2[0][1] * R[1][1] + R2[0][2] * R[2][1],
            R2[0][0] * R[0][2] + R2[0][1] * R[1][2] + R2[0][2] * R[2][2],
        ],
        [
            R2[1][0] * R[0][0] + R2[1][1] * R[1][0] + R2[1][2] * R[2][0],
            R2[1][0] * R[0][1] + R2[1][1] * R[1][1] + R2[1][2] * R[2][1],
            R2[1][0] * R[0][2] + R2[1][1] * R[1][2] + R2[1][2] * R[2][2],
        ],
        [
            R2[2][0] * R[0][0] + R2[2][1] * R[1][0] + R2[2][2] * R[2][0],
            R2[2][0] * R[0][1] + R2[2][1] * R[1][1] + R2[2][2] * R[2][1],
            R2[2][0] * R[0][2] + R2[2][1] * R[1][2] + R2[2][2] * R[2][2],
        ],
    ]

    xyz_out = np.zeros(3)
    xyz_out[0] = R3[0][0] * xyz[0] + R3[0][1] * xyz[1] + R3[0][2] * xyz[2]
    xyz_out[1] = R3[1][0] * xyz[0] + R3[1][1] * xyz[1] + R3[1][2] * xyz[2]
    xyz_out[2] = R3[2][0] * xyz[0] + R3[2][1] * xyz[1] + R3[2][2] * xyz[2]
    return xyz_out


@numba.njit(cache=True, nopython=True)
def spot_inverse_rotation1(
    xyz: np.ndarray,
    longitude: float,
    latitude: float,
    inclination: float,
    phase: float,
    xyz_out: np.ndarray,
):
    """
    Relocate a point (x,y,z) to the 'initial' configuration, i.e. when the
    active region is on the disc center. This consists in rotating the point,
    according to latitude, longitude, inclination and phase, but in the reverse
    order.

    Conventions:
    - when inclination=0 the star rotates around z axis
    - line of sight is along x-axis
    - sky plane = yz-plane
    """

    g2 = (-1.0 + phase) * (2.0 * pi)  # inverse phase ([0-1] -> [rad])
    deg_to_rad = pi / 180.0
    i = inclination * deg_to_rad
    b = latitude * deg_to_rad
    g = -longitude * deg_to_rad
    b2 = -(pi / 2.0 - i)

    sing2, cosg2 = sincos(g2)
    sini, cosi = sincos(i)
    onemcosg2 = 1.0 - cosg2

    R_00 = onemcosg2 * cosi * cosi + cosg2
    R_01 = sing2 * sini
    R_02 = onemcosg2 * cosi * sini
    R_10 = -sing2 * sini
    R_11 = cosg2
    R_12 = sing2 * cosi
    R_20 = onemcosg2 * sini * cosi
    R_21 = -sing2 * cosi
    R_22 = onemcosg2 * sini * sini + cosg2

    sinb, cosb = sincos(b)
    sinb2, cosb2 = sincos(b2)
    sing, cosg = sincos(g)

    R2_00 = cosb * cosg * cosb2 - sinb2 * sinb
    R2_01 = -sing * cosb
    R2_02 = cosb * cosg * sinb2 + sinb * cosb2
    R2_10 = sing * cosb2
    R2_11 = cosg
    R2_12 = sing * sinb2
    R2_20 = -sinb * cosg * cosb2 - cosb * sinb2
    R2_21 = sinb * sing
    R2_22 = -sinb * cosg * sinb2 + cosb * cosb2

    R3_00 = R2_00 * R_00 + R2_01 * R_10 + R2_02 * R_20
    R3_01 = R2_00 * R_01 + R2_01 * R_11 + R2_02 * R_21
    R3_02 = R2_00 * R_02 + R2_01 * R_12 + R2_02 * R_22
    R3_10 = R2_10 * R_00 + R2_11 * R_10 + R2_12 * R_20
    R3_11 = R2_10 * R_01 + R2_11 * R_11 + R2_12 * R_21
    R3_12 = R2_10 * R_02 + R2_11 * R_12 + R2_12 * R_22
    R3_20 = R2_20 * R_00 + R2_21 * R_10 + R2_22 * R_20
    R3_21 = R2_20 * R_01 + R2_21 * R_11 + R2_22 * R_21
    R3_22 = R2_20 * R_02 + R2_21 * R_12 + R2_22 * R_22

    xyz_out[0] = R3_00 * xyz[0] + R3_01 * xyz[1] + R3_02 * xyz[2]
    xyz_out[1] = R3_10 * xyz[0] + R3_11 * xyz[1] + R3_12 * xyz[2]
    xyz_out[2] = R3_20 * xyz[0] + R3_21 * xyz[1] + R3_22 * xyz[2]
    # return xyz_out


@numba.njit(cache=True, nopython=True)
def temperature_spot(Tstar, Tdiff):
    return Tstar - Tdiff


@numba.njit(cache=True, nopython=True)
def temperature_plage(Tstar, Tdiff, r_cos):
    return Tstar + Tdiff - 407.7 * r_cos + 190.9 * r_cos**2


@numba.njit(cache=True, nopython=True)
def spot_scan_flux(
    i,
    u1,
    u2,
    grid,
    s,
    lon,
    lat,
    phase,
    iminy,
    iminz,
    imaxy,
    imaxz,
    magn_feature_type,
    Tstar,
    Tdiff,
    wlll,
    ar_grid,
    prot,
    Rp,
    P,
    t0,
    e,
    w,
    ip,
    a,
    lbda,
    fe,
    fi,
    theta,
    ir,
):
    """
    Scan of the yz-area where the spot is.
    For each grid point (y,z) we need to check whether it belongs to the spot
    or not. Sadly, we do not know the projected geometry of the spot in its
    actual position. Thus, we have to do an inverse rotation to replace the
    grid powhere it would be in the initial configuration. Indeed, in the
    initial configuration, the spot has a well known geometry of a circle
    centered on the x-axis.
    """

    xyzp = planet_position_at_date(phase * prot, P, t0, e, w, ip, a, lbda)
    wlll = wlll * 1e-10
    delta_grid = 2.0 / grid

    planck_star = planck_law(wlll, Tstar)
    # print(planck_star)
    # xyza # actual coordinates
    # xyzi # coordinates transformed back to the initial configuration
    xyza = np.empty(3)
    xyzi = np.empty(3)

    sum_spot = 0.0

    # Constant quantities
    costheta = cos(theta)
    sintheta = sin(theta)
    cosir2 = pow(cos(ir), 2)
    Rp2 = pow(Rp, 2)
    fe2_Rp2 = pow(fe, 2) * Rp2
    fi2_Rp2 = pow(fi, 2) * Rp2

    # Scan of each cell on the grid
    for iy in prange(iminy, imaxy):
        y = -1.0 + iy * delta_grid+delta_grid/2  # y between -1 et 1
        xyza[1] = y

        for iz in prange(iminz, imaxz):
            if ar_grid[iy, iz]:
                continue

            z = -1.0 + iz * delta_grid+delta_grid/2 # z between -1 et 1

            if z * z + y * y < 1.0:
                # sqrt(r^2 - (y^2+z^2)) where r=1.
                # This is equal to 1 at the disc center, and 0 on the limb.
                # Often referred in the literature as cos(theta)
                r_cos, limb = ld(y, z, u1, u2)

                xyza[2] = z
                xyza[0] = r_cos

                dp1 = xyza[1] - xyzp[1]
                dp2 = xyza[2] - xyzp[2]

                costheta_dp1 = costheta * dp1
                costheta_dp2 = costheta * dp2
                sintheta_dp1 = sintheta * dp1
                sintheta_dp2 = sintheta * dp2

                # Check if the planet is in-front off the active region
                a = dp1**2.0 + dp2**2.0
                if Rp != 0:
                    aringin = (costheta_dp1 + sintheta_dp2) ** 2 / fi2_Rp2 + (
                        (sintheta_dp1 - costheta_dp2) ** 2 / (cosir2 * fi2_Rp2)
                    )

                    aringout = (costheta_dp1 + sintheta_dp2) ** 2 / fe2_Rp2 + (
                        (sintheta_dp1 - costheta_dp2) ** 2 / (cosir2 * fe2_Rp2)
                    )
                else:
                    aringin = 0
                    aringout = 1.1

                if (
                    ((aringin < 1) and (a >= Rp2))
                    or ((aringout > 1) and (a >= Rp2))
                    or (xyzp[0] < 0.0)
                ):

                    # Rotate the star so that the spot is on the disc center
                    xyzi = spot_inverse_rotation(xyza, lon, lat, i, phase)

                    # if inside the active region when scanning the grid
                    if xyzi[0] ** 2 >= 1.0 - s**2:
                        ar_grid[iy, iz] = True

                        if magn_feature_type == 0:
                            # intensity of the spot
                            T_spot = temperature_spot(Tstar, Tdiff)
                            intensity = planck_law(wlll, T_spot) / planck_star

                        else:
                            # plages are brighter on the limb (e.g. Meunier 2010)
                            T_plage = temperature_plage(Tstar, Tdiff, r_cos)
                            intensity = planck_law(wlll, T_plage) / planck_star
                            # print(planck_law(wlll, T_plage))

                        # calculates the "non contributing" total flux of the active
                        # region taking into account the limb-darkening and the
                        # active region intensity
                        sum_spot += limb * (1.0 - intensity)

    return sum_spot


@numba.njit(cache=True, nopython=True)
def spot_scan_rv(
    v_eq,
    i,
    u1,
    u2,
    alphaB,
    alphaC,
    cb1,
    grid,
    rv,
    ccf,
    ccf_spot,
    v_interval,
    n_v,
    s,
    lon,
    lat,
    phase,
    iminy,
    iminz,
    imaxy,
    imaxz,
    magn_feature_type,
    Tstar,
    Tdiff,
    wlll,
    ar_grid,
    prot,
    Rp,
    P,
    t0,
    e,
    w,
    ip,
    a,
    lbda,
    fe,
    fi,
    theta,
    ir,
):
    """
    Scan of the yz-area where the spot is.
    For each grid point (y,z) we need to check whether it belongs to the spot
    or not. Sadly, we do not know the projected geometry of the spot in its
    actual position. Thus, we have to do an inverse rotation to replace the
    grid powhere it would be in the initial configuration. Indeed, in the
    initial configuration, the spot has a well known geometry of a circle
    centered on the x-axis.
    """
    i_rad = i * pi / 180.0  # [degree] --> [radian]

    # step of the grid. grid goes from -1 to 1, therefore 2 in total
    delta_grid = 2.0 / grid
    # v_interval is from the velocity 0 to the edge of the spectrum taking into
    # account minimal or maximal rotation (width - v to 0 or 0 to width + v).
    # n_v is the number of points for all the CCF from minimum rotation to
    # maximum one (from width - v to width + v). n_v represent therefore the
    # double than v_interval, we therefore have to multiply v_interval by 2.

    # step in speed of the CCF. There is (n_v-1) intervals
    delta_v = 2.0 * v_interval / (n_v - 1)

    n = rv.size

    wlll = wlll * 1e-10
    planck_star = planck_law(wlll, Tstar)

    # xyza # actual coordinates
    # xyzi # coordinates transformed back to the initial configuration
    xyza = np.empty(3)
    xyzi = np.empty(3)

    f_spot_bconv = np.zeros(n_v)
    f_spot_flux = np.zeros(n_v)
    f_spot_tot = np.zeros(n_v)

    xyzp = planet_position_at_date(phase * prot, P, t0, e, w, ip, a, lbda)

    # Constant quantities
    costheta = cos(theta)
    sintheta = sin(theta)
    cosir2 = pow(cos(ir), 2)
    Rp2 = pow(Rp, 2)
    fe2_Rp2 = pow(fe, 2) * Rp2
    fi2_Rp2 = pow(fi, 2) * Rp2
    rv_vrot = np.linspace(-v_interval, v_interval, n_v)

    # Scan of each cell on the grid
    for iy in range(iminy, imaxy):
        for iz in range(iminz, imaxz):

            if ar_grid[iy, iz]:
                continue

            y = -1.0 + iy * delta_grid+ delta_grid/2  # y between -1 et 1
            xyza[1] = y

            z = -1.0 + iz * delta_grid+ delta_grid/2  # z between -1 et 1
            xyza[2] = z

            # projected radius on the sky smaller than 1: we are on the stellar
            # disc
            if z * z + y * y < 1.0:
                r_cos, limb = ld(y, z, u1, u2)
                xyza[0] = r_cos

                dp1 = xyza[1] - xyzp[1]
                dp2 = xyza[2] - xyzp[2]

                costheta_dp1 = costheta * dp1
                costheta_dp2 = costheta * dp2
                sintheta_dp1 = sintheta * dp1
                sintheta_dp2 = sintheta * dp2

                # Check if the planet is in-front off the active region
                a = dp1**2.0 + dp2**2.0

                if Rp != 0:
                    aringin = (costheta_dp1 + sintheta_dp2) ** 2 / fi2_Rp2 + (
                        (sintheta_dp1 - costheta_dp2) ** 2 / (cosir2 * fi2_Rp2)
                    )

                    aringout = (costheta_dp1 + sintheta_dp2) ** 2 / fe2_Rp2 + (
                        (sintheta_dp1 - costheta_dp2) ** 2 / (cosir2 * fe2_Rp2)
                    )
                else:
                    aringin = 0
                    aringout = 1.1

                if (
                    ((aringin < 1) and (a >= Rp2))
                    or ((aringout > 1) and (a >= Rp2))
                    or (xyzp[0] < 0.0)
                ):

                    # xyza --> xyzi:
                    # Rotate the star so that the spot is on the disc center
                    xyzi = spot_inverse_rotation(xyza, lon, lat, i, phase)

                    # if inside the active region when scanning the grid
                    if xyzi[0] ** 2 >= 1.0 - s**2:
                        ar_grid[iy, iz] = True

                        delta_quiet = vrot(v_eq, r_cos, y, z, alphaB, alphaC, i_rad, cb1)
                        # We have inhibition of convective blueshift on the spot, as such cb1=0
                        delta_spot = vrot(v_eq, r_cos, y, z, alphaB, alphaC, i_rad, 0)
                        if magn_feature_type == 0:
                            # intensity of the spot
                            T_spot = temperature_spot(Tstar, Tdiff)
                            intensity = planck_law(wlll, T_spot) / planck_star
                        else:
                            # plages are brighter on the limb (e.g. Meunier 2010)
                            T_plage = temperature_plage(Tstar, Tdiff, r_cos)
                            intensity = planck_law(wlll, T_plage) / planck_star

                        shifted_quiet = linear_interpolator(
                            rv, ccf, rv_vrot - delta_quiet
                        )
                        shifted_spot = linear_interpolator(
                            rv, ccf_spot, rv_vrot - delta_spot
                        )

                        f_spot_flux += shifted_quiet * limb * (1.0 - intensity)
                        f_spot_bconv += (shifted_quiet - shifted_spot) * limb
                        f_spot_tot += (shifted_quiet - intensity * shifted_spot) * limb

    return f_spot_bconv, f_spot_flux, f_spot_tot


@numba.njit(cache=True, nopython=True)
def spot_scan_spectrum(
    v_eq,
    i,
    u1,
    u2,
    alphaB,
    alphaC,
    cb1,
    grid,
    pixel,
    pixel_spot,
    s,
    lon,
    lat,
    phase,
    iminy,
    iminz,
    imaxy,
    imaxz,
    magn_feature_type,
    Tstar,
    Tdiff,
    wlll,
    ar_grid,
    prot,
    Rp,
    P,
    t0,
    e,
    w,
    ip,
    a,
    lbda,
    fe,
    fi,
    theta,
    ir,
):
    """
    Scan of the yz-area where the spot is.
    For each grid point (y,z) we need to check whether it belongs to the spot
    or not. Sadly, we do not know the projected geometry of the spot in its
    actual position. Thus, we have to do an inverse rotation to replace the
    grid powhere it would be in the initial configuration. Indeed, in the
    initial configuration, the spot has a well known geometry of a circle
    centered on the x-axis.
    """

    xyzp = planet_position_at_date(phase * prot, P, t0, e, w, ip, a, lbda)

    i_rad = i * pi / 180.0  # [degree] --> [radian]

    # step of the grid. grid goes from -1 to 1, therefore 2 in total
    delta_grid = 2.0 / grid
    # v_interval is from the velocity 0 to the edge of the spectrum taking into
    # account minimal or maximal rotation (width - v to 0 or 0 to width + v).
    # n_v is the number of points for all the CCF from minimum rotation to
    # maximum one (from width - v to width + v). n_v represent therefore the
    # double than v_interval, we therefore have to multiply v_interval by 2.

    wlll = wlll * 1e-10
    planck_star = planck_law(wlll, Tstar)

    # xyza # actual coordinates
    # xyzi # coordinates transformed back to the initial configuration
    xyza = np.empty(3)
    xyzi = np.empty(3)

    f_spot_bconv = np.zeros_like(pixel.wave, dtype=np.float64)
    f_spot_flux = np.zeros_like(pixel.wave, dtype=np.float64)
    f_spot_tot = np.zeros_like(pixel.wave, dtype=np.float64)

    # Constant quantities
    costheta = cos(theta)
    sintheta = sin(theta)
    cosir2 = pow(cos(ir), 2)
    Rp2 = pow(Rp, 2)
    fe2_Rp2 = pow(fe, 2) * Rp2
    fi2_Rp2 = pow(fi, 2) * Rp2

    # Scan of each cell on the grid
    for iy in range(iminy, imaxy):
        for iz in range(iminz, imaxz):

            if ar_grid[iy, iz]:
                continue

            y = -1.0 + iy * delta_grid+delta_grid/2  # y between -1 et 1
            z = -1.0 + iz *delta_grid+ delta_grid/2  # z between -1 et 1

            # projected radius on the sky smaller than 1: we are on the stellar
            # disc
            if z * z + y * y < 1.0:
                r_cos, limb = ld(y, z, u1, u2)

                dp1 = y - xyzp[1]
                dp2 = z - xyzp[2]

                costheta_dp1 = costheta * dp1
                costheta_dp2 = costheta * dp2
                sintheta_dp1 = sintheta * dp1
                sintheta_dp2 = sintheta * dp2

                # Check if the planet is in-front off the active region
                a = dp1**2.0 + dp2**2.0

                if Rp != 0:
                    aringin = (costheta_dp1 + sintheta_dp2) ** 2 / fi2_Rp2 + (
                        (sintheta_dp1 - costheta_dp2) ** 2 / (cosir2 * fi2_Rp2)
                    )

                    aringout = (costheta_dp1 + sintheta_dp2) ** 2 / fe2_Rp2 + (
                        (sintheta_dp1 - costheta_dp2) ** 2 / (cosir2 * fe2_Rp2)
                    )
                else:
                    aringin = 0
                    aringout = 1.1

                if (
                    ((aringin < 1) and (a >= Rp2))
                    or ((aringout > 1) and (a >= Rp2))
                    or (xyzp[0] < 0.0)
                ):
                    xyza = np.array([r_cos, y, z])
                    # xyza --> xyzi:
                    # Rotate the star so that the spot is on the disc center
                    xyzi = spot_inverse_rotation(xyza, lon, lat, i, phase)

                    # if inside the active region when scanning the grid
                    if xyzi[0] ** 2 >= 1.0 - s**2:
                        ar_grid[iy, iz] = True

                        delta_quiet = vrot(v_eq, r_cos, y, z, alphaB, alphaC, i_rad, cb1)
                        # We have inhibition of convective blueshift on the spot, as such cb1=0
                        delta_spot = vrot(v_eq, r_cos, y, z, alphaB, alphaC, i_rad, 0)
                        if magn_feature_type == 0:
                            # intensity of the spot
                            T_spot = temperature_spot(Tstar, Tdiff)
                            intensity = planck_law(wlll, T_spot) / planck_star
                        else:
                            # plages are brighter on the limb (e.g. Meunier 2010)
                            T_plage = temperature_plage(Tstar, Tdiff, r_cos)
                            intensity = planck_law(wlll, T_plage) / planck_star

                        shifted_quiet = doppler_shift(
                            pixel.wave, pixel.flux(r_cos), delta_quiet * 1e3
                        )
                        shifted_spot = doppler_shift(
                            pixel.wave, pixel_spot.flux(r_cos), delta_spot * 1e3
                        )

                        f_spot_flux += shifted_quiet * limb * (1.0 - intensity)
                        f_spot_bconv += (shifted_quiet - shifted_spot) * limb
                        f_spot_tot += (shifted_quiet - intensity * shifted_spot) * limb

    return f_spot_bconv, f_spot_flux, f_spot_tot


# @numba.jit()
def active_region_contributions(
    psi,
    star,
    active_regions,
    pixel,
    pixel_spot,
    grid,
    nrho,
    wlll,
    planet,
    ring,
    ccf=True,
    skip_rv=False,
):
    npsi: int = len(psi)
    flux_ar = np.zeros(npsi)

    if ccf:
        pixel_bconv = np.zeros((npsi, pixel.n_v))
        pixel_flux = np.zeros((npsi, pixel.n_v))
        pixel_tot = np.zeros((npsi, pixel.n_v))
        pix_x, pix_y = pixel.rv, pixel.intensity
        pix_y_spot = pixel_spot.intensity
    else:
        pixel_bconv = np.zeros((npsi, pixel.n))
        pixel_flux = np.zeros((npsi, pixel.n))
        pixel_tot = np.zeros((npsi, pixel.n))
        # pix_x, pix_y = pixel.wave, pixel.flux
        # pix_y_spot = pixel_spot.flux
    # import matplotlib.pyplot as plt
    # plt.plot(pix_x,pix_y)
    # plt.plot(pix_x,pix_y_spot)
    # plt.show()

    # from tqdm import tqdm
    for j in range(npsi):
        # pixels in the grid where there are active regions at this phase, to
        # deal with overlaping active regions
        ar_grid_flux = np.zeros((grid, grid), dtype=bool)
        ar_grid_rv = np.zeros((grid, grid), dtype=bool)

        # Calculate the contribution from each active region
        for ar in active_regions:
            if not ar.check:
                continue

            # if isinstance(active_region.size, float):
            #     sizes = active_region.size * np.ones_like(psi)
            # elif callable(active_region.size):
            #     sizes = active_region.size(psi * star.prot)
            # else:
            #     sizes = np.atleast_1d(active_region.size)

            xyz = spot_init(ar.size, ar.lon, ar.lat, star.incl, nrho)
            xyz2 = spot_phase(xyz, star.incl, psi[j])
            vis, iminy, iminz, imaxy, imaxz = spot_area(xyz2, nrho, grid)
            if vis:
                S = spot_scan_flux(
                    star.incl,
                    star.u1,
                    star.u2,
                    grid,
                    ar.size,
                    ar.lon,
                    ar.lat,
                    psi[j],
                    iminy,
                    iminz,
                    imaxy,
                    imaxz,
                    ar.active_region_type,
                    star.teff,
                    ar.temp_diff,
                    wlll,
                    ar_grid_flux,
                    star.prot,
                    planet.Rp,
                    planet.P,
                    planet.t0,
                    planet.e,
                    planet.w,
                    planet.ip,
                    planet.a,
                    planet.lbda,
                    ring.fe,
                    ring.fi,
                    ring.theta,
                    ring.ir,
                )
                flux_ar[j] += S

                # no need to calculate RVs
                if skip_rv:
                    continue

                if ccf:
                    bconv, flux, tot = spot_scan_rv(
                        star.vrot,
                        star.incl,
                        star.u1,
                        star.u2,
                        star.diffrotB,
                        star.diffrotC,
                        star.cb1,
                        grid,
                        pix_x,
                        pix_y,
                        pix_y_spot,
                        pixel.v_interval,
                        pixel.n_v,
                        ar.size,
                        ar.lon,
                        ar.lat,
                        psi[j],
                        iminy,
                        iminz,
                        imaxy,
                        imaxz,
                        ar.active_region_type,
                        star.teff,
                        ar.temp_diff,
                        wlll,
                        ar_grid_rv,
                        star.prot,
                        planet.Rp,
                        planet.P,
                        planet.t0,
                        planet.e,
                        planet.w,
                        planet.ip,
                        planet.a,
                        planet.lbda,
                        ring.fe,
                        ring.fi,
                        ring.theta,
                        ring.ir,
                    )
                else:
                    bconv, flux, tot = spot_scan_spectrum(
                        star.vrot,
                        star.incl,
                        star.u1,
                        star.u2,
                        star.diffrotB,
                        star.diffrotC,
                        star.cb1,
                        grid,
                        pixel,
                        pixel_spot,
                        ar.size,
                        ar.lon,
                        ar.lat,
                        psi[j],
                        iminy,
                        iminz,
                        imaxy,
                        imaxz,
                        ar.active_region_type,
                        star.teff,
                        ar.temp_diff,
                        wlll,
                        ar_grid_rv,
                        star.prot,
                        planet.Rp,
                        planet.P,
                        planet.t0,
                        planet.e,
                        planet.w,
                        planet.ip,
                        planet.a,
                        planet.lbda,
                        ring.fe,
                        ring.fi,
                        ring.theta,
                        ring.ir,
                    )

                pixel_bconv[j] += bconv
                pixel_flux[j] += flux
                pixel_tot[j] += tot
    # plt.close()
    # plt.plot(pixel_tot)
    # plt.show()
    if skip_rv:
        return flux_ar, 0.0, 0.0, 0.0
    else:
        return flux_ar, pixel_bconv, pixel_flux, pixel_tot


def clip_ccfs(ccf, _flux, _bconv, _tot, _quiet):
    # calculate where the extrapolated CCF corresponds to the boundaries of
    # the non-extrapolated CCF.
    istart = (ccf.n_v - len(ccf.rv)) // 2
    # ccf.n_v is odd given our definition, like len(ccf.rv), therefore the
    # difference can be divided by 2. The CCF has been extrapolated the same
    # way on each boundary, so dividing the difference between the
    # extrapolated and non extrapolated CCF by 2 gives by how many points
    # the CCF was extrapolated on each side
    iend = istart + len(ccf.rv)

    # truncate the extrapolated rotating CCF to the same interval as the
    # non-rotating CCF
    flux = _flux[:, istart:iend]
    bconv = _bconv[:, istart:iend]
    tot = _tot[:, istart:iend]
    quiet = _quiet[istart:iend]
    return flux, bconv, tot, quiet


def convolve_ccfs(ccf, resolution, ccf_quiet, _flux, _bconv, _tot):
    flux = np.empty_like(_flux)
    bconv = np.empty_like(_bconv)
    tot = np.empty_like(_tot)
    # convolution with a Gaussian instrumental profile of a given
    # resolution, given by the "inst_reso" attribute
    # this is only done if the CCF is not already convolved
    if resolution != 0 and not ccf.convolved:
        # t1 = time.time()

        # resolution R = lambda / Delta(lambda)
        # R = c/Delta_v -> Delta_v = c/R
        resolution = resolution
        # instrument profile
        ip_FWHM = c / resolution / 1000.0  # (to km/s)
        ip_sigma = ip_FWHM / (2 * np.sqrt(2 * np.log(2)))
        Gaussian_low_reso = np.exp(-ccf.rv**2 / (2 * (ip_sigma) ** 2))

        ccf_quiet_tmp = signal.convolve(
            -ccf_quiet + 1, Gaussian_low_reso, "same", method="direct"
        )
        # normalization
        ccf_quiet = 1 - ccf_quiet_tmp * (1 - min(ccf_quiet)) / max(ccf_quiet_tmp)

        # convolution of the CCF with the Gaussian instrumental profile to
        # reduce the resolution of the CCF
        args = dict(in2=Gaussian_low_reso, mode="same", method="direct")
        for i in range(len(_flux)):
            _flux_tmp = signal.convolve(-_flux[i] + 1, **args)
            flux[i] = 1 - _flux_tmp * (1 - min(_flux[i])) / max(_flux_tmp)

            _bconv_tmp = signal.convolve(-_bconv[i] + 1, **args)
            bconv[i] = 1 - _bconv_tmp * (1 - min(_bconv[i])) / max(_bconv_tmp)

            _tot_tmp = signal.convolve(-_tot[i] + 1, **args)
            tot[i] = 1 - _tot_tmp * (1 - min(_tot[i])) / max(_tot_tmp)

        # print('finished convolutions, took %f sec' % (time.time() - t1))
    return ccf_quiet, flux, bconv, tot


@numba.njit(cache=True, nopython=True)
def calculate_ccf(
    wave1: np.ndarray,
    flux1: np.ndarray,
    wave2: np.ndarray,
    flux2: np.ndarray,
    rvmin=-20e3,
    rvmax=20e3,
    rvstep=0.5e3,
):
    left_edge = doppler_shift_wave(wave1[0], rvmax) - wave1[0]
    right_edge = doppler_shift_wave(wave1[-1], rvmax) - wave1[-1]
    mask = (wave1 > wave1[0] + left_edge) & (wave1 < wave1[-1] - right_edge)
    rv = np.arange(rvmin, rvmax + rvstep, rvstep)
    ccf = np.zeros_like(rv)
    for i, v in enumerate(rv):
        shifted = doppler_shift(wave1[mask], flux1[mask], v)

        # shifted_interp = np.interp(wave2, wave1[mask], shifted)
        shifted_interp = linear_interpolator(wave1[mask], shifted, wave2)
        ccf[i] = (shifted_interp * flux2).sum()
    return rv, ccf


@numba.njit()
def calculate_ccf_with_mask(wave, flux, wave_mask, contrast_mask, rv):
    rvmin, rvmax, rvstep = rv
    _rv = np.arange(rvmin, rvmax, rvstep)
    ccf_out = np.zeros_like(_rv)
    istart = np.searchsorted(doppler_shift_wave(wave_mask, rvmin), wave[0], side="left")
    iend = np.searchsorted(doppler_shift_wave(wave_mask, rvmax), wave[-1], side="right")
    for i, v in enumerate(_rv):
        new_wave_mask = doppler_shift_wave(wave_mask[istart:iend], v)
        interpolated_spectrum = linear_interpolator(wave, flux, new_wave_mask)
        ccf_out[i] = np.nansum(interpolated_spectrum * contrast_mask[istart:iend])
    return _rv, ccf_out, (istart, iend)


###############################################################################
# planet functions


@numba.njit(cache=True)
def planet_position_at_date(t, P, t0, e, w, ip, a, lbda):
    """
    w, ip, a, lbd must be given in degrees!
    """
    phase = (t - t0) / P - int((t - t0) / P)
    anm = 2.0 * np.pi * phase
    E = anm
    E1 = E + (anm + e * sin(E) - E) / (1.0 - e * cos(E))
    while E - E1 > 1e-6:
        E = E1
        E1 = E + (anm + e * sin(E) - E) / (1.0 - e * cos(E))
    nu = 2.0 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E1 / 2.0))

    ip = ip * np.pi / 180
    lbda = lbda * np.pi / 180
    w = w * np.pi / 180

    rp = a * (1.0 - e * e) / (1.0 + e * cos(nu))
    xyz = np.empty(3)
    xyz[0] = rp * (sin(ip) * sin(nu + w))
    xyz[1] = rp * (-cos(lbda) * cos(nu + w) + sin(lbda) * cos(ip) * sin(nu + w))
    xyz[2] = rp * (-sin(lbda) * cos(nu + w) - cos(lbda) * cos(ip) * sin(nu + w))
    return xyz


@numba.njit(cache=True)
def spot_area(xyz, nrho, grid):
    """
    Determine a smaller yz-area of the stellar disk where the active region is
    The different cases are:
    - the active region is completely visible (all x of the circumference >=0)
    - the active region is completely invisible (all x of the circumference < 0)
    - the active region is on the disk edge and partially visible only
    """
    grid_step = 2.0 / grid  # the stellar disc goes from -1 to 1, therefore 2
    # initialize to 'opposite'-extreme values
    miny = 1
    minz = 1
    maxy = -1
    maxz = -1
    # to count how many points of the circumference are visible and invisible
    counton = 0
    countoff = 0
    # scan each point the circumference
    for j in range(nrho):
        if xyz[j, 0] >= 0:  # if x >= 0
            counton += 1
            #  select the extreme points of the circumference
            if xyz[j, 1] < miny:
                miny = xyz[j, 1]
            if xyz[j, 2] < minz:
                minz = xyz[j, 2]
            if xyz[j, 1] > maxy:
                maxy = xyz[j, 1]
            if xyz[j, 2] > maxz:
                maxz = xyz[j, 2]
        else:
            countoff = 1

    # if there are both visible and invisible points
    if counton > 0 and countoff > 0:
        #  --> active region is on the edge
        # in this situation there are cases where the yz-area define above is
        # actually smaller than the real area of the active region on the
        # stellar disk.
        # The minima/maxima are over/under-estimated if the active region is on
        # one of the axis (y or z). Because if on the y axis, the minimum
        # (or maximum) won't be on the circumference of the active region.
        # Same for z axis

        # active region on the z-axis because one pois on the positive side
        # of z, and the other on the negative side of z
        if miny * maxy < 0:
            if minz < 0:
                minz = -1  # active region on the bottom-z axis (z<0)
            else:
                maxz = 1  # active region on the top-z axis (z>=0)

        # active region on the y-axis because one pois on the positive side
        # of y, and the other on the negative side of z
        if minz * maxz < 0:
            if miny < 0:
                miny = -1  # active region on the left hand-y axis (y<0)
            else:
                maxy = 1  # active region on the right hand-y axis (y>=0)

    if counton == 0:
        visible = False
    else:
        visible = True

    # find the indices of miny, minz,... on the grid
    iminy = int(np.floor((1 + miny) / grid_step))
    iminz = int(np.floor((1 + minz) / grid_step))

    imaxy = int(np.ceil((1 + maxy) / grid_step))
    imaxz = int(np.ceil((1 + maxz) / grid_step))

    return visible, iminy, iminz, imaxy, imaxz


@numba.njit(cache=True, nopython=True)
def planet_area(xyz, grid, Rp, fe):
    # Grid step for the stellar disc from -1 to 1
    grid_step = 2.0 / grid
    fe_Rp = fe * Rp  # Factor for planet size and limb darkening

    # Initialize bounding box to extreme values
    miny, minz = 1.0, 1.0
    maxy, maxz = -1.0, -1.0

    # Distance of planet projection on the yz-plane
    d = np.sqrt(xyz[1] ** 2 + xyz[2] ** 2)

    visible = False
    if (xyz[0] >= 0) and (d <= (1 + fe_Rp)):
        visible = True
        miny = xyz[1] - fe_Rp
        minz = xyz[2] - fe_Rp
        maxy = xyz[1] + fe_Rp
        maxz = xyz[2] + fe_Rp

        # Adjust bounds for edge cases
        if d >= (1 - fe_Rp):
            if miny * maxy < 0:  # Crossing the z-axis
                if minz < 0:
                    minz = -1.0
                if maxz > 0:
                    maxz = 1.0
            if minz * maxz < 0:  # Crossing the y-axis
                if miny < 0:
                    miny = -1.0
                if maxy > 0:
                    maxy = 1.0

    # Convert to grid indices, ensuring indices are within grid bounds
    iminy = max(0, int(np.floor((1 + miny) / grid_step)))
    iminz = max(0, int(np.floor((1 + minz) / grid_step)))
    imaxy = min(grid, int(np.ceil((1 + maxy) / grid_step)))
    imaxz = min(grid, int(np.ceil((1 + maxz) / grid_step)))

    return visible, iminy, iminz, imaxy, imaxz


# ---------------------------------------------------------------------------------------------
@numba.njit(cache=True, nopython=True)
def planet_scan_rv(
    v: float64,
    i: float64,
    limba1: float64,
    limba2: float64,
    grid: int,
    rv: np.ndarray,
    ccf: np.ndarray,
    v_interval: float64,
    n_v: int,
    N: int,
    iminy: int,
    iminz: int,
    imaxy: int,
    imaxz: int,
    dp1: float64,
    dp2: float64,
    Rp: float64,
    fe: float64,
    fi: float64,
    theta: float64,
    ir: float64,
    calc_rv: int,
    alphaB: float64,
    alphaC: float64,
    cb1: float64,
) -> np.ndarray:
    # Initialize parameters
    delta_grid = 2.0 / grid
    delta_v = 2.0 * v_interval / n_v
    f_planet = np.zeros(n_v + 1)
    ccf_shifted = np.zeros(N)
    sum_planet = 0.0

    # Precompute constants
    costheta = cos(theta)
    sintheta = sin(theta)
    cosir2 = cos(ir) ** 2
    Rp2 = Rp**2
    fe2_Rp2 = fe**2 * Rp2
    fi2_Rp2 = fi**2 * Rp2
    rv_vrot = np.linspace(-v_interval, v_interval, n_v)

    # Iterate over the grid
    for iy in range(iminy, imaxy):
        y = -1.0 + iy * delta_grid+delta_grid/2
        y2 = y**2

        for iz in range(iminz, imaxz):
            z = -1.0 + iz * delta_grid+delta_grid/2
            z2 = z**2

            xdp1 = y - dp1
            xdp2 = z - dp2

            costheta_xdp1 = costheta * xdp1
            costheta_xdp2 = costheta * xdp2
            sintheta_xdp1 = sintheta * xdp1
            sintheta_xdp2 = sintheta * xdp2

            term1 = (costheta_xdp1 + sintheta_xdp2) ** 2 / fi2_Rp2
            term2 = (sintheta_xdp1 - costheta_xdp2) ** 2 / (cosir2 * fi2_Rp2)
            term3 = (costheta_xdp1 + sintheta_xdp2) ** 2 / fe2_Rp2
            term4 = (sintheta_xdp1 - costheta_xdp2) ** 2 / (cosir2 * fe2_Rp2)

            if (
                (term1 + term2 > 1.0 and term3 + term4 <= 1.0)
                or (xdp1**2 + xdp2**2 <= Rp2)
            ) and (y2 + z2 <= 1.0):
                r_cos, limb = ld(y, z, limba1, limba2)
                sum_planet += limb

                if calc_rv == 1:
                    latitude = z * sin(i) + r_cos * cos(i)
                    delta = vrot(v, r_cos, y, z, alphaB, alphaC, i, cb1)
                    f_planet += linear_interpolator(rv, ccf, rv_vrot - delta) * limb

    # Set the final value
    f_planet[-1] = sum_planet

    return f_planet


@numba.njit(cache=True, nopython=True)
def planet_scan_spectrum(
    v: float64,
    i: float64,
    limba1: float64,
    limba2: float64,
    grid: int,
    wave: np.ndarray,
    pixel,
    v_interval: float64,
    n_v: int,
    N: int,
    iminy: int,
    iminz: int,
    imaxy: int,
    imaxz: int,
    dp1: float64,
    dp2: float64,
    Rp: float64,
    fe: float64,
    fi: float64,
    theta: float64,
    ir: float64,
    calc_rv: int,
    alphaB: float64,
    alphaC: float64,
    cb1: float64,
) -> np.ndarray:
    # Scan of the yz-area where the planet is.
    # For each grid-point_t (y,z) we need to check whether it belongs to the planet
    # or not.

    delta_grid = 2.0 / grid
    delta_v = 2.0 * v_interval / n_v  # vv

    f_planet = np.zeros(len(wave) + 1)
    ccf_shifted = np.zeros(N)

    sum_planet = 0

    # Constant quantities (assuming that the rv array has a constant step)
    diff_CCF_non_v_and_v = int((n_v - N) / 2)

    costheta = cos(theta)
    sintheta = sin(theta)
    cosir2 = pow(cos(ir), 2)
    Rp2 = pow(Rp, 2)
    fe2_Rp2 = pow(fe, 2) * Rp2
    fi2_Rp2 = pow(fi, 2) * Rp2

    # Scan of each cell on the grid
    for iy in prange(iminy, imaxy):
        y = -1.0 + iy * delta_grid+ delta_grid/2 # y between -1 et 1
        y2 = pow(y, 2)
        for iz in range(iminz, imaxz):
            z = -1.0 + iz * delta_grid+ delta_grid/2  # z between -1 et 1
            z2 = pow(z, 2)

            xdp1 = y - dp1
            xdp2 = z - dp2

            costheta_xdp1 = costheta * xdp1
            costheta_xdp2 = costheta * xdp2
            sintheta_xdp1 = sintheta * xdp1
            sintheta_xdp2 = sintheta * xdp2

            if (
                (
                    (
                        (costheta_xdp1 + sintheta_xdp2) ** 2.0 / (fi2_Rp2)
                        + (sintheta_xdp1 - costheta_xdp2) ** 2.0 / (cosir2 * fi2_Rp2)
                    )
                    > 1
                    and (
                        (
                            (costheta_xdp1 + sintheta_xdp2) ** 2.0 / (fe2_Rp2)
                            + (sintheta_xdp1 - costheta_xdp2) ** 2.0
                            / (cosir2 * fe2_Rp2)
                        )
                        <= 1
                    )
                )
                or (xdp1**2.0 + xdp2**2.0) <= Rp2
            ) and (y**2.0 + z**2.0) <= 1:
                r_cos, limb = ld(y, z, limba1, limba2)
                sum_planet += limb

                if calc_rv == 1:

                    latitude = z * sin(i) + r_cos * cos(i)
                    delta = vrot(v, r_cos, y, z, alphaB, alphaC, i, cb1)
                    # Carefull!! Check the shift definition. This may be important for the spots also!
                    #shifted_quiet = doppler_shift(wave, flux, delta * 1e3)
                    shifted_quiet = doppler_shift(
                            pixel.wave, pixel.flux(r_cos), delta* 1e3)
                    # check this later
                    f_planet += shifted_quiet * limb
    # join the flux result for a more efficient output
    f_planet[-1] = sum_planet
    return f_planet


# ----------------------------------------------------------------------------------------------------------------


#@numba.njit(cache=True)
def planet_scan_ndate(v, i, date , ndate, limba1, limba2, 
                       grid, vrad_ccf,pixel,v_interval, n_v, n, Pp, t0,
                       e,w , ip , a,lbda ,Rp, fe, fi, theta, ir, calc_rv, pixel_type, alphaB, alphaC,cb1): 
    xyz2=np.zeros((ndate,3))
    flux_planet=np.zeros((ndate,n_v+1))
    ir=ir*pi/180.
    i=i*pi/180.
    theta=theta*pi/180.
    for idate in range(ndate):
        xyz=planet_position_at_date(date[idate], Pp, t0, e, w, ip, a, lbda)
        out=planet_area(xyz, grid, Rp,fe)

        xyz2[idate]=xyz
        if out[0]==1:
            if pixel_type == 'ccf': 
                flux_planet[idate]=planet_scan_rv(v, i, limba1, limba2, grid, vrad_ccf,pixel, 
                        v_interval, n_v, n, out[1], out[2], out[3], out[4], xyz[1], xyz[2], Rp, fe, fi, theta, ir, calc_rv, alphaB, alphaC,cb1)
            elif pixel_type == 'spectrum':
                flux_planet[idate]=planet_scan_spectrum(v, i, limba1, limba2, grid, vrad_ccf,pixel, 
                        v_interval, n_v, n, out[1], out[2], out[3], out[4], xyz[1], xyz[2], Rp, fe, fi, theta, ir, calc_rv, alphaB, alphaC, cb1)
            else:
                raise RuntimeError("Pixel type not supported. Choose between ccf and spectrum")
        else:
            pass
    # Split results for the flux and the sum_planet
    flux, sum_planet = flux_planet[:, :-1], flux_planet[:, -1]

    return flux, sum_planet, xyz2


def precompile_functions():
    # from .classes import solarFTS
    # spot_scan_spectrum(
    #     1,
    #     90,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     2,
    #     solarFTS().to_numba(),
    #     solarFTS().to_numba(),
    #     0.1,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     5000,
    #     0,
    #     3000,
    #     np.zeros((2 + 1, 2 + 1), dtype=bool),
    #     10,
    #     0,
    #     1,
    #     0,
    #     0,
    #     90,
    #     90,
    #     5,
    #     0,
    #     1,
    #     1,
    #     90,
    #     90,
    # )
    # itot_spectrum_par(1, 90, 0, 0, 0, 0, 0, 2, solarFTS().to_numba())
    return None
