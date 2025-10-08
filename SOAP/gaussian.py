import numba
import numpy as np

DEBUG = False


@numba.njit(cache=True)
def gauss(p: np.ndarray, x: np.ndarray):
    """A Gaussian function with parameters p = [A, x0, σ, offset]."""
    return p[0] * np.exp(-((x - p[1]) ** 2) / (2.0 * p[2] ** 2)) + p[3]


@numba.njit(cache=True)
def unit_gauss(x0: float, fwhm: float, x: np.ndarray):
    """
    A Gaussian function with area=1. Arguments are center position and FWHM.
    """
    sigma = np.abs(fwhm) / (2 * np.sqrt(2 * np.log(2)))
    a = 1.0 / (sigma * np.sqrt(2 * np.pi))
    tau = -((x - x0) ** 2) / (2 * (sigma**2))
    return a * np.exp(tau)


@numba.jit(cache=True, nopython=True)
def BIS_HARPS(rv: np.ndarray, ccf: np.ndarray, p0: tuple) -> float:
    """
    BIS (Bisector Inverse Slope) calculation as performed by the HARPS pipeline.
    This implementation uses a custom quadratic interpolation devised by Christophe Lovis,
    where the quadratic term is set by a Gaussian fit to the profile.

    Parameters:
        rv (np.ndarray): Radial velocities (x-axis values of the CCF).
        ccf (np.ndarray): Cross-correlation function (y-axis values of the CCF).
        p0 (tuple): Initial parameters (k, v0, sigma, c) for the Gaussian profile.

    Returns:
        float: BIS span (difference between the RV at the top and bottom of the profile).
    """
    down_range = (0.1, 0.4)  # Bottom region of CCF for BIS
    up_range = (0.6, 0.9)  # Top region of CCF for BIS

    k, v0, sigma, c = p0

    # Normalize the CCF
    norm_ccf = -c / k * (1.0 - ccf / c)
    n_steps = 100
    margin = 5
    depth = np.arange(n_steps - 2 * margin + 1) / n_steps + margin / n_steps

    # Initialize arrays for coefficients and bisectors
    coefficients = np.zeros((len(ccf), 3))
    bis_blue = np.zeros(len(depth))
    bis_red = np.zeros(len(depth))

    # Calculate quadratic coefficients for the CCF
    for i in range(len(ccf) - 1):
        if (
            max(norm_ccf[i], norm_ccf[i + 1]) >= depth[0]
            and min(norm_ccf[i], norm_ccf[i + 1]) <= depth[-1]
        ):
            v = (rv[i] + rv[i + 1]) / 2.0
            dccf_dRV = -(v - v0) / sigma**2 * np.exp(-((v - v0) ** 2) / (2 * sigma**2))
            d2ccf_dRV2 = (
                ((v - v0) ** 2 / sigma**2 - 1)
                / sigma**2
                * np.exp(-((v - v0) ** 2) / (2 * sigma**2))
            )
            d2RV_dccf2 = -d2ccf_dRV2 / dccf_dRV**3

            coefficients[i, 2] = d2RV_dccf2 / 2.0
            coefficients[i, 1] = (
                rv[i + 1]
                - rv[i]
                - coefficients[i, 2] * (norm_ccf[i + 1] ** 2 - norm_ccf[i] ** 2)
            ) / (norm_ccf[i + 1] - norm_ccf[i])
            coefficients[i, 0] = (
                rv[i]
                - coefficients[i, 1] * norm_ccf[i]
                - coefficients[i, 2] * norm_ccf[i] ** 2
            )

    # Calculate bisectors
    for j, d in enumerate(depth):
        # Blue side
        i_blue = np.argmax(norm_ccf)
        while norm_ccf[i_blue] > d and i_blue > 0:
            i_blue -= 1

        # Red side
        i_red = np.argmax(norm_ccf)
        while i_red < len(ccf) - 2 and norm_ccf[i_red + 1] > d:
            i_red += 1

        bis_blue[j] = (
            coefficients[i_blue, 0]
            + coefficients[i_blue, 1] * d
            + coefficients[i_blue, 2] * d**2
        )
        bis_red[j] = (
            coefficients[i_red, 0]
            + coefficients[i_red, 1] * d
            + coefficients[i_red, 2] * d**2
        )

    # Compute final bisector values
    bisector = (bis_blue + bis_red) / 2.0

    # Handle non-finite values
    if not np.all(np.isfinite(bisector)):
        return 0.0

    # Calculate BIS span
    down_mask = (depth >= down_range[0]) & (depth <= down_range[1])
    up_mask = (depth >= up_range[0]) & (depth <= up_range[1])

    RV_top = np.mean(bisector[up_mask])
    RV_bottom = np.mean(bisector[down_mask])

    return RV_top - RV_bottom


def _precompile_gauss(rv):
    """Call jit-ed functions so that numba compiles them"""
    par = np.array([-1.0, 0.0, 1.0, 0.0])
    _ = gauss(par, rv)
    # _ = BIS_HARPS(rv, rv, par)


def compute_rv_2d(rv: np.ndarray, ccf: np.ndarray):
    return np.array([compute_rv(rv, _ccf) for _ccf in ccf])


def compute_rv_fwhm_2d(rv: np.ndarray, ccf: np.ndarray):
    return np.array([compute_rv_fwhm(rv, _ccf) for _ccf in ccf])


def compute_rv_fwhm_bis_2d(rv: np.ndarray, ccf: np.ndarray):
    return np.array([compute_rv_fwhm_bis(rv, _ccf) for _ccf in ccf])


@numba.njit(cache=True)
def fast_convolve(λ, R, wave, flux, Nfwhm):
    """IP convolution multiplication step for a single wavelength value."""
    fwhm = λ / R
    # mask of wavelength range within Nfwhm of λ
    index_mask = (wave > (λ - Nfwhm * fwhm)) & (wave < (λ + Nfwhm * fwhm))
    flux_2convolve = flux[index_mask]

    # Gaussian Instrument Profile for given resolution and wavelength
    inst_profile = unit_gauss(λ, fwhm, wave[index_mask])

    sum_val = np.sum(inst_profile * flux_2convolve)
    # Correct for the effect of convolution with non-equidistant positions
    unitary_val = np.sum(inst_profile)

    return sum_val / unitary_val


@numba.njit(cache=True, parallel=True)
def ip_convolution(wave: np.ndarray, flux: np.ndarray, R: int, Nfwhm: int = 5):
    """Spectral convolution to resolution `R`.

    Args:
        wave (ndarray): Wavelength array of the spectrum to be convolved.
        flux (ndarray): Flux array of the spectrum to be convolved.
        R (int): Desired resolution to convolve to.
        Nfwhm (int, optional): Number of FWHM of Gaussian convolution kernel used as edge buffer. Default is 5.

    Returns:
        (ndarray, ndarray): Wavelength array and convolved flux array.
    """

    n = wave.size
    flux_conv = np.zeros(
        n, dtype=np.float64
    )  # Directly create the array of the correct size

    for i in numba.prange(n):
        # Convolve at each wavelength point
        flux_conv[i] = fast_convolve(wave[i], R, wave, flux, Nfwhm)

    return wave, flux_conv


def gaussian(x, params):
    """
    Gaussian function with an offset: A * exp(-(x - mu)^2 / (2 * sigma^2)) + C

    Parameters:
    - x: Input values (array).
    - params: Array of parameters [A, mu, sigma, C].

    Returns:
    - Gaussian function values with offset.
    """
    A, mu, sigma, C = params
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + C


def jacobian_gaussian(x, params):
    """
    Jacobian of the Gaussian function with respect to parameters A, mu, sigma, and C.

    Parameters:
    - x: Input values (array).
    - params: Array of parameters [A, mu, sigma, C].

    Returns:
    - Jacobian matrix of size len(x) x 4 (for A, mu, sigma, C).
    """
    A, mu, sigma, C = params
    exp_term = np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    # Partial derivatives with respect to A, mu, sigma, C
    dA = exp_term
    dmu = (x - mu) / (sigma**2) * A * exp_term
    dsigma = (x - mu) ** 2 / (sigma**3) * A * exp_term
    dC = np.ones_like(x)  # Derivative of offset C is 1 for all points

    # Stack partial derivatives as columns
    jacobian = np.vstack([dA, dmu, dsigma, dC]).T
    return jacobian


def levenberg_marquardt(
    func, jacobian, x_data, y_data, theta_init, max_iter=200, tol=1e-8, lambda_init=0.01
):
    """
    Levenberg-Marquardt algorithm for non-linear least squares optimization.

    Parameters:
    - func: A function that returns the model prediction for given parameters.
    - jacobian: A function that returns the Jacobian matrix of the model.
    - x_data: Input data for the model.
    - y_data: Observed data.
    - theta_init: Initial guess for the parameters [A, mu, sigma, C].
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.
    - lambda_init: Initial damping factor.

    Returns:
    - theta: Optimized parameters [A, mu, sigma, C].
    """
    # Initial parameter guess
    theta = np.array(theta_init, dtype=float)
    lambda_ = lambda_init  # Initial damping factor

    for iteration in range(max_iter):
        # Compute residuals and Jacobian at the current parameters
        residuals = func(x_data, theta) - y_data
        jacobian_matrix = jacobian(x_data, theta)

        # Compute the normal equation: (J^T * J + lambda * I) * delta_theta = J^T * residuals
        JTJ = np.dot(jacobian_matrix.T, jacobian_matrix)  # J^T * J
        JTr = np.dot(jacobian_matrix.T, residuals)  # J^T * residuals

        # Identity matrix of the same size as JTJ
        I = np.eye(len(theta))

        # Update step (delta_theta)
        A = JTJ + lambda_ * I
        delta_theta = np.linalg.solve(A, JTr)

        # Update the parameters
        theta_new = theta - delta_theta

        # Compute new residuals and check for convergence
        residuals_new = func(x_data, theta_new) - y_data
        if np.linalg.norm(residuals_new) < np.linalg.norm(residuals):
            # If residuals decreased, accept the new parameters and reduce lambda
            lambda_ *= 0.1
            theta = theta_new
        else:
            # If residuals did not decrease, increase lambda
            lambda_ *= 10

        # Check for convergence (change in parameters is small enough)
        if np.linalg.norm(delta_theta) < tol:
            # print(f"Converged in {iteration} iterations.")
            break

    return theta


def compute_rv_fwhm(rv: np.ndarray, ccf: np.ndarray):
    """Fit a Gaussian to rv, ccf and return the estimated RV and FWHM"""
    res = levenberg_marquardt(
        gaussian,
        jacobian_gaussian,
        rv,
        ccf,
        np.array([abs(np.max(ccf) - np.min(ccf)), 0, 3, np.max(ccf)]),
    )
    # import matplotlib.pyplot as plt
    # plt.plot(rv,ccf,".k")
    # plt.plot(rv,gaussian(rv,res),".r")
    # plt.plot(rv,gaussian(rv,np.array([np.min(ccf)-np.max(ccf),0, 3,np.max(ccf)])),".g")
    # plt.show()
    return res[1], 2 * np.sqrt(2 * np.log(2)) * res[2]


def compute_rv(rv: np.ndarray, ccf: np.ndarray):
    """Fit a Gaussian to rv, ccf and return the estimated RV and FWHM"""
    res = levenberg_marquardt(
        gaussian,
        jacobian_gaussian,
        rv,
        ccf,
        np.array([abs(np.max(ccf) - np.min(ccf)), 0, 3, np.max(ccf)]),
    )

    return res[1]


def compute_rv_fwhm_bis(rv: np.ndarray, ccf: np.ndarray):
    """Fit a Gaussian to rv, ccf and return the estimated RV and FWHM"""
    res = levenberg_marquardt(
        gaussian,
        jacobian_gaussian,
        rv,
        ccf,
        np.array([np.min(ccf) - np.max(ccf), 0, 3, np.max(ccf)]),
    )
    # import matplotlib.pyplot as plt
    # plt.plot(rv,ccf,".k")
    # plt.plot(rv,gaussian(rv,res),".r")
    # plt.plot(rv,gaussian(rv,np.array([np.min(ccf)-np.max(ccf),0, 3,np.max(ccf)])),".g")
    # plt.show()
    bis = BIS_HARPS(rv, ccf, res)
    return res[1], 2 * np.sqrt(2 * np.log(2)) * res[2], bis
