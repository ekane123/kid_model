import numpy as np
from scipy.special import kn

# SI units
k = 1.381e-23
hbar = 1.055e-34
h = hbar * 2*np.pi
eV = 1.602e-19
mu_0 = 1.257e-6


def get_NEP_freq(nu_o, R, T_RJ, eta_o, P_g, eta_a, T_a, T, T_c, N_0,
                nstar, tau_max, S_TLS, alpha, gamma, V_ind, Q_c, f_0):
    """
    calculates the NEP in the frequency direction

    INPUTS:
        nu_o: center frequency of mm-wave filter
        R: spectral resolution of mm-wave filter
        T_RJ: Rayleigh-Jeans temperature of sky at filter frequencies
        eta_o: efficiency with which mm-wave power out of the filter
                generates quasiparticles
        P_g: generator tone power at which resonator bifurcates [dBm]
        eta_a: efficiency with which absorbed readout power generates
                quasiparticles
        T_a: amplifier noise temperature
        T: detector temperature
        T_c: superconductor critical temperature
        N_0: single-spin density of states at the Fermi level
        nstar: characteristic quasiparticle density
        tau_max: maximum quasiparticle lifetime
        S_TLS: power spectral density of TLS (approximated to be constant)
        alpha: kinetic inductance fraction
        gamma: KID geometry parameter
                1 --> thin film limit
                1/2 --> local limit
                1/3 --> extreme anomalous limit
        V_ind: inductor volume
        Q_c: coupling quality factor
        f_0: resonant frequency of KID

    RETURNS:

    """

    omega = 2*np.pi*f_0
    Delta_0 = 1.76*k*T_c

    P_o = k*T_RJ*nu_o/R
    n_o = k*T_RJ/(h*nu_o)

    n_th = get_n_th(T, Delta_0, N_0)
    n_qp = get_n_qp(T, Delta_0, N_0, nstar, tau_max, V_ind, eta_o, P_o)
    N_th = n_th * V_ind
    N_qp = n_qp * V_ind
    T_eff = get_T_eff(T, Delta_0, N_0, n_qp)
    S_1 = get_S_1(T_eff, Delta_0, omega)
    S_2 = get_S_2(T_eff, Delta_0, omega)
    beta = S_2/S_1
    Q_i = 2*N_0*Delta_0/(alpha*gamma*S_1*n_qp)

    chi_c = 4*Q_c*Q_i/(Q_c+Q_i)**2
    chi_qp = 1.

    P_g = 1e-3 * 10**(P_g/10)
    P_a = P_g*chi_c/2
    P_a = chi_c/2 * P_g

    tau_th = get_tau_qp(n_th, nstar, tau_max)
    tau_qp = get_tau_qp(n_qp, nstar, tau_max)
    Gamma_th = N_th/2*(1/tau_max + 1/tau_th)
    Gamma_r = N_qp/2*(1/tau_max + 1/tau_qp)


    NEP_photon_g = np.sqrt(2*P_o*h*nu_o*(1+n_o))

    NEP_microwave_g = np.sqrt(4*eta_a*chi_qp*Delta_0*P_a/eta_o**2)

    NEP_thermal_g = np.sqrt(4*Gamma_th*Delta_0**2/eta_o**2)

    NEP_total_r = np.sqrt(4*Gamma_r*Delta_0**2/eta_o**2)

    NEP_amplifier = np.sqrt(8*N_qp**2*Delta_0**2*k*T_a \
                    /(beta**2*eta_o**2*chi_c*chi_qp**2*tau_qp**2*P_a))

    NEP_TLS = np.sqrt(8*N_qp**2*Delta_0**2*Q_i**2*S_TLS \
                    /(beta**2*eta_o**2*chi_c*chi_qp**2*tau_qp**2))

    return NEP_photon_g, NEP_microwave_g, NEP_thermal_g, NEP_total_r, \
            NEP_amplifier, NEP_TLS


def get_n_th(T, Delta_0, N_0):
    """
    density of thermally generated quasiparticles
    """
    if T > 0:
        n_th = 2*N_0*np.sqrt(2*np.pi*k*T*Delta_0)*np.exp(-Delta_0/(k*T))
    elif T == 0:
        n_th = 0
    return n_th


def get_n_qp(T, Delta_0, N_0, nstar, tau_max, V_inductor, eta_o, P_o):
    """
    total density of quasiparticles (thermally generated +
        generated by optical radiation)
    """
    n_th = get_n_th(T, Delta_0, N_0)
    n_qp = np.sqrt((nstar+n_th)**2 + \
            2*nstar*eta_o*P_o*tau_max/(Delta_0*V_inductor)) - nstar
    return n_qp


def get_T_eff(T, Delta_0, N_0, n_qp):
    '''
    Given a total amount of quatiparticles n_qp
        (thermally generated + generated by incident radiation),
        returns the temperature T_eff that would be required to thermally
        generate n_qp quasiparticles.
    '''
    n_th = get_n_th(T, Delta_0, N_0)
    error = np.abs((n_th - n_qp)/n_qp)
    T_c = Delta_0/(3.52*k)
    T_eff = max(T, T_c/10.0)
    converged = False

    ii = 0
    while not converged:
        if np.abs(error) < 1e-3 or ii>10:
            converged = True
        else:
            dT_eff = T_eff/50
            derivative = ( get_n_th(T_eff+dT_eff/2, Delta_0, N_0) \
                         - get_n_th(T_eff-dT_eff/2, Delta_0, N_0) ) \
                         / dT_eff
            T_eff = max(0, T_eff + (n_qp - n_th) / derivative)
            n_th = get_n_th(T_eff, Delta_0, N_0)
            error = np.abs((n_th - n_qp)/n_qp)
        ii += 1
    return T_eff


def get_S_1(T_eff, Delta_0, omega):
    S_1 = 2/np.pi*np.sqrt(2*Delta_0/(np.pi*k*T_eff)) \
        * np.sinh(hbar*omega/(2*k*T_eff))*kn(0, hbar*omega/(2*k*T_eff))
    return S_1


def get_S_2(T_eff, Delta_0, omega):
    S_2 = 1 + np.sqrt(2*Delta_0/(np.pi*k*T_eff)) \
            * np.exp(-hbar*omega/(2*k*T_eff))*np.i0(hbar*omega/(2*k*T_eff))
    return S_2


def get_tau_qp(n_qp, nstar, tau_max):
    return tau_max/(1+n_qp/nstar)