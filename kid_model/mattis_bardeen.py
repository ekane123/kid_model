import numpy as np
from scipy.special import kn

# SI units
k_B = 1.381e-23
hbar = 1.055e-34
eV = 1.602e-19
mu_0 = 1.257e-6

def get_S1(T, Tc, omega):
    Delta0 = 1.76*k_B*Tc
    a = hbar*omega/(2*k_B*T)
    S1 = 2/np.pi * np.sqrt(2*Delta0/(np.pi*k_B*T)) * np.sinh(a) * kn(0,a)
    return S1

def get_S2(T, Tc, omega):
    Delta0 = 1.76*k_B*Tc
    a = hbar*omega/(2*k_B*T)
    S2 = 1 + np.sqrt(2*Delta0/(np.pi*k_B*T)) * np.exp(-a) * np.i0(a)
    return S2

def get_nqp_thermal(T, Tc, N0):
    Delta0 = 1.76*k_B*Tc
    nqp = 2*N0*np.sqrt(2*np.pi*k_B*T*Delta0)*np.exp(-Delta0/(k_B*T))
    return nqp

def get_Teff(nqp_total, T_actual, Tc, N0):
    '''
    Given a total amount of quatiparticles nqp_total, returns the temperature
    Teff that would be required to thermally generate nqp_total quasiparticles.
    '''
    Delta0 = 1.76*k_B*Tc
    nth = get_nqp_thermal(T_actual, Tc, N0)
    error = np.abs((nth - nqp_total)/nqp_total)
    Teff = max(T_actual, Tc/10.0)
    converged = False

    last2 = [Teff,Teff]
    i = 0
    while not converged:
        if np.abs(error) < 1e-4 or i>100:
            converged = True
        else:
            dTeff = Teff/50
            derivative = (get_nqp_thermal(Teff+dTeff/2,Tc,N0) \
                        - get_nqp_thermal(Teff-dTeff/2,Tc,N0)) \
                         / dTeff
            Teff = Teff + (nqp_total - nth) / derivative
            if Teff <= 0:
                Teff = (last2[0]+last2[1])/2
                last2[1] = Teff
            else:
                last2.append(Teff)
                last2 = last2[1:]
            nth = get_nqp_thermal(Teff,Tc,N0)
            error = np.abs((nth - nqp_total)/nqp_total)
        i+=1
    return Teff

def get_Gamma_r(nqp, tau_max, nstar):
    Gamma_r = nqp/tau_max * (1 + nqp/(2*nstar))
    return Gamma_r
