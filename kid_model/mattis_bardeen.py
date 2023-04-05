import numpy as np
from scipy.special import kn
from scipy.integrate import quad

# work in SI units
k_B = 1.381e-23
hbar = 1.055e-34
eV = 1.602e-19
mu_0 = 1.257e-6

def fFD(E,T):
    '''Fermi-Dirac Distribution'''
    return 1/(1+np.exp(E/(k_B*T)))

def get_S1(T, Tc, omega):
    '''
    T<<Tc approximation for the function describing
    the real part of the conductivity's response to quasiparticles.
    '''
    Delta0 = 1.76*k_B*Tc
    a = hbar*omega/(2*k_B*T)
    S1 = 2/np.pi * np.sqrt(2*Delta0/(np.pi*k_B*T)) * np.sinh(a) * kn(0,a)
    return S1

def get_S2(T, Tc, omega):
    '''
    T<<Tc approximation for the function describing
    the imaginary part of the conductivity's response to quasiparticles.
    '''
    Delta0 = 1.76*k_B*Tc
    a = hbar*omega/(2*k_B*T)
    S2 = 1 + np.sqrt(2*Delta0/(np.pi*k_B*T)) * np.exp(-a) * np.i0(a)
    return S2

def get_nth(T, Tc, N0, exact_nth=False, exact_delta=False, omega_D=9.66e13):
    '''
    Calculates the thermal quasiparticle density.

    If exact_nth==True, uses the BCS integral for thermal quasiparticle
    density, otherwise uses the T<<Tc approximation.

    If exact_delta==True, also uses the exact expression for Delta(T) when
    calculating the BCS integral. Otherwise, uses the zero-temperature
    value of Delta.

    Note that a change of variables has been used to remove the singularity
    present in the BCS density of states, in order to allow for easy
    numerical integration.
    '''
    if T == 0:
        nth = 0.
    elif exact_nth:
        if exact_delta:
            Delta = get_Delta(T, Tc, omega_D)
        else:
            Delta = 1.76*k_B*Tc
        def integrand(E):
            return 1/(1+np.exp(np.sqrt(Delta**2+E**2)/(k_B*T)))
        # The upper bound of 10*Delta is arbitrary but it works
        # across virtually the entire range of T=[0,Tc].
        integral, abserr = quad(integrand,0,10*Delta)
        nth = 4*N0*integral
    else:
        Delta0 = 1.76*k_B*Tc
        nth = 2*N0*np.sqrt(2*np.pi*k_B*T*Delta0)*np.exp(-Delta0/(k_B*T))
    return nth

def get_Delta(T, Tc, omega_D=9.66e13):
    '''
    Returns the superconducting energy gap at temperature T
    given the critical temperature Tc and Debye frequency omega_D.
    '''
    Delta_0 = 1.76*k_B*Tc
    if T==0:
        return Delta_0

    Beta = 1/(k_B*T)
    Delta_T = Delta_0
    A = np.arcsinh(hbar*omega_D/Delta_0)

    def integrand(E,Delta_T):
        return np.tanh(Beta*np.sqrt(E**2 + Delta_T**2)/2)/(2*np.sqrt(E**2 + Delta_T**2))

    integral = quad(lambda E: integrand(E, Delta_T),-hbar*omega_D,hbar*omega_D)[0] / A
    error = integral - 1
    converged = False

    ii = 0
    while not converged:
        if np.abs(error) < 1e-8 or ii>10:
            converged = True
        else:
            dDelta_T = Delta_T/100
            derivative = (quad(lambda E: integrand(E, Delta_T+dDelta_T/2),-hbar*omega_D,hbar*omega_D)[0] / A \
                         -quad(lambda E: integrand(E, Delta_T-dDelta_T/2),-hbar*omega_D,hbar*omega_D)[0] / A ) \
                         / dDelta_T
            Delta_T = max(0, Delta_T + (1 - integral) / derivative)
            integral = quad(lambda E: integrand(E, Delta_T),-hbar*omega_D,hbar*omega_D)[0] / A
            error = integral - 1
        ii+=1
    return Delta_T

def get_Teff(nqp_total, T_actual, Tc, N0, exact_nth=False, exact_delta=False,
            omega_D=9.66e13):
    '''
    Given a total amount of quatiparticles nqp_total, returns the temperature
    Teff that would be required to thermally generate nqp_total quasiparticles.
    '''
    Delta0 = 1.76*k_B*Tc
    nth = get_nth(T_actual,Tc,N0,exact_nth,exact_delta)
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
            derivative = (get_nth(Teff+dTeff/2,Tc,N0,exact_nth,exact_delta) \
                        - get_nth(Teff-dTeff/2,Tc,N0,exact_nth,exact_delta)) \
                         / dTeff
            Teff = Teff + (nqp_total - nth) / derivative
            if Teff <= 0:
                Teff = (last2[0]+last2[1])/2
                last2[1] = Teff
            else:
                last2.append(Teff)
                last2 = last2[1:]
            nth = get_nth(Teff,Tc,N0,exact_nth,exact_delta)
            error = np.abs((nth - nqp_total)/nqp_total)
        i+=1
    return Teff

def get_Gamma_r(nqp, tau_max, nstar):
    Gamma_r = nqp/tau_max * (1 + nqp/(2*nstar))
    return Gamma_r
