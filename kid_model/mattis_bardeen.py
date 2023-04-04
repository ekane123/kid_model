import numpy as np
from scipy.special import kn
from scipy.integrate import trapz
import math

# SI units
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

def get_nqp_thermal(T, Tc, N0):
    '''
    T<<Tc approximation for the thermal quasiparticle density.
    '''
    Delta0 = 1.76*k_B*Tc
    nqp = 2*N0*np.sqrt(2*np.pi*k_B*T*Delta0)*np.exp(-Delta0/(k_B*T))
    return nqp

def get_nqp_thermal_exact(T, Tc, N0):
    '''
    Calculates the exact integral for nqp_thermal. There is still a
    low-temperature approximation being made here, in that we use the
    zero-temperature value of Delta. This should be nearly exact at
    temperatures T<Tc/2, if I remember correctly.
    '''
    if T == 0:
        nqp_thermal = 0.
    else:
        Delta0 = 1.76*k_B*Tc
        def integrand(E):
            BCS_DOS = E/np.sqrt(E**2-Delta0**2)
            return fFD(E,T)*BCS_DOS
        # The upper bound of 5*Delta0 is arbirtrary but seems to
        # be high enough for all practical purposes.
        nqp_thermal = 4*N0*trapz2(integrand,(Delta0,5*Delta0))
    return nqp_thermal

def get_Teff(nqp_total, T_actual, Tc, N0, exact=False):
    '''
    Given a total amount of quatiparticles nqp_total, returns the temperature
    Teff that would be required to thermally generate nqp_total quasiparticles.

    If exact==True, the function get_nqp_thermal_exact() is used to calculate
    thermal quasiparticle density. Otherwise, the T<<Tc approximation
    get_nqp_thermal() is used.
    '''
    if exact:
        get_nth = get_nqp_thermal_exact
    else:
        get_nth = get_nqp_thermal
    Delta0 = 1.76*k_B*Tc
    nth = get_nth(T_actual, Tc, N0)
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
            derivative = (get_nth(Teff+dTeff/2,Tc,N0) \
                        - get_nth(Teff-dTeff/2,Tc,N0)) \
                         / dTeff
            Teff = Teff + (nqp_total - nth) / derivative
            if Teff <= 0:
                Teff = (last2[0]+last2[1])/2
                last2[1] = Teff
            else:
                last2.append(Teff)
                last2 = last2[1:]
            nth = get_nth(Teff,Tc,N0)
            error = np.abs((nth - nqp_total)/nqp_total)
        i+=1
    return Teff

def get_Gamma_r(nqp, tau_max, nstar):
    Gamma_r = nqp/tau_max * (1 + nqp/(2*nstar))
    return Gamma_r

def trapz2(func, bounds, conv_cond=0.05, n=8, Int1=None):
    '''
    Recursively splits an integral into halves and decreases dx
    until the integral converges.
    This is needed to calculate sigma1n because scipy.integrate.quad was giving
    erroneous answers dependent upon the upper integration bound.

    INPUTS:
        func: 1D function to be integrated
        bounds: bounds of integration

        conv_cond: convergence parameter, should be in range 0.02-0.1
        n: Sets the initial number of points to sample, 2**n + 1.
           Making this too high/low will slow down the function.
        Int1: Lower-resolution integral to be compared against for convergence.
              DON'T PROVIDE AN INPUT TO THIS VARIABLE

    OUTPUTS:
        The value of the integral
    '''
    if Int1 is None:
        x1 = np.linspace(bounds[0],bounds[1],2**(n-1)+1)
        y1 = func(x1)
        i1 = [i for i in range(len(y1)) if not math.isinf(y1[i])]
        Int1 = trapz(y1[i1], x1[i1])

    x2 = np.linspace(bounds[0],bounds[1],2**n+1)
    y2 = func(x2)
    i2 = np.array([i for i in range(len(y2)) if not math.isinf(y2[i])])
    imid = i2[np.argmin(np.abs(i2-len(x2)//2))]
    Int2a = trapz(y2[i2[:imid+1]],x2[i2[:imid+1]])
    Int2b = trapz(y2[i2[imid:]],x2[i2[imid:]])
    Int2 = Int2a+Int2b

    if np.abs((Int1-Int2)/Int1) < conv_cond:
        return Int2
    else:
        return trapz2(func,(x2[0],x2[imid]),Int1=Int2a) \
             + trapz2(func,(x2[imid],x2[-1]),Int1=Int2b)
