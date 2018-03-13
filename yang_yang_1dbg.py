import numpy as np
from scipy import constants, signal
from numba import vectorize

# Fundamental Constants
hbar = constants.codata.value('Planck constant over 2 pi')
kB = constants.codata.value('Boltzmann constant')
pi = np.pi    
 
class harmonic_trap(object):
    """ Simple 1D harmonic oscillator methods """
    
    def __init__(self, frequency, mass):
        self.omega = 2*pi*frequency
        self.mass = mass
        
    def oscillator_length(self):
        return np.sqrt(hbar/(self.mass*self.omega))
    
    def get_potential(self, xlim, N_points):
        x = np.linspace(-xlim, xlim, N_points)
        return 0.5*self.mass*self.omega**2*x**2
    
    def get_levels(self, number_of_levels):
        levels = []
        for i in range(number_of_levels):
            levels.append(hbar*self.omega*(i+0.5))
        return np.array(levels)

class Homo_1DBG(harmonic_trap):
    """ Zero temperature homogeneous one-dimensional Bose 
        gas methods; assume transverse harmonic potential"""
    
    def __init__(self, trans_freq, mass, scatt_length):
        harmonic_trap.__init__(self, trans_freq, mass)
        self.trans_freq = trans_freq
        self.mass = mass
        self.a_trans = self.oscillator_length()
        self.a3D = scatt_length
            
    def effective_1D_scattering_length(self):
        C = 1.4603/np.sqrt(2) # Confinement resonance constant
        corr = (1-C*(self.a3D/self.a_trans))
        return -(self.a_trans**2/self.a3D)*corr
        
    def regularized_1d_coupling_constant(self):
        a1D = self.effective_1D_scattering_length()
        return -2*hbar**2/(self.mass*a1D)
    
    def c_constant(self):
        g1D = self.regularized_1d_coupling_constant()
        return self.mass*g1D/hbar**2

@vectorize('float64(float64)')
def log_factor(eps_q):
    if eps_q > 0:
        return np.log(1+np.exp(-eps_q))
    else:
        return np.log(1+np.exp(eps_q)) - eps_q

class bethe_integrator(Homo_1DBG):
    
    def __init__(self, trans_freq, mass, temperature, 
                 chemical_potential, scatt_length):
        Homo_1DBG.__init__(self, trans_freq, mass, scatt_length)
        self.mass = mass
        self.c = self.c_constant()
        self.g1D = self.regularized_1d_coupling_constant()
        self.mu = 2*pi*chemical_potential*hbar
        self.E_thermal = kB*temperature
        self.k_space = self.get_k_space(1e1*self.get_k_thermal(), 2**10)

        # Rescaled quantities
        self.mutilde = self.mu/self.E_thermal
        self.ctilde = self.c/self.get_k_thermal()
        self.kappa = self.k_space/self.get_k_thermal()
        self.dkappa = self.kappa[1]-self.kappa[0]
        # Zero temperature dispersion
        self.eps0 = self.kappa**2 - self.mutilde
        # Initial k-distribution
        self.epsk = self.eps_solver(eps_tol=1e-10)
        self.bose_factor = np.exp(log_factor(-self.epsk))
        self.f0=1/(2*pi*self.bose_factor)

    def get_g1D(self):
        return self.g1D
        
    def get_k_space(self, klim, N):
        return np.linspace(-klim, klim, N)
    
    def get_k_thermal(self):
        return np.sqrt(2*self.mass*self.E_thermal)/hbar
    
    def lieb_kernel(self, k0):
        return 1/(pi*self.ctilde)/(1+((k0-self.kappa)/self.ctilde)**2)

    def epsilon_update(self, eps_q, method='convolution'):
        g_tilde = log_factor(eps_q)*self.dkappa
        if method == 'integral':
            return np.array([np.sum(g_tilde*self.lieb_kernel(k0=ki)) for ki in self.kappa])
        elif method == 'convolution':
            return signal.fftconvolve(g_tilde, self.lieb_kernel(k0=0), "same")
    
    def f_update(self, f_q, method='convolution'):
        if method == 'integral':
            return np.array([np.sum(f_q*self.dkappa*self.lieb_kernel(k0=ki)) for ki in self.kappa])
        elif method == 'convolution':
            return signal.fftconvolve(f_q*self.dkappa, self.lieb_kernel(k0=0), "same")
        
    def eps_solver(self, eps_tol=1e-3):
        def iterator(eps_it):
            eps_next = self.eps0 - self.epsilon_update(eps_it)
            eps_error = np.sqrt(np.mean((eps_it-eps_next)**2))
            return eps_error, eps_next 
        min_iteration, eps_err = 50, 1.0
        eps_i = self.eps0
        for i in range(min_iteration):
            _, eps_f = iterator(eps_i)
            eps_i = eps_f
        while (eps_err > eps_tol) and (min_iteration < 1000):
            min_iteration += 1
            eps_err, eps_f = iterator(eps_i)
            #print min_iteration, eps_err
            eps_i = eps_f
        return eps_f
    
    def f_solver(self, f_tol=1e-3):
        def iterator(f_it):
            f_next = self.f0 + self.f_update(f_it)/self.bose_factor
            f_error = np.sqrt(np.mean((f_it-f_next)**2))
            return f_error, f_next
        min_iteration, f_err = 50, 1.0
        f_i = self.f0
        for i in range(min_iteration):
            f_err, f_f = iterator(f_i)
            f_i = f_f
        while (f_err > f_tol) and (min_iteration < 1000):
            min_iteration += 1
            f_err, f_f = iterator(f_i)
            f_i = f_f
            #print min_iteration, f_err
        return f_f
    
    def density(self):
        dk = self.dkappa*self.get_k_thermal()
        return np.trapz(self.f_solver(f_tol=1e-10), dx=dk), self.get_g1D()

    def entropy_per_particle(self):
        eps = self.eps_solver(eps_tol=1e-3)
        fp = self.f_solver(f_tol=1e-3)
        fh = fp*np.exp(eps)
        f = fp + fh
        n, _ = self.density()
        dk = (self.kappa[1]-self.kappa[0])*self.get_k_thermal()
        S0 = np.sum(dk*f*np.log(1+np.exp(-eps)))
        S1 = np.sum(dk*fp*eps)
        return (S0 + S1)/n

    def pressure(self):
        T = self.E_thermal/kB
        dk = (self.kappa[1]-self.kappa[0])*self.get_k_thermal()
        eps = self.eps_solver(eps_tol=1e-3)
        P0 = np.sum(dk*np.log(1+np.exp(-eps)))
        n, _ = self.density()
        return T*P0/(2*pi*n)