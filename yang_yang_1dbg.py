import numpy as np
from scipy import constants, signal

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
        
        # Zero temperature dispersion
        self.eps0 = self.kappa**2 - self.mutilde

    def get_g1D(self):
        return self.g1D
        
    def get_k_space(self, klim, N):
        return np.linspace(-klim, klim, N)
    
    def get_k_thermal(self):
        return np.sqrt(2*self.mass*self.E_thermal/hbar**2)
    
    def lieb_kernel(self, k0):
        return 1/(pi*self.ctilde)/(1+((k0-self.kappa)/self.ctilde)**2)
    
    def epsilon_update(self, eps_q, method='convolution'):
        dk = self.kappa[1]-self.kappa[0]
        g_tilde = np.log(1 + np.exp(-eps_q))*dk
        if method == 'integral':
            return np.array([np.sum(g_tilde*self.lieb_kernel(k0=ki)) for ki in self.kappa])
        elif method == 'convolution':
            return signal.fftconvolve(g_tilde, self.lieb_kernel(k0=0), "same")
    
    def f_update(self, f_q, method='convolution'):
        dk = self.kappa[1] - self.kappa[0]
        f_q *= dk
        if method == 'integral':
            return np.array([np.sum(f_q*self.lieb_kernel(k0=ki)) for ki in self.kappa])
        elif method == 'convolution':
            return signal.fftconvolve(f_q, self.lieb_kernel(k0=0), "same")
        
    def eps_solver(self, eps_tol=1e-10):
        eps_convergence = 1.0
        eps_i = self.eps0
        while eps_convergence > eps_tol:
            eps_f = self.eps0 - self.epsilon_update(eps_i)
            eps_convergence = np.sqrt(np.sum((eps_i - eps_f)**2))
            #print(eps_convergence)
            eps_i = eps_f
        return eps_f
    
    def f_solver(self, f_tol=1e-3):
        f_convergence = 1.0
        eps_k = self.eps_solver(eps_tol=1e-3)
        f_i = 1/(2*pi*(1+np.exp(eps_k)))
        f_buffer = [np.amax(f_i), 0.]
        while f_convergence > f_tol:
            f_f = 1/(2*pi*(1+np.exp(eps_k))) + self.f_update(f_i)/(1+np.exp(eps_k))
            f_buffer[0] = f_buffer[1]
            f_buffer[1] = np.amax(f_f)
            f_convergence = np.abs(f_buffer[1] - f_buffer[0])
            f_i = f_f
        return f_f
    
    def density(self):
        dk = (self.kappa[1]-self.kappa[0])*self.get_k_thermal()
        return np.sum(self.f_solver()*dk), self.get_g1D()

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