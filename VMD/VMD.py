# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 12:59:05 2014

@author: nabobalis
"""

from __future__ import division
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def VMD(signal, alpha, tau, K, DC, init, tol):
    """
    Computes Something from Something!
    
    Parameters
    ----------
    signal : array-like
        The time domain signal (1D) to be decomposed
    alpha : float
        The balancing paramter of the data-fidelity
    tau : float
        Time-step of the dual ascent. 
        Pick 0 for noise-slack
    K : int
        Number of modes to be recovered.
    DC : Bool
        True if the first mode is put and kept at DC (0-freq)
    init : int
        0 : all omegas start at 0.
        1 : all omegas initialized unifromaly distributed.
        2 : all omegas initialized randomly.
        Variable length argument list.
    tol : float
        tolerance of convergence criterion; typcially around 1e-6
    
    Returns
    -------
    u : array-like
        The collection of decompiosed modes
    u_hat : array-like?
        Spectra of the modes
    omega : float
        Estimated mode centre-frequnices
    """
    ## Preparations
    
    # Period and sampling frequency of input signal
    save_t = len(signal)
    fs = 1/save_t

    # Extend signal by mirroring
    T = save_t
    f_mirror = signal[0:T/2+1][::-1]
    f_mirror = np.append(f_mirror,signal)
    f_mirror = np.append(f_mirror, signal[T/2+1:][::-1])
    f = f_mirror
    
    # Time domain 0 to T (of mirrored signal)
    T = len(f)
    t = np.arange(0,T)/T
    
    # Spectral Domain discretization
    freqs = (t-0.5)-(1/T)
    
    # Maxiimum number of iterations (if not converged yet, then it won't anyway!)
    N = 500
    
    # For future generalziations: individual alpha for each mode
    Alpha = alpha*np.ones([1,K])
    
    # Contruct and cnter f_hat
    f_hat = fftpack.fftshift(fftpack.fft(f))
    f_hat_plus = f_hat
    f_hat_plus[0:T/2] = 0
    
    # Matrix keeping track of every iterant, could be discarded for memory
    u_hat_plus = np.zeros([N, len(freqs),K])
    
    # Initializastion of omega_k
    omega_plus = np.zeros([N,K])
    
    if init == 1:
        for i in range(K):
            omega_plus[0,i] = (0.5/K)*(i-1)
    elif init == 2:
        omega_plus[0:] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.random(K)))
    else:
        omega_plus[0,:] = 0
    
    # If DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0,0] = 0
        
    # Start with empty dual varaibles
    lambda_hat = np.zeros([N, len(freqs)])
    
    # Other inits
    uDiff = 2* tol + np.spacing(1) #update step
    n = 0 # loop counter
    sum_uk = 0 # accumulator
    
    # Main loop for iterative updates
    while (uDiff.any() > tol) or (n < N-1): # Not converged and below iterations limit
        #update first mode accumulator
        k = 1
        sum_uk = u_hat_plus[n,:,K-1] + sum_uk - u_hat_plus[n,:,0]
        
        # Update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2) / (1 + Alpha[0,:k] * (freqs - omega_plus[n,k])**2)
        
        # Update first omega if not held at 0
        if DC == False:
            omega_plus[n,k] = np.dot(freqs[T/2+1:T],(np.abs(u_hat_plus[n, T/2+1:T, k])**2).T) / \
                                 np.sum(np.abs(u_hat_plus[n,T/2+1:T,k])**2)
            
        for k in range(0,K):
            
            # Accumlautor
            sum_uk = u_hat_plus[n,:,k-1] + sum_uk - u_hat_plus[n,:,k]
            
            # Mode spectrum
            u_hat_plus[n, :, k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2) / (1 + Alpha[0,k] * (freqs - omega_plus[n,k])**2)
            
            # Center frequcies 
            omega_plus[n,k] = np.dot(freqs[T/2+1:T],(np.abs(u_hat_plus[n, T/2+1:T, k])**2).T) \
                                / np.sum(np.abs(u_hat_plus[n,T/2+1:T,k])**2)
        
        # Dual ascent
        lambda_hat[n,:] = lambda_hat[n,:] + tau * (np.sum(u_hat_plus[n,:,:],axis=1) - f_hat_plus)

        
        #loop counter
        n = n + 1
            
        # Not converged yet?
        uDiff = np.spacing(1)
        for i in range(0,K):
            uDiff = uDiff + 1/T*(u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i])*np.conjugate(u_hat_plus[n,:,i] - u_hat_plus[n-1,:,i]).T 
        uDiff = np.abs(uDiff)
        
    # Postprocessing and cleanup
    
    #Discard empty space if converged early
    N = np.min(N,n)
    omega = omega_plus[0:N,:]
    
    # Signal recontruction
    u_hat = np.zeros([T,K])
    u_hat[T/2+1:T,:] = u_hat_plus[N-1,T/2+1:T,:].squeeze()
    u_hat[2:T/2+1,:] = np.conjugate(u_hat_plus[N-1,T/2+1:T,:]).squeeze()
    u_hat[1,:] = np.conjugate(u_hat[-1,:])
    
    u = np.zeros([K,T])
    for k in range(0,K):
        u[k,:] = np.real(fftpack.ifft(fftpack.ifftshift(u_hat[:,k])))
    
    # Remove mirror part
    u = u[:,T/4+1:3*T/4]
    
    # Recompute spectrum
    del u_hat
    u_hat = np.zeros(u.shape[::-1])
    for k in range(0,K):
        u_hat[:,k] = fftpack.fftshift(fftpack.fft(u[k,:])).T
        
    return u, u_hat, omega
    
T = 1000
fs = 1/T
t = np.arange(0,T)/T
freqs = 2*np.pi*(t-0.5-1/T)/(fs)

# center frequencies of components
f_1 = 2
f_2 = 24
f_3 = 288

# modes
v_1 = (np.cos(2*np.pi*f_1*t))
v_2 = 1/4*(np.cos(2*np.pi*f_2*t))
v_3 = 1/16*(np.cos(2*np.pi*f_3*t))

# composite signal, including noise
f = v_1 + v_2 + v_3 + 0.1*np.random.randn(len(v_1))
f_hat = fftpack.fftshift((fftpack.fft(f)))

# some sample parameters for VMD
alpha = 2000 # moderate bandwidth constraint
tau = 0 #noise-tolerance (no strict fidelity enforcement)
K = 3 # 3 modes
DC = 0 # no DC part imposed
init = 1 # initialize omegas uniformly
tol = 1e-7

#Run actual VMD code

plt.plot(f)
plt.show()
u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
