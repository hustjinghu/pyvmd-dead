from __future__ import print_function, absolute_import, division

import numpy as np
from scipy import fftpack

def VMD(signal, alpha, tau, K, DC, init, tol):
    """
     Variational Mode Decomposition
     Authors: Konstantin Dragomiretskiy and Dominique Zosso
     zosso@math.ucla.edu --- http://www.math.ucla.edu/~zosso
     Initial release 2013-12-12 (c) 2013

     Input and Parameters:
     ---------------------
     signal  - the time domain signal (1D) to be decomposed
     alpha   - the balancing parameter of the data-fidelity constraint
     tau     - time-step of the dual ascent ( pick 0 for noise-slack )
     K       - the number of modes to be recovered
     DC      - true if the first mode is put and kept at DC (0-freq)
     init    - 0 = all omegas start at 0
                        1 = all omegas start uniformly distributed
                        2 = all omegas initialized randomly
     tol     - tolerance of convergence criterion typically around 1e-6

     Output:
     -------
     u       - the collection of decomposed modes
     u_hat   - spectra of the modes
     omega   - estimated mode center-frequencies

     When using this code, please do cite our paper:
     -----------------------------------------------
     K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Trans.
     on Signal Processing (in press)
     please check here for update reference:
              http://dx.doi.org/10.1109/TSP.2013.2288675
    """

    #---------- Preparations

    # Period and sampling frequency of input signal
    save_T = len(signal)
    fs = 1/save_T

    # extend the signal by mirroring
    T = save_T
    f_mirror = signal[T/2::-1]
    f_mirror = np.append(f_mirror,signal)
    f_mirror = np.append(f_mirror,signal[:T/2:-1])
    f = f_mirror

    # Time Domain 0 to T (of mirrored signal)
    T = len(f)
    t = np.arange(T)/T

    # Spectral Domain discretization
    freqs = t-0.5-1/T

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N = 500

    # For future generalizations: individual alpha for each mode
    Alpha = alpha*np.ones((1,K))

    # Construct and center f_hat
    f_hat = fftpack.fftshift((fftpack.fft(f)))
    f_hat_plus = f_hat
    f_hat_plus[:T/2] = 0

    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros((N, len(freqs), K))
    # Initialization of omega_k
    omega_plus = np.zeros((N, K))
    if init == 1:
        for i in range(K):
            omega_plus[0,i] = (0.5/K)*(i-1)
    elif init == 2:
        omega_plus[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(K)))
    else:
        omega_plus[0,:] = 0

    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0,0] = 0

    # start with empty dual variables
    lambda_hat = np.zeros((N, len(freqs)))

    # other inits
    eps = 2.220446049250313e-16
    uDiff = tol + eps #update step
    n = 0  #loop counter
    sum_uk = 0  #accumulator
    # ----------- Main loop for iterative updates

    while ( uDiff > tol and  n < N - 1 ): #not converged and below iterations limit

        # update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n,:,K-1] + sum_uk - u_hat_plus[n,:,0]

        # update spectrum of first mode through Wiener filter of residuals

        u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1+np.dot(Alpha[0,k],(freqs - omega_plus[n,k])**2))

        # update first omega if not held at 0
        if ~DC:
            omega_plus[n+1,k] = (np.dot(freqs[T/2+1:T],(np.abs(u_hat_plus[n+1, T/2+1:T, k])**2).conj().T))/np.sum(np.abs(u_hat_plus[n+1,T/2+1:T,k])**2)

        # update of any other mode
        for k in range(K):

            # accumulator
            sum_uk = u_hat_plus[n+1,:,k-1] + sum_uk - u_hat_plus[n,:,k]

            # mode spectrum
            u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1+np.dot(Alpha[0,k],(freqs - omega_plus[n,k])**2))

            # center frequencies
            omega_plus[n+1,k] = (np.dot(freqs[T/2+1:T],(np.abs(u_hat_plus[n+1, T/2+1:T, k])**2).conj().T))/np.sum(np.abs(u_hat_plus[n+1,T/2+1:T,k])**2)

        # Dual ascent
        lambda_hat[n+1,:] = lambda_hat[n,:] + tau*(np.sum(u_hat_plus[n+1,:,:],axis=-1) - f_hat_plus)

        # loop counter
        n = n+1

        # converged yet?
        uDiff = eps
        for i in range(K):
            uDiff = uDiff + 1/T*(np.dot(u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i],np.conj((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i])).conj().T))

        uDiff = np.abs(uDiff)

    #------ Postprocessing and cleanup

    # discard empty space if converged early
    N = np.min((N,n))
    omega = omega_plus[:N,:]

    # Signal reconstruction
    u_hat = np.zeros((T, K))
    u_hat[(T/2+1):T,:] = np.squeeze(u_hat_plus[N,(T/2+1):T,:])
    import pdb; pdb.set_trace()
    u_hat[(T/2+1):1:-1,:] = np.squeeze(np.conj(u_hat_plus[N,(T/2+1):T,:]))
    u_hat[0,:] = np.conj(u_hat[-1,:]) # used end asusumed -1 index

    u = np.zeros(K,len(t))

    for k in range(K):
        u[k,:]=np.real(fftpack.ifft(fftpack.ifftshift(u_hat[:,k])))

    # remove mirror part
    u = u[:,T/4+1:3*T/4]

    # recompute spectrum
    u_hat *= 0
    for k in range(K):
        u_hat[:,k]=fftpack.fftshift(fftpack.fft(u[k,:])).conj().T

    return u, u_hat, omega

# Time Domain 0 to T
T = 1000
fs = 1/T
t = np.arange(T)/T
freqs = 2*np.pi*(t-0.5-1/T)/(fs)

# center frequencies of components
f_1 = 2
f_2 = 24
f_3 = 288

# modes
v_1 = (np.cos(2*np.pi*f_1*t))
v_2 = 1/4*(np.cos(2*np.pi*f_2*t))
v_3 = 1/16*(np.cos(2*np.pi*f_3*t))

## for visualization purposes
#fsub = {}
#wsub = {}
#fsub{1} = v_1
#fsub{2} = v_2
#fsub{3} = v_3
#wsub{1} = 2*np.pi*f_1
#wsub{2} = 2*np.pi*f_2
#wsub{3} = 2*np.pi*f_3

# composite signal, including noise
f = v_1 + v_2 + v_3 + 0.1*np.random.randn(np.size(v_1))
f_hat = fftpack.fftshift((fftpack.fft(f)))

# some sample parameters for VMD
alpha = 2000        # moderate bandwidth constraint
tau = 0            # noise-tolerance (no strict fidelity enforcement)
K = 3              # 3 modes
DC = 0          # no DC part imposed
init = 1          # initialize omegas uniformly
tol = 1e-7

u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
