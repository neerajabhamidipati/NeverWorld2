"""
Functions for filtering 2D model output to coarser resolution
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
import matplotlib.pylab as pylab

def filterSpec(N,dxMin,Lf,shape="Gaussian",X=np.pi):
    """
    Inputs: 
    N is the number of total steps in the filter
    dxMin is the smallest grid spacing - should have same units as Lf
    Lf is the filter scale, which has different meaning depending on filter shape
    shape can currently be one of two things:
        Gaussian: The target filter has kernel ~ e^{-|x/Lf|^2}
        Taper: The target filter has target grid scale Lf. Smaller scales are zeroed out. 
               Scales larger than pi*Lf/2 are left as-is. In between is a smooth transition.
    X is the width of the transition region in the "Taper" filter; per the CPT Bar&Prime doc the default is pi.
    Note that the above are properties of the *target* filter, which are not the same as the actual filter.
    
    Outputs:
    NL is the number of Laplacian steps
    sL is s_i for the Laplacian steps; units of sL are one over the units of dxMin and Lf, squared
    NB is the number of Biharmonic steps
    sB is s_i for the Biharmonic steps; units of sB are one over the units of dxMin and Lf, squared
    """
    # Code only works for N>2
    if N <= 2:
        print("Code requires N>2")
        return 
    # First set up the mass matrix for the Galerkin basis from Shen (SISC95)
    M = (np.pi/2)*(2*np.eye(N-1) - np.diag(np.ones(N-3),2) - np.diag(np.ones(N-3),-2))
    M[0,0] = 3*np.pi/2
    # The range of wavenumbers is 0<=|k|<=sqrt(2)*pi/dxMin. Nyquist here is for a 2D grid. 
    # Per the notes, define s=k^2.
    # Need to rescale to t in [-1,1]: t = (2/sMax)*s -1; s = sMax*(t+1)/2
    sMax = 2*(np.pi/dxMin)**2
    # Set up target filter
    if shape == "Gaussian":
        F = lambda t: np.exp(-(sMax*(t+1)/2)*(Lf/2)**2)
    elif shape == "Taper":
        F = interpolate.PchipInterpolator(np.array([-1,(2/sMax)*(np.pi/(X*Lf))**2 -1,(2/sMax)*(np.pi/Lf)**2 -1,2]),np.array([1,1,0,0]))
    else:
        print("Please input a valid shape")
        return
    # Compute inner products of Galerkin basis with target
    b = np.zeros(N-1)
    points, weights = np.polynomial.chebyshev.chebgauss(N+1)
    for i in range(N-1):
        tmp = np.zeros(N+1)
        tmp[i] = 1
        tmp[i+2] = -1
        phi = np.polynomial.chebyshev.chebval(points,tmp)
        b[i] = np.sum(weights*phi*(F(points)-((1-points)/2 + F(1)*(points+1)/2)))
    # Get polynomial coefficients in Galerkin basis
    cHat = np.linalg.solve(M,b)
    # Convert back to Chebyshev basis coefficients
    p = np.zeros(N+1)
    p[0] = cHat[0] + (1+F(1))/2
    p[1] = cHat[1] - (1-F(1))/2
    for i in range(2,N-1):
        p[i] = cHat[i] - cHat[i-2]
    p[N-1] = -cHat[N-3]
    p[N] = -cHat[N-2]
    # Now plot the target filter and the approximate filter
    x = np.linspace(-1,1,10000)
    k = np.sqrt((sMax/2)*(x+1))
    plt.plot(k,F(x),'g',label='target filter',linewidth=4)
    plt.plot(k,np.polynomial.chebyshev.chebval(x,p),'m',label='approximation',linewidth=4)
    left, right = plt.xlim()
    plt.xlim(left=0)
    bottom,top = plt.ylim()
    plt.ylim(bottom=-0.1)
    plt.ylim(top=1.1)
    plt.xlabel('k', fontsize=18)
    plt.grid(True)
    plt.legend()
    
    # Get roots of the polynomial
    r = np.polynomial.chebyshev.chebroots(p)
    # convert back to s in [0,sMax]
    s = (sMax/2)*(r+1)
    # Separate out the real and complex roots
    NL = np.size(s[np.where(np.abs(np.imag(r)) < 1E-12)]) 
    sL = np.real(s[np.where(np.abs(np.imag(r)) < 1E-12)])
    NB = (N - NL)//2
    sB_re,indices = np.unique(np.real(s[np.where(np.abs(np.imag(r)) > 1E-12)]),return_index=True)
    sB_im = np.imag(s[np.where(np.abs(np.imag(r)) > 1E-12)])[indices]
    sB = sB_re + sB_im*1j
    return NL,sL,NB,sB

def simple_Laplacian(phi,wetMask):

    try:
        from cupy import get_array_module as _get_array_module
    except ImportError:
        import numpy as np

        def _get_array_module(*args):
            return np	
    xp = _get_array_module(phi)
    """Laplacian for regular grid.
    
    Parameters
    ----------
    phi : array_like
    wetMask: array_like, same dimensions as phi
    
    Returns
    -------
    array_like
        Laplacian of `phi`
    """
    out = phi.copy()
    out = xp.nan_to_num(out) 
    out = wetMask * out 
    
    fac = (xp.roll(wetMask, -1, axis=-1) 
            + xp.roll(wetMask, 1, axis=-1) 
            + xp.roll(wetMask, -1, axis=-2) 
            + xp.roll(wetMask, 1, axis=-2)  
    )
        
    out = (- fac * out
            + xp.roll(out, -1, axis=-1) 
            + xp.roll(out, 1, axis=-1) 
            + xp.roll(out, -1, axis=-2) 
            + xp.roll(out, 1, axis=-2)
    )
        
    out = wetMask * out
    return out


