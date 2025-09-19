import numpy as np
import math
import numpy.matlib
from scipy import signal
import powerlaw


def mle (x, xmin=None, xmax=None):
    if (xmin==None):
        xmin = np.min(x)
    if (xmax==None):
        xmax = np.max(x)
    tauRange=np.array([1,5])
    precision = 10**(-3)
    # Error check the precision
    if math.log10(precision) != round(math.log10(precision)):
        print('The precision must be a power of ten.')
    
    x = np.reshape(x, len(x))
    
#Determine data type
    if np.count_nonzero(np.absolute(x - np.round(x)) > 3*(np.finfo(float).eps)) > 0:
        dataType = 'CONT'
    else:
        dataType = 'INTS'
        x = np.round(x)

#     print(dataType)
    #Truncate
    z = x[(x>=xmin) & (x<=xmax)]
    unqZ = np.unique(z)
    nZ = len(z)
    nUnqZ = len(unqZ)
    allZ = np.arange(xmin,xmax+1)
    nallZ = len(allZ)
    
    #MLE calculation
    
    r = xmin / xmax
    nIterations = int(-math.log10(precision))
    
    for iIteration in range(1, nIterations+1):
        
        spacing = 10**(-iIteration)
        
        if iIteration == 1:
            taus = np.arange(tauRange[0], tauRange[1]+spacing, spacing)
            
        else: 
            if tauIdx == 0:
                taus = np.arange(taus[0], taus[1]+spacing, spacing)
                #return (taus,0,0,0)
            elif tauIdx == len(taus):    
                taus = np.arange(taus[-2], taus[-1]+spacing, spacing)#####
            else:
                taus = np.arange(taus[tauIdx-1], taus[tauIdx+1]+spacing, spacing)

        #return(dataType)        
        nTaus = len(taus)
        
        if dataType=='INTS':
            #replicate arrays to equal size
            allZMat = np.matlib.repmat(np.reshape(allZ,(nallZ,1)),1,nTaus)
            tauMat = np.matlib.repmat(taus,nallZ,1)
        
            #compute the log-likelihood function
            #L = - np.log(np.sum(np.power(allZMat,-tauMat),axis=0)) - (taus/nZ) * np.sum(np.log(z))
            L = - nZ*np.log(np.sum(np.power(allZMat,-tauMat),axis=0)) - (taus) * np.sum(np.log(z))
            
            
        elif dataType=='CONT':
            #return (taus,r, nZ,z)
            L = np.log( (taus - 1) / (1 - r**(taus - 1)) )- taus * (1/nZ) * np.sum(np.log(z)) - (1 - taus) * np.log(xmin)
            
            if numpy.in1d(1,taus):
                L[taus == 1] = -np.log(np.log(1/r)) - (1/nZ) * np.sum(np.log(z))
        tauIdx=np.argmax(L)
        
    tau = taus[tauIdx]
    
#     return (taus,L,tau)
    return ([tau, L[tauIdx]])

def get_mexican_hat_kernel(T,J=None):
    import numpy as np
    if (J==None):
        J=4*T
    sigma1 = T;
    mu=0
    x=np.arange(-4*J,4*J,0.001)
    a1 = 1/(sigma1*np.sqrt(2*np.pi))
    k1 = a1*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma1, 2.)))
    #A1=np.trapz(x,k1)
    #k1 = k1/A1

    sigma2 = np.sqrt(np.power(T, 2.) + np.power(J, 2.))
    a2 = 1/(sigma2*np.sqrt(2*np.pi))
    k2=a2*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma2, 2.)))
    #A2=np.trapz(x,k2)
    #k2 = k2/A2

    k = k1-k2
    return (k)


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))







def plot_pdf(data, ax=None, original_data=False,
             linear_bins=False, **kwargs):
    """
    Plots the probability density function (PDF) or the data to a new figure
    or to axis ax if provided.
    Parameters
    ----------
    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.
    original_data : bool, optional
        Whether to use all of the data initially passed to the Fit object.
        If False, uses only the data used for the fit (within xmin and
        xmax.)
    linear_bins : bool, optional
        Whether to use linearly spaced bins (True) or logarithmically
        spaced bins (False). False by default.
    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    return plot_pdf2(data, ax=ax, linear_bins=linear_bins, **kwargs)


def plot_pdf2(data, ax=None, linear_bins=False, **kwargs):
    """
    Plots the probability density function (PDF) to a new figure or to axis ax
    if provided.
    Parameters
    ----------
    data : list or array
    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.
    linear_bins : bool, optional
        Whether to use linearly spaced bins (True) or logarithmically
        spaced bins (False). False by default.
    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    bins = None

    edges, hist = pdf(data, linear_bins=linear_bins, bins=bins, **kwargs)
    bin_centers = (edges[1:]+edges[:-1])/2.0
    from numpy import nan
    hist[hist==0] = nan
    if not ax:
        import matplotlib.pyplot as plt
        plt.plot(bin_centers, hist, **kwargs)
        ax = plt.gca()
    else:
        ax.plot(bin_centers, hist, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return bin_centers, hist





def trim_to_range(data, xmin=None, xmax=None, **kwargs):
    """
    Removes elements of the data that are above xmin or below xmax (if present)
    """
    from numpy import asarray
    data = asarray(data)
    if xmin:
        data = data[data>=xmin]
    if xmax:
        data = data[data<=xmax]
    return data

def pdf(data, xmin=None, xmax=None, linear_bins=False, bins=None, **kwargs):
    """
    Returns the probability density function (normalized histogram) of the
    data.
    Parameters
    ----------
    data : list or array
    xmin : float, optional
        Minimum value of the PDF. If None, uses the smallest value in the data.
    xmax : float, optional
        Maximum value of the PDF. If None, uses the largest value in the data.
    linear_bins : float, optional
        Whether to use linearly spaced bins, as opposed to logarithmically
        spaced bins (recommended for log-log plots).
    Returns
    -------
    bin_edges : array
        The edges of the bins of the probability density function.
    probabilities : array
        The portion of the data that is within the bin. Length 1 less than
        bin_edges, as it corresponds to the spaces between them.
    """
    from numpy import logspace, histogram, floor, unique,asarray
    from math import ceil, log10
    data = asarray(data)
    if not xmax:
        xmax = max(data)
    if not xmin:
        xmin = min(data)

    if xmin<1:  #To compute the pdf also from the data below x=1, the data, xmax and xmin are rescaled dividing them by xmin.
        xmax2=xmax/xmin
        xmin2=1
    else:
        xmax2=xmax
        xmin2=xmin

    if bins is not None:
        bins = bins
    elif linear_bins:
        bins = range(int(xmin2), ceil(xmax2)+1)
    else:
        log_min_size = log10(xmin2)
        log_max_size = log10(xmax2)
        number_of_bins = ceil((log_max_size-log_min_size)*10)
        bins = logspace(log_min_size, log_max_size, num=number_of_bins)
        bins[:-1] = floor(bins[:-1])
        bins[-1] = ceil(bins[-1])
        bins = unique(bins)

    if xmin<1: #Needed to include also data x<1 in pdf.
        hist, edges = histogram(data/xmin, bins, density=True)
        edges=edges*xmin # transform result back to original
        hist=hist/xmin # rescale hist, so that np.sum(hist*edges)==1
    else:
        hist, edges = histogram(data, bins, density=True)

    return edges, hist



def construct_phase(t,tk,tk1):
    if(tk1-tk)==0:
        return 0.0
    else:
        return 2.0*np.pi*(float(t-tk)/float(tk1-tk))

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]