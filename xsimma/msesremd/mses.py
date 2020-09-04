
import numpy as np
import mdtraj as md
from pymbar import MBAR
from scipy.special import logsumexp
from matplotlib import pyplot as plt


__all__ = ['compute_emses','compute_qcontacts','compute_ultrans',
           'pltaa','compute_nk', 'compute_ukn', 'compute_weights']


def compute_emses(traj,atcg_list,forc=1.0,fmax=0.5,rswi=2.0,sexp=1.0):
    """ 
    Compute the multiscale atomistic(AT) and coarse-grained(CG) coupling energy.

    Parameters
    ----------
    traj : md.Trajectory, shape = (len(traj),)
        the input trajectories
    atcg_list : np.array, shape = (len(atcg_pairs),4)
        the indices of AT and CG atoms,
        for each pair, it follows [AT_atom1, AT_atom2, CG_atom1, CG_atom2]
    forc : np.array, shape = (1,)
    rswi : np.array, shape = (1,)
    sexp : np.array, shape = (1,)
    
    Returns
    -------
    emses : np.array, shape = (len(traj),)
        the multiscale AT-CG coupling energy
    
    Examples
    --------
    
    """
    
    # ignore the warning (divide by zero) and it is safe here
    np.seterr(divide='ignore')
    
    # calculate two variables
    softA = rswi*rswi*(0.5+1/sexp) - fmax*rswi*(1+2/sexp)
    softB = np.power(rswi,sexp+1) * (fmax-rswi)/sexp
    
    # the distance between AT and CG pairs
    dat = md.compute_distances(traj, atcg_list[:,0:2].astype(int))
    dcg = md.compute_distances(traj, atcg_list[:,2:4].astype(int))
    datcg = md.utils.in_units_of(np.abs(dat-dcg),'nanometers','angstrom')
    
    # the multiscale AT-CG coupling energy
    emses = np.where(datcg < rswi,
                     0.5*datcg*datcg,
                     softA + softB/np.power(datcg,sexp) + fmax*datcg)
    emses = forc * np.sum(emses,axis=1)
    
    return emses


def compute_qcontacts(traj, reference, pairs, beta=4, lamb=-1, gamma=-7/4):
    """
    Compute the fraction of pairs contacts using the following 
    definition,

    Q(X) = xx.

    Parameters
    ----------
    traj : md.Trajectory, shape = (len(traj),)
        input trajectory
    reference : md.Trajectory, shape = 1
        input trajectory of reference state (usualy is native state)
    pairs : np.array, shape = (len(pairs),2)

    Returns
    -------
    qc : np.array, shape = (len(traj),)
        The fraction of pairs contacts for each frame in the input trajectory
        qc -> [0,1]

    Examples
    --------

    """

    # the distances of pairs for reference trajectory
    r_reference = md.compute_distances(reference, pairs)
    r_reference = md.utils.in_units_of(r_reference,'nanometers','angstrom')

    # the distances of pairs for input trajectory and normalized
    r_traj = md.compute_distances(traj,pairs)
    r_traj = md.utils.in_units_of(r_traj,'nanometers','angstrom')
    r_traj = r_traj + lamb*r_reference + gamma
    r_traj *= beta
    
    # avoid the "RuntimeWarning: overflow encountered in exp"
    EXPLIM = 50
    r_traj[r_traj>EXPLIM] = EXPLIM
    r_traj[r_traj<-EXPLIM] = -EXPLIM
    
    # calculate the fraction of pairs contacts 
    qc = np.mean(1.0 / (1 + np.exp(r_traj)), axis=1) 

    return qc


def compute_ultrans(dat, upper=1.0, lower=0.0, half_size=0, prnlev=1):
    """
    Compute the reversible upper/lower transitions for a given array.


    Parameters
    ----------
    dat : 1-D array_like
        input array.
    upper : array-like, optional
        all the values (>= upper) will be recorded to a upper status.
    lower : array-like, optional  
        all the values (<= lower) will be recorded to a lower status.
    half_size : array-like, int, optional
        half_size >= 0 is reasonable,
        the range dat[(max(0,pos-half_size), min(len(dat),pos+half_size+1)] around the pos index 
        will be averged and considered as a status.   
    prnlev : int, optional
        prnlev = 0, the output will not be printed,
        prnlev = 1, the transition status will be printed

    Returns
    -------
    ultrans : array-like
        the total number of reversible upper/lower transitions

    Examples
    --------
    >>> compute_ultrans([1,2,3,5,4,3,1,4,3,6],upper=3,lower=2,half_size=0,prnlev=1)
    record transition status: index, cur_value, avg_value =  2 3 3.0
    record transition status: index, cur_value, avg_value =  6 1 1.0
    record transition status: index, cur_value, avg_value =  7 4 4.0
    1

    """
    
    # the length of 1-D array
    ndat = len(dat)

    # compute the initial status 
    ix = 0
    iy = min(ndat, half_size)
    x1 = np.mean(dat[ix:iy])
    if x1 >= upper: 
        flag_init = 1
    else:
        flag_init = -1
 
    ultrans = 0
    flag_last = flag_init
    flag_cur = flag_init
    for i in range(0,ndat):

        # compute the current status
        ix = max(0, i - half_size)
        iy = min(ndat, i + half_size + 1)
        x1 = np.mean(dat[ix:iy])
        if x1 >= upper: 
            flag_cur = 1
        elif x1 <= lower: 
            flag_cur = -1

        # the current status needs to be recorded?        
        if flag_cur != flag_last:
            flag_last = flag_cur
            if prnlev > 0: print('record transition status: ' + 'index, cur_value, avg_value = ', i, dat[i], x1)
            if flag_last == flag_init: ultrans += 1


    return ultrans


def pltaa(dat, bins=np.arange(-0.5,8.5), ns=50000, ne=100000, nframe=100000, nrow=8):
    """
    plot timeseries and histogram of all simulations

    """

    ncol = 3

    runs = ['ctrl','fold']
    for k in range(0,nrow):
        # time series
        # ctrl
        plt.subplot(nrow,ncol,k*ncol+1)
        inx = k*nframe
        iny = k*nframe + nframe
        plt.plot(dat[0,inx:iny])
        if k == 0: plt.title('ctrl')
        plt.ylabel('aa'+str(k+1))
        # fold
        plt.subplot(nrow,ncol,k*ncol+2)
        inx = k*nframe
        iny = k*nframe + nframe
        plt.plot(dat[1,inx:iny])
        if k == 0: plt.title('fold')

        # histogram
        plt.subplot(nrow,ncol,k*ncol+3)
        # ctrl
        inx = k*nframe + ns
        iny = k*nframe + ne
        plt.hist(dat[0,inx:iny], bins=bins, density=True, histtype='step', label=runs[0])
        # fold
        inx = k*nframe + ns
        iny = k*nframe + ne
        plt.hist(dat[1,inx:iny], bins=bins, density=True, histtype='step', label=runs[1])
        if k == 0: plt.title('histogram: '+str(ns)+'-'+str(ne)); plt.legend(loc=9)


def compute_nk(xn, conds, temperature,return_counts=True):
    """
    caclulate the number of samples (nk)

    """

    ncond = conds.shape[0]

    uni_temperature,count_temperature=np.unique(temperature[xn],return_counts=True)
    nk = count_temperature
    for k in range(0,ncond):
        temp = uni_temperature[k]
        print(str(k+1)+'th simulation: temperature, lambda_mses, #sample = ',
            temp,conds[conds[:,0]==temp,1],nk[k])

    return nk


def compute_ukn(xn, conds, temperature, energy, energy_mses):
    """
    calculate the relative free energies (fk)

    """

    # variables
    kB = 0.593/298
    ncond = conds.shape[0]
    nk_sum = xn.shape[0]

    # calculate the ukn = uk(xn)
    # uk(x) = beta_k * (V_AT(x) + V_CG(x) + lambda_k*V_MSES(x))
    #    = beta_k * (V_ATCG(x) + lambda_k*V_MSES(x))
    scale = np.zeros(temperature.shape)
    for j in range(0,ncond): scale[temperature==conds[j,0]]=conds[j,1]
    energy_atcg = energy - scale * energy_mses
    beta_k = 1/(kB * conds[:,0])
    lambda_k = conds[:,1]
    ukn = \
        np.matmul(np.reshape(beta_k,(ncond,1)),np.reshape(energy_atcg[xn],(1,nk_sum))) + \
        np.matmul(np.reshape(beta_k*lambda_k,(ncond,1)),np.reshape(energy_mses[xn],(1,nk_sum)))

    return ukn


def compute_weights(xn, u0, conds, temperature, energy, energy_mses):
    """
    calculate the relative weights using MBAR
    
    """

    # 1) calculate nk
    nk = compute_nk(xn, conds, temperature)

    # 2) calculate the ukn = uk(xn)
    # uk(x) = beta_k * (V_AT(x) + V_CG(x) + lambda_k*V_MSES(x))
    ukn = compute_ukn(xn, conds, temperature, energy, energy_mses)

    # 3) calculate the fk
    mbar = MBAR(ukn, nk)
    fk = mbar.f_k
    print('target free energies of all baised simulations = ',fk)

    # 4) calculate the target weights
    wn = np.zeros(energy.shape)
    wn[xn] = np.exp(-logsumexp(fk-(ukn-u0).T,b=nk,axis=1))
    
    return wn


