"""plot: plotting 1-D trajectory property data.

"""

#
# Authors
# -------
# Xiping Gong <xipinggong@umass.edu> 02/02/2021
#

import numpy as np
from matplotlib import pyplot as plt

__all__ = ['plttimeseries', 'plthist']

def plttimeseries(dat, nplots=1, nsubplots=1, label='dat', **kwargs):
    """plot time series and histograms of a given property data.

    Parameters
    ----------
    dat : shape = (N,), type = numpy.array.
        It is a 1-D array of a given property.

    nplots : shape = (1,), dtype = int, default = 1.
        It defines how many plots in total.

    nsubplots : shape = (1,), dtype = int, default = 1.
        It defines how many subplots each plot has.

    label : type = string, default = 'dat'.
        This label will be shown in the title of each plot.

    kwargs : pyplot.hist property
        The histogram plotting used the pyplot.hist function.

    Notes
    -----
    Given a property data array, the time series and histograms can be plotted together.
    The common property includes the conditions, total energies, the number of native
    hydrogen bonds, etc.

    Examples
    --------
    >>> # loading packages
    >>> import xsimma as xsim
    >>> import mdtraj as md
    >>> import glob

    >>> # loading data
    >>> print('Extracting the data files >>')
    >>> wdir = '/home/ping/programs/xsimma/tests/data/' # modify here
    >>> dat_files = wdir + 'aaqaa3*.dat'
    >>> dat_files = glob.glob(dat_files)
    >>> dat_files = sorted(dat_files)
    >>> for k in range(0,len(dat_files)): print(dat_files[k])
    >>> dat = [np.loadtxt(f) for f in dat_files]
    >>> dat = np.concatenate(dat)
    >>> aax = np.array(dat[:,1],dtype=int) # replicas from REMD simulations
    >>> condx = np.array(dat[:,2],dtype=int) # conditions from REMD simulations
    >>> energy = dat[:,3] # total energies
    >>> print('aax.shape = ', aax.shape)
    >>> print('condx.shape = ', condx.shape)
    >>> print('energy.shape = ', energy.shape)
    #Extracting the data files >>
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa01.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa02.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa03.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa04.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa05.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa06.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa07.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa08.dat
    #aax.shape =  (800,)
    #condx.shape =  (800,)
    #energy.shape =  (800,)

    >>> # plotting time series of conditions array
    >>> xsim.plttimeseries(condx, nsubplots=8, bins=50, label='condx') # 8 subplots
    >>> xsim.plttimeseries(condx, nplots=2, nsubplots=4, bins=50, label='condx') # 2 plots and 4 subplots for each plot

    """

    ndat = int(dat.shape[0] / (nplots*nsubplots))
    xmin = np.min(dat)
    xmax = np.max(dat)
    dh = (xmax - xmin) * 0.1
    xmin -= dh
    xmax += dh
    k = 0
    for kplot in range(0, nplots):

        # plot each condition
        plt.figure()
        for kseg in range(0,nsubplots):
            # specify data range
            inx = k * ndat
            iny = inx + ndat
            k = k + 1
            xdat = dat[inx:iny]
            # time series
            xpos = kseg * 2 + 1
            plt.subplot(nsubplots,2,xpos)
            if kseg==0: plt.title(label)
            if kseg != nsubplots-1: plt.xticks([])
            if kseg == nsubplots-1: plt.xlabel('cycles')
            plt.plot(xdat)
            plt.ylim([xmin, xmax])
            # histogram
            plt.subplot(nsubplots,2,xpos+1)
            if kseg != nsubplots-1: plt.xticks([])
            if kseg == nsubplots-1: plt.xlabel('histogram')
            if 'histtype' not in kwargs: kwargs.update(histtype='step')
            if 'density' not in kwargs: kwargs.update(density=True)
            plt.hist(xdat, **kwargs)
            plt.xlim([xmin, xmax])


def plthist(dat, nplots=1, nsegs=1, label='dat', **kwargs):
    """plot histogram distributions of a given property data.

    Parameters
    ----------
    dat : shape = (N,), type = numpy.array.
        It is a 1-D array of a given property.

    nplots : shape = (1,), dtype = int, default = 1.
        It defines how many plots in total.

    nsegs : shape = (1,), dtype = int, default = 1.
        It defines how many segments for each plot.

    label : type = string, default = 'dat'.
        This label will be shown in the title of each plot.

    kwargs : pyplot.hist property
        The histogram plotting used the pyplot.hist function.

    Notes
    -----
    Given a property data array, it will divided into multiple segments (nsegs) and
    then their histograms will be plotted together.
    The common property includes the total energies, the number of native
    hydrogen bonds, etc. It is therefore useful to monitor the convergence of multiple
    simulations.

    The average and stdandard deviation of each segment will be shown in the legend.

    Examples
    --------
    >>> # loading packages
    >>> import xsimma as xsim
    >>> import mdtraj as md
    >>> import glob

    >>> # loading data
    >>> print('Extracting the data files >>')
    >>> wdir = '/home/ping/programs/xsimma/tests/data/' # modify here
    >>> dat_files = wdir + 'aaqaa3*.dat'
    >>> dat_files = glob.glob(dat_files)
    >>> dat_files = sorted(dat_files)
    >>> for k in range(0,len(dat_files)): print(dat_files[k])
    >>> dat = [np.loadtxt(f) for f in dat_files]
    >>> dat = np.concatenate(dat)
    >>> aax = np.array(dat[:,1],dtype=int) # replicas from REMD simulations
    >>> condx = np.array(dat[:,2],dtype=int) # conditions from REMD simulations
    >>> energy = dat[:,3] # total energies
    >>> print('aax.shape = ', aax.shape)
    >>> print('condx.shape = ', condx.shape)
    >>> print('energy.shape = ', energy.shape)
    #Extracting the data files >>
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa01.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa02.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa03.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa04.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa05.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa06.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa07.dat
    #/home/ping/programs/xsimma/tests/data/aaqaa3aa08.dat
    #aax.shape =  (800,)
    #condx.shape =  (800,)
    #energy.shape =  (800,)

    >>> # plotting histogram distributions of energy array
    >>> xsim.plthist(energy, nsegs=8, bins=50, label='energy') # 8 segments
    >>> xsim.plthist(energy, nplots=2, nsegs=4, bins=50, label='energy') # 2 plots, 4 segments for each plot

    """

    ndat = int(dat.shape[0] / (nplots*nsegs))
    k = 0
    for kplot in range(0, nplots):

        # loop each plot
        plt.figure()
        for kseg in range(0,nsegs):
            inx = k * ndat
            iny = inx + ndat
            k = k + 1
            xdat = dat[inx:iny]
            xmean = np.mean(xdat)
            xstd = np.std(xdat)
            xlabel = ' inx = ' + str(inx) + ':' + str(iny) +\
                     ' mean = ' + '{:.2f}'.format(xmean) + \
                     ' std = ' + '{:.2f}'.format(xstd)
            if 'histtype' not in kwargs: kwargs.update(histtype='step')
            if 'density' not in kwargs: kwargs.update(density=True)
            plt.hist(dat[inx:iny], label=xlabel, **kwargs)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(label)

