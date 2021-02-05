"""remd: Trajectory analysis of replica exchange molecular dynamics (REMD) simulations.

"""

#
# Authors
# -------
# Xiping Gong <xipinggong@umass.edu> 02/02/2021
#
#
# Notes
# -----
#

__all__ = ['compute_rg', 'compute_qcontacts']
__version__ = '0.0.1'
__author__ = 'Xiping Gong'

import numpy as np
from scipy.special import expit
import mdtraj as md

def compute_rg(traj, atomsinx=None, masses=None):
    """compute the property: radius of gyration, rg(m,x).

    Parameters
    ----------
    traj : shape = (N,), type = mdtraj.Trajectory.
        It is a mdtraj array and the mdtraj is a Python package.

    atomsinx : shape = (-1,), type = numpy.array, default = None.
        It is an input array of the choosen atom indices, default all atoms.

    masses : shape = (M,), type = numpy.array, default = None.
        It is an input array of all atomic masses, M is the number of atoms.

    Returns
    -------
    rg : shape = (N,), type = numpy.array, unit = nm.
        Rg values for all trajectory frames.

    Notes
    -----
    Suppose a molecular confiration consists of particles and the coordinate and
    effective mass of :math:`k^{th}` particle are :math:`x_{k}` and :math:`mass_{k}`, respectively,
    then the square of radius of gyration can be written as,

    .. math::
        R_{g}^{2}(m,x) = \sum_{k}m_{k}(x_{k}-\sum_{k}m_{k}x_{k})^{2}

        =\sum_{k}m_{k}x_{k}^{2} - (\sum_{k}m_{k}x_{k})^{2},

    where :math:`m_{k} = mass_{k}/\sum_{i}mass_{i}, i, k = 1, 2, ...,M`,
    and :math:`M` is the number of particles.
    When the masses of part of particles are set to zero, it means that we will not take them into account.

    References
    ----------
    .. [1] Radius of gyration: https://en.wikipedia.org/wiki/Radius_of_gyration

    Examples
    --------
    >>> # loading packages
    >>> import xsimma as xsim
    >>> import mdtraj as md
    >>> import glob

    >>> # loading trajectories
    >>> print('Extracting the trajectory files >>')
    >>> wdir = '/home/ping/programs/xsimma/tests' # modify here
    >>> traj_files = wdir + '/data/aaqaa3*.dcd'
    >>> natpdb = wdir + '/data/aaqaa3.pdb'
    >>> traj_files = glob.glob(traj_files)
    >>> traj_files = sorted(traj_files)
    >>> for traj_file in traj_files: print(traj_file)
    >>> traj = md.load(traj_files, top=natpdb)
    >>> print('traj = ', traj)
    Extracting the trajectory files >>
    /home/ping/programs/xsimma/tests/data/aaqaa3aa01.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa02.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa03.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa04.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa05.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa06.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa07.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa08.dcd
    traj =  <mdtraj.Trajectory with 800 frames, 180 atoms, 15 residues, without unitcells>

    >>> # calculating radius of gyration using all atoms and default masses
    >>> rg = xsim.compute_rg(traj)
    >>> print(rg[0:5])
    [1.20029026 1.02205584 1.17694111 1.05258003 1.08400387]

    >>> # calculating radius of gyration using backbone atoms and default masses
    >>> atomsinx = traj[0].topology.select('backbone')
    >>> rg = xsim.compute_rg(traj, atomsinx=atomsinx)
    >>> print(rg[0:5])
    [1.12240211 0.94310565 1.14143905 1.03187016 1.04209682]

    >>> # calculating radius of gyration using backbone atoms and equal masses
    >>> atomsinx = traj[0].topology.select('backbone')
    >>> masses = np.ones(traj[0].n_atoms)
    >>> rg = xsim.compute_rg(traj, masses=masses)
    >>> print(rg[0:5])
    [1.20896776 1.03303262 1.2001014  1.06826148 1.10217055]

    """

    # default particle indices
    if atomsinx is None:
        atomsinx = traj[0].topology.select('all')

    # default particle masses
    if masses is None:
        masses = np.array([a.element.mass for a in traj[0].topology.atoms])

    # particle coordinates and normalized masses
    xyz = traj.xyz[:,atomsinx,:]
    masses = masses[atomsinx]
    masses /= np.sum(masses)

    # radius of gyration (slow ***)
    rg = np.average(xyz, weights=masses, axis=1)
    rg = np.sum(rg*rg, axis=1)
    rg = np.average(np.sum(xyz*xyz, axis=2), weights=masses, axis=1) - rg
    rg = np.sqrt(rg)

    return rg

def compute_qcontacts(traj, reference, pairs, beta=50, gamma=1.8, epsilon=0.0):
    """Compute the property: the number of native contacts, Q(x).

    Parameters
    ----------
    traj : shape = (N,), type = mdtraj.Trajectory.
        It is a mdtraj array and the mdtraj is a Python package.

    reference : shape = (1,), type = mdtraj.Trajectory.
        It is an input trajectory of reference state (e.g., native state).

    pairs : shape = (M,2), type = numpy.array, dtype = int.
        It includes the indices of all pairs, e.g.,
        pairs = numpy.array([[AT1, AT2], [AT3, AT4], ...], dtype=int).

    beta : shape = (1,), default = 50, unit = :math:`nm^{-1}`.
        The exponential smoothing factor.

    gamma : shape = (1,), default = 1.8.
        The scaling factor, 1.8 for all-atom system and 1.2 for coarse-grain system.

    epsilon : shape = (1,), default = 0.0, unit = nm.
        The shifed factor.

    Returns
    -------
    qcontacts : shape = (N,), type = numpy.array.
        The number of native contacts for trajectories.

    Notes
    -----
    Given a trajectory (x), the number of native contacts is defined as, [1]_

    .. math::
        Q(x) = \sum_{(i,j)} 1/(1+e^{\beta [r_{ij}(x) - \lambda * r_{ij}^{0} - \epsilon ]}),

    where the sum loops over the M pairs of native contacts,
    :math:`r_{ij}(x)` is the distance (unit: nm)
    between atom i and j in the pair :math:`(i,j)` of this trajectory,
    :math:`r_{ij}^{0}` is the distance (unit: nm)
    between atom i and j in the pair :math:`(i,j)` of reference trajectory :math:`(x_{ref})`,
    :math:`\beta = 50 nm^{-1}` is a smoothing parameter,
    :math:`\lambda` is used to scale the strength of native contact pair,
    taken to be 1.2 for the coarse-grain and 1.8 for the all-atom system, and
    :math:`\epsilon = 0.0` a shifted fluctuation.

    References
    ----------
    .. [1] Best, R. B., Hummer, G., & Eaton, W. A. (2013),
    Native contacts determine protein folding mechanisms in atomistic simulations.
    Proceedings of the National Academy of Sciences, 110(44), 17874-17879.
    Link: https://www.pnas.org/content/110/44/17874

    Examples
    --------
    >>> # loading packages
    >>> import xsimma as xsim
    >>> import mdtraj as md
    >>> import glob

    >>> # loading trajectories
    >>> print('Extracting the trajectory files >>')
    >>> wdir = '/home/ping/programs/xsimma/tests/data/' # modify here
    >>> traj_files = wdir + 'aaqaa3*.dcd'
    >>> natpdb = wdir + 'aaqaa3.pdb'
    >>> traj_files = glob.glob(traj_files)
    >>> traj_files = sorted(traj_files)
    >>> for traj_file in traj_files: print(traj_file)
    >>> traj = md.load(traj_files, top=natpdb)
    >>> print('traj = ', traj)
    Extracting the trajectory files >>
    /home/ping/programs/xsimma/tests/data/aaqaa3aa01.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa02.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa03.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa04.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa05.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa06.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa07.dcd
    /home/ping/programs/xsimma/tests/data/aaqaa3aa08.dcd
    traj =  <mdtraj.Trajectory with 800 frames, 180 atoms, 15 residues, without unitcells>

    >>> # loading native trajectory
    >>> native = md.load_pdb(natpdb)
    >>> print('native =', native)
    native = <mdtraj.Trajectory with 1 frames, 180 atoms, 15 residues, without unitcells>

    >>> # generating native contact pairs
    >>> pairs = md.baker_hubbard(native)[:,[0,2]] # Baker-Hubbard Hydrogen Bond pairs
    >>> print('pairs.shape = ', pairs.shape)
    >>> for k in range(0,len(pairs)):
    >>>     print('pair '+str(k)+' -> ', pairs[k,:], ' -> ',
    >>>         (native.topology.atom(pairs[k,0]), native.topology.atom(pairs[k,1])))
    pairs.shape =  (9, 2)
    pair 0 ->  [63 25]  ->  (ALA6-N, ALA2-O)
    pair 1 ->  [73 42]  ->  (ALA7-N, GLN3-O)
    pair 2 ->  [83 52]  ->  (GLN8-N, ALA4-O)
    pair 3 ->  [100  62]  ->  (ALA9-N, ALA5-O)
    pair 4 ->  [110  72]  ->  (ALA10-N, ALA6-O)
    pair 5 ->  [120  82]  ->  (ALA11-N, ALA7-O)
    pair 6 ->  [130  99]  ->  (ALA12-N, GLN8-O)
    pair 7 ->  [140 119]  ->  (GLN13-N, ALA10-O)
    pair 8 ->  [157 129]  ->  (ALA14-N, ALA11-O)

    >>> # calculating the number of native contacts
    >>> qcontacts = xsim.compute_qcontacts(traj, native, pairs=pairs)
    >>> print(qcontacts[0:5])
    [0.00342021 0.01861136 0.11058011 0.0003493  0.00252101]

    """

    # the distances (nm) of native contact pairs for reference trajectory
    r_reference = md.compute_distances(reference, pairs)

    # the distances (nm) of native contact pairs for input trajectories and normalized
    r_traj = md.compute_distances(traj, pairs)
    r_traj = r_traj - gamma*r_reference - epsilon
    r_traj *= -beta

    # calculate total native contacts
    qcontacts = np.sum(expit(r_traj), axis=1)

    return qcontacts

