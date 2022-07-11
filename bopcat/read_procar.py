# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import numpy as np


def read(filename='PROCAR'):
    """
    Read the procar and determines the character of each band
    in each k-point in terms of its weight.
    the format of the output:
        weight[kpt][band][ion(last index is total)][orb]
        orb : 0 - s
              1 - py
              2 - pz
              3 - px
              4 - dxy
              5 - dyz
              6 - dz2
              7 - dxz
              8 - dx2
    """
    f = open(filename)
    l = f.readlines()
    f.close()
    # initialize array
    nkpts = int(l[1].split()[3])
    nbands = int(l[1].split()[7])
    nions = int(l[1].split()[11])
    nspin = 0
    for i in range(len(l)):
        if '# of' in l[i]:
            nspin += 1
    if nions > 1:
        nions += 1  # for the total
    weight = np.zeros((nspin, nkpts, nbands, 9))
    sp = -1
    for i in range(len(l)):
        if '# of' in l[i]:
            sp += 1
        if 'k-point ' in l[i]:
            nk = int(l[i].split()[1]) - 1
        if 'band ' in l[i]:
            nb = int(l[i].split()[1]) - 1
        if 'ion ' in l[i]:
            # for j in range(nions):
            temp = l[i + nions].split()
            weight[sp][nk][nb] = temp[1:10]
            tot = float(temp[10])
        # if tot > 0.:
        #    weight[nk][nb] /= float(temp[10])
    return weight
