# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import numpy as np
from copy import deepcopy
from .plotting import plot_electronic_band_structure


def arrange_bands(eig, orb, required):
    """
    Arrange the eigenvalues with respect to required.
        
    eig[nbands] is 1-d array of eigenvalues, orb[nbands][9] 
    is 2-d array of orbital character projected unto 
    s0, p-1, p0, p1, d-2, d-1, d0, d1, d2.
    
    Due to hybridization, it is not sufficient to simply
    assign each band to the orbital with highest character.
    Each band is first sorted in decreasing character and 
    subsequently assigned to a particular orbital.
    """
    # get required occupied bands
    iocc = []
    ureq = []
    for i in required:
        if i not in ureq:
            ureq.append(i)
    count = 0
    lasti = 0
    minsum = 0.5
    while len(iocc) < len(required):
        count += 1
        minsum -= 0.1 * count
        for i in range(lasti, len(eig)):
            sumreq = 0.0
            for j in ureq:
                sumreq += orb[i][j]
            # determine if orb is occupied
            if np.sum(orb[i]) == 0.0:
                continue
            if sumreq / np.sum(orb[i]) > minsum and i not in iocc:
                iocc.append(i)
                lasti = i
            if len(iocc) >= len(required):
                break
    # iocc is list of relevant bands, but should be arranged
    # in order of required 
    eig = np.array([eig[i] for i in iocc])
    oorb = np.array([orb[i] for i in iocc])
    # disregard minor differences in orb_char
    # orb = np.around(oorb,5)
    brequired = list(range(len(eig)))
    crequired = list(required)
    # sort bands by decreasing character 
    new_eig = []
    new_orb = []
    req_pb = []
    for i in range(9):
        if i not in required:
            continue
        # if i in [7,8]:
        #    continue
        # elif i == 4:
        #    p = 8
        # elif i == 5:
        #    p = 7
        # else:
        #    p = i
        c_b = []
        # determine which bands has i character
        minw = 0.5
        count = 0
        done = []
        while len(c_b) < 1 * required.count(i) + 3:
            minw -= 0.1 * count
            for b in range(len(eig)):
                cw = oorb[b][i] / np.sum(orb[b])
                # if i != p:
                #    cw += orb[b][p]/np.sum(orb[b])
                if cw >= minw and b not in done:
                    c_b.insert(0, (cw, b))
                    done.append(b)
            count += 1
        # c_b.sort(reverse=True)
        bi = np.array(c_b, dtype=int).T[1]
        bi = list(bi[::-1])
        req_pb.append((len(bi), i, bi))
        # if i != p:
        #    req_pb.append((len(bi),p,bi))
    req_pb.sort()
    req_b = []
    for i in range(len(req_pb)):
        temp = []
        for j in range(len(req_pb[i][2])):
            if req_pb[i][2][j] in brequired:
                temp.append(req_pb[i][2][j])
                brequired.pop(brequired.index(req_pb[i][2][j]))
            if len(temp) == required.count(req_pb[i][1]):
                break
        temp.sort()
        req_b.append((req_pb[i][1], temp))
    brequired = list(range(len(eig)))
    for i in required:
        for j in range(len(req_b)):
            if req_b[j][0] == i:
                # print '>>>', i, req_b[j][1]
                for k in req_b[j][1]:
                    if k in brequired:
                        # print i, k
                        new_eig.append(eig[k])
                        new_orb.append(oorb[k][i])
                        brequired.pop(brequired.index(k))
                        break
                break
    # assign weighted average of d (m=1,-1 and m=2,-2)
    """
    done = []
    for i in range(len(required)):
        # find consecutive pairs with indices 4-8 and 5-7
        if required[i] not in [4,5]:
            continue
        if i not in done: 
            if required[i]==4:
                p = 8
            elif required[i]==5:
                p = 7
        for j in range(len(required)):
            if required[j] not in [7,8]:
                continue
            if j not in done and required[j]==p:
                break
        done.append(i)
        done.append(j)
        ave = (new_eig[i] + new_eig[j])/2.
        new_eig[i] = ave
        new_eig[j] = ave
    """
    return new_eig, new_orb


def _arrange_bands(eig, orb, required):
    iocc = []
    ureq = []
    for i in required:
        if i not in ureq:
            ureq.append(i)
    count = 0
    lasti = 0
    minsum = 0.5
    while len(iocc) < len(required):
        count += 1
        minsum -= 0.1 * count
        for i in range(lasti, len(eig)):
            sumreq = 0.0
            for j in ureq:
                sumreq += orb[i][j]
            # determine if orb is occupied
            if np.sum(orb[i]) == 0.0:
                continue
            if sumreq / np.sum(orb[i]) > minsum and i not in iocc:
                iocc.append(i)
                lasti = i
            if len(iocc) >= len(required):
                break
    # iocc is list of relevant bands, but should be arranged
    # in order of required 
    eig = np.array([eig[i] for i in iocc])
    oorb = np.array([orb[i] for i in iocc])
    # disregard minor differences in orb_char
    orb = np.around(oorb, 2)
    brequired = list(range(len(eig)))
    crequired = list(required)
    # print '++++++++++++++++', np.shape(orb), iocc
    # for i in range(len(oorb)):
    #    for j in range(len(oorb[i])):
    #        print '%6.3f'%oorb[i][j],
    #    print '\n'
    new_eig = []
    new_orb = []
    req_pb = []
    for i in ureq:
        c_b = []
        # determine which bands has i character
        minw = 0.5
        count = 0
        done = []
        while len(c_b) < required.count(i) + 3:
            minw -= 0.1 * count
            for b in range(len(eig)):
                cw = orb[b][i] / np.sum(orb[b])
                if cw >= minw and b not in done:
                    c_b.append((cw, b))
                    done.append(b)
            count += 1
        c_b.sort(reverse=True)
        bi = list(np.array(c_b, dtype=int).T[1])
        req_pb.append((len(bi), i, bi))
    req_pb.sort()
    req_b = []
    for i in range(len(req_pb)):
        # print i, req_pb[i]
        temp = []
        for j in range(len(req_pb[i][2])):
            if req_pb[i][2][j] in brequired:
                temp.append(req_pb[i][2][j])
                brequired.pop(brequired.index(req_pb[i][2][j]))
            if len(temp) == required.count(req_pb[i][1]):
                break
        temp.sort()
        req_b.append((req_pb[i][1], temp))

    brequired = list(range(len(eig)))
    for i in required:
        for j in range(len(req_b)):
            if req_b[j][0] == i:
                for k in req_b[j][1]:
                    if k in brequired:
                        # print i, k
                        new_eig.append(eig[k])
                        new_orb.append(oorb[k][i])
                        brequired.pop(brequired.index(k))
                break
    # print np.shape(new_eig), np.shape(new_orb), iocc
    return new_eig, new_orb


def ____arrange_bands(eig, orb, required):
    iocc = []
    ureq = []
    for i in required:
        if i not in ureq:
            ureq.append(i)
    count = 0
    lasti = 0
    minsum = 0.5
    while len(iocc) < len(required):
        count += 1
        minsum -= 0.1 * count
        for i in range(lasti, len(eig)):
            sumreq = 0.0
            for j in ureq:
                sumreq += orb[i][j]
            # determine if orb is occupied
            if np.sum(orb[i]) == 0.0:
                continue
            if sumreq / np.sum(orb[i]) > minsum and i not in iocc:
                iocc.append(i)
                lasti = i
            if len(iocc) >= len(required):
                break
    """
    iocc = []
    # get first relevant bands
    crequired = list(required)
    for i in range(len(eig)):
        maxc = list(orb[i]).index(np.amax(orb[i]))
        if maxc in crequired:
            print maxc, '<<<<<'
            iocc.append(i) 
            crequired.pop(crequired.index(maxc))
            if len(iocc) == len(required):
                break
    count = 0
    lasti = 0
    minsum = 0.5
    # sometimes bands are heavily hybridized
    while len(iocc) < len(required):
      for i in range(len(eig)):
        if i not in iocc:
            count += 1
            minsum -= 0.1*count
            sumb = 0.
            for j in range(9):
                if j in required:
                    sumb += orb[i][j]
            if sumb/np.sum(orb[i]) >= minsum:
                if i in crequired:
                iocc.append(i)
    iocc.sort()         
    """
    # iocc is list of relevant bands, but should be arranged
    # in order of required 
    eig = np.array([eig[i] for i in iocc])
    oorb = np.array([orb[i] for i in iocc])
    # disregard minor differences in orb_char
    orb = np.around(oorb, 2)
    brequired = list(range(len(eig)))
    crequired = list(required)

    new_eig = []
    new_orb = []
    """
    for i in required:
        c_b = []
        # determine which bands has i character
        for b in range(len(eig)):
          if b in brequired:
            #if list(orb[b]).index(np.amax(orb[b])) == i:
            if orb[b][i]/np.sum(orb[b]) >= 0.5:    
                c_b.append((orb[b][i],b))
        c_b.sort(reverse=True)
        if len(c_b) < 1:
            bi2 = []
            for b in brequired:
                if orb[b][i] > 0.0:
                #    if orb[b][i]/np.amax(orb[b]) > 
                        c_b.append((orb[b][i]/np.sum(orb[b]),b))
            c_b.sort(reverse=True)
            print i, c_b,brequired
            bi = list(np.array(c_b,dtype=int).T[1])
            #if c_b[0][0]/c_b[-1][0] > 0.8:
            #  # hard to distinguish p-1,p1 and d-2,d2 and d-1,d1
            #  # we always assign lower bands to - and higher bands to +
            #  bi = bi[:2]
            #  bi.sort()
            #  if i in [1,4,5]:
            #      bi = [bi[0]]
            #  elif i in [3,7,8]:
            #      bi = [bi[-1]]
        else:
            bi = list(np.array(c_b,dtype=int).T[1])
            bi.sort()
            bi = bi[:1]
        print i, bi[0]
        new_eig.append(eig[bi[0]])
        new_orb.append(oorb[bi[0]][i])
        brequired.pop(brequired.index(bi[0]))
        crequired.pop(crequired.index(i))
    """
    req_pb = []
    for i in ureq:
        c_b = []
        # determine which bands has i character
        for b in range(len(eig)):
            cw = orb[b][i] / np.sum(orb[b])
            if cw >= 0.2:
                c_b.append((cw, b))
        c_b.sort(reverse=True)
        bi = list(np.array(c_b, dtype=int).T[1])
        req_pb.append((len(bi), i, bi))

    req_pb.sort()
    req_b = []
    for i in range(len(req_pb)):
        temp = []
        for j in range(required.count(req_pb[i][1])):
            if req_pb[i][2][j] in brequired:
                temp.append(req_pb[i][2][j])
                brequired.pop(brequired.index(req_pb[i][2][j]))
        temp.sort()
        req_b.append((req_pb[i][1], temp))

    brequired = list(range(len(eig)))
    for i in required:
        for j in range(len(req_b)):
            if req_b[j][0] == i:
                for k in req_b[j][1]:
                    if k in brequired:
                        new_eig.append(eig[k])
                        new_orb.append(oorb[k][i])
                        brequired.pop(brequired.index(k))
                break
    return new_eig, new_orb


def __arrange_bands(eig, orb, required, debug=False):
    iocc = []
    ureq = []
    for i in required:
        if i not in ureq:
            ureq.append(i)
    count = 0
    lasti = 0
    minsum = 0.5
    while len(iocc) < len(required):
        count += 1
        minsum -= 0.1 * count
        for i in range(lasti, len(eig)):
            sumreq = 0.0
            for j in ureq:
                sumreq += orb[i][j]
            # determine if orb is occupied
            if np.sum(orb[i]) == 0.0:
                continue
            if sumreq / np.sum(orb[i]) > minsum and i not in iocc:
                iocc.append(i)
                lasti = i
            if len(iocc) >= len(required):
                break
        # sometimes bopfox calculates zero weight for a band for all orb
        if minsum <= 0:
            lasti = lasti + 1
            iocc.append(lasti)
    """
    # loop over required orbital types
    ieigs = []
    for i in range(len(required)):
        maxw = 0.0
        for j in iocc:
            if orb[j][required[i]] > maxw and j not in ieigs:
                maxw = orb[j][required[i]]
                ei = j
        ieigs.append(ei)
    new_eig = []
    new_orb = []
    for i in range(len(ieigs)):
        new_eig.append(eig[ieigs[i]])
        #orbi = [orb[i][j]/np.sum(orb[i]) for j in range(len(orb[i]))]
        orbi = orb[ieigs[i]][required[i]]
        new_orb.append(orbi)
    new_eig = np.array(new_eig)
    new_orb = np.array(new_orb)

#    temp = zip(new_eig,new_orb)
#    temp.sort()
#    temp = np.array(temp).T
#    new_eig = temp[0]
#    new_orb = temp[1]    
    """
    # determine character of each band
    ieigs = list(iocc)
    orbc = list(required)
    temp = []
    iorb = []
    for i in ieigs:
        maxw = 0.0
        for j in required:
            if orb[i][j] >= maxw:
                if j in orbc:
                    maxw = orb[i][j]
                    maxj = j
                else:
                    # check if weight of assigned orb is greater than current
                    # get all orbs and check largest
                    # temp list of orbchar of band
                    continue
                    # max2w = maxw
                    # for k in range(len(temp)):
                    #    if temp[k] == j:
                    #        if orb[k][j] > max2w:
                    #            max2w = orb[k][j]
                    #            max2j = k
                    # if max2w > np.amax(np.array(orb).T[j]):
                    #    print i,j,max2j,max2w,maxj
                    #    #raise
        # print i, j
        temp.append(maxj)
        iorb.append(orbc.pop(orbc.index(maxj)))
        # print '<<<<<', i, maxj, orb[i][maxj]
    ieigs = list(temp)
    # print ieigs
    # raise
    """
    ieigs = list(iocc)       
    orbc = list(ieigs)
    temp = []
    for j in required:
        maxw = 0.0
        for i in ieigs:
            if orb[i][j] > maxw and i in orbc:
                maxw = orb[i][j] 
                maxi = i  
        temp.append(maxi)
        orbc.pop(maxi) 
        print j, maxi   
    ieigs = list(temp)
    """
    # arrange bands according to required
    temp = []
    for i in range(len(required)):
        for j in range(len(ieigs)):
            if required[i] == ieigs[j] and iocc[j] not in temp:
                temp.append(iocc[j])
                break

    ieigs = list(temp)

    new_eig = []
    new_orb = []
    for i in range(len(ieigs)):
        new_eig.append(eig[ieigs[i]])
        new_orb.append(orb[ieigs[i]][iorb[i]])

    return new_eig, new_orb


def _arrange_bands(eig, orb, required, debug=False):
    c_e = np.zeros((len(eig), 3))
    DONE = []
    done = []
    for i in range(len(eig)):
        if i % len(orb[i]) == 0:
            if len(done) > 0: DONE.append(list(done))
            done = []
        temp = [[orb[i][j], j] for j in range(len(orb[i]))]
        temp.sort()
        if (temp[-1][0] - temp[-2][0]) > 0.3:
            c_e[i] = [temp[-1][1], eig[i], temp[-1][0]]
            done.append(temp[-1][1])
            continue
        start = int(i / len(orb[i])) * len(orb[i])
        end = start + len(orb[i])
        if end > len(eig):
            end = len(eig)
        for j in range(len(temp) - 1, -1, -1):
            add = True
            for k in range(len(eig)):
                if orb[k][temp[j][1]] > temp[j][0]:
                    add = False
            if add and temp[j][1] not in done:
                done.append(temp[j][1])
                c_e[i] = [temp[j][1], eig[i], temp[j][0]]
                break
        if not add:
            c_e[i] = [100, eig[i], 100]
    DONE.append(done)
    for i in range(len(c_e)):
        if c_e[i][0] == 100:
            temp = [[orb[i][j], j] for j in range(len(orb[i]))]
            temp.sort()
            for j in range(len(temp) - 1, -1, -1):
                if temp[j][1] not in DONE[int(i / len(orb[i]))]:
                    c_e[i] = [temp[j][1], c_e[i][1], temp[j][0]]
                    DONE[i / len(orb[i])].append(temp[j][1])
                    break
    new_eig = []
    new_orb = []
    done = [False] * len(c_e)
    for i in required:
        test = False
        for j in range(len(c_e)):
            if i == c_e[j][0] and not done[j]:
                test = True
                done[j] = True
                break
        if test:
            new_eig.append(c_e[j][1])
            new_orb.append(c_e[j][2])
        else:
            # extraordinary case when assignment of bands fail
            for k in range(len(eig)):
                if orb[k][i] > 0.1:
                    new_eig.append(eig[k])
                    new_orb.append(orb[k][i])
                    break
    new_eig = np.array(new_eig)
    new_orb = np.array(new_orb)
    return new_eig, new_orb


def sort_bands(dft, bop, orb):
    temp = list(zip(dft, orb))
    temp.sort()
    temp = np.array(temp).T
    dft = temp[0]
    orb = temp[1]
    bop = list(bop)
    bop.sort()
    bop = np.array(bop)
    return dft, bop, orb


def get_relevant_bands_atom(dft_eigs, bop_eigs, orb_char, required):
    # eigs have shape: [nkpts][nbands]
    temp = []
    tempo = []
    for i in range(len(dft_eigs)):
        ne, no = arrange_bands(dft_eigs[i], orb_char[i], required=required)
        be = bop_eigs[i]
        # ne,be,no = sort_bands(ne,be,no)
        for j in range(len(no)):
            if no[j] < 0.0:
                ne[j] = 0.0
                be[j] = 0.0
        temp.append(ne)
        tempo.append(no)
        bop_eigs[i] = be
    new_dft_eigs = np.array(temp)
    new_orb_char = np.array(tempo)

    debug = False
    if debug:
        import matplotlib.pyplot as pl
        cols = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i in [0, 1, 2, 3, 4]:
            pl.plot(bop_eigs.T[i], '%sx' % cols[i], lw=2)
        for i in [0, 1, 2, 3, 4]:
            pl.plot(new_dft_eigs.T[i], '%so' % cols[i])
        pl.ylabel('Energy(eV)')
        pl.xlabel('k')
        pl.savefig('bandstructure.png')
        pl.show()
        raise
    del temp
    del tempo
    return new_dft_eigs, new_orb_char


def get_relevant_orbs(atom, modelsbx):
    sym = atom.get_chemical_symbols()
    required = []
    val = {}
    for abx in modelsbx.atomsbx:
        v = abx.valenceorbitals
        if isinstance(v, list):
            v = v[0]
        val[abx.atom] = v
    for i in range(len(sym)):
        if val[sym[i]] == 1:
            required += [0]
        elif val[sym[i]] == 3:
            required += [1, 2, 3]
        elif val[sym[i]] == 5:
            required += [5, 7, 4, 8, 6]
        elif val[sym[i]] == 4:
            required += [0, 1, 2, 3]
        elif val[sym[i]] == 6:
            required += [0, 5, 7, 4, 8, 6]
        elif val[sym[i]] == 8:
            required += [1, 2, 3, 4, 5, 6, 7, 8]
        elif val[sym[i]] == 9:
            required += [0, 1, 2, 3, 4, 5, 6, 7, 8]
        else:
            raise NotImplementedError('No options for norb=%d' % val[sym[i]])
    # required.sort()
    return required


def get_relevant_bands(calc, dft_eigs, orb_char, debug=False):
    bop_eigs = calc.get_property(shift_fermi=False)
    dft_atoms = [deepcopy(atom) for atom in calc.get_atoms()]
    bop_atoms = calc.get_atoms()
    # bop and dft eigs have shape: [nstrucs][nspin][nkpts][nbands]
    # orb: [nstrucs][9]
    new_dft_eigs = []
    new_orb_char = []
    done = False
    for si in range(len(dft_eigs)):
        new_eigs = []
        new_orbs = []
        nat = len(dft_atoms[si])
        required = get_relevant_orbs(dft_atoms[si], calc.get_model())
        for spin in range(len(dft_eigs[si])):
            # print orb_char[si][0][0][9]
            n_e, n_o = get_relevant_bands_atom(dft_eigs[si][spin]
                                               , bop_eigs[si][spin]
                                               , orb_char[si][spin], required)
            # for k in range(len(n_e)):
            #    if not done:
            #        print dft_eigs[si][spin][k]
            #        print n_e[k]
            #        done = True
            # n_o = np.ones(np.shape(n_o))
            # print n_e
            # print n_o
            # raise
            new_eigs.append(n_e)
            new_orbs.append(n_o)
        new_eigs = list(new_eigs)
        new_orbs = list(new_orbs)
        info = dft_atoms[si].info
        info.update({'eigenvalues': new_eigs, 'orbital_character': new_orbs})
        dft_atoms[si].info = info
        new_dft_eigs.append(np.array(new_eigs))
        new_orb_char.append(np.array(new_orbs))
    debug = False
    if debug:
        plot_electronic_band_structure(dft_atoms[:1], filename='ebs_dft.png')
        plot_electronic_band_structure(bop_atoms[:1], filename='ebs_bop.png')
        raise
    return new_dft_eigs, new_orb_char
