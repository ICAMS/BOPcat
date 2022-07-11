#!/usr/bin/env python

# Definition of the atomsbx bondsbx and modelsbx

# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import os
import copy
from numpy import random as nran


###########################################################################
class atomsbx:
    """
    Defines an atoms.bx object for BOPfox.
    """

    def __init__(self, **kwargs):
        self.version = None
        self.atom = None
        self.mass = None
        self.densityr = None
        self.densitya = None
        self.pairrepr = None
        self.pairrepa = None
        self.embedding = None
        self.eamcutoff = None
        self.valenceorbitals = None
        self.valenceelectrons = None
        self.es = None
        self.ep = None
        self.ed = None
        self.stonerintegral = None
        self.jii = None
        self.onsitelevels = None
        self.delta_onsitelevels = [0.]
        self.norbscreen = None
        self.set(**kwargs)

    def __eq__(self, other):
        if not isinstance(other, atomsbx):
            return False
        for key in list(self.get_atomspar().keys()):
            if self.get_atomspar()[key] != other.get_atomspar()[key]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def set(self, **kwargs):
        for key in kwargs:
            if key.lower() == 'version':
                self.version = kwargs[key]
            elif key.lower() == 'atom':
                self.atom = kwargs[key]
            elif key.lower() == 'mass':
                self.mass = kwargs[key]
            elif key.lower() == 'densityr':
                self.densityr = kwargs[key]
            elif key.lower() == 'densitya':
                self.densitya = kwargs[key]
            elif key.lower() == 'pairrepr':
                self.pairrepr = kwargs[key]
            elif key.lower() == 'pairrepa':
                self.pairrepa = kwargs[key]
            elif key.lower() == 'embedding':
                self.embedding = kwargs[key]
            elif key.lower() == 'eamcutoff':
                self.eamcutoff = kwargs[key]
            elif key.lower() == 'valenceorbitals':
                self.valenceorbitals = kwargs[key]
            elif key.lower() == 'valenceelectrons':
                self.valenceelectrons = kwargs[key]
            elif key.lower() == 'ed':
                self.ed = kwargs[key]
            elif key.lower() == 'stonerintegral':
                self.stonerintegral = kwargs[key]
            elif key.lower() == 'es':
                self.es = kwargs[key]
            elif key.lower() == 'ep':
                self.ep = kwargs[key]
            elif key.lower() == 'jii':
                self.jii = kwargs[key]
            elif key.lower() == 'onsitelevels':
                self.onsitelevels = kwargs[key]
            elif key.lower() == 'delta_onsitelevels':
                self.delta_onsitelevels = kwargs[key]
            elif key.lower() == 'norbscreen':
                self.norbscreen = kwargs[key]
            else:
                raise TypeError('Unrecognized parameter: ' + key)

    def set_atomspar(self, atomspar):
        for key, val in list(atomspar.items()):
            if key.lower() == 'densityr':
                self.set(densityr=val)
            elif key.lower() == 'densitya':
                self.set(densitya=val)
            elif key.lower() == 'pairrepr':
                self.set(pairrepr=val)
            elif key.lower() == 'pairrepa':
                self.set(pairrepa=val)
            elif key.lower() == 'embedding':
                self.set(embedding=val)
            elif key.lower() == 'eamcutoff':
                self.set(eamcutoff=val)
            elif key.lower() == 'valenceorbitals':
                self.set(valenceorbitals=val)
            elif key.lower() == 'valenceelectrons':
                self.set(valenceelectrons=val)
            elif key.lower() == 'onsitelevels':
                self.set(onsitelevels=val)
            elif key.lower() == 'delta_onsitelevels':
                self.set(delta_onsitelevels=val)
            elif key.lower() == 'ed':
                self.set(ed=val)
            elif key.lower() == 'stonerintegral':
                self.set(stonerintegral=val)
            elif key.lower() == 'es':
                self.set(es=val)
            elif key.lower() == 'ep':
                self.set(ep=val)
            elif key.lower() == 'jii':
                self.set(jii=val)
            elif key.lower() == 'norbscreen':
                self.set(norbscreen=val)
            elif key.lower() == 'mass':
                self.set(mass=val)
            else:
                raise TypeError('Unrecognized parameter: %s' % key)

    def get_version(self):
        """Returns the version of the atoms.bx object"""
        return self.version

    def get_atom(self):
        """Returns the element in the atoms.bx object"""
        return self.atom

    def get_mass(self):
        """Returns the mass of the element"""
        if not isinstance(self.mass, float):
            self.mass = self.mass[0]
        return self.mass

    def get_atomspar(self):
        if self.onsitelevels is not None:
            try:
                self.onsitelevels = [i + self.delta_onsitelevels[0] \
                                     for i in self.onsitelevels]
            except:
                pass
            self.onsitelevels = list(self.onsitelevels)

        atomspar = {'densityr': self.densityr
            , 'densitya': self.densitya
            , 'pairrepr': self.pairrepr
            , 'pairrepa': self.pairrepa
            , 'embedding': self.embedding
            , 'eamcutoff': self.eamcutoff
            , 'mass': self.mass
            , 'valenceorbitals': self.valenceorbitals
            , 'valenceelectrons': self.valenceelectrons
            , 'onsitelevels': self.onsitelevels
            , 'delta_onsitelevels': self.delta_onsitelevels
            , 'ed': self.ed
            , 'stonerintegral': self.stonerintegral
            , 'es': self.es
            , 'ep': self.ep
            , 'jii': self.jii
            , 'norbscreen': self.norbscreen}
        return atomspar

    def get_keys(self):
        return list(self.get_atomspar().keys())

    def get_norbtype(self):
        if self.valenceorbitals in [1, 3, 5]:
            return 1
        elif self.valenceorbitals in [4, 6, 8]:
            return 2
        elif self.valenceorbitals in [9]:
            return 3

    def copy(self):
        return copy.deepcopy(self)

    def rattle(self, var='all', factor='random', maxf=0.1):
        """
        Generate a modified atomsbx with parameters defined in var modified 
        by factor.
        
        :Parameters:
        
        - *var*: dictionary of key, constraints   
        
        - *factor*: float   
        """
        if var == 'all':
            var = self.get_atomspar()
            for key, val in list(var.items()):
                if key.lower() in ['valenceorbitals', 'mass']:
                    del var[key]
                    continue
                if val is None:
                    del var[key]
                    continue
                if isinstance(val, float) or isinstance(val, int):
                    val = [val]
                var[key] = [True] * (len(val))
        new = self.copy()
        newpar = new.get_atomspar()
        for key, val in list(newpar.items()):
            if key not in list(var.keys()):
                continue
            if not isinstance(val, list):
                # expecting here a functions object
                num = val.get_numbers()
            else:
                num = list(val)
            if factor == 'random':
                f = maxf * nran.random_sample(len(num)) * nran.choice([-1, 1])
            elif factor == 'gaussian':
                f = nran.normal(0., maxf, len(num))
            elif isinstance(factor, float):
                f = [factor] * len(num)
            for i in range(len(num)):
                if var[key][i]:
                    num[i] *= (1.0 + f[i])
            if not isinstance(val, list):
                # expecting here a functions object
                val.set(parameters=num, constraints=[True] * len(num))
            else:
                val = list(num)
            newpar[key] = val
        new.set_atomspar(newpar)
        return new

    def write(self, filename='atoms.bx', update=False):
        """
        Writes out atomsbx object to file.
        The keyword update does not necessarily mean that atomsbx
        is updated. It just means that succeeding entries will not
        have the header 'Version'.
        """
        dash = '/-----------------------------------------------------------\n'
        if update:
            atoms = read_atomsbx(filename=filename)
            found = False
            for j in range(len(atoms)):
                if atoms[j].get_version() == self.get_version() and \
                        atoms[j].get_atom() == self.get_atom():
                    atoms[j] = self.copy()
                    found = True
                    break
            if not found:
                atoms.append(self.copy())
        else:
            atoms = [self.copy()]
        versions = []
        for i in range(len(atoms)):
            version = atoms[i].get_version()
            if version not in versions:
                versions.append(version)
        f = open(filename, 'w')
        f.write('/atoms.bx written in ASE\n')
        for i in range(len(versions)):
            marker = 0
            for j in range(len(atoms)):
                if atoms[j].get_version() == versions[i]:
                    if marker == 0:
                        f.write()
                        f.write('Version = %s\n' % versions[i])
                        f.write(dash)
                        f.write('\n')
                    f.write('Atom             = %s\n' % atoms[j].get_atom())
                    f.write('Mass             = %s\n' % atoms[j].get_mass())
                    for key, list_val in list(atoms[j].get_atomspar().items()):
                        if key in ['valenceorbitals', 'norbscreen'] \
                                and list_val is not None:
                            f.write('{0:16} = {1:10}\n'.format(key
                                                               , int(list_val[0])))
                        elif key in ['onsitelevels'] \
                                and list_val is not None:
                            f.write('{0:16} =  '.format(key))
                            for v in range(len(list_val)):
                                f.write('%18.16e ' % list_val[v])
                            f.write('\n')
                        elif list_val is not None and type(list_val) is list:
                            f.write('{0:16} = '.format(key))
                            for val in list_val:
                                f.write('%18.16e ' % val)
                            f.write('\n')
                        elif list_val is not None and \
                                type(list_val) is not list:
                            f.write('{0:16} = {1:10}\n'.format(key, list_val))
                    f.write('\n')
                    marker += 1
                    f.write('\n')
        f.close()


class bondsbx:
    """
    Defines a bonds.bx object for BOPfox.
    """

    def __init__(self, **kwargs):
        self.version = None
        self.bondform = None
        self.repform = None
        self.attform = None
        self.couform = None
        self.chaform = None
        self.bond = None
        self.valence = None
        self.scaling = 1
        self.ddsigma = None
        self.ddpi = None
        self.dddelta = None
        self.sssigma = None
        self.sdsigma = None
        self.dssigma = None
        self.spsigma = None
        self.ppsigma = None
        self.pppi = None
        self.pssigma = None
        self.dpsigma = None
        self.pdsigma = None
        self.dppi = None
        self.pdpi = None
        self.ddsigmaoverlap = None
        self.ddpioverlap = None
        self.dddeltaoverlap = None
        self.sssigmaoverlap = None
        self.sdsigmaoverlap = None
        self.dssigmaoverlap = None
        self.spsigmaoverlap = None
        self.ppsigmaoverlap = None
        self.pppioverlap = None
        self.pssigmaoverlap = None
        self.dpsigmaoverlap = None
        self.pdsigmaoverlap = None
        self.dppioverlap = None
        self.pdpioverlap = None
        self.pairrepulsion = None
        self.eamattraction = None
        self.eamcoupling = None
        self.chargetransfer = None
        self.rcut = None
        self.dcut = None
        self.r2cut = None
        self.d2cut = None
        self.cutoffversion = None
        self.cutoff2version = None
        self.rep1 = None
        self.rep2 = None
        self.rep3 = None
        self.rep4 = None
        self.rep5 = None
        self.rep6 = None
        self.rep7 = None
        self.rep8 = None
        self.rep9 = None
        self.rep10 = None
        self.rep11 = None
        self.rep12 = None
        self.rep13 = None
        self.rep14 = None
        self.rep15 = None
        self.rep16 = None
        self.rep17 = None
        self.rep18 = None
        self.rep19 = None
        self.rep20 = None
        self.functions = {}
        self.set(**kwargs)

    def __eq__(self, other):
        if not isinstance(other, bondsbx):
            return False
        for key in self.get_bondspar():
            if self.get_bondspar()[key] != other.get_bondspar()[key]:
                return False
        for key in self.get_overlappar():
            if self.get_overlappar()[key] != other.get_overlappar()[key]:
                return False
        for key in self.get_repetal():
            if self.get_repetal()[key] != other.get_repetal()[key]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def set(self, **kwargs):
        # for key in kwargs:
        #    if isinstance(kwargs[key],str):
        #        exec("self.%s = '%s'"%(key.lower(),kwargs[key]))
        #    else:
        #        exec("self.%s = %s"%(key.lower(),kwargs[key]))
        for key in kwargs:
            if key.lower() == 'version':
                self.version = kwargs[key]
            elif key.lower() == 'bondform':
                self.bondform = kwargs[key]
            elif key.lower() == 'repform':
                self.repform = kwargs[key]
            elif key.lower() == 'attform':
                self.attform = kwargs[key]
            elif key.lower() == 'couform':
                self.couform = kwargs[key]
            elif key.lower() == 'chaform':
                self.chaform = kwargs[key]
            elif key.lower() == 'bond':
                self.bond = kwargs[key]
            elif key.lower() == 'valence':
                self.valence = kwargs[key]
            elif key.lower() == 'scaling':
                self.scaling = kwargs[key]
            elif key.lower() == 'ddsigma':
                self.ddsigma = kwargs[key]
            elif key.lower() == 'ddpi':
                self.ddpi = kwargs[key]
            elif key.lower() == 'dddelta':
                self.dddelta = kwargs[key]
            elif key.lower() == 'sssigma':
                self.sssigma = kwargs[key]
            elif key.lower() == 'sdsigma':
                self.sdsigma = kwargs[key]
            elif key.lower() == 'dssigma':
                self.dssigma = kwargs[key]
            elif key.lower() == 'spsigma':
                self.spsigma = kwargs[key]
            elif key.lower() == 'ppsigma':
                self.ppsigma = kwargs[key]
            elif key.lower() == 'pppi':
                self.pppi = kwargs[key]
            elif key.lower() == 'pssigma':
                self.pssigma = kwargs[key]
            elif key.lower() == 'dpsigma':
                self.dpsigma = kwargs[key]
            elif key.lower() == 'pdsigma':
                self.pdsigma = kwargs[key]
            elif key.lower() == 'dppi':
                self.dppi = kwargs[key]
            elif key.lower() == 'pdpi':
                self.pdpi = kwargs[key]
            elif key.lower() == 'ddsigmaoverlap':
                self.ddsigmaoverlap = kwargs[key]
            elif key.lower() == 'ddpioverlap':
                self.ddpioverlap = kwargs[key]
            elif key.lower() == 'dddeltaoverlap':
                self.dddeltaoverlap = kwargs[key]
            elif key.lower() == 'sssigmaoverlap':
                self.sssigmaoverlap = kwargs[key]
            elif key.lower() == 'sdsigmaoverlap':
                self.sdsigmaoverlap = kwargs[key]
            elif key.lower() == 'dssigmaoverlap':
                self.dssigmaoverlap = kwargs[key]
            elif key.lower() == 'spsigmaoverlap':
                self.spsigmaoverlap = kwargs[key]
            elif key.lower() == 'ppsigmaoverlap':
                self.ppsigmaoverlap = kwargs[key]
            elif key.lower() == 'pppioverlap':
                self.pppioverlap = kwargs[key]
            elif key.lower() == 'pssigmaoverlap':
                self.pssigmaoverlap = kwargs[key]
            elif key.lower() == 'dpsigmaoverlap':
                self.dpsigmaoverlap = kwargs[key]
            elif key.lower() == 'pdsigmaoverlap':
                self.pdsigmaoverlap = kwargs[key]
            elif key.lower() == 'dppioverlap':
                self.dppioverlap = kwargs[key]
            elif key.lower() == 'pdpioverlap':
                self.pdpioverlap = kwargs[key]
            elif key.lower() == 'pairrepulsion':
                self.pairrepulsion = kwargs[key]
            elif key.lower() == 'eamattraction':
                self.eamattraction = kwargs[key]
            elif key.lower() == 'eamcoupling':
                self.eamcoupling = kwargs[key]
            elif key.lower() == 'chargetransfer':
                self.chargetransfer = kwargs[key]
            elif key.lower() == 'rcut':
                self.rcut = kwargs[key]
            elif key.lower() == 'dcut':
                self.dcut = kwargs[key]
            elif key.lower() == 'r2cut':
                self.r2cut = kwargs[key]
            elif key.lower() == 'd2cut':
                self.d2cut = kwargs[key]
            elif key.lower() == 'cutoffversion':
                self.cutoffversion = kwargs[key]
            elif key.lower() == 'cutoff2version':
                self.cutoff2version = kwargs[key]
            elif key.lower() == 'rep1':
                self.rep1 = kwargs[key]
            elif key.lower() == 'rep2':
                self.rep2 = kwargs[key]
            elif key.lower() == 'rep3':
                self.rep3 = kwargs[key]
            elif key.lower() == 'rep4':
                self.rep4 = kwargs[key]
            elif key.lower() == 'rep5':
                self.rep5 = kwargs[key]
            elif key.lower() == 'rep6':
                self.rep6 = kwargs[key]
            elif key.lower() == 'rep7':
                self.rep7 = kwargs[key]
            elif key.lower() == 'rep8':
                self.rep8 = kwargs[key]
            elif key.lower() == 'rep9':
                self.rep9 = kwargs[key]
            elif key.lower() == 'rep10':
                self.rep10 = kwargs[key]
            elif key.lower() == 'rep11':
                self.rep11 = kwargs[key]
            elif key.lower() == 'rep12':
                self.rep12 = kwargs[key]
            elif key.lower() == 'rep13':
                self.rep13 = kwargs[key]
            elif key.lower() == 'rep14':
                self.rep14 = kwargs[key]
            elif key.lower() == 'rep15':
                self.rep15 = kwargs[key]
            elif key.lower() == 'rep16':
                self.rep16 = kwargs[key]
            elif key.lower() == 'rep17':
                self.rep17 = kwargs[key]
            elif key.lower() == 'rep18':
                self.rep18 = kwargs[key]
            elif key.lower() == 'rep19':
                self.rep19 = kwargs[key]
            elif key.lower() == 'rep20':
                self.rep20 = kwargs[key]
            elif key.lower() == 'functions':
                self.functions.update(kwargs[key])
            else:
                raise TypeError('Unrecognized parameter: %s' % key)

    def set_bondspar(self, bondspar):
        for key, val in list(bondspar.items()):
            if key.lower() == 'scaling':
                self.set(scaling=val)
            elif key.lower() == 'ddsigma':
                self.set(ddsigma=val)
            elif key.lower() == 'ddpi':
                self.set(ddpi=val)
            elif key.lower() == 'dddelta':
                self.set(dddelta=val)
            elif key.lower() == 'sssigma':
                self.set(sssigma=val)
            elif key.lower() == 'sdsigma':
                self.set(sdsigma=val)
            elif key.lower() == 'dssigma':
                self.set(dssigma=val)
            elif key.lower() == 'spsigma':
                self.set(spsigma=val)
            elif key.lower() == 'ppsigma':
                self.set(ppsigma=val)
            elif key.lower() == 'pppi':
                self.set(pppi=val)
            elif key.lower() == 'pssigma':
                self.set(pssigma=val)
            elif key.lower() == 'dpsigma':
                self.set(dpsigma=val)
            elif key.lower() == 'pdsigma':
                self.set(pdsigma=val)
            elif key.lower() == 'dppi':
                self.set(dppi=val)
            elif key.lower() == 'pdpi':
                self.set(pdpi=val)
            elif key.lower() == 'ddsigmaoverlap':
                self.set(ddsigmaoverlap=val)
            elif key.lower() == 'ddpioverlap':
                self.set(ddpioverlap=val)
            elif key.lower() == 'dddeltaoverlap':
                self.set(dddeltaoverlap=val)
            elif key.lower() == 'sssigmaoverlap':
                self.set(sssigmaoverlap=val)
            elif key.lower() == 'sdsigmaoverlap':
                self.set(sdsigmaoverlap=val)
            elif key.lower() == 'dssigmaoverlap':
                self.set(dssigmaoverlap=val)
            elif key.lower() == 'spsigmaoverlap':
                self.set(spsigmaoverlap=val)
            elif key.lower() == 'ppsigmaoverlap':
                self.set(ppsigmaoverlap=val)
            elif key.lower() == 'pppioverlap':
                self.set(pppioverlap=val)
            elif key.lower() == 'pssigmaoverlap':
                self.set(pssigmaoverlap=val)
            elif key.lower() == 'dpsigmaoverlap':
                self.set(dpsigmaoverlap=val)
            elif key.lower() == 'pdsigmaoverlap':
                self.set(pdsigmaoverlap=val)
            elif key.lower() == 'dppioverlap':
                self.set(dppioverlap=val)
            elif key.lower() == 'pdpioverlap':
                self.set(pdpioverlap=val)
            elif key.lower() == 'pairrepulsion':
                self.set(pairrepulsion=val)
            elif key.lower() == 'eamattraction':
                self.set(eamattraction=val)
            elif key.lower() == 'eamcoupling':
                self.set(eamcoupling=val)
            elif key.lower() == 'chargetransfer':
                self.set(chargetransfer=val)
            elif key.lower() == 'rep1':
                self.set(rep1=val)
            elif key.lower() == 'rep2':
                self.set(rep2=val)
            elif key.lower() == 'rep3':
                self.set(rep3=val)
            elif key.lower() == 'rep4':
                self.set(rep4=val)
            elif key.lower() == 'rep5':
                self.set(rep5=val)
            elif key.lower() == 'rep6':
                self.set(rep6=val)
            elif key.lower() == 'rep7':
                self.set(rep7=val)
            elif key.lower() == 'rep8':
                self.set(rep8=val)
            elif key.lower() == 'rep9':
                self.set(rep9=val)
            elif key.lower() == 'rep10':
                self.set(rep10=val)
            elif key.lower() == 'rep11':
                self.set(rep11=val)
            elif key.lower() == 'rep12':
                self.set(rep12=val)
            elif key.lower() == 'rep13':
                self.set(rep13=val)
            elif key.lower() == 'rep14':
                self.set(rep14=val)
            elif key.lower() == 'rep15':
                self.set(rep15=val)
            elif key.lower() == 'rep16':
                self.set(rep16=val)
            elif key.lower() == 'rep17':
                self.set(rep17=val)
            elif key.lower() == 'rep18':
                self.set(rep18=val)
            elif key.lower() == 'rep19':
                self.set(rep19=val)
            elif key.lower() == 'rep20':
                self.set(rep20=val)
            else:
                raise TypeError('Unrecognized parameter: %s' % key)

    def set_cutpar(self, cutpar):
        for key, val in list(cutpar.items()):
            if key.lower() == 'rcut':
                self.rcut = val
            elif key.lower() == 'dcut':
                self.dcut = val
            elif key.lower() == 'r2cut':
                self.r2cut = val
            elif key.lower() == 'd2cut':
                self.d2cut = val
            elif key.lower() == 'cutoffversion':
                self.cutoffversion = val
            elif key.lower() == 'cutoff2version':
                self.cutoff2version = val
            else:
                raise TypeError('Unrecognized parameter: %s' % key)

    def get_version(self):
        """Returns the version of the bonds object"""
        return self.version

    def get_bond(self):
        """Returns the elements in the bonds object"""
        return self.bond

    def get_cutpar(self):
        cutpar = {'rcut': self.rcut
            , 'dcut': self.dcut
            , 'r2cut': self.r2cut
            , 'd2cut': self.d2cut
            , 'cutoffversion': self.cutoffversion
            , 'cutoff2version': self.cutoff2version}
        return cutpar

    def get_bondspar(self):
        self._symmetrize()
        bondspar = {'sssigma': self.sssigma
            , 'sdsigma': self.sdsigma
            , 'dssigma': self.dssigma
            , 'spsigma': self.spsigma
            , 'ppsigma': self.ppsigma
            , 'pppi': self.pppi
            , 'pssigma': self.pssigma
            , 'ddsigma': self.ddsigma
            , 'ddpi': self.ddpi
            , 'dddelta': self.dddelta
            , 'dpsigma': self.dpsigma
            , 'pdsigma': self.pdsigma
            , 'dppi': self.dppi
            , 'pdpi': self.pdpi
                    }
        return bondspar

    def get_overlappar(self):
        self._symmetrize()
        overlappar = {'sssigmaoverlap': self.sssigmaoverlap
            , 'sdsigmaoverlap': self.sdsigmaoverlap
            , 'dssigmaoverlap': self.dssigmaoverlap
            , 'spsigmaoverlap': self.spsigmaoverlap
            , 'ppsigmaoverlap': self.ppsigmaoverlap
            , 'pppioverlap': self.pppioverlap
            , 'pssigmaoverlap': self.pssigmaoverlap
            , 'ddsigmaoverlap': self.ddsigmaoverlap
            , 'ddpioverlap': self.ddpioverlap
            , 'dddeltaoverlap': self.dddeltaoverlap
            , 'dpsigmaoverlap': self.dpsigmaoverlap
            , 'pdsigmaoverlap': self.pdsigmaoverlap
            , 'dppioverlap': self.dppioverlap
            , 'pdpioverlap': self.pdpioverlap
                      }
        return overlappar

    def get_repetal(self):
        repetal = {'pairrepulsion': self.pairrepulsion
            , 'eamattraction': self.eamattraction
            , 'eamcoupling': self.eamcoupling
            , 'chargetransfer': self.chargetransfer
            , 'rep1': self.rep1
            , 'rep2': self.rep2
            , 'rep3': self.rep3
            , 'rep4': self.rep4
            , 'rep5': self.rep5
            , 'rep6': self.rep6
            , 'rep7': self.rep7
            , 'rep8': self.rep8
            , 'rep9': self.rep9
            , 'rep10': self.rep10
            , 'rep11': self.rep11
            , 'rep12': self.rep12
            , 'rep13': self.rep13
            , 'rep14': self.rep14
            , 'rep15': self.rep15
            , 'rep16': self.rep16
            , 'rep17': self.rep17
            , 'rep18': self.rep18
            , 'rep19': self.rep19
            , 'rep20': self.rep20
                   }
        return repetal

    def get_functions(self):
        bondkeys = list(self.get_bondspar().keys())
        overlapkeys = list(self.get_overlappar().keys())
        repkeys = list(self.get_repetal().keys()) + bondkeys + overlapkeys
        if self.functions is None:
            self.functions = {}
            for key in repkeys:
                self.functions[key] = None
        else:
            for key in repkeys:
                if key not in self.functions:
                    self.functions[key] = None
        return self.functions

    def get_keys(self):
        keys = list(self.get_bondspar().keys()) + \
               list(self.get_overlappar().keys()) + \
               list(self.get_repetal().keys())
        return keys

    def _symmetrize(self):
        if self.bond is None:
            return
        if self.bond[0] == self.bond[1]:
            if self.sdsigma is not None:
                if isinstance(self.sdsigma, list):
                    self.dssigma = list(self.sdsigma)
                    self.functions['dssigma'] = self.functions['sdsigma']
                else:
                    # expecting function
                    self.dssigma = self.sdsigma.copy()
            if self.sdsigmaoverlap is not None:
                if isinstance(self.sdsigmaoverlap, list):
                    self.dssigmaoverlap = list(self.sdsigmaoverlap)
                    self.functions['dssigmaoverlap'] = \
                        self.functions['sdsigmaoverlap']
                else:
                    # expecting function
                    self.dssigmaoverlap = self.sdsigmaoverlap.copy()

    def copy(self):
        return copy.deepcopy(self)

    def _check_bonds(self):
        if self.version is None:
            raise ValueError('Version should be set to proceed.')
        if self.bond is None:
            raise ValueError('Bond should be set to proceed.')
        if self.valence is None:
            raise ValueError('Valence should be set to proceed.')
        else:
            for i in range(2):
                if self.valence[i] == 's':
                    if self.sssigma is None:
                        raise ValueError('sssigma should be set to proceed.')
                elif self.valence[i] == 'sp':
                    if self.sssigma is None:
                        raise ValueError('sssigma should be set to proceed.')
                    if self.ppsigma is None:
                        raise ValueError('ppsigma should be set to proceed.')
                    if self.pppi is None:
                        raise ValueError('pppi should be set to proceed.')
                    if self.spsigma is None:
                        raise ValueError('spsigma should be set to proceed.')
                    if self.pssigma is None:
                        raise ValueError('pssigma should be set to proceed.')
                elif self.valence[i] == 'd':
                    if self.ddsigma is None:
                        raise ValueError('ddsigma should be set to proceed.')
                    if self.ddpi is None:
                        raise ValueError('ddpi should be set to proceed.')
                    if self.dddelta is None:
                        raise ValueError('dddelta should be set to proceed.')
                elif self.valence[i] == 'sd':
                    if self.ddsigma is None:
                        raise ValueError('ddsigma should be set to proceed.')
                    if self.ddpi is None:
                        raise ValueError('ddpi should be set to proceed.')
                    if self.dddelta is None:
                        raise ValueError('dddelta should be set to proceed.')
                    if self.sssigma is None:
                        raise ValueError('sssigma should be set to proceed.')
                    if self.sdsigma is None:
                        raise ValueError('sdsigma should be set to proceed.')
                if self.valence[0] == self.valence[1]:
                    break

    def rattle(self, var='all', factor='random', maxf=0.1):
        """
        Generate a modified bondsbx with parameters defined in var modified 
        by factor.
        
        :Parameters:
        
        - *var*: dictionary of key, constraints   
        
        - *factor*: float   
        """
        if var == 'all':
            var = self.get_bondspar()
            var.update(self.get_overlappar())
            var.update(self.get_repetal())
            for key, val in list(var.items()):
                if val is None:
                    del var[key]
                    continue
                if not isinstance(val, list):
                    cons = val.get_constraints()
                    # cons = [True]*len(val.get_numbers())
                else:
                    cons = [True] * len(val)
                var[key] = cons
        new = self.copy()
        newpar = new.get_bondspar()
        newpar.update(self.get_overlappar())
        newpar.update(self.get_repetal())
        for key, val in list(newpar.items()):
            if key not in list(var.keys()):
                continue
            par = []
            if not isinstance(val, list):
                # expecting here a functions object
                num = val.get_numbers()
            else:
                num = list(val)
            if factor == 'random':
                f = maxf * nran.random_sample(len(num)) * nran.choice([-1, 1])
            elif factor == 'gaussian':
                f = nran.normal(0., maxf, len(num))
            elif isinstance(factor, float):
                f = [factor] * len(num)
            for i in range(len(num)):
                if var[key][i]:
                    num[i] *= (1.0 + f[i])
                    par.append(num[i])
            if not isinstance(val, list):
                # expecting here a functions object
                val.set(parameters=par, numbers=num, constraints=var[key])
            else:
                val = list(num)
            newpar[key] = val
        new.set_bondspar(newpar)
        return new

    def write(self, filename='bonds.bx', update=False):
        """
        Writes bondsbx object to file
        update means that you read the current bonds.bx file
        and write it with bopbonds
        """
        dash = '/-----------------------------------------------------------\n'
        if update:
            bonds = read_bondsbx(filename=filename)
            # Check if necessary parameters were defined otherwise cannot
            # write bonds.bx
            self._check_bonds()
            found = False
            for j in range(len(bonds)):
                old_bonds = [bonds[j].bond[0], bonds[j].bond[1]]
                new_bonds = [self.bond[0], self.bond[1]]
                if (bonds[j].get_version() == self.get_version()) \
                        and (old_bonds == new_bonds):
                    bonds[j] = self.copy()
                    found = True
                    break
            if not found:
                bonds.append(self.copy())

        else:
            bonds = [self.copy()]
        versions = []
        for i in range(len(bonds)):
            version = bonds[i].get_version()
            if version not in versions:
                versions.append(version)
        f = open(filename, 'w')
        f.write('/bonds.bx written in ASE\n')
        for i in range(len(versions)):
            marker = 0
            for j in range(len(bonds)):
                if bonds[j].get_version() == versions[i]:
                    if marker == 0:
                        f.write()
                        f.write('Version = %s\n' % versions[i])
                        f.write(dash)
                        f.write('\n')
                    f.write('bond           =   %s %s \n' % (bonds[j].bond[0]
                                                             , bonds[j].bond[1]))
                    f.write('valence        =   %s %s \n' % (bonds[j].valence[0]
                                                             , bonds[j].valence[1]))
                    f.write('scaling        =%18.16e \n' % bonds[j].scaling[0])
                    for key, list_val in list(bonds[j].get_bondspar().items()):
                        if list_val is not None:
                            f.write('{0:14} = '.format(key))
                            for val in list_val:
                                f.write('%18.16e ' % val)
                            f.write('\n')
                    for key, list_val in list(bonds[j].get_overlappar().items()):
                        if list_val is not None:
                            f.write('{0:14} = '.format(key))
                            for val in list_val:
                                f.write('%18.16e ' % val)
                            f.write('\n')
                    for key, list_val in list(bonds[j].get_repetal().items()):
                        if list_val is not None:
                            f.write('{0:14} = '.format(key))
                            for val in list_val:
                                f.write('%18.16e ' % val)
                            f.write('\n')
                    marker += 1
                    f.write('\n')
            f.write('\n')
        f.close()


class modelsbx:
    """
    Defines a modelsbx object in BOPfox
    """

    def __init__(self, **kwargs):
        self.model = None
        self.infox_parameters = {}
        self.atomsbx = None
        self.bondsbx = None
        self.annotations = []
        self.set(**kwargs)

    def __eq__(self, other):
        if not isinstance(other, modelsbx):
            return False
        if [abx.atom for abx in self.atomsbx] != \
                [abx.atom for abx in other.atomsbx]:
            return False
        if [bbx.bond for bbx in self.bondsbx] != \
                [bbx.bond for bbx in other.bondsbx]:
            return False
        for sabx in self.atomsbx:
            for oabx in other.atomsbx:
                if sabx.atom != oabx.atom:
                    continue
                if sabx != oabx:
                    return False
        for sbbx in self.bondsbx:
            for obbx in other.bondsbx:
                if sbbx.bond != obbx.bond:
                    continue
                if sbbx != obbx:
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def set(self, **kwargs):
        for key in kwargs:
            if key.lower() == 'model':
                self.model = kwargs[key]
            elif key.lower() == 'infox_parameters':
                self.infox_parameters = kwargs[key]
            elif key.lower() == 'atomsbx':
                self.atomsbx = kwargs[key]
            elif key.lower() == 'bondsbx':
                self.bondsbx = kwargs[key]
            elif key.lower() == 'annotations':
                self.annotations = kwargs[key]
            else:
                raise ValueError('No options for key %s' % key)
        if isinstance(self.annotations, str):
            self.annotations = [self.annotations]
        for s in self.annotations:
            if not isinstance(s, str):
                raise ValueError('Annotations can only be of type strings.')

    def copy(self):
        return copy.deepcopy(self)

    def get_keys(self):
        bond_keys = self.bondsbx[0].get_keys()
        atom_keys = self.atomsbx[0].get_keys()
        infox_keys = list(self.infox_parameters.keys())
        return bond_keys + atom_keys + infox_keys

    def get_infox(self):
        return dict(self.infox_parameters)

    def get_bonds(self):
        bonds = []
        for bond in self.bondsbx:
            bonds.append(bond.get_bond())
        return bonds

    def get_atoms(self):
        atoms = []
        for atom in self.atomsbx:
            atoms.append(atom.get_atom())
        return atoms

    def get_parameters(self, name, atom=None, bond=None):
        out = None
        if atom is not None and bond is not None:
            print("Can only get parameters for atom or bond")
            out = None
        if atom is not None:
            for atomsbx in self.atomsbx:
                if atomsbx.get_atom() == atom:
                    out = atomsbx.get_atomspar()[name]
                    break
        if bond is not None:
            bond = list(bond)
            bond.sort()
            for bondsbx in self.bondsbx:
                bbx = bondsbx.get_bond()
                bbx.sort()
                if bbx == bond:
                    if name in list(bondsbx.get_bondspar().keys()):
                        out = bondsbx.get_bondspar()[name]
                    elif name in list(bondsbx.get_overlappar().keys()):
                        out = bondsbx.get_overlappar()[name]
                    elif name in list(bondsbx.get_repetal().keys()):
                        out = bondsbx.get_repetal()[name]
                    elif name in list(bondsbx.get_cutpar().keys()):
                        out = bondsbx.get_cutpar()[name]
        return out

    def bond_parameters_to_functions(self, bondlist='all', variables=None):
        """
        Convert list of parameters in bondsbx to corresponding function 
        (see :mod functions)
        """
        from .functions import list_to_func
        bbxs = self.bondsbx
        if bondlist == 'all':
            bondlist = [bbx.bond for bbx in bbxs]
        assert (type(bondlist) == list)
        for i in range(len(bondlist)):
            bondlist[i] = list(bondlist[i])
            bondlist[i].sort()
        bondkeys = list(bbxs[0].get_bondspar().keys())
        bondkeys += list(bbxs[0].get_overlappar().keys())
        bondkeys += list(bbxs[0].get_repetal().keys())
        for i in range(len(bbxs)):
            bond = bbxs[i].bond
            bond.sort()
            funnames = bbxs[i].get_functions()
            if bond not in bondlist:
                continue
            temp = {}
            for name in bondkeys:
                if name in ['chargetransfer']:
                    continue
                par = self.get_parameters(name, bond=bond)
                if par is None:
                    continue
                if not isinstance(par, list):
                    print(("Warning: parameters for %s is a %s" % (name, type(par))))
                    print("         expecting list.")
                    continue
                var = None
                if variables is not None:
                    for j in range(len(variables)):
                        if 'bond' not in variables[j]:
                            continue
                        bondv = variables[j]['bond']
                        bondv.sort()
                        if bondv == bond:
                            break
                    for key, val in list(variables[j].items()):
                        if key == name:
                            var = variables[j][key]
                            break
                fun = list_to_func(par, funnames[name], var)
                temp[name] = fun
            bbxs[i].set_bondspar(temp)
        self.bondsbx = bbxs

    def rattle(self, var='all', factor='random', maxf=0.1):
        """
        Generate a modified bondsbx with parameters defined in var modified 
        by factor.
        
        :Parameters:
        
        - *var*: list of dictionary of key, constraints 

            e.g. [{'bond':['Fe','Fe'],'sssigma':[True,True,True]}
                 ,{'atom':['Fe'], 'onsitelevels':[True]}]
        
        - *factor*: float 
        """
        mbx = self.copy()
        abxs = list(mbx.atomsbx)
        bbxs = list(mbx.bondsbx)
        # if var == 'all':
        #    var = abxs[0].get_atomspar().keys()
        #    var = bbxs[0].get_bondspar().keys()
        #    keys+= bbxs[0].get_overlappar().keys()
        #    keys+= bbxs[0].get_repetal().keys()
        # if factor.lower() == 'random':
        #    factor = nran.random() 
        if isinstance(var, list):
            atomlist = [v['atom'] for v in var if 'atom' in v]
            bondlist = [v['bond'] for v in var if 'bond' in v]
        else:
            atomlist = [abx.atom for abx in abxs]
            bondlist = [bbx.bond for bbx in bbxs]
        for i in range(len(bondlist)):
            bondlist[i].sort()
        for i in range(len(abxs)):
            if abxs[i].atom not in atomlist:
                continue
            if isinstance(var, list):
                for v in var:
                    if 'atom' not in v:
                        continue
                    if v['atom'] == abxs[i].atom:
                        break
            else:
                v = var
            abxs[i] = abxs[i].rattle(v, factor, maxf)
        for i in range(len(bbxs)):
            bond = bbxs[i].bond
            bond.sort()
            if bond not in bondlist:
                continue
            if isinstance(var, list):
                for v in var:
                    if 'bond' not in v:
                        continue
                    bo = v['bond']
                    bo.sort()
                    if bo == bond:
                        break
            else:
                v = var
            bbxs[i] = bbxs[i].rattle(v, factor, maxf)
        mbx.atomsbx = abxs
        mbx.bondsbx = bbxs
        return mbx

    def get_index(self, atom=None, bond=None):
        out = None
        if atom is not None and bond is not None:
            err = '''Cannot determine index of atom and bond simultaneously.'''
            raise ValueError(err)
        if atom is not None:
            for i in range(len(self.atomsbx)):
                if self.atomsbx[i].atom == atom:
                    out = i
                    break
        elif bond is not None:
            for i in range(len(self.bondsbx)):
                b0 = self.bondsbx[i].bond
                b1 = list(bond)
                b0.sort()
                b1.sort()
                if b0 == b1:
                    out = i
                    break
        return out

    def write(self, filename='models.bx', printf='%18.16e '):
        f = open(filename, 'w')
        f.write('/models.bx written in ASE\n')
        f.write('/------------------------------------------------------\n')
        f.write('/------------------------------------------------------\n')
        f.write('model = %s\n' % self.model)
        f.write('/------------------------------------------------------\n')
        for s in self.annotations:
            f.write('/%s \n' % s)
        f.write('/------------------------------------------------------\n')
        inpara = self.infox_parameters
        # exclude = ['strucfile','modelfile','model','task','tbkpointfile'
        #          ,'tbkpointmesh','scfsteps','tbkpointfolding','rcut'
        #          ,'dcut','r2cut','d2cut','cutoffversion','cutoff2version'
        #          ,'magconfig']
        include = ['version', 'eamversion', 'atomsversion', 'bondsversion'
            , 'repversion', 'screening', 'screeningversion', 'jijversion'
            , 'moments', 'global_a_inf', 'global_b_inf', 'bandwidth'
            , 'globalbandwidth', 'terminator', 'nexpmoments', 'bopkernel'
            , 'ecut', 'decut', 'orthogonal']
        for key, val in list(inpara.items()):
            if (key.lower() in include) and (val is not None):
                if isinstance(val, str):
                    f.write('{0:30} = {1}\n'.format(key, val))
        for key, val in list(inpara.items()):
            if (key.lower() in include) and (val is not None):
                if isinstance(val, bool):
                    if val:
                        f.write('{0:30} = {1}\n'.format(key, 'T'))
                    else:
                        f.write('{0:30} = {1}\n'.format(key, 'F'))
        for key, val in list(inpara.items()):
            if (key.lower() in include) and (val is not None):
                if isinstance(val, int) and not isinstance(val, bool):
                    f.write('{0:30} = {1}\n'.format(key, val))
        for key, val in list(inpara.items()):
            if (key.lower() in include) and (val is not None):
                if isinstance(val, float):
                    f.write('{0:30} = {1:10.20f}\n'.format(key, val))
        for key, val in list(inpara.items()):
            if (key.lower() in include) and (val is not None):
                if isinstance(val, list):
                    f.write('{0:30} ='.format(key))
                    for i in range(len(val)):
                        f.write('{0:4} '.format(val[i]))
                    f.write('\n')
        f.write('\n')
        atoms = self.atomsbx
        for j in range(len(atoms)):
            f.write('/-----------------------------------------------------\n')
            f.write('Atom             = %s\n' % atoms[j].get_atom())
            f.write('/-----------------------------------------------------\n')
            f.write('Mass             = %s\n' % atoms[j].get_mass())
            for key, list_val in list(atoms[j].get_atomspar().items()):
                if key in ['valenceorbitals', 'norbscreen'] \
                        and list_val is not None:
                    if type(list_val) is not list:
                        list_val = [list_val]
                    f.write('{0:16} = {1:10}\n'.format(key, int(list_val[0])))
                elif key in ['onsitelevels'] and list_val is not None:
                    f.write('{0:16} =  '.format(key))
                    if type(list_val) is not list:
                        list_val = [list_val]
                    for v in range(len(list_val)):
                        f.write('%18.16e ' % list_val[v])
                    f.write('\n')
                elif list_val is not None and type(list_val) is list:
                    f.write('{0:16} = '.format(key))
                    for val in list_val:
                        f.write('%18.16e ' % val)
                    f.write('\n')
                elif list_val is not None and type(list_val) is not list:
                    f.write('{0:16} = {1:10}\n'.format(key, list_val))
            f.write('\n')

        bonds = self.bondsbx
        for j in range(len(bonds)):
            f.write('/-----------------------------------------------------\n')
            f.write('bond           =   %s %s \n' % (bonds[j].bond[0]
                                                     , bonds[j].bond[1]))
            f.write('/-----------------------------------------------------\n')
            f.write('valence        =   %s %s \n' % (bonds[j].valence[0]
                                                     , bonds[j].valence[1]))
            if isinstance(bonds[j].scaling, int):
                f.write('scaling        = %18.16e \n' % bonds[j].scaling)
            else:
                f.write('scaling        = %18.16e \n' % bonds[j].scaling[0])
            for key, list_val in list(bonds[j].get_bondspar().items()):
                if list_val is not None:
                    if isinstance(list_val, list):
                        func = bonds[j].get_functions()[key]
                        para = list(list_val)
                    else:
                        func = list_val.get_name(environment='bopfox-ham')
                        para = list_val.get_numbers()
                    if func is not None:
                        f.write('{0:14} = {1:30}'.format(key, func))
                    else:
                        f.write('{0:14} = '.format(key))
                    for val in para:
                        # f.write('%18.16e '%val)
                        f.write(printf % val)
                    f.write('\n')
            for key, list_val in list(bonds[j].get_overlappar().items()):
                if list_val is not None:
                    if isinstance(list_val, list):
                        func = bonds[j].get_functions()[key]
                        para = list(list_val)
                    else:
                        func = list_val.get_name(environment='bopfox-ham')
                        para = list_val.get_numbers()
                    if func is not None:
                        f.write('{0:14} = {1:30}'.format(key, func))
                    else:
                        f.write('{0:14} = '.format(key))
                    for val in para:
                        # f.write('%18.16e '%val)
                        f.write(printf % val)
                    f.write('\n')
            for key, list_val in list(bonds[j].get_repetal().items()):
                if list_val is not None:
                    if isinstance(list_val, list):
                        func = bonds[j].get_functions()[key]
                        para = list(list_val)
                    else:
                        func = list_val.get_name(environment='bopfox-rep')
                        para = list_val.get_numbers()
                    if func is not None:
                        f.write('{0:14} = {1:30}'.format(key, func))
                    else:
                        f.write('{0:14} = '.format(key))
                    for val in para:
                        # f.write('%18.16e '%val)
                        f.write(printf % val)
                    f.write('\n')
            for key, list_val in list(bonds[j].get_cutpar().items()):
                if list_val is not None:
                    if isinstance(list_val, str):
                        f.write('{0:14} = {1:15}\n'.format(key, list_val))
                    elif isinstance(list_val, float):
                        f.write('{0:14} = {1:12.5f}\n'.format(key, list_val))
                    else:
                        for val in list_val:
                            f.write('{0:14} = {1:12.5f}\n'.format(key, val))
            f.write('\n')
        f.close()


###########################################################################

def filetodic(filename):
    """
    Returns a dictionary that contains the data in a bonds.bx or
    atoms.bx file. The data are arranged as;
        {version:{'bond' or 'atom':{key1:[], key2:[]}}}
    """
    from ase.calculators.bopfox import update_bopfox_keys
    bk_s, bk_b, bk_i, bk_f = update_bopfox_keys()
    infox_keys = bk_s + bk_b + bk_i + bk_f
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    data = {}
    keys_bo = [
        # bonds
        'valence'
        , 'scaling'
        , 'ddsigma'
        , 'ddpi'
        , 'dddelta'
        , 'sssigma'
        , 'sdsigma'
        , 'dssigma'
        , 'spsigma'
        , 'ppsigma'
        , 'pppi'
        , 'pssigma'
        , 'dpsigma'
        , 'pdsigma'
        , 'dppi'
        , 'pdpi'
        , 'ddsigmaoverlap'
        , 'ddpioverlap'
        , 'dddeltaoverlap'
        , 'sssigmaoverlap'
        , 'sdsigmaoverlap'
        , 'dssigmaoverlap'
        , 'spsigmaoverlap'
        , 'ppsigmaoverlap'
        , 'pppioverlap'
        , 'pssigmaoverlap'
        , 'dpsigmaoverlap'
        , 'pdsigmaoverlap'
        , 'dppioverlap'
        , 'pdpioverlap'
        , 'pairrepulsion'
        , 'eamattraction'
        , 'eamcoupling'
        , 'chargetransfer'
    ]
    repN = ['rep%s' % (i + 1) for i in range(100)]
    keys_bo += repN
    keys_at = [
        # atoms
        'mass'
        , 'densityr'
        , 'densitya'
        , 'pairrepr'
        , 'pairrepa'
        , 'embedding'
        , 'eamcutoff'
        , 'valenceorbitals'
        , 'valenceelectrons'
        , 'ed'
        , 'stonerintegral'
        , 'es'
        , 'ep'
        , 'jii'
        , 'onsitelevels'
        , 'delta_onsitelevels'
        , 'norbscreen'
    ]
    keys_cut = [
        'rcut'
        , 'dcut'
        , 'r2cut'
        , 'd2cut'
    ]

    def is_modelsbx():
        out = False
        for i in range(len(lines)):
            if 'model' in lines[i].lower():
                out = True
                break
        return out

    commentsid = ['/', '!', '#']
    version = None

    if is_modelsbx():
        head = 'model'
    else:
        head = 'version'
    for i in range(len(lines)):
        # avoid blanks
        if len(lines[i].strip()) > 3:
            split = lines[i].replace('=', ' ').split()
            key = split[0].strip()
            val = split[1:]
            if key.lower() == head:
                version = val[0]
                if head == 'version':
                    data[version] = {}
                elif head == 'model':
                    data[version] = {'infox_parameters': {key: val[0]}
                        , 'annotations': []}
            elif key.lower() == 'bond':
                bond = val[0] + '-' + val[1]
                data[version][bond] = {}
                data[version][bond]['version'] = {}
            elif key.lower() == 'atom':
                atom = val[0]
                data[version][atom] = {}
            elif key.lower() in keys_at:
                for j in range(len(val)):
                    try:
                        val[j] = float(val[j])
                    except:
                        val[j] = val[j]
                try:
                    data[version][atom][key] = val
                except:
                    pass
            elif key.lower() in keys_bo:
                try:
                    func = float(val[0])
                    func = None
                except:
                    func = val[0]
                if head == 'model':
                    data[version][bond]['version'][key.lower()] = func
                else:
                    data[version][bond]['version'] = version
                for j in range(len(val)):
                    try:
                        val[j] = float(val[j])
                    except:
                        val[j] = val[j]
                try:
                    data[version][bond][key] = val
                except:
                    pass
            elif key.lower() in keys_cut:
                for j in range(len(val)):
                    try:
                        val[j] = float(val[j])
                    except:
                        val[j] = val[j]
                try:
                    data[version][bond][key] = val
                except:
                    pass
            elif key.lower() in infox_keys and \
                    head == 'model':
                data[version]['infox_parameters'][key] = convert_value(val[0])
            elif version is not None and lines[i][0] in commentsid:
                data[version]['annotations'].append(lines[i][1:])
    return data


def convert_value(val):
    out = val
    if not isinstance(val, str):
        return out
    try:
        out = float(val)
    except:
        if out.lower() in ['t', 'true']:
            out = True
        elif out.lower() in ['f', 'false']:
            out = False
    if isinstance(out, float):
        if (out - int(out)) == 0:
            out = int(out)
    return out


def read_atomsbx(version=[], system=[], filename='atoms.bx'):
    """
    Extracts atoms parameters and stores them as BOPatoms objects.
    convert file into a dictionary of atoms parameters
    filename can either be:
       1. path/name
       2. path to directory where 'atoms.bx' is located
       3. name of file containing atoms database
    """
    cwd = os.getcwd()
    l = filename.split('/')
    if len(l) > 1 and os.path.isfile(filename) is True:
        path = ''
        for i in range(len(l) - 1):
            path += (l[i] + '/')
        os.chdir(path)
        filename = l[-1]
    elif len(l) > 1 and os.path.isdir(filename) is True:
        os.chdir(filename)
        filename = 'atoms.bx'
    elif len(l) == 1:
        pass
    else:
        raise Exception('Cannot find path to atoms file.')

    # filename can also be a path to a directory 
    atoms_data = filetodic(filename)
    os.chdir(cwd)
    # remove versions that are not specified version
    if version != []:
        for ver in list(atoms_data.keys()):
            if ver not in version:
                del atoms_data[ver]

    # remove atoms that are not specified in system 
    if system != []:
        for ver in list(atoms_data.keys()):
            for atom in list(atoms_data[ver].keys()):
                if atom not in system:
                    del atoms_data[ver][atom]

    bop_atoms = []
    # store data in atoms_data as atomsbx objects
    for ver in list(atoms_data.keys()):
        for atom in list(atoms_data[ver].keys()):
            abx = atomsbx(version=ver)
            abx.set(atom=atom)
            for para in atoms_data[ver][atom]:
                if para.lower() == 'mass':
                    abx.set(mass=atoms_data[ver][atom][para][0])
                else:
                    val = atoms_data[ver][atom][para]
                    val_list = []
                    for i in range(len(val)):
                        if type(val[i]) == float:
                            val_list.append(val[i])
                    exec("abx.set(%s=%s)" % (para.lower(), val_list))
                    """
                    if para.lower() == 'densityr':
                        abx.set(densityr=val_list)  
                    elif para.lower() == 'densitya' :
                        abx.set(densitya=val_list)
                    elif para.lower() == 'pairrepr':
                        abx.set(pairrepr=val_list) 
                    elif para.lower() == 'pairrepa':
                        abx.set(pairrepa=val_list)
                    elif para.lower() == 'embedding':
                        abx.set(embedding=val_list)
                    elif para.lower() == 'eamcutoff':
                        abx.set(eamcutoff=val_list) 
                    elif para.lower() == 'valenceorbitals':
                        abx.set(valenceorbitals=val_list)
                    elif para.lower() == 'valenceelectrons':
                        abx.set(valenceelectrons=val_list)
                    elif para.lower() == 'ed':
                        abx.set(ed=val_list)
                    elif para.lower() == 'stonerintegral':
                        abx.set(stonerintegral=val_list)
                    elif para.lower() == 'es':
                        abx.set(es=val_list)
                    elif para.lower() == 'ep':
                        abx.set(ep=val_list)
                    elif para.lower() == 'jii':
                        abx.set(jii=val_list)
                    elif para.lower() == 'onsitelevels':
                        abx.set(onsitelevels = val_list)
                    """
            bop_atoms.append(abx)
    return bop_atoms


def read_bondsbx(version=[], system=[], filename='bonds.bx'):
    """
    Extracts bonds parameters and stores them as bondsbx objects.
    """
    # generate all possible bonds combination in system
    required_bonds = []
    for i in system:
        for j in system:
            required_bonds.append(i + '-' + j)

    cwd = os.getcwd()
    l = filename.split('/')
    if len(l) > 1 and os.path.isfile(filename) is True:
        path = ''
        for i in range(len(l) - 1):
            path += (l[i] + '/')
        os.chdir(path)
        filename = l[-1]
    elif len(l) > 1 and os.path.isdir(filename) is True:
        os.chdir(filename)
        filename = 'bonds.bx'
    elif len(l) == 1:
        pass
    else:
        raise Exception('Cannot find path to bonds file.')

    bonds_data = filetodic(filename)
    os.chdir(cwd)
    # remove versions that are not specified version
    if version != []:
        for ver in list(bonds_data.keys()):
            if ver not in version:
                del bonds_data[ver]

    # remove bonds that are not specified in system 
    if system != []:
        for ver in list(bonds_data.keys()):
            for bond in list(bonds_data[ver].keys()):
                if bond not in required_bonds:
                    del bonds_data[ver][bond]

    bop_bonds = []
    # store data in bonds_data as bondsbx objects
    for ver in list(bonds_data.keys()):
        for bond in list(bonds_data[ver].keys()):
            bbx = bondsbx(version=ver)
            bbx.bond = bond.split('-')
            for para in bonds_data[ver][bond]:
                if para.lower() == 'valence':
                    bbx.valence = bonds_data[ver][bond][para][0:2]
                elif para.lower() == 'version':
                    bbx.version = bonds_data[ver][bond][para]
                else:
                    val = bonds_data[ver][bond][para]
                    val_list = []
                    for i in range(len(val)):
                        if type(val[i]) == float:
                            val_list.append(val[i])
                    exec("bbx.set(%s=%s)" % (para.lower(), val_list))
                    """
                    if para.lower() == 'scaling':
                        bbx.set(scaling=val_list)  
                    elif para.lower() == 'ddsigma' :
                        bbx.set(ddsigma=val_list)
                    elif para.lower() == 'ddpi':
                        bbx.set(ddpi=val_list) 
                    elif para.lower() == 'dddelta':
                        bbx.set(dddelta=val_list)
                    elif para.lower() == 'sssigma':
                        bbx.set(sssigma=val_list)
                    elif para.lower() == 'spsigma':
                        bbx.set(spsigma=val_list) 
                    elif para.lower() == 'ppsigma':
                        bbx.set(ppsigma=val_list)
                    elif para.lower() == 'dpsigma':
                        bbx.set(dpsigma=val_list)
                    elif para.lower() == 'pppi':
                        bbx.set(pppi=val_list)
                    elif para.lower() == 'dppi':
                        bbx.set(dppi=val_list)
                    elif para.lower() == 'pssigma':
                        bbx.set(pssigma=val_list)
                    elif para.lower() == 'ddsigmaoverlap':
                        bbx.set(ddsigmaoverlap=val_list)
                    elif para.lower() == 'ddpioverlap':
                        bbx.set(ddpioverlap=val_list)
                    elif para.lower() == 'dddeltaoverlap':
                        bbx.set(dddeltaoverlap=val_list)
                    elif para.lower() == 'sssigmaoverlap':
                        bbx.set(sssigmaoverlap=val_list)
                    elif para.lower() == 'spsigmaoverlap':
                        bbx.set(spsigmaoverlap=val_list)
                    elif para.lower() == 'ppsigmaoverlap':
                        bbx.set(ppsigmaoverlap=val_list)
                    elif para.lower() == 'dpsigmaoverlap':
                        bbx.set(dpsigmaoverlap=val_list)
                    elif para.lower() == 'dppioverlap':
                        bbx.set(dppioverlap=val_list)
                    elif para.lower() == 'pppioverlap':
                        bbx.set(pppioverlap=val_list)
                    elif para.lower() == 'pssigmaoverlap':
                        bbx.set(pssigmaoverlap=val_list)
                    elif para.lower() == 'pairrepulsion':
                        bbx.set(pairrepulsion=val_list)
                    elif para.lower() == 'eamattraction':
                        bbx.set(eamattraction=val_list)
                    elif para.lower() == 'eamcoupling':
                        bbx.set(eamcoupling=val_list)
                    elif para.lower() == 'chargetransfer':
                        bbx.set(chargetransfer=val_list)
                    """
            bop_bonds.append(bbx)
    return bop_bonds


def read_modelsbx(model=[], system=[], filename='models.bx'):
    """
    Extracts models parameters and stores them as modelsbx objects.
    """
    # generate all possible bonds combination in system
    required_bonds = []
    for i in system:
        for j in system:
            required_bonds.append(i + '-' + j)

    cwd = os.getcwd()

    if os.path.isfile(filename):
        pass
    else:
        filename = os.path.expanduser(filename)
        path = os.path.dirname(filename)
        filename = os.path.basename(filename)
        if not os.path.isdir(path):
            os.chdir(cwd)
            raise ValueError('Cannot find path to models file: %s' % path)
        os.chdir(path)
    models_data = filetodic(filename)
    os.chdir(cwd)

    #    l = filename.split('/')
    #    if len(l) > 1 and os.path.isfile(filename) is True:
    #        path = ''
    #        for i in range(len(l)-1):
    #            path += (l[i]+'/')
    #        os.chdir(path)
    #        filename = l[-1]
    #    elif len(l) > 1 and os.path.isdir(filename) is True:
    #        os.chdir(filename)
    #        filename = 'bonds.bx'
    #    elif len(l) == 1:
    #        pass
    #    else:
    #        raise Exception('Cannot find path to models file.')
    #    models_data = filetodic(filename)
    #    os.chdir(cwd)

    # remove models that are not specified in model
    if model != []:
        for ver in list(models_data.keys()):
            if ver not in model:
                del models_data[ver]

    # remove atoms/bonds that are not specified in system 
    if system != []:
        for ver in list(models_data.keys()):
            for key in list(models_data[ver].keys()):
                if key.lower() in ['infox_parameters', 'annotations']:
                    pass
                elif '-' in key:
                    bond = key
                    if bond not in required_bonds:
                        del models_data[ver][bond]
                else:
                    atom = key
                    if atom not in system:
                        del models_data[ver][atom]
    bop_models = []
    # store data in bonds_data as bondsbx objects
    for ver in list(models_data.keys()):
        bop_atoms = []
        bop_bonds = []
        for key in list(models_data[ver].keys()):
            if key == 'infox_parameters':
                infox_parameters = models_data[ver][key]
            elif key == 'annotations':
                annotations = models_data[ver][key]
            elif '-' in key:
                bond = key
                # version is irrelevant
                bbx = bondsbx(version=None)
                bbx.bond = bond.split('-')
                for para in models_data[ver][bond]:
                    if para.lower() == 'valence':
                        bbx.valence = models_data[ver][bond][para][0:2]
                    elif para.lower() == 'version':
                        bbx.functions = models_data[ver][bond][para]
                        bbx.version = bbx.functions[list(bbx.functions.keys())[0]]
                    elif para.lower() in ['valence', 'scaling', 'pairrepulsion'
                        , 'eamattraction', 'eamcoupling'
                        , 'chargetransfer']:
                        val = models_data[ver][bond][para]
                        val_list = []
                        for i in range(len(val)):
                            if type(val[i]) == float:
                                val_list.append(val[i])
                        exec("bbx.set(%s=%s)" % (para.lower(), val_list))
                    elif para.lower() in ['rcut', 'dcut'
                        , 'r2cut', 'd2cut']:
                        val = float(models_data[ver][bond][para][0])
                        exec("bbx.set(%s=%s)" % (para.lower(), val))
                    else:
                        val = models_data[ver][bond][para]
                        val_list = []
                        for i in range(len(val)):
                            if type(val[i]) == float:
                                val_list.append(val[i])
                        exec("bbx.set(%s=%s)" % (para.lower(), val_list))
                bop_bonds.append(bbx)
            else:
                atom = key
                # for compatibility purposes
                abx = atomsbx(version='canonicaltb')
                abx.set(atom=atom)
                for para in models_data[ver][atom]:
                    if para.lower() == 'mass':
                        abx.set(mass=models_data[ver][atom][para][0])
                    else:
                        val = models_data[ver][atom][para]
                        val_list = []
                        for i in range(len(val)):
                            if type(val[i]) == float:
                                val_list.append(val[i])
                        exec("abx.set(%s=%s)" % (para.lower(), val_list))
                bop_atoms.append(abx)
        mbx = modelsbx(atomsbx=bop_atoms, bondsbx=bop_bonds
                       , infox_parameters=infox_parameters, model=ver
                       , annotations=annotations)
        bop_models.append(mbx)
    return bop_models


def average_models(models):
    """
    Returns modelsbx with the average parameters of models
    """
    import numpy as np
    # get model name
    name = models[0].model
    for i in range(1, len(models)):
        if name != models[i].model:
            raise ValueError('Cannot add different models')
    # infox
    infox = {}
    toave = []
    for i in range(len(models)):
        infoxm = models[i].get_infox()
        for key, val in list(infoxm.items()):
            if val is None:
                continue
            if not (key in infox):
                infox[key] = val
            else:
                if infox[key] != infoxm[key]:
                    toave.append(key)
                    # infox keys with different values, get most frequent
    for key in toave:
        vals = []
        counts = []
        for i in range(len(models)):
            infoxm = models[i].get_infox()
            if key in infoxm:
                if infoxm[key] in vals:
                    counts[vals.index(infox[key])] += 1
                else:
                    vals.append(infox[key])
                    counts.append(1)
        val = vals[counts.index(max(counts))]
        infox[key] = val
    # atoms
    # get elements first
    atoms_mass = {}
    for i in range(len(models)):
        abx = models[i].atomsbx
        for j in range(len(abx)):
            if abx[j].atom not in atoms_mass:
                atoms_mass[abx[j].atom] = abx[j].mass
                # get average of atoms parameters
    abx = []
    akeys = atomsbx().get_keys()
    for atom, mass in list(atoms_mass.items()):
        abxi = atomsbx(atom=atom, mass=mass)
        apar = {}
        for j in range(len(akeys)):
            aval = []
            for k in range(len(models)):
                val = models[k].get_parameters(akeys[j], atom=atom)
                if val is None:
                    continue
                aval.append(val)
            if len(aval) > 0:
                aval = np.average(aval, axis=0)
                if isinstance(aval, np.ndarray):
                    aval = list(aval)
                apar[akeys[j]] = aval
        abxi.set_atomspar(apar)
        abx.append(abxi)
        # bonds
    # get bonds first
    bonds = []
    valence = {}
    scaling = {}
    for i in range(len(models)):
        bbx = models[i].bondsbx
        for j in range(len(bbx)):
            bbxb = bbx[j].bond
            if bbxb not in bonds:
                bonds.append(bbxb)
                valence['%s-%s' % (bbxb[0], bbxb[1])] = bbx[j].valence
                scaling['%s-%s' % (bbxb[0], bbxb[1])] = bbx[j].scaling
    # get average of bonds parameters
    bbx = []
    bkeys = bondsbx().get_keys()
    for i in range(len(bonds)):
        bbxi = bondsbx(bond=bonds[i]
                       , valence=valence['%s-%s' % (bonds[i][0], bonds[i][1])]
                       , scaling=scaling['%s-%s' % (bonds[i][0], bonds[i][1])])
        bpar = {}
        for j in range(len(bkeys)):
            bval = []
            for k in range(len(models)):
                val = models[k].get_parameters(bkeys[j], bond=bonds[i])
                if val is None:
                    continue
                if not isinstance(val, list):
                    # expecting functions
                    val = val.get_numbers()
                bval.append(val)
            if len(bval) > 0:
                bval = np.average(bval, axis=0)
                if isinstance(bval, np.ndarray):
                    bval = list(bval)
                bpar[bkeys[j]] = bval
        bbxi.set_bondspar(bpar)
        bbx.append(bbxi)
    # generate model   
    model = modelsbx(infox_parameters=infox, atomsbx=abx, bondsbx=bbx, model=name)
    return model


if __name__ == "__main__":
    bonds = read_bondsbx(version=['bochumtb'], system=['Fe', 'Nb']
                         , filename='bonds.bx')
    bonds[0].write(filename='bonds_temp.bx')
    atoms = read_atomsbx(version=['canonicaltb'], system=['Fe', 'Nb']
                         , filename='atoms.bx')
    atoms[0].write(filename='atoms_temp.bx')
    bopfox_para = {
        'ini_magmoms': {'W': 1.0},
        'rcut': 4.2,
        'r2cut': 6.0,
        'dcut': 0.5,
        'd2cut': 0.5,
        'bandwidth': 'findeminemax',
        'terminator': 'constantabn',
        'bopkernel': 'jackson',
        'nexpmoments': 200,
        'scfsteps': 500,
        'moments': 9}
    models = modelsbx(atomsbx=atoms, bondsbx=bonds
                      , infox_parameters=bopfox_para, model='test')
    models.write()
    # average models
    bbx1 = [bondsbx(bond=['Fe', 'Fe'], valence=['d', 'd']
                    , ddsigma=[-10., -1., 0.5])]
    bbx2 = [bondsbx(bond=['Fe', 'Fe'], valence=['d', 'd'], ddsigma=[-20., -1., 0.5])]
    abx1 = [atomsbx(atom='Fe', mass=55.847, valenceelectrons=[7]
                    , stonerintegral=[0.8, 0.8, 0.8])]
    abx2 = [atomsbx(atom='Fe', mass=55.847, valenceelectrons=[6.8]
                    , stonerintegral=[0.7, 0.7, 0.7])]
    bp1 = dict(bopfox_para)
    bp2 = dict(bopfox_para)
    bp2['moments'] = 12
    model1 = modelsbx(infox_parameters=bp1, bondsbx=bbx1, atomsbx=abx1
                      , model='test')
    model2 = modelsbx(infox_parameters=bp2, bondsbx=bbx2, atomsbx=abx2
                      , model='test')
    model = average_models([model1, model2])
    model.write()
