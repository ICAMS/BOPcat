#!/usr/bin/env python

# Definition of the CATKernel, CATobjective and CAToptimizer objects

# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import numpy as np
from . import functions as Funcs
import os


###########################################################################
class Beta:
    """
    Defines a set of bond integrals for constructing the hamiltonian. 

    It reads the data points from a file stored in pathtobetas. 
    A set of functions in fitfuncs is parametrized with respect to the data.

    The betas file should be named as 
    [struc]_[ele1][ele2]_[basis]_[betatype].betas

    See the folder betas in the examples folder for reference. 

    :Parameters:
    
        - *structure*: str

            structure from which the bond integrals were derived

        - *bondpair*: tuple

            pair of chemical symbols

        - *basis*: str

            basis used for downfolding

        - *betatype*: str

            can be unscreened, overlap, loewdin or other beta type

        - *valence*: tuple

            pair of valency, e.g. ``sd``

        - *pathtobetas*: str

            path to betas file
             
    """

    def __init__(self, **kwargs):
        self.structure = None
        self.bondpair = None
        self.basis = None
        self.betatype = 'unscreened'
        self.valence = None
        self.pathtobetas = None
        self.bondlengths = None
        self.sssigma = None
        self.spsigma = None
        self.sdsigma = None
        self.pssigma = None
        self.ppsigma = None
        self.pppi = None
        self.pppi_minus = None
        self.pdsigma = None
        self.pdpi = None
        self.pdpi_minus = None
        self.dssigma = None
        self.dpsigma = None
        self.dppi = None
        self.dppi_minus = None
        self.ddsigma = None
        self.ddpi = None
        self.ddpi_minus = None
        self.dddelta = None
        self.dddelta_minus = None
        self.fitfuncs = None
        self.bondparam = None
        self.set(**kwargs)

    def set(self, **kwargs):
        for key in kwargs:
            if key.lower() == 'structure':
                self.structure = kwargs[key]
            if key.lower() == 'bondpair':
                self.bondpair = kwargs[key]
            if key.lower() == 'basis':
                self.basis = kwargs[key]
            if key.lower() == 'betatype':
                self.betatype = kwargs[key]
            if key.lower() == 'valence':
                self.valence = kwargs[key]
            if key.lower() == 'pathtobetas':
                self.pathtobetas = kwargs[key]
            if key.lower() == 'bondlengths':
                self.bondlengths = kwargs[key]
            if key.lower() == 'sssigma':
                self.sssigma = kwargs[key]
            if key.lower() == 'spsigma':
                self.spsigma = kwargs[key]
            if key.lower() == 'sdsigma':
                self.sdsigma = kwargs[key]
            if key.lower() == 'pssigma':
                self.pssigma = kwargs[key]
            if key.lower() == 'ppsigma':
                self.ppsigma = kwargs[key]
            if key.lower() == 'pppi':
                self.pppi = kwargs[key]
            if key.lower() == 'pppi_minus':
                self.pppi_minus = kwargs[key]
            if key.lower() == 'pdsigma':
                self.pdsigma = kwargs[key]
            if key.lower() == 'pdpi':
                self.pdpi = kwargs[key]
            if key.lower() == 'pdpi_minus':
                self.pdpi_minus = kwargs[key]
            if key.lower() == 'dssigma':
                self.dssigma = kwargs[key]
            if key.lower() == 'dpsigma':
                self.dpsigma = kwargs[key]
            if key.lower() == 'dppi':
                self.dppi = kwargs[key]
            if key.lower() == 'dppi_minus':
                self.dppi_minus = kwargs[key]
            if key.lower() == 'ddsigma':
                self.ddsigma = kwargs[key]
            if key.lower() == 'ddpi':
                self.ddpi = kwargs[key]
            if key.lower() == 'ddpi_minus':
                self.ddpi_minus = kwargs[key]
            if key.lower() == 'dddelta':
                self.dddelta = kwargs[key]
            if key.lower() == 'dddelta_minus':
                self.dddelta_minus = kwargs[key]
            if key.lower() == 'fitfuncs':
                self.fitfuncs = kwargs[key]
            else:
                raise NameError("Unrecognized keyword %s" % key)

    def set_fitfuncs(self, fitfuncs):
        """
        Set fitting function , default(sum_exponential)
        """
        betas = self.zip_betas()
        if fitfuncs is None:
            fit_funcs = {}
            for name, beta in list(betas.items()):
                fit_funcs[name] = Funcs.exponential()
            self.fitfuncs = dict(fit_funcs)
        elif isinstance(fitfuncs, dict):
            for name, beta in list(betas.items()):
                if name not in fitfuncs:
                    exp0 = Funcs.exponential()
                    # exp1 = Funcs.exponential(parameters=[1,1]
                    #        ,constraints=[True,True])
                    # exp2 = Funcs.exponential(parameters=[1,1]
                    #        ,constraints=[True,True,True])
                    # exp3 = Funcs.exponential(parameters=[1,1]
                    #        ,constraints=[True,True])
                    # sumexp = Funcs.sum_funcs(functions=[exp0,exp1,exp2,exp3])
                    sumexp = exp0
                    self.fitfuncs[name] = sumexp
                else:
                    self.fitfuncs[name] = fitfuncs[name]
        else:  # expecting here a functional class
            self.fitfuncs = {}
            for name, beta in list(betas.items()):
                self.fitfuncs[name] = fitfuncs.copy()

    def _done_read(self):
        empty = True
        betas = self.zip_betas()
        for name, beta in list(betas.items()):
            if beta is not None:
                empty = False
                break
        return not (empty)

    def zip_betas(self):
        """ 
        Returs a dictionary of the bond integrals
        """
        names = ['sssigma', 'spsigma', 'sdsigma', 'pssigma'
            , 'ppsigma', 'pppi', 'pdsigma', 'pdpi', 'dssigma'
            , 'dpsigma', 'dppi', 'ddsigma', 'ddpi', 'dddelta']
        betas = {}
        for name in names:
            if self.betatype.lower() == "overlap":
                name += 'overlap'
            if "sssigma" in name:
                betas[name] = self.sssigma
            elif "spsigma" in name:
                betas[name] = self.spsigma
            elif "sdsigma" in name:
                betas[name] = self.sdsigma
            elif "pssigma" in name:
                betas[name] = self.pssigma
            elif "ppsigma" in name:
                betas[name] = self.ppsigma
            elif "pppi" in name:
                betas[name] = self.pppi
            elif "pdsigma" in name:
                betas[name] = self.pdsigma
            elif "pdpi" in name:
                betas[name] = self.pdpi
            elif "dssigma" in name:
                betas[name] = self.dssigma
            elif "dpsigma" in name:
                betas[name] = self.dpsigma
            elif "dppi" in name:
                betas[name] = self.dppi
            elif "ddsigma" in name:
                betas[name] = self.ddsigma
            elif "ddpi" in name:
                betas[name] = self.ddpi
            elif "dddelta" in name:
                betas[name] = self.dddelta
        return betas

    def read_betas(self):
        """
        Reads in bond integrals from a betamaker output file(.betas) 
        saved in pathtobeta. If not provided, will read from folder:
        /betas. The filenames must follow the format: 
        [structure]_[elementA][elementB]_[basis]_[betatype].betas
        """
        cwd = os.getcwd()
        self.pathtobetas = os.path.expanduser(self.pathtobetas)
        if not os.path.isdir(self.pathtobetas):
            msg = "Provided path; %s does not exist." % self.pathtobetas
            raise ValueError(msg)
        os.chdir(self.pathtobetas)
        b1 = "%s%s" % (self.bondpair[0], self.bondpair[1])
        b2 = "%s%s" % (self.bondpair[1], self.bondpair[0])
        temp = os.listdir(os.getcwd())
        success = False
        for d in temp:
            if (self.structure in d) and (b1 in d or b2 in d) \
                    and (self.betatype.lower() == \
                         d.split("_")[3].split(".")[0].lower()) \
                    and (self.basis.lower() == d.split("_")[2].lower()):
                l = open(d).readlines()
                for i in range(len(l)):
                    if "#" in l[i]:
                        l[i] = 'None'
                        continue
                    l[i] = l[i].split()
                    if len(l[i]) < 2:
                        l[i] = 'None'
                        continue
                    for j in range(len(l[i])):
                        l[i][j] = float(l[i][j])
                l = [li for li in l if li != 'None']
                l.sort()
                l = np.array(l).T
                self.bondlengths = l[0]
                if len(l) == 9:
                    self.sssigma = l[1]
                    self.sdsigma = l[2]
                    self.dssigma = l[3]
                    self.ddsigma = l[4]
                    self.ddpi = l[5]
                    self.ddpi_minus = l[6]
                    self.dddelta = l[7]
                    self.dddelta_minus = l[8]
                    if self.valence is None:
                        self.valence = ["sd", "sd"]
                elif len(l) == 7:
                    self.sssigma = l[1]
                    self.spsigma = l[2]
                    self.pssigma = l[3]
                    self.ppsigma = l[4]
                    self.pppi = l[5]
                    self.pppi_minus = l[6]
                    if self.valence is None:
                        self.valence = ["sp", "sp"]
                elif len(l) == 20:
                    self.sssigma = l[1]
                    self.spsigma = l[2]
                    self.sdsigma = l[3]
                    self.pssigma = l[4]
                    self.ppsigma = l[5]
                    self.pppi = l[6]
                    self.pppi_minus = l[7]
                    self.pdsigma = l[8]
                    self.pdpi = l[9]
                    self.pdpi_minus = l[10]
                    self.dssigma = l[11]
                    self.dpsigma = l[12]
                    self.dppi = l[13]
                    self.dppi_minus = l[14]
                    self.ddsigma = l[15]
                    self.ddpi = l[16]
                    self.ddpi_minus = l[17]
                    self.dddelta = l[18]
                    self.dddelta_minus = l[19]
                    if self.valence is None:
                        self.valence = ["spd", "spd"]
                elif len(l) == 2:
                    self.sssigma = l[1]
                else:
                    print("Expecting only 9 columns for sd-sd")
                    print("               7 columns for sp-sp")
                    print("              20 columns for spd-spd")
                    print("               1 column  for s-s")
                    msg = "Cannot determine valency for %d columns" % len(l)
                    raise ValueError(msg)
                success = True
                break
        os.chdir(cwd)
        if not success:
            print("##########################################")
            print("Error in reading betas.")
            print("##########################################")
            raise ValueError('module beta error')

    def _is_required(self, name):
        val = list(self.valence)
        val.sort()
        if val == ['s', 's']:
            req = ['sssigma']
        elif val == ['p', 'p']:
            req = ['ppsigma', 'pppi']
        elif val == ['d', 'd']:
            req = ['ddsigma', 'ddpi', 'dddelta']
        elif val == ['p', 's']:
            req = ['spsigma', 'pssigma']
        elif val == ['d', 's']:
            req = ['sdsigma', 'dssigma']
        elif val == ['d', 'p']:
            req = ['pdsigma', 'dpsigma', 'pdpi', 'dppi']
        elif val == ['s', 'sp']:
            req = ['sssigma', 'spsigma', 'pssigma']
        elif val == ['s', 'sd']:
            req = ['sssigma', 'sdsigma', 'dssigma']
        elif val == ['s', 'pd']:
            req = ['spsigma', 'pssigma', 'sdsigma', 'dssigma']
        elif val == ['p', 'sp']:
            req = ['spsigma', 'pssigma', 'ppsigma', 'pppi']
        elif val == ['p', 'sd']:
            req = ['spsigma', 'pssigma', 'pdsigma', 'dpsigma', 'pdpi', 'dppi']
        elif val == ['p', 'pd']:
            req = ['ppsigma', 'pppi', 'pdsigma', 'dpsigma', 'pdpi', 'dppi']
        elif val == ['d', 'sp']:
            req = ['sdsigma', 'dssigma', 'pdsigma', 'dpsigma', 'pdpi', 'dppi']
        elif val == ['d', 'sd']:
            req = ['sdsigma', 'dssigma', 'ddsigma', 'ddpi', 'dddelta']
        elif val == ['d', 'pd']:
            req = ['dpsigma', 'pdsigma', 'ddsigma', 'ddpi', 'dddelta']
        elif val == ['sp', 'sp']:
            req = ['sssigma', 'spsigma', 'pssigma', 'ppsigma', 'pppi']
        elif val == ['sd', 'sp']:
            req = ['sssigma', 'spsigma', 'pssigma', 'sdsigma', 'dssigma'
                , 'dpsigma', 'pdsigma', 'dppi', 'pdpi']
        elif val == ['pd', 'sp']:
            req = ['spsigma', 'pssigma', 'sdsigma', 'dssigma', 'ppsigma', 'pppi'
                , 'pdsigma', 'dpsigma', 'pdpi', 'dppi']
        elif val == ['sd', 'sd']:
            req = ['sssigma', 'sdsigma', 'dssigma', 'ddsigma', 'ddpi', 'dddelta']
        elif val == ['pd', 'sd']:
            req = ['spsigma', 'pssigma', 'pdsigma', 'dpsigma', 'pdpi', 'dppi'
                , 'dssigma', 'sdsigma', 'ddsigma', 'ddpi', 'dddelta']
        elif val == ['pd', 'pd']:
            req = ['ppsigma', 'pppi', 'pdsigma', 'dpsigma', 'pdpi', 'dppi'
                , 'ddsigma', 'ddpi', 'dddelta']
        elif val == ['sp', 'spd']:
            req = ['sssigma', 'spsigma', 'pssigma', 'sdsigma', 'dssigma', 'ppsigma'
                , 'pppi', 'pdsigma', 'dpsigma', 'pdpi', 'dppi']
        elif val == ['sd', 'spd']:
            req = ['sssigma', 'spsigma', 'pssigma', 'sdsigma', 'dssigma'
                , 'dpsigma', 'pdsigma', 'dppi', 'pdpi', 'ddsigma', 'ddpi'
                , 'dddelta']
        elif val == ['pd', 'spd']:
            req = ['pssigma', 'spsigma', 'ppsigma', 'ppi', 'pdsigma', 'dpsigma'
                , 'pdpi', 'dppi', 'dssigma', 'sdsigma', 'ddsigma', 'ddpi'
                , 'dddelta']
        elif val == ['spd', 'spd']:
            req = ['sssigma', 'spsigma', 'pssigma', 'sdsigma', 'dssigma', 'ppsigma'
                , 'pppi', 'pdsigma', 'dpsigma', 'pdpi', 'dppi', 'ddsigma', 'ddpi'
                , 'dddelta']
        else:
            raise ValueError('Invalid valency. This should never happen.')
        out = False
        for b in req:
            if b in name:
                out = True
                break
        return out

    def _fit(self, betaname, ftype, **kwargs):
        betas = self.zip_betas()[betaname]
        # dimer overlap is too large
        if self.betatype == 'overlap':
            betas = betas / 1.
        if betas is not None:
            Rij = self.bondlengths
            if 'sum' in ftype.get_name():
                param = Funcs.fit_sum(Rij, betas, ftype, **kwargs)
            else:
                param = Funcs.fit(Rij, betas, ftype, **kwargs)
        else:
            param = None
        return param

    def gen_fit_params(self, **kwargs):
        self.bondparam = {}
        if not (self._done_read()):
            self.read_betas()
            self.set_fitfuncs(self.fitfuncs)
        betas = self.zip_betas()
        for name, beta in list(betas.items()):
            if beta is not None and self._is_required(name):
                newfunc = self.fitfuncs[name].copy()
                param = self._fit(name, self.fitfuncs[name], **kwargs)
                newfunc.set_parameters(param)
                self.bondparam[name] = newfunc
            else:
                self.fitfuncs[name] = None
                self.bondparam[name] = None

    #     if self.betatype == 'unscreened':
    #         try:
    #             x = self.bondlengths
    #             pl.plot(x,betas[name],'o',label=name)
    #             pl.plot(x,self.bondparam[name](x))
    #         except:
    #             pass
    # if self.betatype == 'unscreened':
    #     pl.legend(loc='best')
    #     pl.show()
    #     raise

    def get_bondparam(self):
        if self.bondparam is None:
            self.gen_fit_params()
        return self.bondparam


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    beta = Beta()
    cwd = os.getcwd()
    beta.structure = 'dimer'
    beta.bondpair = ['Fe', 'Fe']
    beta.basis = 'tz0'
    beta.betatype = 'overlap'
    beta.valence = ['sd', 'sd']
    beta.pathtobetas = cwd + '/betas'
    exp0 = Funcs.exponential()
    exp1 = Funcs.exponential(parameters=[1, 1, 1], constraints=[True, True, True])
    exp2 = Funcs.exponential(parameters=[1, 1, 1], constraints=[True, True, True])
    exp3 = Funcs.exponential(parameters=[1, 1, 1], constraints=[True, True, True])
    sumexp = Funcs.sum_funcs(functions=[exp0, exp1, exp2, exp3])
    beta.set_fitfuncs(sumexp)
    beta.gen_fit_params()
    fit = beta.get_bondparam()
    betas = beta.zip_betas()
    for name, func in list(fit.items()):
        if func is not None:
            x = beta.bondlengths
            y0 = betas[name]
            y1 = func(x)
            pl.plot(x, y0, 'o')
            pl.plot(x, y1, '-', label=name)
    pl.legend(loc='best')
    pl.xlabel(r'R($\AA$)')
    pl.ylabel(r'$\beta$(eV)')
    pl.show()
