#!/usr/bin/env python

# Definition of the CATParam object

# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import os
from .beta import Beta
import numpy as np
from . import bopmodel
from .catcalc import CATCalc
from . import functions as Funcs
from .variables import atomic_properties, pathtobetas, repulsive_keys_bopfox
from .variables import pathtoonsites
from .utils import make_pairs, neigh_dist
# import matplotlib.pyplot as pl
from .output import print_format
from .sedt import energy_diff


###########################################################################
class CATParam:
    """
    Defines the model for optimization.
    
    It includes functionalities to read or generate models.
    
    The models are stored as list of model objects, e.g. modelsbx.
    
    .. todo:: generalize models format for other calculators

    :Parameters:
    
        - *controls*: instance of CATControls   
        
            CATControls object to initialize parameters      
                                  
        - *elements*: list   
            
            list of chemical symbols

        - *model*: str
        
            model version

        - *model_filename*: str
        
            file containing model, can also be path to file 
                
        - *valences*: dict
        
            dictionary of element:valency

        - *valenceelectrons*: dict
        
            dictionary of element:number of valence electrons
                
                ``None``: will read valenceelectrons from 
                :func:`variables.atomic_properties`

        - *functions*: dict   
              
            dictionary of keyword:bondpair:function or keyword:function, 
            function is one of the instances in functions module. The latter
            implies that all bond pairs will have the same functional form

        - *cutoff*: dict
        
            dictionary of cutoff keyword:bondpair:value or keyword:value.
            The latter implies that all bond pairs will have the same value
          
        - *pathtobetas*: str
          
            path to betas files
            
            ``None``: will use :func:`variables.pathtobetas`

        - *pathtoonsites*: str
        
            path to onsites files 
            
            ``None``: will use :func:`variables.pathtoonsites`

        - *betafitstruc*: str
        
            structure from which the betas were derived (see format of betas 
            files) 
            
            ``None``: will default to 'dimer'

        - *betatype*: str
        
            method used in beta generation
            
            ``None``: will default to 'loewdin'

        - *betabasis*: str
        
            basis used in beta generation 
            ``None``: will default to 'tz0'

        - *calculator*: str
        
            name of the calculator 

        - *calculator_settings*: dict
        
            dictionary of controls passed to the calculator
                
        - *orthogonal*: bool
        
            orthogonal or non-orthogonal model

        - *data*: instance of CATData   

            CATData object to required to generate models      

    """

    def __init__(self, **kwargs):
        self.controls = None
        self.elements = None
        self.model = None
        self.model_filename = 'models.bx'
        self.pairs = None
        self.valences = None
        self.valenceelectrons = None
        self.scaling = 1
        self.functions = {}
        self.cutoff = {}
        self.pathtobetas = None
        self.pathtoonsites = None
        self.betafitstruc = None
        self.orthogonal = True
        self.betatype = None
        self.betabasis = None
        self.betas = {}
        self.repulsion = {}
        self.calculator = 'bopfox'
        self.data = None
        self.models = []
        self.scores = {}
        self._init_msg()
        self.set(**kwargs)

    def _init_msg(self):
        print_format('Building CATParam', level=1)

    def set(self, **kwargs):
        if 'controls' in kwargs:
            self.controls = kwargs['controls']
            if self.controls is not None:
                self._unfold_controls()
        for key, val in list(kwargs.items()):
            # if val is None:
            #    continue
            if key.lower() == 'elements':
                self.elements = val
            elif key.lower() == 'model':
                self.model = val
            elif key.lower() == 'model_filename':
                self.model_filename = val
            elif key.lower() == 'data':
                self.data = val
            elif key.lower() == 'valences':
                self.valences = val
            elif key.lower() == 'valenceelectrons':
                self.valenceelectrons = val
            elif key.lower() == 'scaling':
                self.scaling = val
            elif key.lower() == 'functions':
                self.functions = val
            elif key.lower() == 'cutoff':
                self.cutoff = val
            elif key.lower() == 'pathtobetas':
                self.pathtobetas = val
            elif key.lower() == 'pathtoonsites':
                self.pathtoonsites = val
            elif key.lower() == 'betafitstruc':
                self.betafitstruc = val
            elif key.lower() == 'orthogonal':
                self.orthogonal = val
            elif key.lower() == 'betatype':
                self.betatype = val
            elif key.lower() == 'betabasis':
                self.betabasis = val
            elif key.lower() == 'calculator_settings':
                self.calculator_settings = val
            elif key.lower() == 'controls':
                pass
            else:
                raise ValueError('Unrecognized key %s' % key)
        self.pairs = make_pairs(self.elements)
        if self.model is None:
            print_format('Generating models.', level=2)
            self.models.append(self.gen_model())
        else:
            print_format('Fetching models.', level=2)
            self.models.append(self.read_model())

    def _unfold_controls(self):
        self.elements = self.controls.elements
        self.valences = self.controls.model_valences
        self.valenceelectrons = self.controls.model_valenceelectrons
        self.functions = self.controls.model_functions
        self.pathtobetas = self.controls.model_pathtobetas
        self.pathtoonsites = self.controls.model_pathtoonsites
        self.betafitstruc = self.controls.model_betafitstruc
        self.orthogonal = self.controls.model_orthogonal
        self.betatype = self.controls.model_betatype
        self.betabasis = self.controls.model_betabasis
        self.calculator_settings = self.controls.calculator_settings
        self.model = self.controls.model
        self.model_filename = self.controls.model_pathtomodels
        self.cutoff = self.controls.model_cutoff
        if self.functions is None and (self.model is None or \
                                       self.model_filename is None):
            self.functions = self.controls.gen_functions()

    def get_fitfunc(self, bondpair):
        """
        Returns a dictionary of keyword:functions 
        
        :Parameters:
        
            - *bondpair*: list 
        
                pair of chemical symbols
        """
        funcs = self.functions
        out = {}
        for key in funcs:
            if isinstance(funcs[key], dict):
                for bp, val in list(funcs[key].items()):
                    if (bp == '%s-%s' % (bondpair[0], bondpair[1])) or \
                            (bp == '%s-%s' % (bondpair[1], bondpair[0])):
                        out[key] = val.copy()
            else:
                out[key] = funcs[key].copy()
        return out

    def _get_beta_param(self, betatype, bondpair):
        beta = Beta()
        beta.structure = self.betafitstruc
        beta.bondpair = bondpair
        beta.basis = self.betabasis
        beta.betatype = betatype
        beta.pathtobetas = self.pathtobetas
        beta.fitfuncs = self.get_fitfunc(bondpair)
        if self.valences is not None:
            if bondpair[0] in self.valences and \
                    bondpair[1] in self.valences:
                beta.valence = [self.valences[bondpair[0]]
                    , self.valences[bondpair[1]]]
        if beta.valence is None:
            print_format('Warning: No valency found.', level=3)
            print_format('Will generate from betas file.', level=3)
        if beta.bondpair is None:
            raise ValueError("Provide value for bondpair.")
        if beta.structure is None:
            print_format("Structure for betas is not provided", level=2)
            print_format("Setting to dimer", level=3)
            beta.structure = "dimer"
        if beta.pathtobetas is None:
            print_format("Path to betas is not provided.", level=2)
            print_format("Setting to %s" % pathtobetas(), level=3)
            beta.pathtobetas = pathtobetas()
        if beta.betatype is None:
            print_format("Beta type is not provided.", level=2)
            print_format("Assuming loewdin", level=3)
            beta.betatype = 'loewdin'
        if beta.basis is None:
            print_format("Basis name is not provided.", level=2)
            print_format("Assuming tz0", level=3)
            beta.basis = 'tz0'
        bondparam = beta.get_bondparam()
        bondkey = '%s-%s' % (bondpair[0], bondpair[1])
        if bondkey not in self.betas:
            self.betas[bondkey] = {}
            self.betas[bondkey][betatype] = bondparam
        return bondparam

    def get_initial_bondparam(self, bondpair):
        """
        Returns a dictionary of keyword:functions resulting from the initial
        fitting of the bond/overlap integrals.
        
        :Parameters:
        
            - *bondpair*: list 
        
                pair of chemical symbols    
        """
        if self.orthogonal:
            betatypes = [self.betatype]
        else:
            betatypes = ["unscreened", "overlap"]
        bond_param = {}
        for betatype in betatypes:
            bond_param.update(self._get_beta_param(betatype, bondpair))
        return bond_param

    def _initialize_funcs(self, func, bondpair):
        unique = []
        for b in bondpair:
            if b not in unique:
                unique.append(b)
        if func.get_name(environment='generic') == 'spline':
            gs = self.data.get_ground_state(elements=unique)
            nlist = neigh_dist(gs, N=5)
            num = list(func.numbers)
            num[5:] = nlist
            func.set_parameters(num)
            func.set(parameters=num[:5], constraints=[True] * 5 + [False] * 5)
        elif func.get_name(environment='generic') == 'env_lin':
            func.set(parameters=[], constraints=[False] * len(func.numbers))
        return func

    def get_initial_repparam(self, bondpair):
        """
        Returns a dictionary of keyword:functions resulting from the initial
        fitting of the dimer repulsive energy
        
        :Parameters:
            
            - *bondpair*: list 
            
                pair of chemical symbols            
        """
        func = self.get_fitfunc(bondpair)
        # necessary to order repulsive functions since these are fitted
        # recursively
        repkeys = []
        for k in repulsive_keys_bopfox():
            if k in func:
                repkeys.append(k)
        repfuncs = []
        for k in repkeys:
            repfuncs.append(func[k])
            # initialize functions
        # for i in range(len(repfuncs)):
        #    repfuncs[i] = self._initialize_funcs(repfuncs[i],bondpair)
        repfunc = Funcs.sum_funcs(functions=repfuncs)

        """            
        # get dimer structures
        if bondpair[0] == bondpair[1]:
            dimers = '%s2/*/1/*/*'%(bondpair[0])
        else:
            dimers = '%s%s/*/1/*/*'%(bondpair[0],bondpair[1])
        # reference total energy
        refatoms = self.data.get_ref_atoms(structures=[dimers]
                                              ,quantities=['energy'])
        iso_ene = self.data.get_free_atom_energies()
        # for setting cut-off, but this should simply be from 
        # variables.minimum_distance_isolated_atom()
        #maxd = 0.
        #for atom in refatoms:
        #    d = atom.get_distance(0,1,mic=True) 
        #    if d > maxd:
        #        maxd = d 
        # quickfix BOPfox error (cannot overwrite rcut when bondsversion=None)
        if True:
            maxd = 5.0
            refatoms = [at for at in refatoms if at.get_distance(0,1) < maxd]
        # set up calculation of bond energy
        modelsbx = self.gen_model(part='bond')
        old_infox = dict(modelsbx.infox_parameters)
        modelsbx.infox_parameters['repversion'] = 'None'
        # do tight-binding 
        modelsbx.infox_parameters['version'] = 'tight-binding'
        modelsbx.infox_parameters['scfsteps'] = 0        
        modelsbx.infox_parameters['efermisteps'] = 0        
        modelsbx.bondsbx[0].rcut = maxd + 0.1
        modelsbx.bondsbx[0].r2cut = maxd + 0.2
        modelsbx.bondsbx[0].dcut = 0.
        modelsbx.bondsbx[0].d2cut = 0.
        calc = CATCalc(controls=self.controls,model=modelsbx)
        calc.set_atoms([at.copy() for at in refatoms])
        # calculate bond energy
        bondene = calc.get_property(contribution='bond')
        calc.docalculate= True
        promene = calc.get_property(contribution='prom')
        bondene = np.array(bondene)
        promene = np.array(promene)
        refdata = []
        # set free atom energy to zero?
        for atom in refatoms:
            ene = atom.get_potential_energy()
            sym = atom.get_chemical_symbols()
            for i in range(len(atom)):
                ene -= iso_ene[sym[i]]
            #ene /= len(atom)
            d = atom.get_distance(0,1)
            refdata.append((d,ene))
        refdata = np.array(refdata).T
        repeneref = refdata[1]-bondene-promene 
        repeneref /= 2
        #param =  Funcs.fit_sum(refdata[0],repeneref,repfunc,tol=2
        #         ,fit_all=True,debug=False,minimizer='leastsq'
        #         ,constrain=False)
        #param =  Funcs.fit(refdata[0],repeneref,repfunc,tol=2,debug=False
        #                  ,minimizer='leastsq',constrain=True)
        #repfunc.set_parameters(param)        
        calc.docalculate = True
        #pl.plot(refdata[0],refdata[1],'ro')
        #pl.plot(refdata[0],2*repfunc.functions[0](refdata[0]),'go')
        #pl.plot(refdata[0],2*repfunc.functions[1](refdata[0]),'gs')
        #pl.plot(refdata[0],2*repfunc.functions[2](refdata[0]),'gd')
        #pl.plot(refdata[0],2*repfunc(refdata[0])+bondene,'g-')
        #pl.show()
        #print param
        #raise

        # set old infox
        modelsbx.infox_parameters = dict(old_infox)
        """
        reppar = {}
        for i in range(len(repkeys)):
            reppar[repkeys[i]] = repfunc.functions[i]
        bondkey = '%s-%s' % (bondpair[0], bondpair[1])
        if bondkey not in self.repulsion:
            self.repulsion[bondkey] = {}
            self.repulsion[bondkey] = reppar
        return reppar

    def _read_onsites(self, elem):
        cwd = os.getcwd()
        if self.pathtoonsites is None:
            print_format("Path to onsites is not provided.", level=2)
            print_format("Setting to %s" % pathtoonsites(), level=3)
            self.pathtoonsites = pathtoonsites()
        os.chdir(os.path.expanduser(self.pathtoonsites))
        fn = '%s_%s%s_%s_%s.onsites' % (self.betafitstruc, elem, elem
                                        , self.betabasis, self.betatype)
        if not os.path.isfile(fn):
            raise ValueError('No onsite file found: %s' % fn)
        l = open(fn).readlines()
        for i in range(len(l)):
            if '#' in l[i]:
                skip_first = True
                continue
            l[i] = l[i].split()
            for j in range(len(l[i])):
                l[i][j] = float(l[i][j])
        if skip_first:
            l = l[1:]
        l.sort()
        l = np.array(l).T
        es = 0.0
        ep = 0.0
        ed = 0.0
        # 
        if len(l) == 7:
            es = l[1]
            ed1 = l[2]
            ed2 = l[3]
            ed3 = l[4]
            ed4 = l[5]
            ed5 = l[6]
            es = np.average(es)
            ed = np.average([ed1, ed2, ed3, ed4, ed5])
        elif len(l) == 5:
            es = l[1]
            ep1 = l[2]
            ep2 = l[3]
            ep3 = l[4]
            es = np.average(es)
            ep = np.average([ep1, ep2, ep3])
        elif len(l) == 10:
            es = l[1]
            ep1 = l[2]
            ep2 = l[3]
            ep3 = l[4]
            ed1 = l[5]
            ed2 = l[6]
            ed3 = l[7]
            ed4 = l[8]
            ed5 = l[9]
            es = np.average(es)
            ed = np.average([ed1, ed2, ed3, ed4, ed5])
        else:
            print_format("Incorrect size of onsites in file: %s" % fn, level=3)
            raise
        os.chdir(cwd)
        return {'s': es, 'p': ep, 'd': ed}

    def gen_onsite(self, elem):
        """
        Returns a dictionary of the valence:onsites read from the onsite file.
        
        It calls the function :func:`_read_onsites`.        
        
        The filename must follow the format
        
            ``[structure]_[elem][elem]_[basis]_[betatype].onsites``
        
        The structure, basis and betatype are similar to the bond integrals.
        The onsite is taken as the average over a range of distances.
        
        .. todo:: fit distance dependence of onsites

        .. note:: Determine if onsite should just be set to zero.        
        
        :Parameters:
        
            - *elem*: str
            
                chemical symbol
        """
        ve = self.valences[elem]
        onsite = {}
        # ei = self._read_onsites(elem)
        for i in range(len(ve)):
            # onsite[ve[i]] = ei[ve[i]]
            onsite[ve[i]] = 0.1
        return onsite

    def gen_stoner(self, elem, struc='gs'):
        """
        Returns the Stoner parameter. This is read from 
        :func:`variables.atomic_properties`. 

        ..todo:: fit to mag-non mag energy difference of struc

        :Parameters:
        
            - *elem*: str
        
                chemical symbol
        """
        atprop = atomic_properties()[elem]
        if 'stoner' in atprop:
            return atprop['stoner']
        else:
            return 0.0

    def gen_jii(self, elem):
        """
        Returns the Jii parameter. This is read from 
        :func:`variables.atomic_properties`. 

        :Parameters:
        
            - *elem*: str
        
                chemical symbol
        """
        atprop = atomic_properties()[elem]
        if 'jii' in atprop:
            return atprop['jii']
        else:
            return 8.0

    def gen_valenceelectrons(self, elem):
        """
        Returns the number of valence electrons and orbitals. This is read from
        :func:`variables.atomic_properties`. 

        :Parameters:
        
            - *elem*: str
            
                chemical symbol
        """
        atprop = atomic_properties()[elem]
        valence = self.valences[elem]
        orbs = {'s': 1, 'p': 3, 'd': 5}
        val = 0.0
        norb = 0
        for v in valence:
            val += atprop['valence'][v]
            norb += orbs[v]
        if self.valenceelectrons is not None:
            val = self.valenceelectrons[elem]
        return val, norb

    def get_atoms_properties(self, elem):
        """
        Returns the mass, number of orbitals and valence electrons, Jii,
        onsite and Stoner parameters. This calls :func:`gen_valenceelectrons`,
        :func:`gen_jii`, :func:`gen_stoner`, :func:`gen_onsite`.

        :Parameters:
        
            - *elem*: str
                
                chemical symbol
        """
        atprop = atomic_properties()[elem]
        onsite = self.gen_onsite(elem)
        try:
            mass = atprop['mass']
        except:
            print_format('Mass not set for %s.' % elem, level=2)
        jii = self.gen_jii(elem)
        stoner = self.gen_stoner(elem)
        val, norb = self.gen_valenceelectrons(elem)
        return mass, val, norb, jii, onsite, stoner

    def _gen_atomsbx(self):
        atomsbx = []
        for elem in self.elements:
            mass, valence, norb, jii, onsite, stoner = self.get_atoms_properties(elem)
            onsite = [onsite[n] for n in self.valences[elem]]
            # version is not anymore relevant in modelsbx
            if 'atomsversion' in self.calculator_settings:
                aversion = self.calculator_settings['atomsversion']
            else:
                aversion = 'canonicaltb'
            abx = bopmodel.atomsbx(version=aversion
                                   , atom=elem, mass=mass
                                   , valenceelectrons=[valence]
                                   , valenceorbitals=[norb]
                                   , jii=jii
                                   , onsitelevels=onsite
                                   , stonerintegral=[stoner] * 3)
            atomsbx.append(abx)
        return atomsbx

    def calc_rcut(self, pair, debug=False):
        """
        Returns a dictionary of cut-off keyword: value. 
        
        `rcut` is set to the distance when the value of the most long-ranged 
        bond integral is 5 percent its value at the equilibrium 
        dimer distance. The hierarcy of the range of the bond integrals (m=0)
        for simplicity is assumed as follows:
        
        ``ppsigma, spsigma, sssigma, pdsigma, sdsigma, ddsigma``
        
        The cut-off function starts at 15 percent. `r2cut` is 
        `1.5*rcut` and `d2cut` is `2*dcut`
        
        It also sets the cutoff versions to cosine.
        
        :Parameters:
        
            - *bondpair*: list 
            
                pair of chemical symbols 
            
            - *debug*: bool
        
                if debug will plot the cut-off values with the reference 
                bond integral
        """
        heirarchy = ['ppsigma', 'spsigma', 'sssigma', 'pdsigma', 'sdsigma'
            , 'ddsigma']
        if pair[0] == pair[1]:
            dim = '%s2/*/1/*/*' % (pair[0])
        else:
            dim = '%s%s/*/1/*/*' % (pair[0], pair[1])
        r0 = self.data.get_equilibrium_distance(dim)
        if r0 is None:
            print_format('Dimer equilibrium distance not found.', level=3)
            gs = self.data.get_ground_state(elements=pair, out='system_ID'
                                            , spin=1)
            r0 = self.data.get_equilibrium_distance(gs)
        bondkey = '%s-%s' % (pair[0], pair[1])
        if bondkey in self.betas:
            for key in list(self.betas[bondkey].keys()):
                if key not in ['overlap']:
                    break
            maxb = 0
            maxbname = list(self.betas[bondkey].keys())[0]  # safety net in case fails
            for beta in self.betas[bondkey][key]:
                if beta in heirarchy:
                    func = self.betas[bondkey][key][beta]
                    if func is None:
                        continue
                    # to identity range simply evaluate value at 3*r0
                    temp = abs(func(3 * r0))
                    if temp > maxb:
                        maxbname = beta
                        maxb = temp
            if maxb == 0:
                print_format('No sigma bond integral found!', level=3)
                print_format('Setting rcut ')
            #    
            func = self.betas[bondkey][key][maxbname]
            x = np.linspace(r0, 5 * r0, 1000)
            temp = func(x)
            temp /= temp[0]
            rcut = None
            dcut = None
            for i in range(len(temp)):
                if temp[i] < 0.15 and dcut is None:
                    dcut = x[i]
                if temp[i] < 0.05:
                    rcut = x[i]
                    break
            dcut = rcut - dcut
            if debug:
                import matplotlib.pyplot as pl
                pl.plot(x, temp, label=maxbname)
                pl.plot([rcut, rcut], [min(temp), max(temp)], label='rcut')
                pl.plot([rcut - dcut, rcut - dcut], [min(temp), max(temp)])
                pl.plot([r0, r0], [min(temp), max(temp)], label='r0')
                pl.legend(loc='best')
                pl.show()
                raise Exception('debug mode.')
        else:
            # safety net
            print_format("No beta information for %s-%s" % (pair[0]
                                                            , pair[1]), level=2)
            print_format("Setting rcut = 1.5 * dimer equilibrium distance"
                         , level=3)
            rcut = 1.5 * r0
            dcut = 0.2 * rcut
        # needs better way to calculate r2cut    
        r2cut = 1.5 * rcut
        d2cut = 2.0 * dcut
        out = {'rcut': rcut, 'dcut': dcut, 'r2cut': r2cut, 'd2cut': d2cut
            , 'cutoffversion': 'cosine', 'cutoff2version': 'cosine'}
        return out

    def get_cutoffpara(self, bondpair):
        """
        Returns a dictionary of cut-off keyword: value. Calls :func:`calc_rcut`
        
        :Parameters:
        
            - *bondpair*: list 
        
                pair of chemical symbols
        """
        cs = self.cutoff
        csc = self.calc_rcut(bondpair)
        cutpar = {}
        if self.calculator.lower() == 'bopfox':
            cuts = ['rcut', 'r2cut', 'dcut', 'd2cut'
                , 'cutoffversion', 'cutoff2version']
        else:
            raise NotImplementedError("No option for %s" % self.calculator)
        for cut in cuts:
            if cut in cs:
                if isinstance(cs[cut], dict):
                    for bp, val in list(cs[cut].items()):
                        if (bp == '%s-%s' % (bondpair[0], bondpair[1])) or \
                                (bp == '%s-%s' % (bondpair[1], bondpair[0])):
                            cutpar[cut] = val
                            break
                else:
                    cutpar[cut] = cs[cut]
            else:
                print_format('No %s found for %s-%s' % (cut, bondpair[0]
                                                        , bondpair[1]), level=2)
                # this is necessary to accomodate keys for other calculators
                # add relevant key in list
                if cut in ['rcut']:
                    cutpar[cut] = csc['rcut']
                elif cut in ['dcut']:
                    cutpar[cut] = csc['dcut']
                elif cut in ['r2cut']:
                    cutpar[cut] = csc['r2cut']
                elif cut in ['d2cut']:
                    cutpar[cut] = csc['d2cut']
                elif cut in ['cutoffversion']:
                    cutpar[cut] = csc['cutoffversion']
                elif cut in ['cutoff2version']:
                    cutpar[cut] = csc['cutoff2version']
        return dict(cutpar)

    def _gen_bondsbx(self, **kwargs):
        bond_keys = ['sssigma', 'spsigma', 'sdsigma', 'pssigma'
            , 'ppsigma', 'pppi', 'pdsigma', 'pdpi', 'dssigma'
            , 'dpsigma', 'dppi', 'ddsigma', 'ddpi', 'dddelta']
        bondsbx = []
        part = 'all'
        if 'part' in kwargs:
            part = kwargs['part'].lower()
        for bondpair in self.pairs:
            bx = bopmodel.bondsbx()
            bx.bond = bondpair
            valence = [self.valences[p] for p in bondpair]
            bx.valence = valence
            bx.scaling = [self.scaling]
            if part == 'bond':
                bondspar = self.get_initial_bondparam(bondpair)
                bx.set_bondspar(bondspar)
                params = bondspar
            elif part == 'rep':
                reppar = self.get_initial_repparam(bondpair)
                bx.set_bondspar(reppar)
                params = reppar
            else:
                bondspar = self.get_initial_bondparam(bondpair)
                bx.set_bondspar(bondspar)
                reppar = self.get_initial_repparam(bondpair)
                bx.set_bondspar(reppar)
                params = bondspar
            cutpar = self.get_cutoffpara(bondpair)
            bx.set_cutpar(cutpar)
            if 'bondsversion' in self.calculator_settings:
                bversion = self.calculator_settings['bondsversion']
            else:
                for key, val in list(params.items()):
                    if val is not None:
                        if key in bond_keys:
                            env = 'bopfox-ham'
                        else:
                            env = 'bopfox-rep'
                        bversion = val.get_name(environment=env)
                        break
            bx.version = bversion
            bondsbx.append(bx)
        return bondsbx

    def gen_modelsbx(self, **kwargs):
        """
        Returns a modelsbx object. See :class:`bopmodel.modelsbx`
        
        Generates modelsbx. Calls :func:`_gen_atomsbx` and :func:`_gen_bondsbx`
        
        :Parameters:
        
            - *kwargs* : dict
        
                directives for future extensions. For example, `part` can be 
                set to `bond`, `rep` or `all` to specify part of modelsbx that 
                is generated.
        """
        name = 'test'  # model name is irrelevant if generated
        self.calculator_settings.update({'model': name})
        modelsbx = bopmodel.modelsbx(model=name
                                     , infox_parameters=dict(self.calculator_settings)
                                     , atomsbx=self._gen_atomsbx()
                                     , bondsbx=self._gen_bondsbx(**kwargs))
        if not self.orthogonal:
            modelsbx.infox_parameters.update({'screening': True})
        return modelsbx

    def read_modelsbx(self, **kwargs):
        """
        Returns a modelsbx object. See :class:`bopmodel.modelsbx`

        Reads modelsbx. Calls :func:`bopmodel.read_modelsbx`

        :Parameters:
        
            - *kwargs* : dict
        
                directives for future extensions.
        """
        modelsbx = bopmodel.read_modelsbx(model=[self.model]
                                          , system=self.elements
                                          , filename=self.model_filename)
        if len(modelsbx) > 1:
            raise ValueError('Found more than one model.Check %s' \
                             % self.model_filename)
        modelsbx = modelsbx[0]
        return modelsbx

    def gen_model(self, **kwargs):
        """
        Returns model specific to the calculator. 
        
        Generates model. Calls calculator-specific function to generate model. 
        
        .. todo:: Extend to other calculators.
        
       :Parameters:
        
            - *kwargs* : dict
        
                directives for future extensions.
        """
        if self.calculator.lower() == 'bopfox':
            model = self.gen_modelsbx(**kwargs)
        else:
            raise NotImplementedError("No option for %s" % self.calculator)
        return model

    def read_model(self, **kwargs):
        """
        Returns model specific to the calculator. 
        
        Reads model. Calls calculator-specific function to read model. 
        
        .. todo:: Extend to other calculators.
        
       :Parameters:
        
            - *kwargs* : dict
            
                directives for future extensions.
        """

        if self.calculator.lower() == 'bopfox':
            model = self.read_modelsbx(**kwargs)
        else:
            raise NotImplementedError("No option for %s" % self.calculator)
        return model

    def get_model(self, iteration=0):
        """
        Returns the model generated at N=iteration step.
        
       :Parameters:
        
            - *iteration* : int
        
                index of the model in list 
        """
        return self.models[iteration].copy()

    def average_modelsbx(self, iterations='all'):
        """
        Returns a model by averaging models in iterations.
        
       :Parameters:
        
            - *iterations* : list
        
                indices of the models to be averaged
            
                `all` : include all models
        """
        if iterations == 'all':
            iterations = list(range(len(self.models)))
        models = [self.models[i] for i in iterations]
        if self.calculator.lower() == 'bopfox':
            model = bopmodel.average_models(models)
        else:
            raise NotImplementedError("No option for %s" % self.calculator)
        return model

    def get_SED(self, elem, model, strucs=None):
        """
        Returns an array of the bond energy differences of the structures
        in strucs. The reference is the first structure not necessarily the
        lowest in energy in order to simplify fitting where one does not
        need the index of the lowest energy structure.
        
        Calls :func:`sedt.energy_diff` to calculate bond energies of 
        structures adjusted to have the same average second moment. 
        
        :Parameters:  
        
            - *elem* : str
        
                chemical symbol
        
            - *model* : calculator-specific object
        
                model to be used by the calculator
        
            - *strucs* : list
        
                list of system_ID's corresponding to relaxed structures
            
                `None`: will include all relaxed structures        
        """
        if strucs is None:
            # get all relaxed structures
            strucs = ['%s*/*/0/0/*' % elem]
        atoms = self.data.get_ref_atoms(structures=strucs
                                        , quantities=['energy'])
        ref_ene = self.data.get_ref_data()
        ref_ene = [ref_ene[i] / len(atoms[i]) for i in range(len(atoms))]
        # imin = ref_ene.index(min(ref_ene))
        calc = CATCalc(controls=self.controls, model=model)
        if self.calculator.lower() == 'bopfox':
            model_ene = energy_diff(atoms, calc)
        else:
            raise NotImplementedError("No option for %s" % self.calculator)
        model_ene = [model_ene[i] / len(atoms[i]) for i in range(len(atoms))]
        # energy difference wrt to first structure
        ref_ene = [e - ref_ene[0] for e in ref_ene]
        model_ene = [e - model_ene[0] for e in model_ene]
        # write out comparison
        print_format('Structural energy differences for %s' % elem, level=2)
        for i in range(len(atoms)):
            sID = atoms[i].get_strucname()
            print_format('%30s :  reference : %8.4f  model : %8.4f' \
                         % (sID, ref_ene[i], model_ene[i]), level=3)
        out = np.array(model_ene) - np.array(ref_ene)
        return out
