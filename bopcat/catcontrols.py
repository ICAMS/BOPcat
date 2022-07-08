#!/usr/bin/env python

# Definition of the CATControls object

# This module is part of the BOPcat package

from .utils import make_pairs
from .variables import data_key_code, available_calculators
import os
from .output import print_format


class CATControls:
    """
    Handles all input controls.
    
    Parameters are categorized as either data-, opt- calculator-
    and model-specific

    :Parameters:
    
        - *elements*: list
            
            list of chemical symbols. Bond pairs will be generated from the
            list.
            
        - *calculator_settings*: dict
        
            options to be passed to the calculator
            
        - *calculator*: str
        
            name of the calculator
            
        - *calculator_nproc*: int
                
            number of parallel processes

            ``None``: will not parallelize
            
        - *data_parameters*: dict
        
            specifications for the data, e.g.::
            
                data_parameters = {'xc_functional':30, 'encut':400}
                
        - *data_system_parameters*: dict

            specifications for the structures, e.g.::
                
                data_system_parameters = {'stoichiometry':['Fe2','Fe']}
                
        - *data_filename*: str

            filename or path to file containing the structures and properties
            
        - *data_free_atom_energies*: dict
        
            dictionary of element:free atom energies
            
            ``dimer``: will generate free_atom_energies from dimers
            
            ``None``: will read from atomic_properties in :mod:`variables`
            
        - *opt_variables*: list
        
            list of dictionary of contraints on the parameters::
             
                variables = [{'bond':['Fe','Fe'],'rep1':[True,True]}
                            ,{'bond':['Fe','Nb'],'rep2':[True,True,True]}
                            ,{'atom':['Fe'],'valenceelectrons':[True]}] 
                            
        - *opt_structures*: list

            list of structures in target set::
            
                structures = ['Fe*/*/0/1/*']

        - *opt_test_structures*: list

            list of structures in test set

        - *opt_optimizer*: str

            name of optimizer  
    
        - *opt_optimizer_options*: dict
        
            controls to be passed to the optimizer                        
                
        - *opt_objective*: str

            name of the objective function
            
        - *model*: str
        
            name of the model
            
        - *model_pathtomodels*: str

            filename or path to file containing model

        - *model_pathtobetas*: str

            filename or path to file containing bond integrals

        - *model_pathtoonsites*: str

            filename or path to file containing onsites

        - *model_functions*: dict

            dictionary of keyword:bondpair:function or keyword:function, 
            function is one of the instances in functions module. The latter
            implies that all bond pairs will have the same functional form::
            
                import functions as funcs
                functions = {'ddsigma':{'Fe-Fe':funcs.exponential()
                                       ,'Nb-Nb':funcs.GSP()}
                                       ,'rep1':funcs.rep_exponential()}
                                       
        - *model_valences*: dict
        
            dictionary of element:valency::
         
                valences = {'Fe':'d','Nb':'d'} 

        - *model_valenceelectrons*: dict
        
            dictionary of element:number of valence electrons::
         
                valenceelectrons = {'Fe':7.1,'Nb':4.0} 
                
            ``None``: will read valenceelectrons from 
            :func:`variables.atomic_properties`
                
        - *model_orthogonal*: bool

            orthogonal or non-orthogonal model
            
        - *model_betafitstruc*: str
        
            structure from which the betas were derived (see format of betas 
            files)

        - *model_betatype*: str
        
            method used in bond integral generation
            
        - *model_betabasis*: str
        
            basis used in bond integral generation    
            
        - *model_cutoff*: dict

            dictionary of cutoff keyword:bondpair:value or keyword:value.

            The latter implies that all bond pairs will have the same value::

                cutoff = {'rcut' : {'Nb-Nb':4.5,'Fe-Fe':4.5,'Fe-Nb':4.5}
                         ,'dcut' :0.5}

        - *verbose*: int
        
            prints details at different levels
            
    """

    def __init__(self, **kwargs):
        self._init_msg()
        self.elements = None
        self.calculator_settings = {}
        self.calculator_nproc = 'default'
        self.calculator_parallel = 'serial'
        self.calculator_queue = 'serial'
        self.calculator = 'bopfox'
        self.data_parameters = {}
        self.data_system_parameters = {}
        self.data_ini_magmoms = {}
        self.data_filename = None
        self.data_free_atom_energies = None
        self.opt_variables = None
        self.opt_structures = []
        self.opt_test_structures = []
        self.opt_optimizer = 'least_squares'
        self.opt_optimizer_options = {}
        self.opt_objective = 'default'
        self.verbose = 1
        self.model = None
        self.model_pathtomodels = 'models.bx'
        self.model_pathtobetas = None
        self.model_pathtoonsites = None
        self.model_functions = None
        self.model_valences = None
        self.model_valenceelectrons = None
        self.model_orthogonal = True
        self.model_betafitstruc = 'dimer'
        self.model_betatype = 'loewdin'
        self.model_betabasis = 'tz0'
        self.model_cutoff = None
        self.set(**kwargs)

    def _init_msg(self):
        print_format('Preparing input controls', level=1)

    def set(self, **kwargs):
        for key in kwargs:
            if key.lower() == 'elements':
                self.elements = kwargs[key]
            elif key.lower() == 'calculator_settings':
                self.calculator_settings = kwargs[key]
            elif key.lower() == 'calculator':
                self.calculator = kwargs[key]
            elif key.lower() == 'calculator_nproc':
                self.calculator_nproc = kwargs[key]
            elif key.lower() == 'calculator_queue':
                self.calculator_queue = kwargs[key]
            elif key.lower() == 'data_parameters':
                self.data_parameters = kwargs[key]
            elif key.lower() == 'data_system_parameters':
                self.data_system_parameters = kwargs[key]
            elif key.lower() == 'data_ini_magmoms':
                self.data_ini_magmoms = kwargs[key]
            elif key.lower() == 'data_filename':
                self.data_filename = kwargs[key]
            elif key.lower() == 'data_free_atom_energies':
                self.data_free_atom_energies = kwargs[key]
            elif key.lower() == 'opt_variables':
                self.opt_variables = kwargs[key]
            elif key.lower() == 'opt_structures':
                self.opt_structures = kwargs[key]
            elif key.lower() == 'opt_test_structures':
                self.opt_test_structures = kwargs[key]
            elif key.lower() == 'opt_optimizer':
                self.opt_optimizer = kwargs[key]
            elif key.lower() == 'opt_optimizer_options':
                self.opt_optimizer_options = kwargs[key]
            elif key.lower() == 'opt_objective':
                self.opt_objective = kwargs[key]
            elif key.lower() == 'model':
                self.model = kwargs[key]
            elif key.lower() == 'model_pathtomodels':
                self.model_pathtomodels = kwargs[key]
            elif key.lower() == 'model_pathtobetas':
                self.model_pathtobetas = kwargs[key]
            elif key.lower() == 'model_pathtoonsites':
                self.model_pathtoonsites = kwargs[key]
            elif key.lower() == 'model_functions':
                self.model_functions = kwargs[key]
            elif key.lower() == 'model_valences':
                self.model_valences = kwargs[key]
            elif key.lower() == 'model_valenceelectrons':
                self.model_valenceelectrons = kwargs[key]
            elif key.lower() == 'model_betafitstruc':
                self.model_betafitstruc = kwargs[key]
            elif key.lower() == 'model_orthogonal':
                self.model_orthogonal = kwargs[key]
            elif key.lower() == 'model_betatype':
                self.model_betatype = kwargs[key]
            elif key.lower() == 'model_betabasis':
                self.model_betabasis = kwargs[key]
            elif key.lower() == 'model_cutoff':
                self.model_cutoff = kwargs[key]
            elif key.lower() == 'verbose':
                self.verbose = kwargs[key]
            else:
                raise ValueError('Unrecognized key %s' % key)

    def initialize(self):
        """
        Initializes input controls. 
        
        Calls :func:`check_input`, :func:`make_lower`, :func:`make_pairs`,
        :func:`convert_params`
        """
        self.check_input()
        self.bond_pairs = make_pairs(self.elements)
        self.make_lower()
        self.convert_params()

    def make_lower(self):
        """
        Change case of all strings in calculator_settings.
        """
        # calculator
        self.calculator = self.calculator.lower()
        # calculator settings
        cs = self.calculator_settings
        new = {}
        for key, val in list(cs.items()):
            new[key.lower()] = val
        self.calculator_settings = dict(new)

    def convert_params(self):
        """
        Convert data and system parameters into integers.
        """
        key_code = data_key_code()
        dpar = self.data_parameters
        spar = self.data_system_parameters
        for par in [dpar, spar]:
            for key, val in list(par.items()):
                if isinstance(val, str) and key != 'stoichiometry':
                    val = val.strip().lower()
                    par[key] = key_code[val]

    def gen_functions(self):
        """
        Set default functions. Returns dictonary of key:functions
        """
        from . import functions
        bonds = self.bond_pairs
        funcs = {}
        exp0 = functions.exponential(parameters=[1, 1]
                                     , constraints=[True, True, False])
        exp1 = functions.exponential(parameters=[1, 1, 1]
                                     , constraints=[True, True, True])
        exp2 = functions.exponential(parameters=[1, 1, 1]
                                     , constraints=[True, True, True])
        exp3 = functions.exponential(parameters=[1, 1, 1]
                                     , constraints=[True, True, True])
        sum_exp = functions.sum_funcs(functions=[exp0, exp1, exp2, exp3])
        sum_exp.set_name('sum_exp')
        # sum_exp = exp0.copy()
        # rep1 = functions.env_lin()
        # rep2 = functions.spline()
        rep1 = functions.rep_exponential()
        rep2 = functions.emb_sqrt()
        keys = [
            'sssigma'
            , 'spsigma'
            , 'sdsigma'
            , 'pssigma'
            , 'ppsigma'
            , 'pppi'
            , 'pdsigma'
            , 'pdpi'
            , 'dssigma'
            , 'dpsigma'
            , 'dppi'
            , 'ddsigma'
            , 'ddpi'
            , 'dddelta'
        ]
        for key in keys:
            funcs[key] = {}
            for bond in bonds:
                bp = '%s-%s' % (bond[0], bond[1])
                funcs[key][bp] = sum_exp.copy()
        for key in ['rep1', 'rep2']:
            for bond in bonds:
                bp = '%s-%s' % (bond[0], bond[1])
                funcs[key][bp] = rep1
        return funcs

    def check_input(self):
        """
        Checks the consistency of input data type. 
        """
        all_keys = {'calculator settings': (self.calculator_settings
                                            , [dict])
            , 'elements': (self.elements
                           , [list])
            , 'calculator': (self.calculator
                             , [str])
            , 'data_parameters': (self.data_parameters
                                  , [dict])
            , 'data_system_parameters': (self.data_system_parameters
                                         , [dict])
            , 'data_filename': (self.data_filename
                                , [str])
            , 'data_free_atom_energies': (self.data_free_atom_energies
                                          , [str, dict])
            , 'opt_variables': (self.opt_variables
                                , [list])
            , 'opt_structures': (self.opt_structures
                                 , [list])
            , 'opt_optimizer': (self.opt_optimizer
                                , [str])
            , 'opt_optimizer_options': (self.opt_optimizer_options
                                        , [dict])
            , 'opt_objective': (self.opt_objective
                                , [str])
            , 'model': (self.model
                        , [str, type(None)])
            , 'model_pathtomodels': (self.model_pathtomodels
                                     , [str, type(None)])
            , 'model_pathtobetas': (self.model_pathtobetas
                                    , [str, type(None)])
            , 'model_pathtoonsites': (self.model_pathtoonsites
                                      , [str, type(None)])
            , 'model_functions': (self.model_functions
                                  , [dict, type(None)])
            , 'model_valences': (self.model_valences
                                 , [dict, type(None)])
            , 'model_valenceelectrons': (self.model_valenceelectrons
                                         , [dict, type(None)])
            , 'model_betafitstruc': (self.model_betafitstruc
                                     , [str, type(None)])
            , 'model_orthogonal': (self.model_orthogonal
                                   , [bool])
            , 'model_betatype': (self.model_betatype
                                 , [str])
            , 'model_betabasis': (self.model_betabasis
                                  , [str])
            , 'model_cutoff': (self.model_cutoff
                               , [dict, type(None)])
                    }

        for key in all_keys:
            if type(all_keys[key][0]) not in all_keys[key][1]:
                print_format('Invalid input for %s' % key, level=2)
                print_format('Must of %s' % str(all_keys[key][1]), level=3)

        avail_calc = available_calculators()
        if self.calculator not in avail_calc:
            print_format("%s is not currently implemented" \
                         % self.calculator, level=2)
            raise
        # system_parameters and data_parameters
        for par in [self.data_parameters, self.data_system_parameters]:
            for key, val in list(par.items()):
                if not isinstance(val, list):
                    par[key] = [val]
        # 
        if self.model is None:
            print_format("No model_initial provided.", level=2)
            print_format("will try to generate.", level=3)
        #
        if self.model_pathtobetas is None and self.model is None:
            from .variables import pathtobetas
            print_format("Path to betas is not provided.", level=2)
            print_format("Setting to %s" % pathtobetas(), level=3)
            self.model_pathtobetas = pathtobetas()
            if not os.path.isdir(self.model_pathtobetas):
                print_format("Cannot find betas at %s" \
                             % self.model_pathtobetas, level=3)
                raise
        #
        if self.model_pathtoonsites is None and self.model is None:
            from .variables import pathtoonsites
            print_format("Path to onsites is not provided.", level=2)
            print_format("Setting to %s" % pathtoonsites(), level=3)
            self.model_pathtoonsites = pathtoonsites()
            if not os.path.isdir(self.model_pathtoonsites):
                print_format("Cannot find onsites", level=3)
                raise

        if self.calculator_parallel.lower() == 'mpi':
            try:
                import mpi4py
            except:
                print_format("Install mpi4py to use mpi.", level=3)
                raise
