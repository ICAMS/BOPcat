#!/usr/bin/env python

# Definition of the CATKernel, CATobjective and CAToptimizer objects

# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import numpy as np
from scipy import optimize
from .utils import gen_param_model, get_par_keys, calc_diff
from .utils import gen_penalty_coeffs_list, penalty_term
from .variables import fit_weights
from .output import cattxt, print_format
from inspect import getargspec
from .bopmodel import modelsbx
from copy import deepcopy as cdeepcopy

try:
    from collections import OrderedDict as dict
except:
    pass


###########################################################################
class CATKernel:
    """
    Defines the optimization kernel.
    
    It generates the objective function and the optimizer.

    :Parameters:
    
        - *controls*: instance of CATControls
        
            CATControls object to initialize parameters 
            
        - *calc*: instance of CATcalc
        
            CATcalc object to define the calculator
            
        - *ref_data*: list
        
            list of calculated properties 
            
        - *variables*: list
        
            list of dictionary of contraints on the parameters
                            
        - *objective*: callable object
        
            should take parameters as argument and return the error 
            
            None: will generate default objective function
            
            str: name of the objective function, should be implemented in 
            :class:`CATobjective`                   
            
        - *optimizer*: callable object
       
            should return optimized parameters
           
            str: name of the optimizer, should be implemented in 
            :class:`CAToptimizer`

        - *optimizer_options*: dict
        
            controls to be passed to the optimizer

        - *weights*: list
            
            weight of each structure in the error 
           
            None: will assign weight from :func:`variables.fit_weights`

        - *log*: str
        
            output file for writing parameters and error at each optimization
            step 

        - *verbose*: int

            controls verbosity e.g.
            
            prints on screen rms at each optimization step if verbose >= 1

        - *dump_min_model*: bool
        
            logical flag to switch on dumping of best model at each iteration

    """

    def __init__(self, **kwargs):
        self.calc = None
        self.ref_data = None
        self.variables = None
        self.penalty_coeffs = None
        self.objective = None
        self.optimizer = 'leastsq'
        self.optimizer_options = {}
        self.weights = None
        self.controls = None
        self.log = None
        self.error_vec_log = None
        self.optimized_model = None
        self.verbose = 2
        self.objective = None
        self.array = False
        self.error = None
        self.error_type = 'rms'
        self.param_weights = None
        self.ignore_cutoff = False
        self.cutoff_reference = 'rcut'
        self.dump_min_model = False
        self.shift_min_ene = False
        self.gui_log = None
        self.set(**kwargs)

    def _init_msg(self, mode='optimization'):
        print_format('Performing parameter %s' % mode, level=1)

    def set(self, **kwargs):
        if 'controls' in kwargs:
            self.controls = kwargs['controls']
            if self.controls is not None:
                self._unfold_controls()
        for key in kwargs:
            if key.lower() == 'calc':
                self.calc = kwargs[key]
            elif key.lower() == 'ref_data':
                self.ref_data = kwargs[key]
            elif key.lower() == 'variables':
                self.variables = kwargs[key]
            elif key.lower() == 'penalty_coeffs':
                self.penalty_coeffs = kwargs[key]
            elif key.lower() == 'objective':
                self.objective = kwargs[key]
            elif key.lower() == 'optimizer':
                self.optimizer = kwargs[key]
            elif key.lower() == 'optimizer_options':
                self.optimizer_options = kwargs[key]
            elif key.lower() == 'weights':
                self.weights = kwargs[key]
            elif key.lower() == 'log':
                self.log = kwargs[key]
            elif key.lower() == 'error_vec_log':
                self.error_vec_log = kwargs[key]
            elif key.lower() == 'verbose':
                self.verbose = kwargs[key]
            elif key.lower() == 'error_type':
                self.error_type = kwargs[key]
            elif key.lower() == 'ignore_cutoff':
                self.ignore_cutoff = kwargs[key]
            elif key.lower() == 'cutoff_reference':
                self.cutoff_reference = kwargs[key]
            elif key.lower() == 'param_weights':
                self.param_weights = kwargs[key]
            elif key.lower() == 'dump_min_model':
                self.dump_min_model = kwargs[key]
            elif key.lower() == 'shift_min_ene':
                self.shift_min_ene = kwargs[key]
            elif key.lower() == 'gui_log':
                self.gui_log = kwargs[key]
            elif key.lower() == 'controls':
                pass
            else:
                raise ValueError('Cannot recognize key %s' % key)
        # initialize models and parameters        
        if self.calc is not None and self.variables is not None:
            self.model = self.calc.get_model()
            self.init_model_data = []
        if isinstance(self.optimizer, str):
            if self.optimizer.lower() in ['leastsq', 'least_squares']:
                self.array = True
        if self.ref_data is None:
            raise RuntimeError("Provide reference data.")
        # exclude structures with min distance < cutoff
        if not self.ignore_cutoff:
            self._check_for_cutoff()

    def _unfold_controls(self):
        self.optimizer = self.controls.opt_optimizer
        self.objective = self.controls.opt_objective
        self.optimizer_options = self.controls.opt_optimizer_options
        self.variables = self.controls.opt_variables
        self.verbose = self.controls.verbose

    def gen_weights(self):
        if self.weights is None:
            if self.verbose > 0:
                warn = '''Warning! Weights not set. BOPcat will generate them.
            Weight is product of weight of corresponding property (see weights 
            in module variables) and weight of structure. Energy also 
            normalized by number of atoms. Weight of structure normalized by 
            mean of corresponding property resulting in dimensionless error. 
            To rid of this warning set your weights manually. 
            '''
                print_format(warn, level=2)
            weights = []
            atoms = self.calc.get_atoms()
            properties = []
            for i in range(len(atoms)):
                # prop = atoms[i].get_required_property()
                prop = atoms[i].info['required_property']
                if prop not in properties:
                    properties.append(prop)
            # get mean of data for every property
            averages = {}
            for prop in properties:
                all_data = []
                for i in range(len(atoms)):
                    # if prop != atoms[i].get_required_property():
                    if prop != atoms[i].info['required_property']:
                        continue
                    rd = self.ref_data[i]
                    if prop.lower() == 'energy':
                        rd /= len(atoms[i])
                    if not isinstance(rd, float):
                        rd = np.amax(np.abs(rd), axis=0)
                    all_data.append(rd)
                averages[prop] = np.average(np.abs(all_data))
            for i in range(len(atoms)):
                # try to check if atom has weight
                if 'weight' in atoms[i].info:
                    wi = atoms[i].info['weight']
                    weights.append(wi)
                else:
                    prop = atoms[i].info['required_property']
                    wi = fit_weights(prop)
                    # normalize wrt to number of atoms
                    if atoms[i].info['required_property'] == 'energy':
                        wi /= len(atoms[i])
                        wi = [wi]
                    # forces array vector will add 3*N entries so 
                    # also need to normalize
                    elif atoms[i].info['required_property'] == 'forces':
                        wi /= (3 * len(atoms[i]))
                        wi = [wi] * (3 * len(atoms[i]))
                    # stress array vector will add 6 entries so 
                    # also need to normalize
                    elif atoms[i].info['required_property'] == 'stress':
                        wi /= 6
                        wi = [wi] * 6
                    # rms should be dimensionless
                    if averages[prop] > 0.:
                        # weights.append(wi/averages[prop])
                        weights += [w / averages[prop] for w in wi]
                    else:
                        # weights.append(wi)
                        weights += wi
            self.weights = np.array(weights)
            # atoms = self.calc.get_atoms()

    def _get_cutoff(self):
        model = self.calc.get_model()
        cutoff = {}
        if isinstance(model, modelsbx):
            bbxs = model.bondsbx
            for bbx in bbxs:
                bond = bbx.bond
                bond.sort()
                bond = "%s-%s" % (bond[0], bond[1])
                rcut = bbx.get_cutpar()[self.cutoff_reference.lower()]
                if rcut is None:
                    rcut = 4.0
                cutoff[bond] = rcut
        else:
            raise NotImplementedError
        return cutoff

    def _is_fit_cutoff(self, atom, cutoff):
        atom = atom.copy()
        # check first unit cell dimensions
        sym = atom.get_chemical_symbols()
        c = atom.get_cell()
        norm = np.array([np.linalg.norm(c[i]) for i in range(3)])
        for i in range(len(atom)):
            if (norm < cutoff["%s-%s" % (sym[i], sym[i])]).any():
                return True
        out = False
        for i in range(len(atom)):
            for j in range(len(atom)):
                if j > i:
                    d = atom.get_distance(i, j)
                    bond = [sym[i], sym[j]]
                    bond.sort()
                    bond = "%s-%s" % (bond[0], bond[1])
                    if d < cutoff[bond]:
                        out = True
                        break
        return out

    def _check_for_cutoff(self):
        if self.calc.get_model() is None:
            return
        atoms = self.calc.get_atoms()
        if atoms is None:
            return
        cutoff = self._get_cutoff()
        ok = []
        for i in range(len(atoms)):
            if self._is_fit_cutoff(atoms[i], cutoff):
                ok.append(i)
        if len(ok) < len(atoms) and self.verbose > 1:
            warn = '''Warning: No bond less than cutoff for following:
                      (Will remove from reference.)
                   '''
            print_format(warn, level=2)
            for i in range(len(atoms)):
                if i not in ok:
                    # struc = atoms[i].get_property('strucname')
                    struc = atoms[i].info['strucname']
                    print_format("%s" % struc, level=3)
        atoms = [atoms[i] for i in ok]
        self.calc.set_atoms(atoms)
        self.ref_data = [self.ref_data[i] for i in ok]

    def write_summary(self):
        """
        Prints to screen the starting and ending parameters.
        """
        rms = self.objective.rms[-1]
        penalty = self.objective.penalty[-1]
        par = self.objective.par[-1]
        p0 = self.objective.p0
        par_keys = get_par_keys(self.variables)
        assert (len(par_keys) == len(par))
        try:
            optimizer_name = self.optimizer.name
        except:
            optimizer_name = 'unknown'
        print_format('optimizer       : %s' % optimizer_name, level=3)
        print_format('N function calls: %d' % len(self.objective.rms), level=3)
        print_format('function value  : %10.5f' % rms, level=3)
        print_format('Penalty: %10.5f' % penalty, level=3)
        print_format('%15s %15s %15s ' % ('Parameters', 'Start', 'End'), level=3)
        for i in range(len(par)):
            print_format('%15s %15.5f %15.5f' % (par_keys[i], p0[i], par[i])
                         , level=3)
        atoms = self.calc.get_atoms()
        print_format('Structures in set: %d' % len(atoms), level=3)
        if self.verbose > 0:
            for i in range(len(atoms)):
                print_format('%8d %30s ' % (i + 1, atoms[i].info['system_ID']))

    def get_optimized_model(self):
        """
        Returns the optimized model.
        """
        return self.optimized_model

    def gen_objective(self):
        """"
        Generates the objective function. Standard setting is unknown.
        """
        self.gen_weights()
        verbose = False
        if self.verbose > 1:
            verbose = True
        objective = CATobjective(calc=self.calc, ref_data=self.ref_data
                                 , variables=self.variables, penalty_coeffs=self.penalty_coeffs
                                 , weights=self.weights, array=self.array
                                 , log=self.log, error_vec_log=self.error_vec_log, name=self.objective
                                 , verbose=verbose, error_type=self.error_type
                                 , dump_min_model=self.dump_min_model, shift_min_ene=self.shift_min_ene
                                 , gui_log=self.gui_log)
        if self.param_weights is None:
            self.param_weights = np.ones(len(objective.p0))
        self.objective = objective

    def gen_optimizer(self, x0=None):
        if x0 is None:
            x0 = np.ones(len(self.objective.p0))
        optimizer = CAToptimizer(name=self.optimizer, x0=x0
                                 , objective=self.objective, options=self.optimizer_options)
        self.optimizer = optimizer

    def run(self):
        if self.variables is None:
            # test 
            self.test()
        else:
            self.optimize()

    def optimize(self):
        """
        Builds objective and optimizer objects if not initialized and 
        runs optimization. The resulting model is assigned to optimized_model.
        """
        self._init_msg()
        if self.objective is None or isinstance(self.objective, str):
            # build the objective function
            self.gen_objective()
        if self.optimizer is None or isinstance(self.optimizer, str):
            # build the optimizer
            self.gen_optimizer(self.param_weights)
        # run
        x1 = self.optimizer()
        p0 = self.objective.p0
        # update par and rms in objective
        self.objective(x1)
        # generate new model
        new_par = np.array(x1) * np.array(p0)
        new_model = gen_param_model(self.model, self.variables, newparam=new_par)
        if self.verbose > 0:
            self.write_summary()
        if self.log is not None or self.error_vec_log is not None:
            self.objective.write()
        self.optimized_model = new_model
        if self.verbose > 1:
            print_format('done.', level=2)
        self.error = self.objective.rms[-1]
        self.calc.clean()

    def test(self, param_weights=None):
        """
        Calculates the error for current model with parameters weighted by
        param_weights         
        """
        self._init_msg(mode='testing')
        if self.objective is None or isinstance(self.objective, str):
            # build the objective function
            self.gen_objective()
        if param_weights is None:
            param_weights = np.array(self.param_weights)
        self.objective(param_weights)
        if self.verbose > 0:
            self.write_summary()
        if self.log is not None:
            self.objective.write()
        if self.verbose > 1:
            print_format('done.', level=2)
        self.error = self.objective.rms[-1]
        self.calc.clean()

    def copy(self):
        return cdeepcopy(self)


class CATobjective:
    """
    Defines the objective function. Callable with the parameters as argument, 
    will return the corresponding error.

    :Parameters:
            
        - *calc*: instance of CATcalc
        
            CATcalc object to define the calculator
            
        - *ref_data*: list
        
            list of calculated properties 
            
        - *variables*: list
        
            list of dictionary of contraints on the parameters
                            
        - *penalty_coeffs*: list

            list of penalty coefficients for individual coefficients
                            
        - *weights*: list
            
            weight of each structure in the error 
           
            None: will assign weight from :func:`variables.fit_weights`

        - *log*: str
        
            output file for writing parameters and error at each optimization
            step 
         
        - *error_vec_log*: str
        
            output file for writing error for each structure at each 
            optimization step 

        - *verbose*: bool

            prints on screen rms at each optimization step 
            
        - *array*: bool
        
            if error is vector or scalar

        - *name*: str
        
            name of the objective function

        - *dump_min_model*: bool
        
            logical flag to switch on dumping of best model at each iteration
    """

    def __init__(self, **kwargs):
        self.calc = None
        self.ref_data = None
        self.init_model_data = []
        self.variables = None
        self.penalty_coeffs = None
        self.weights = None
        self.array = True
        self.verbose = False
        self.log = None
        self.error_vec_log = None
        self.name = 'default'
        self.error_type = 'rms'
        self.error_details = {}
        self.rms = []
        self.penalty = []
        self.error_vec = []
        self.par = []
        self.ref_data_diff = None
        self.dump_min_model = False
        self.shift_min_ene = False
        self.save = 0
        self.xtol = 1e-6
        self.gui_log = None
        self.set(**kwargs)

    def set(self, **kwargs):
        for key in kwargs:
            if key.lower() == 'calc':
                self.calc = kwargs[key]
            elif key.lower() == 'ref_data':
                self.ref_data = kwargs[key]
            elif key.lower() == 'variables':
                self.variables = kwargs[key]
            elif key.lower() == 'penalty_coeffs':
                self.penalty_coeffs = kwargs[key]
            elif key.lower() == 'weights':
                self.weights = kwargs[key]
            elif key.lower() == 'array':
                self.array = kwargs[key]
            elif key.lower() == 'verbose':
                self.verbose = kwargs[key]
            elif key.lower() == 'error_vec_log':
                self.error_vec_log = kwargs[key]
            elif key.lower() == 'log':
                self.log = kwargs[key]
            elif key.lower() == 'name':
                self.name = kwargs[key]
            elif key.lower() == 'error_type':
                self.error_type = kwargs[key]
            elif key.lower() == 'dump_min_model':
                self.dump_min_model = kwargs[key]
            elif key.lower() == 'shift_min_ene':
                self.shift_min_ene = kwargs[key]
            elif key.lower() == 'save':
                self.save = kwargs[key]
            elif key.lower() == 'xtol':
                self.xtol = kwargs[key]
            elif key.lower() == 'gui_log':
                self.gui_log = kwargs[key]
            else:
                raise ValueError('Cannot recognize key %s' % key)
        if len(self.init_model_data) > 0:
            return
        if self.variables is None:
            self.variables = []
        if self.calc is not None and self.variables is not None:
            self.model = self.calc.get_model()
            self.p0 = gen_param_model(self.model, self.variables)
        # convert variables to OrderedDict
        self.variables = [dict(var) for var in self.variables]
        # check if rep_only calculation, i.e. if init_model_ene has non
        # zero components
        if not self.is_bond_required() and self.has_bond_integrals():
            self.init_model_data = \
                self.calc.get_property(contribution='electronic')
        if len(self.init_model_data) < 1:
            self.contribution = 'binding'
        else:
            self.contribution = 'empirical'
        if self.shift_min_ene:
            self.ref_data = self._shift_energies(self.ref_data)
        self._memory = [[None] * self.save, [None] * self.save]

    def __call__(self, x0, grad=None):
        if self.name == 'default':
            return self.default(x0)
        if self.name == 'difference':
            return self.difference(x0)
        else:
            raise NotImplementedError('No options for %s' % self.name)

    def has_bond_integrals(self):
        out = False
        if isinstance(self.model, modelsbx):
            for bond in self.model.bondsbx:
                for val in list(bond.get_bondspar().values()):
                    if val not in [None, [], 'None', 'none']:
                        out = True
                        break
        else:
            raise NotImplementedError
        return out

    def is_bond_required(self):
        """ 
        Checks if bond-related parameters are included in fit
        """
        bond_keys = ['sssigma', 'sdsigma', 'dssigma', 'spsigma', 'ppsigma', 'pppi'
            , 'pssigma', 'ddsigma', 'ddpi', 'dddelta', 'dpsigma', 'pdsigma'
            , 'dppi', 'pdpi', 'rcut', 'dcut', 'stonerintegral'
            , 'onsitelevels', 'valenceelectrons']
        if self.variables in [None, []]:
            return True
        out = False
        for i in range(len(self.variables)):
            for key in list(self.variables[i].keys()):
                for bk in bond_keys:
                    if bk in key:
                        out = True
                        break
        return out

    def write(self):
        """
        Dumps the parameters and errors at each step on files.
        """
        if self.log is not None:
            self.log = cattxt(filename=self.log)
            for i in range(len(self.rms)):
                self.log.add([i, self.rms[i]] + list(self.par[i]))
            self.log.write()

        if self.error_vec_log is not None:
            self.error_vec_log = cattxt(filename=self.error_vec_log)
            for i in range(len(self.error_vec)):
                self.error_vec[i]
                error_str = ' , '.join(str(e) for e in self.error_vec[i])
                self.error_vec_log.add([i, " , " + error_str])
            self.error_vec_log.write()

    @staticmethod
    def error_vector(ref, model, weights):
        assert (len(ref) == len(model))
        out = []
        for i in range(len(ref)):
            diff = (model[i] - ref[i]) * weights[i]
            if isinstance(diff, float):
                out.append(diff)
            else:
                # diff = np.sign(diff.flatten()[np.argmax(np.abs(diff))])*np.amax(np.abs(diff))
                # out.append(diff)
                diff = np.array(diff)
                diff = diff.reshape(np.size(diff))
                out += list(diff)
        out = np.array(out)
        out = out.reshape(np.size(out))
        return out

    @staticmethod
    def eval_error(array, mode='rms'):
        if mode.lower() == 'rms':
            out = np.sqrt(np.average(array ** 2))
        elif mode.lower() == 'mae':
            out = np.average(np.absolute(array))
        elif mode.lower() == 'max':
            out = np.amax(np.absolute(array))
        elif mode.lower() == 'max_square':
            out = np.amax(np.absolute(array))
        elif mode.lower() == 'min':
            out = np.amin(np.absolute(array))
        elif mode.lower() == 'std':
            out = np.std(array)
        elif mode.lower() == 'average':
            out = np.average(array)
        else:
            raise NotImplementedError('No options for %s' % mode)
        return out

    def _shift_energies(self, data):
        atoms = self.calc.get_atoms()
        energies = []
        for i in range(len(atoms)):
            if atoms[i].info['required_property'] == 'energy':
                energies.append(data[i] / len(atoms[i]))
        mine = np.amin(energies)
        for i in range(len(atoms)):
            if atoms[i].info['required_property'] == 'energy':
                data[i] -= mine * len(atoms[i])
        return data

    def _calc_model_data(self, p1):
        found = False
        model = gen_param_model(self.model, self.variables, newparam=p1)
        for i in range(len(self._memory[0])):
            if self._memory[0][i] is None:
                continue
            diff = np.abs(np.array(p1) - np.array(self._memory[0][i]))
            if (diff < self.xtol).all():
                model_data = self._memory[1][i]
                found = True
                break

        if not found:
            # set new model
            self.calc.set_model(model)

            # calculate model_ene
            model_data = self.calc.get_property(contribution=self.contribution)
            if self.contribution == 'empirical':
                for i in range(len(model_data)):
                    model_data[i] += self.init_model_data[i]

            if self.shift_min_ene:
                model_data = self._shift_energies(model_data)

            if self.save > 0:
                if None in self._memory[0]:
                    l = self._memory[0].index(None)
                else:
                    l = len(self.rms) % self.save
                self._memory[0][l] = p1
                self._memory[1][l] = model_data
        return model_data, model

    def default(self, x0):
        """
        Default objective function. 
        
        Returns the diffferences between the calculated properties and the
        reference.
        """
        if self.penalty_coeffs is not None:
            penalty_coeffs_list = gen_penalty_coeffs_list(self.model,
                                                          self.variables,
                                                          self.penalty_coeffs)
            pt = penalty_term(x0, self.p0, penalty_coeffs_list)
            if len(pt) > 0:
                self.penalty.append(np.average(pt))
        else:
            pt = [0]
            self.penalty.append(0.)
        p1 = np.array(x0) * np.array(self.p0)

        model_data, model = self._calc_model_data(p1)
        # get error vector   
        err = CATobjective.error_vector(self.ref_data, model_data, self.weights)

        rms = float(np.sqrt(np.average(err ** 2)))

        if self.dump_min_model == True:
            if len(self.rms) == 0 or rms < np.amin(self.rms):
                model.write(filename='model_min.bx')
        elif isinstance(self.dump_min_model, str):
            if len(self.rms) == 0 or rms < np.amin(self.rms):
                model.write(filename=self.dump_min_model)

        # check nan
        if rms != rms:
            self.rms.append(float('inf'))
        else:
            self.rms.append(rms)
        self.par.append(p1)

        if self.gui_log is not None:
            if rms < 100:
                with open(self.gui_log, 'a') as f:
                    if len(p1) > 0:
                        f.write('%5d %20.10f ' % (len(self.rms), rms))
                        for p in p1:
                            f.write('%20.10f ' % p)
                        f.write('\n')

        if self.error_vec_log is not None:
            self.error_vec.append(err)

        if self.verbose:
            keys = ['mae', 'std', 'max', 'min']
            msg = 'iter: %8d rms: %15.8f     ' % (len(self.rms), self.rms[-1])
            for key in keys:
                val = CATobjective.eval_error(err, mode=key)
                self.error_details[key] = val
                msg += '%s: %15.8f     ' % (key, val)
            msg += ' penalty: %15.8f     ' % (self.penalty[-1])
            print_format(msg, level=2)
        if self.array:
            if len(x0) > 0:
                scaled_pt = np.sqrt(len(err) / len(x0)) * np.array(pt)
            else:
                scaled_pt = np.array(pt)
            return np.hstack((err, scaled_pt))
        elif self.error_type.lower() in list(self.error_details.keys()):
            return self.error_details[self.error_type.lower()] + pt
        else:
            return CATobjective.eval_error(err, mode=self.error_type.lower()) + pt

    def _get_diff(self):
        if self.ref_data_diff is None:
            self.ref_data_diff = calc_diff(self.ref_data)
        return self.ref_data_diff

    def difference(self, x0):
        """
        Objective function for energy differences calculations.
        
        Returns the differences between delta(calculated properties) and
        delta(reference)::
        
            delta_data = data[:len(data)/2] - data[len(data)/2:]
        """
        if self.penalty_coeffs is not None:
            penalty_coeffs_list = gen_penalty_coeffs_list(self.model,
                                                          self.variables,
                                                          self.penalty_coeffs)
            pt = penalty_term(x0, self.p0, penalty_coeffs_list)
            if len(pt) > 0:
                self.penalty.append(np.average(pt))
        else:
            self.penalty.append(0.)
        p1 = np.array(x0) * np.array(self.p0)
        model_data, model = self._calc_model_data(p1)
        # calculate difference of model_data
        model_data = calc_diff(model_data)
        # calculate difference
        ref_data = self._get_diff()
        # get error vector
        err = CATobjective.error_vector(ref_data, model_data, self.weights)

        rms = np.sqrt(np.average(err ** 2))

        if self.dump_min_model:
            if len(self.rms) == 0 or rms < np.amin(self.rms):
                model.write(filename='model_min.bx')

        # check nan
        if rms != rms:
            self.rms.append(float('inf'))
        else:
            self.rms.append(rms)
        self.par.append(p1)

        if self.error_vec_log is not None:
            self.error_vec.append(err)

        if self.verbose:
            keys = ['mae', 'std', 'max', 'min']
            msg = 'iter: %8d rms: %15.8f     ' % (len(self.rms), self.rms[-1])
            for key in keys:
                val = CATobjective.eval_error(err, mode=key)
                self.error_details[key] = val
                msg += '%s: %15.8f     ' % (key, val)
            msg += ' penalty: %15.8f     ' % (self.penalty[-1])
            print_format(msg, level=2)
        if self.array:
            if len(x0) > 0:
                scaled_pt = np.sqrt(len(err) / len(x0)) * np.array(pt)
            else:
                scaled_pt = np.array(pt)
            return np.hstack((err, scaled_pt))
        elif self.error_type.lower() in list(self.error_details.keys()):
            return self.error_details[self.error_type.lower()] + pt
        else:
            return CATobjective.eval_error(err, mode=self.error_type.lower()) + pt


class CAToptimizer:
    """
    Defines the optimizer. Callable and returns the optimized parameters.
      
    :Parameters:
                            
        - *objective*: callable object
        
            should take parameters as argument and return the error 
                        
        - *x0*: list
            
            initial guess for parameters               
            
        - *optimizer_options*: dict
        
            controls to be passed to the optimizer

        - *name*: str
        
            name of the optimizer
    """

    def __init__(self, **kwargs):
        from numpy import inf
        self.name = None
        self.objective = None
        self.x0 = None
        self.x1 = None
        self.options = None
        self.set(**kwargs)

    def set(self, **kwargs):
        for key in kwargs:
            if key.lower() == 'name':
                self.name = kwargs[key]
            elif key.lower() == 'objective':
                self.objective = kwargs[key]
            elif key.lower() == 'x0':
                self.x0 = kwargs[key]
            elif key.lower() == 'options':
                self.options = kwargs[key]
            else:
                raise ValueError('Cannot recognize key %s' % key)
        self.set_defaults()
        self.set_options()

    def set_defaults(self):
        if self.name.lower() == 'leastsq':
            args = getargspec(optimize.leastsq)
        elif self.name.lower() == 'least_squares':
            args = getargspec(optimize.least_squares)
        elif self.name.lower() == 'nelder-mead':
            args = getargspec(optimize.fmin)
        elif self.name.lower() == 'powell':
            args = getargspec(optimize.fmin_powell)
        elif self.name.lower() == 'cg':
            args = getargspec(optimize.fmin_cg)
        elif self.name.lower() == 'bfgs':
            args = getargspec(optimize.fmin_bfgs)
        elif self.name.lower() == 'ncg':
            args = getargspec(optimize.fmin_ncg)
        elif self.name.lower() == 'l-bfgs-b':
            args = getargspec(optimize.fmin_l_bfgs_b)
        elif self.name.lower() == 'tnc':
            args = getargspec(optimize.fmin_tnc)
        elif self.name.lower() == 'slsqp':
            args = getargspec(optimize.fmin_slsqp)
        elif self.name.lower() == 'differential_evolution':
            args = getargspec(optimize.differential_evolution)
        elif self.name.lower() == 'basinhopping':
            args = getargspec(optimize.basinhopping)
        elif self.name.lower() == 'brute':
            args = getargspec(optimize.brute)
        elif 'nlopt' in self.name.lower():
            return
        elif 'psopt' in self.name.lower():
            return
        elif 'genetic' in self.name.lower():
            return
        else:
            raise NotImplementedError('No options for %s' % self.name)
        keys = args.args
        defaults = args.defaults
        # this is a dirty way of setting optimizer defaults
        for i in range(len(defaults), 0, -1):
            exec("self._%s = defaults[-i]" % (keys[-i]))

    def set_options(self):
        # this is a dirty way of setting optimizer options
        for key in self.options:
            exec("self._%s = self.options['%s']" % (key, key))

    def set_options_nlopt(self, opt):
        for key, val in list(self.options.items()):
            if key.lower() in ['lower_bounds', 'upper_bounds']:
                if isinstance(val, float):
                    val = [val] * len(self.x0)
            try:
                exec("opt.set_%s(%s)" % (key, val))
            except:
                print_format('Warning: Cannot set optimizer option %s' % key, 3)
        return opt

    def set_options_opt(self, opt):
        for key, val in list(self.options.items()):
            if key.lower() in ['lower_bounds', 'upper_bounds']:
                if isinstance(val, float):
                    val = [val] * len(self.x0)
            elif key.lower() == 'catkernel':
                opt.catkernel = val
                continue
            try:
                if isinstance(val, str):
                    exec("opt.set(%s='%s')" % (key, val))
                else:
                    exec("opt.set(%s=%s)" % (key, val))
            except:
                print_format('Warning: Cannot set optimizer option %s' % key, 3)
        return opt

    def __call__(self):
        # Least squares is obsolete, use least_squares and set method='lm'
        if self.name.lower() == 'leastsq':
            x1 = optimize.leastsq(self.objective, self.x0, args=self._args
                                  , Dfun=self._Dfun
                                  , full_output=self._full_output, col_deriv=self._col_deriv
                                  , ftol=self._ftol, xtol=self._xtol, gtol=self._gtol
                                  , maxfev=self._maxfev
                                  , epsfcn=self._epsfcn, factor=self._factor, diag=self._diag)
            x1 = x1[0]
        elif self.name.lower() == 'least_squares':
            x1 = optimize.least_squares(self.objective, self.x0, jac=self._jac
                                        , bounds=self._bounds, method=self._method, ftol=self._ftol
                                        , xtol=self._xtol, gtol=self._gtol, x_scale=self._x_scale
                                        , loss=self._loss, f_scale=self._f_scale
                                        , diff_step=self._diff_step, tr_solver=self._tr_solver
                                        , tr_options=self._tr_options
                                        , jac_sparsity=self._jac_sparsity, max_nfev=self._max_nfev

                                        , verbose=self._verbose, args=self._args
                                        , kwargs=self._kwargs)
            x1 = x1.x
        elif self.name.lower() == 'nelder-mead':
            x1 = optimize.fmin(self.objective, self.x0, args=self._args
                               , xtol=self._xtol, ftol=self._ftol, maxiter=self._maxiter
                               , maxfun=self._maxfun, full_output=self._full_output
                               , disp=self._disp, retall=self._retall
                               , callback=self._callback)
        elif self.name.lower() == 'powell':
            x1 = optimize.fmin_powell(self.objective, self.x0, args=self._args
                                      , xtol=self._xtol, ftol=self._ftol, maxiter=self._maxiter
                                      , maxfun=self._maxfun, full_output=self._full_output
                                      , disp=self._disp, retall=self._retall
                                      , callback=self._callback, direc=self._direc)
        elif self.name.lower() == 'cg':
            x1 = optimize.fmin_cg(self.objective, self.x0, fprime=self._fprime
                                  , args=self._args, gtol=self._gtol, norm=self._norm
                                  , epsilon=self._epsilon, maxiter=self._maxiter
                                  , full_output=self._full_output, disp=self._disp
                                  , retall=self._retall, callback=self._callback)
        elif self.name.lower() == 'bfgs':
            x1 = optimize.fmin_bfgs(self.objective, self.x0, fprime=self._fprime
                                    , args=self._args, gtol=self._gtol, norm=self._norm
                                    , epsilon=self._epsilon, maxiter=self._maxiter
                                    , full_output=self._full_output, disp=self._disp
                                    , retall=self._retall, callback=self._callback)
        elif self.name.lower() == 'ncg':
            x1 = optimize.fmin_ncg(self.objective, self.x0, fprime=self._fprime
                                   , fhess_p=self._fhess_p, fhess=self._fhess
                                   , args=self._args, avextol=self._avextol
                                   , epsilon=self._epsilon, maxiter=self._maxiter
                                   , full_output=self._full_output, disp=self._disp
                                   , retall=self._retall, callback=self._callback)
        elif self.name.lower() == 'l-bfgs-b':
            x1 = optimize.fmin_l_bfgs_b(self.objective, self.x0
                                        , fprime=self._fprime, args=self._args
                                        , approx_grad=self._approx_grad
                                        , bounds=self._bounds, m=self._m, factr=self._factr
                                        , pgtol=self._pgtol, epsilon=self._epsilon
                                        , iprint=self._iprint, maxfun=self._maxfun
                                        , maxiter=self._maxiter, disp=self._disp
                                        , callback=self._callback, maxls=self._maxls)
            x1 = x1[0]
        elif self.name.lower() == 'tnc':
            x1 = optimize.fmin_tnc(self.objective, self.x0, fprime=self._fprime
                                   , args=self._args, approx_grad=self._approx_grad
                                   , bounds=self._bounds, epsilon=self._epsilon
                                   , scale=self._scale, offset=self._offset
                                   , messages=self._messages, maxCGit=self._maxCGit
                                   , maxfun=self._maxfun, eta=self._eta, stepmx=self._stepmx
                                   , accuracy=self._accuracy, fmin=self._fmin, ftol=self._ftol
                                   , xtol=self._xtol, pgtol=self._pgtol, rescale=self._rescale
                                   , disp=self._disp, callback=self._callback)
            x1 = x1[0]
        elif self.name.lower() == 'slsqp':
            x1 = optimize.fmin_slsqp(self.objective, self.x0, eqcons=self._eqcons
                                     , f_eqcons=self._f_eqcons, ieqcons=self._ieqcons
                                     , f_ieqcons=self._f_ieqcons, bounds=self._bounds
                                     , fprime=self._fprime, fprime_eqcons=self._fprime_eqcons
                                     , fprime_ieqcons=self._fprime_ieqcons, args=self._args
                                     , iter=self._iter, acc=self._acc, iprint=self._iprint
                                     , disp=self._disp, full_output=self._full_output
                                     , epsilon=self._epsilon, callback=self._callback)
        elif self.name.lower() == 'differential_evolution':
            if 'bounds' in self.options:
                self.bounds = self.options['bounds']
                assert (len(self.bounds) == len(self.x0))
            x1 = optimize.differential_evolution(self.objective, self.bounds
                                                 , args=self._args, strategy=self._strategy
                                                 , maxiter=self._maxiter, popsize=self._popsize
                                                 , tol=self._tol, mutation=self._mutation
                                                 , recombination=self._recombination, seed=self._seed
                                                 , callback=self._callback, disp=self._disp
                                                 , polish=self._polish, init=self._init)
            x1 = x1.x
        elif self.name.lower() == 'basinhopping':
            x1 = optimize.basinhopping(self.objective, self.x0
                                       , niter=self._niter, T=self._T, stepsize=self._stepsize
                                       , minimizer_kwargs=self._minimizer_kwargs
                                       , take_step=self._take_step, accept_test=self._accept_test
                                       , callback=self._callback, interval=self._interval
                                       , disp=self._disp, niter_success=self._niter_success)
            x1 = x1.x
        elif self.name.lower() == 'brute':
            if 'bounds' in self.options:
                self.bounds = self.options['bounds']
                assert (len(self.bounds) == len(self.x0))
            x1 = optimize.brute(self.objective, self.bounds, args=self._args
                                , Ns=self._Ns, full_output=self._full_output
                                , finish=self._finish, disp=self._disp)
            x1 = x1.x
        elif 'nlopt' in self.name.lower():
            import nlopt
            s = self.name.lower().strip('nlopt').strip('_').strip('.')
            try:
                exec("opt = nlopt.opt(nlopt.%s,%d)" % (s.upper(), len(self.x0)))
            except AttributeError:
                print_format("Unrecognized optimizer %s" % self.name, 3)
            except:
                raise
            opt = self.set_options_nlopt(opt)
            opt.set_min_objective(self.objective)
            x1 = opt.optimize(self.x0)
        elif self.name.lower() == 'psopt':
            from .psopt import PSopt
            opt = PSopt(objective=self.objective, x0=self.x0)
            opt = self.set_options_opt(opt)
            opt.optimize()
            x1 = opt.x
        elif self.name.lower() == 'geneticalgo':
            from .geneticalgo import Geneticalgo
            opt = Geneticalgo(objective=self.objective, x0=self.x0)
            opt = self.set_options_opt(opt)
            opt.optimize()
            x1 = opt.x
        else:
            raise NotImplementedError('No options for %s' % self.name)
        self.x1 = x1
        return x1

    def call__(self):
        # Least squares is obsolete, use least_squares and set method='lm'
        if self.name.lower() == 'leastsq':
            x1 = optimize.leastsq(self.objective, self.x0, args=self._args
                                  , Dfun=self._Dfun
                                  , full_output=self._full_output, col_deriv=self._col_deriv
                                  , ftol=self._ftol, xtol=self._xtol, gtol=self._gtol
                                  , maxfev=self._maxfev
                                  , epsfcn=self._epsfcn, factor=self._factor, diag=self._diag)
            x1 = x1[0]
        elif self.name.lower() == 'least_squares':
            x1 = optimize.least_squares(self.objective, self.x0, jac=self._jac
                                        , bounds=self._bounds, method=self._method, ftol=self._ftol
                                        , xtol=self._xtol, gtol=self._gtol, x_scale=self._x_scale
                                        , loss=self._loss, f_scale=self._f_scale
                                        , diff_step=self._diff_step, tr_solver=self._tr_solver
                                        , tr_options=self._tr_options
                                        , jac_sparsity=self._jac_sparsity, max_nfev=self._max_nfev

                                        , verbose=self._verbose, args=self._args
                                        , kwargs=self._kwargs)
            x1 = x1.x
        elif self.name.lower() in ['nelder-mead', 'powell', 'cg', 'bfgs'
            , 'newton-cg', 'l-bfgs', 'tnc', 'cobyla', 'slsqp'
            , 'dogleg', 'trust-ncg']:
            x1 = optimize.minimize(self.objective, self.x0, args=self._args
                                   , method=self.name, options=self.options)
            x1 = x1.x
        elif self.name.lower() == 'differential_evolution':
            if 'bounds' in self.options:
                self.bounds = self.options['bounds']
                assert (len(self.bounds) == len(self.x0))
            x1 = optimize.differential_evolution(self.objective, self.bounds
                                                 , args=self._args, strategy=self._strategy
                                                 , maxiter=self._maxiter, popsize=self._popsize
                                                 , tol=self._tol, mutation=self._mutation
                                                 , recombination=self._recombination, seed=self._seed
                                                 , callback=self._callback, disp=self._disp
                                                 , polish=self._polish, init=self._init)
            x1 = x1.x
        elif self.name.lower() == 'basinhopping':
            x1 = optimize.basinhopping(self.objective, self.x0
                                       , niter=self._niter, T=self._T, stepsize=self._stepsize
                                       , minimizer_kwargs=self._minimizer_kwargs
                                       , take_step=self._take_step, accept_test=self._accept_test
                                       , callback=self._callback, interval=self._interval
                                       , disp=self._disp, niter_success=self._niter_success)
            x1 = x1.x
        elif self.name.lower() == 'brute':
            if 'bounds' in self.options:
                self.bounds = self.options['bounds']
                assert (len(self.bounds) == len(self.x0))
            x1 = optimize.brute(self.objective, self.bounds, args=self._args
                                , Ns=self._Ns, full_output=self._full_output
                                , finish=self._finish, disp=self._disp)
            x1 = x1.x
        elif 'nlopt' in self.name.lower():
            import nlopt
            s = self.name.lower().strip('nlopt').strip('_').strip('.')
            try:
                exec("opt = nlopt.opt(nlopt.%s,%d)" % (s.upper(), len(self.x0)))
            except AttributeError:
                print_format("Unrecognized optimizer %s" % self.name, 3)
            except:
                raise
            opt = self.set_options_nlopt(opt)
            opt.set_min_objective(self.objective)
            x1 = opt.optimize(self.x0)
        elif self.name.lower() == 'psopt':
            from .psopt import PSopt
            opt = PSopt(objective=self.objective, x0=self.x0)
            opt = self.set_options_opt(opt)
            opt.optimize()
            x1 = opt.x
        elif self.name.lower() == 'geneticalgo':
            from .geneticalgo import Geneticalgo
            opt = Geneticalgo(objective=self.objective, x0=self.x0)
            opt = self.set_options_opt(opt)
            opt.optimize()
            x1 = opt.x
        else:
            raise NotImplementedError('No options for %s' % self.name)
        self.x1 = x1
        return x1
