#!/usr/bin/env python

# Definition of the CATJob object

# This module is part of the BOPcat package

from .catdata import CATData
from .catparam import CATParam
from .catcalc import CATCalc
from .catkernel import CATKernel
import numpy as np
from .bopmodel import modelsbx
from .process_management import Process_catkernel, Process_catkernels
from collections import OrderedDict
from .functions import Function
from .utils import gen_param_model
from .output import print_format, cattxt
from .functions import polynomial, constant, sum_funcs, fit
from math import ceil as mceil
from math import exp as mexp


###########################################################################
class CATProcess:
    """
    Defines a main job protocol.

    The general parameterization strategy is implemented here

    :Parameters:
	
	- *controls*: instance of CATControls   
        
            CATControls object to initialize parameters      
    """

    def __init__(self, **kwargs):
        self._controls = None
        self._catdata = None
        self._catparam = None
        self._catcalc = None
        self._variables = None
        self._optimizer = None
        self._nstrucs_fit = None
        self._properties_fit = ['energy']
        self._structures_fit = ['*']
        self._atoms_fit = None
        self._success_criteria_fit = None
        self._success_criteria_test = None
        self._structures_test = ['*']
        self._properties_test = ['energy']
        self._elements = None
        self._queue = 'cluster_serial'
        self._directives = ['source ~/.bashrc']
        self._nproc = 8
        # self._nstrucmax            = 1000000000000
        self._nstrucmax = 1000
        self._refmodel = None
        self._nvar = None
        self._correlation_data = None
        self._correlation_funcs = None
        self._variables_split = None
        self._nsamples = None
        self._job = None
        self.set(**kwargs)
        self._init_msg()

    def set(self, **kwargs):
        if 'controls' in kwargs:
            self._controls = kwargs['controls']
            if self._controls is not None:
                self._unfold_controls()
        for key in kwargs:
            if key.lower() == 'catdata':
                self._catdata = kwargs[key]
            elif key.lower() == 'catparam':
                self._catparam = kwargs[key]
            elif key.lower() == 'catcalc':
                self._catcalc = kwargs[key]
            elif key.lower() == 'variables':
                self._variables = kwargs[key]
            elif key.lower() == 'nstrucs_fit':
                self._nstrucs_fit = kwargs[key]
            elif key.lower() == 'success_criteria_fit':
                self._success_criteria_fit = kwargs[key]
            elif key.lower() == 'success_criteria_test':
                self._success_criteria_test = kwargs[key]
            elif key.lower() == 'structures_test':
                self._structures_test = kwargs[key]
            elif key.lower() == 'properties_test':
                self._properties_test = kwargs[key]
            elif key.lower() == 'elements':
                self._elements = kwargs[key]
            elif key.lower() == 'queue':
                self._queue = kwargs[key]
            elif key.lower() == 'directives':
                self._directives = kwargs[key]
            elif key.lower() == 'nproc':
                self._nproc = kwargs[key]
            elif key.lower() == 'jobid':
                self.reload_job(kwargs[key])
            elif key.lower() == 'nstrucmax':
                self._nstrucmax = kwargs[key]
            elif key.lower() == 'refmodel':
                self._refmodel = kwargs[key]
            elif key.lower() == 'nsamples':
                self._nsamples = kwargs[key]
            elif key.lower() == 'controls':
                pass
            else:
                raise ValueError('Unrecognized key %s' % key)
        if self._catdata is None:
            self._gen_catdata()
        if self._catparam is None:
            self._gen_param()

    def _init_msg(self):
        print_format('CATProcess built', level=1)

    def _unfold_controls(self):
        self._nproc = self._controls.calculator_nproc
        # self._variables = self._controls.opt_variables
        self._optimizer = self._controls.opt_optimizer
        self._elements = self._controls.elements

    def _gen_catdata(self):
        self._catdata = CATData(controls=self._controls)

    def _gen_param(self):
        self._catparam = CATParam(controls=self._controls, data=self._catdata)

    def _gen_catcalc(self, model=0):
        if isinstance(model, int):
            model = self._catparam.models[model]
        calc = CATCalc(controls=self._controls, model=model)
        return calc

    def reload_job(self, jid):
        self._job = Process_catkernels(pid=jid)
        self._nstrucs_fit = len(
            self._job._procs[0]._kernel._catkernel.calc._atoms)

    def _gen_kernel(self, model, fit_atoms, fit_data, var=None, global_opt=False):
        calc = self._gen_catcalc(model)
        calc.set_atoms(fit_atoms)
        # dynamically allocate nproc to len of atoms
        # if len(fit_atoms) < self._nproc:
        #    calc.nproc = len(fit_atoms) 
        kernel = CATKernel(calc=calc, ref_data=fit_data, variables=var
                           , controls=self._controls, verbose=1)
        if global_opt:
            kernel.optimizer = 'psopt'
            # kernel.optimizer = 'basinhopping'
            # kernel.optimizer_options = {'niter':2000,'T':0.1,'stepsize':0.05
            #                     ,'minimizer_kwargs':{'method':'nelder-mead'}}
        return kernel

    def get_nvar(self, refresh=False):
        if self._nvar is None or refresh:
            self._nvar = 0
            variables = self.get_fit_variables()
            for i in range(len(variables)):
                for key, val in list(variables[i].items()):
                    if key in ['atom', 'bond']:
                        continue
                    self._nvar += val.count(True)
        return self._nvar

    def get_refmodel(self):
        if self._refmodel is None:
            refmodel = self._catparam.models[0].copy()
            if isinstance(refmodel, modelsbx):
                refmodel.bond_parameters_to_functions(
                    variables=self.get_fit_variables())
            else:
                raise NotImplementedError(
                    'No options for %s' % type(self.get_refmodel()))
            self._refmodel = refmodel
        return self._refmodel

    def _gen_fit_variables_modelsbx(self, mbx, atoms, data):
        # include only bonds parameters?
        bbx = mbx.bondsbx
        fitvar = []
        dp = 0.9
        # get reference error
        kernels = [self._gen_kernel(mbx, atoms, data, None)]
        # calculate error corresponding to each change in parameter
        for i in range(len(bbx)):
            # I do not want to use OrderedDict
            fv = OrderedDict({})
            pars = bbx[i].get_bondspar()
            pars.update(bbx[i].get_overlappar())
            pars.update(bbx[i].get_repetal())
            fv['bond'] = bbx[i].bond
            for key, val in list(pars.items()):
                if val is None:
                    continue
                if isinstance(val, Function):
                    num = val.get_numbers()
                else:
                    num = list(val)
                fv[key] = [False] * len(num)
                # tweak each parameter a bit and see if energy changes
                for j in range(len(num)):
                    var = list(fv[key])
                    var[j] = True
                    var = [{'bond': bbx[i].bond, key: var}]
                    mbxt = mbx.rattle(var=var, factor=dp)
                    kernels.append(self._gen_kernel(mbxt, atoms, data, None))
            fitvar.append(fv)
        # raise
        # check if order of dict is preserved!     
        err = self.run_kernels(kernels)
        count = 1
        tol = 100  # if varible contributes to very high derr, will fix

        for i in range(len(fitvar)):
            for key, val in list(fitvar[i].items()):
                if key in ['bond', 'atom']:
                    continue
                for j in range(len(val)):
                    diff = abs(err[count] - err[0])
                    if diff > 0 and diff < tol:
                        val[j] = True
                    count += 1
        assert (len(err) == count)
        self._variables = fitvar

    def run_kernels(self, kernels, success_criteria={}, out='function_value'):
        if self._job is None:
            procs = []
            for kernel in kernels:
                # cores should match nproc in calc
                # cores = kernel.calc.nproc
                cores = self._nproc
                procs.append(Process_catkernel(catkernel=kernel
                                               , success_criteria=success_criteria
                                               , queue=self._queue, cores=cores
                                               , directives=self._directives))
            self._job = Process_catkernels(procs=procs)
            self._job.run()
        results = self._job.get_results(wait=True)
        success = [p.is_success() for p in self._job._procs]
        results['procs'] = [p for p in self._job._procs if p.is_success()]
        assert (len(results[out]) == success.count(True))
        # self._job.clean()
        # reset job
        self._job = None
        return results[out]

    def get_fit_variables(self, model=0):
        """
        identify model parameters that contribute to change in test_properties
        set them free. Use only one structure
        """
        if self._variables is None:
            atoms, data = self.gen_strucs_random(1, 'test')
            if isinstance(model, int):
                # get ith model        
                model = self._catparam.models[model]
            if isinstance(model, modelsbx):
                self._gen_fit_variables_modelsbx(model, atoms, data)
            else:
                raise NotImplementedError("No options for %s" % type(model))
        return self._variables

    def _exclude_fit_atoms(self, atoms, data):
        # this does not work
        exclude = self._atoms_fit
        if exclude is None:
            exclude = []
        tempa = []
        tempd = []
        for i in range(len(atoms)):
            if atoms[i] in exclude:
                continue
            tempa.append(atoms[i])
            tempd.append(data[i])
        return tempa, tempd

    def gen_struc_smap(self, nstrucs, target):
        """
        """
        if target == 'fit':
            structures = self._structures_fit
            quantities = self._properties_fit
        elif target == 'test':
            structures = self._structures_test
            quantities = self._properties_test
        else:
            raise ValueError('unknown target %s' % target)
        atoms = self._catdata.get_ref_atoms(structures=structures
                                            , quantities=quantities)
        data = self._catdata.get_ref_data()
        # get distance from reference structure
        gs = self._catdata.get_ground_state(elements=self._elements)
        if gs is None:
            raise ValueError('Cannot find ground state')
        dist = self._catdata.get_structuremap_distance(gs, atoms)
        # sort structures in increasing distance from gs
        temp = [(dist[i], i) for i in range(len(atoms))]
        temp.sort()
        atoms = [atoms[temp[i][1]] for i in range(len(temp))]
        dist = [temp[i][0] for i in range(len(temp))]
        data = [data[temp[i][1]] for i in range(len(temp))]
        if target == 'test':
            atoms, data = self._exclude_fit_atoms(atoms, data)
        atoms = atoms[:nstrucs]
        data = data[:nstrucs]
        return atoms, data

    def gen_strucs_random(self, nstrucs, target):
        atoms = []
        data = []
        included = []
        if target == 'fit':
            structures = self._structures_fit
            quantities = self._properties_fit
        elif target == 'test':
            structures = self._structures_test
            quantities = self._properties_test
        else:
            raise ValueError('unknown target %s' % target)
        all_atoms = self._catdata.get_ref_atoms(structures=structures
                                                , quantities=quantities)
        all_data = self._catdata.get_ref_data()
        if target == 'test':
            all_atoms, all_data = self._exclude_fit_atoms(all_atoms, all_data)
        while len(atoms) < nstrucs:
            rs = np.random.choice(list(range(len(all_atoms))))
            if rs in included:
                continue
            included.append(rs)
            atoms.append(all_atoms[rs])
            data.append(all_data[rs])
        atoms = atoms[:nstrucs]
        data = data[:nstrucs]
        return atoms, data

    def gen_strucs_sorted(self, nstrucs, target, sort_by):
        atoms = []
        data = []
        if target == 'fit':
            structures = self._structures_fit
            quantities = self._properties_fit
        elif target == 'test':
            structures = self._structures_test
            quantities = self._properties_test
        else:
            raise ValueError('unknown target %s' % target)
        all_atoms = self._catdata.get_ref_atoms(structures=structures
                                                , quantities=quantities
                                                , sort_by=sort_by)
        all_data = self._catdata.get_ref_data()
        if target == 'test':
            all_atoms, all_data = self._exclude_fit_atoms(all_atoms, all_data)
        atoms = all_atoms[:nstrucs]
        data = all_data[:nstrucs]
        return atoms, data

    def gen_strucs(self, nstrucs, method='structure_map', target='fit'):
        """
        Generates nstrucs fit strucs
        mode : structure_map, random
        """
        if self._structures_fit is None:
            self._structures_fit = ['*'.join(self.elements) + '*']
        if method.lower() == 'structure_map':
            atoms, data = self.gen_struc_smap(nstrucs, target)
        elif method.lower() == 'random':
            atoms, data = self.gen_strucs_random(nstrucs, target)
        elif method.lower() == 'sorted_by_energy':
            atoms, data = self.gen_strucs_sorted(nstrucs, target, 'energy')
        else:
            raise ValueError('unknown method %s' % method)
        return atoms, data

    def get_fit_model_optimized(self, return_score=False):
        score = 1E99
        best = 0
        for i in range(len(self._catparam.models)):
            if i not in self._catparam.scores:
                continue
            if self._catparam.scores[i] < score:
                score = self._catparam.scores[i]
                best = i
        if return_score:
            return self._catparam.models[best].copy(), score
        else:
            return self._catparam.models[best].copy()

    def gen_fit_model_random(self, seed=0):
        if isinstance(seed, int):
            model = self._catparam.models[seed]
        else:
            model = seed
        model = model.copy()
        if isinstance(model, modelsbx):
            model = model.rattle(var=self.get_fit_variables(), factor='random'
                                 , maxf=2)
        else:
            raise NotImplementedError("No options for %s" % type(model))
        return model

    def gen_model(self, method='optimized', seed=0, newparam=None):
        if isinstance(newparam, list):
            if isinstance(seed, int):
                m0 = self._catparam.models[seed]
            else:
                m0 = seed
            model = gen_param_model(m0, self.get_fit_variables()
                                    , newparam=newparam)
        elif method == 'optimized':
            model = self.get_fit_model_optimized()
        elif method == 'random':
            model = self.gen_fit_model_random(seed=seed)
        else:
            raise NotImplementedError("No options for %s" % method)
        return model

    def run_process(self, newparam, nsamples, nstrucs, strucs_sampling
                    , model_sampling, success_criteria, update_models, global_opt):
        kernels = []
        target = 'test'
        if newparam is None:
            target = 'fit'
        # if starting from saved process do not construct kernels
        if self._job is None:
            if target == 'fit':
                var = self.get_fit_variables()
            elif target == 'test':
                var = None
                nsamples = len(newparam)
            # generate kernels
            for i in range(nsamples):
                # get reference structures and properties
                atoms, data = self.gen_strucs(nstrucs, method=strucs_sampling
                                              , target=target)
                if target == 'fit':
                    model = self.gen_model(method=model_sampling
                                           , seed=self.get_refmodel())
                elif target == 'test':
                    model = self.gen_model(seed=self.get_refmodel()
                                           , newparam=newparam[i])
                kernels.append(self._gen_kernel(model, atoms, data, var=var
                                                , global_opt=global_opt))
        # run kernels
        procs = self.run_kernels(kernels, success_criteria=success_criteria
                                 , out='procs')
        if update_models:
            # add models to catparam
            N = len(self._catparam.models)
            for i in range(len(procs)):
                if procs[i].is_success():
                    if target == 'test':
                        new_model = \
                            procs[i]._kernel._catkernel.calc.get_model()
                    elif target == 'fit':
                        new_model = \
                            procs[i]._kernel._catkernel.get_optimized_model()
                    self._catparam.models.append(new_model)
                    self._catparam.scores[N] = \
                        procs[i].get_results()['function_value']
                    N += 1
        if len(procs) == 0:
            msg = """No successful jobs detected \
            . Try increasing nsamples or decreasing success criteria."""
            raise RuntimeError(msg)
        return procs

    def get_success_criteria(self, key='fit'):
        if key == 'fit':
            if self._success_criteria_fit is None:
                self._success_criteria_fit = {'err': 0.05}
            return self._success_criteria_fit
        elif key == 'test':
            if self._success_criteria_test is None:
                self._success_criteria_test = {'err': 1.}
            return self._success_criteria_test

    def fit(self, nsamples='default', strucs_sampling='structure_map'
            , model_sampling='random', nstrucs='default'
            , success_criteria='default', global_opt=False):
        if nsamples == 'default':
            nsamples = self._nvar * 100
        if nstrucs == 'default':
            nstrucs = self.get_nstrucs_fit()
        if success_criteria == 'default':
            success_criteria = self.get_success_criteria('fit')
        procs = self.run_process(None, nsamples, nstrucs
                                 , strucs_sampling, model_sampling
                                 , success_criteria, False, global_opt)
        new_par = [p.get_results()['optimized_parameters'] for p in procs]
        return new_par

    def test(self, newparam, strucs_sampling='structure_map', nstrucs='default'
             , success_criteria='default', update_models=True):
        if nstrucs == 'default':
            nstrucs = self._nstrucmax
        if success_criteria == 'default':
            success_criteria = self.get_success_criteria('test')
        procs = self.run_process(newparam, len(newparam), nstrucs
                                 , strucs_sampling, 'optimized'
                                 , success_criteria, update_models, False)
        terr = [p.get_results()['function_value'] for p in procs]
        return terr

    def gen_optimum_nstrucs(self, tol=0.01, nsamples='default'
                            , strucs_sampling='structure_map'
                            , model_sampling='random', show_plot=False):
        """
        Vary nstrucs, optimize parameters, calculate test score
        stops if change in test score is less than tolerance
        """
        if nsamples == 'default':
            nsamples = self.get_nvar() * 4
        global_opt = False
        if nsamples == 1:
            global_opt = True
        nstruc0 = self.get_nvar()
        nstrucs = nstruc0
        niter = 0
        err0 = 0.
        derr = 1000000.
        print_format('Determining optimum number of parameters ', level=2)
        if show_plot:
            import matplotlib.pyplot as pl
            pl.ion()
            pl.xlabel("Number of structures")
            pl.ylabel("Test error")
        while derr > tol or niter < 3:
            # optimize, all samples should be included 
            new_pars = self.fit(nsamples=nsamples
                                , strucs_sampling=strucs_sampling, nstrucs=nstrucs
                                , success_criteria={}, global_opt=global_opt)
            # get test error
            testerr = self.test(new_pars, strucs_sampling=strucs_sampling
                                , nstrucs=self._nstrucmax, success_criteria={}
                                , update_models=False)
            testerr = np.array(testerr)
            sigma = np.sqrt(np.average((testerr - np.average(testerr)) ** 2))
            derr = np.abs(np.average(testerr) - err0)
            err0 = np.average(testerr)
            print_format("""nstrucs: %7d valid samples: %7d mean error: %8.4f \
                sigma: %8.4f  d_error: %8.4f""" % (nstrucs, len(testerr)
                                                                            , err0, sigma, derr), level=3)
            if show_plot:
                pl.errorbar(nstrucs, err0, yerr=sigma, fmt='bs', ecolor='b')
                # pl.show()
            # increment number of strucs in fit 
            nstrucincr = nstruc0 * (1.0 - mexp(-5 * derr))
            nstrucs += int(mceil(nstrucincr))
            niter += 1
        self._nstrucs_fit = nstrucs

    def get_nstrucs_fit(self):
        if self._nstrucs_fit is None:
            self.gen_optimum_nstrucs()
        return self._nstrucs_fit

    def get_nsamples(self):
        if self._nsamples is None:
            self._nsamples = 20 * self.get_fit_variables()
        return self._nsamples

    def _nullify_var(self):
        variables = self.get_fit_variables()
        var = [variables()[i].copy() for i in range(len(variables))]
        for i in range(len(var)):
            for key, val in list(var[i].items()):
                if key in ['atom', 'bond']:
                    continue
                var[i][key] = [False] * len(val)
        return var

    def _free_var_pair(self, var):
        fit_var = self._nullify_var()
        for i in range(2):
            a0 = var[i][0]
            if isinstance(a0, str):
                a0 = [a0]
            a0.sort()
            assigned = False
            for j in range(len(fit_var)):
                a1, a2 = [], []
                if 'atom' in fit_var[j]:
                    a1 = fit_var[j]['atom']
                    if isinstance(a1, str):
                        a1 = [a1]
                if 'bond' in fit_var[j]:
                    a2 = fit_var[j]['bond']
                    if isinstance(a2, str):
                        a2 = [a2]
                a1.sort()
                a2.sort()
                if a0 == a1 or a0 == a2:
                    assigned = True
                    break
            if not assigned:
                raise ValueError('Cannot find atom/bond %s in variables' % a0)
            if var[i][1] not in fit_var[j]:
                raise ValueError('Variables do not have key %s' % var[i][1])
            fit_var[j][var[i][1]][var[i][2]] = True
        return fit_var

    def get_variables_split(self):
        if self._variables_split is not None:
            return self._variables_split
        pvar = []
        variables = self.get_fit_variables()
        for i in range(len(variables)):
            if 'bond' in variables[i]:
                ab = variables[i]['bond']
            if 'atom' in variables[i]:
                ab = variables[i]['atom']
            for key, val in list(variables[i].items()):
                if key in ['bond', 'atom']:
                    continue
                for j in range(len(val)):
                    if val[j]:
                        pvar.append((ab, key, j))
        self._variables_split = list(pvar)
        return self._variables_split

    def _fit_correlation_pair(self, apar):
        # TODO: generalize this to fit to other functions
        tol = abs(np.average(apar[1]) * 0.1)
        poly = polynomial(constraints=[True, False, True], parameters=[1, 1])
        cons = constant()
        func = sum_funcs(functions=[cons, poly.copy(), poly.copy()])
        # p1 = polynomial(constraints=[True,False,False]
        # ,parameters=[1],numbers=[1,0,2])
        # p2 = polynomial(constraints=[True,False,False]
        # ,parameters=[1],numbers=[1,0,2])
        # p3 = polynomial(constraints=[True,False,False]
        # ,parameters=[1],numbers=[1,0,2])
        # p4 = polynomial(constraints=[True,False,False]
        # ,parameters=[1],numbers=[1,0,2])
        # func = sum_funcs(functions=[p1,p2,p3,p4])
        # func = poly.copy()
        popt = fit(apar[0], apar[1], func, minimizer='leastsq', tol=None)
        popt = [round(p, 4) for p in popt]
        func.set_parameters(popt)
        rms = np.sqrt(np.average((func(apar[0]) - apar[1]) ** 2))
        # x = np.linspace(np.amin(apar[0]),np.amax(apar[0]),100)
        # import matplotlib.pyplot as pl
        # pl.clf()
        # pl.plot(apar[0],apar[1],'o')
        # pl.plot(x,func(x),'-')
        # pl.show()
        # raise
        # print rms, tol
        out = None
        if rms <= tol:
            out = func
        return out

    def _set_newpar_modelsbx(self, var, fstr):
        refmodel = self.get_refmodel()
        bbx = refmodel.bondsbx
        ab0 = var[0][0]
        key0 = var[0][1]
        if len(ab0) == 2:
            par = None
            for i in range(len(bbx)):
                b = bbx[i].bond
                b.sort()
                if b == ab0:
                    bpar = bbx[i].get_bondspar()
                    bpar.update(bbx[i].get_overlappar())
                    bpar.update(bbx[i].get_repetal())
                    bi = i
                    break
            fpar = bpar[key0]
            if fpar is None:
                raise ValueError('No match found!')
        else:
            raise NotImplementedError("""Cannot only apply constraints for \
                                         bonds parameters.""")

        num = list(fpar._numbers)
        if isinstance(num[var[0][2]], str) or isinstance(num[var[1][2]], str):
            return
        num[var[1][2]] = fstr
        con = fpar.get_constraints()
        con[var[1][2]] = False
        par = []
        for i in range(len(con)):
            if con[i]:
                par.append(num[i])
        fpar.set(numbers=num, constraints=con, parameters=par)
        bpar[key0] = fpar
        bbx[bi].set_bondspar(bpar)
        refmodel.bondsbx = bbx

        for i in range(len(self._variables)):
            if 'bond' in self._variables[i]:
                ab = self._variables[i]['bond']
            ab.sort()
            if ab == ab0:
                self._variables[i][key0][var[1][2]] = False
                break
        self._refmodel = refmodel
        self._refmodel.write()
        # update nvar
        self._nvar = self.get_nvar(refresh=True)

    def _set_newpar(self, var, fstr):
        if isinstance(self.get_refmodel(), modelsbx):
            self._set_newpar_modelsbx(var, fstr)
        else:
            raise NotImplementedError('No options for %s' % type(
                self.get_refmodel()))

    def _print_correlation(self, lvari, lvarj, fstr):
        if len(lvari[0]) == 1:
            print_format('atom: %2s      key: %s  index: %5d %5d  %s' % (
                lvari[0], lvarj[1], lvari[2], lvarj[2], fstr), level=3)
        else:
            print_format('bond: %2s-%2s  key: %s  index: %5d %5d  %s' % (
                lvari[0][0], lvari[0][1], lvarj[1], lvari[2], lvarj[2], fstr)
                         , level=3)

    def get_correlation_data(self):
        return self._correlation_data

    def get_correlation_funcs(self):
        return self._correlation_funcs

    def _fit_correlation(self):
        self._correlation_funcs = []
        apar = self.get_correlation_data()
        mins = 10
        if len(apar[0]) < mins:
            raise ValueError('Needs at least %d samples to continue.' % mins)
        self.get_refmodel().write()
        lvar = self.get_variables_split()
        assert (len(apar) == len(lvar))
        print_format('Determining correlations.', level=2)
        for i in range(len(lvar)):
            for j in range(len(lvar)):
                if i == j:
                    continue
                # can only correlate same function on same pair
                if lvar[i][0] != lvar[j][0]:
                    continue
                if lvar[i][1] != lvar[j][1]:
                    continue
                try:
                    func = self._fit_correlation_pair([apar[i], apar[j]])
                except:
                    raise
                    func = None
                self._correlation_funcs.append((lvar[i], lvar[j], func))
                if func is not None:
                    fstr = func.print_function()
                    fstr = fstr.replace("X", "numbers[%d]" % (lvar[i][2]))
                    self._print_correlation(lvar[i], lvar[j], fstr)
                    # set val
                    self._set_newpar((lvar[i], lvar[j]), fstr)

    def get_correlation(self, var1, var2, show_plot=False):
        if var1[0] != var2[0] or var1[1] != var2[1]:
            print_format('Can only find correlations within similar function.')
            return
        if self.get_correlation_data() is None:
            self.gen_correlation()
        if self.get_correlation_funcs() is None:
            self._fit_correlation()
        func = None
        cf = self.get_correlation_funcs()
        for i in range(len(cf)):
            v1 = cf[i][0]
            v2 = cf[i][1]
            if var1 == v1 and var2 == v2:
                func = cf[i][-1]
                break
        if func is not None and show_plot:
            import matplotlib.pyplot as pl
            vi = self.get_variables_split().index(var1)
            vj = self.get_variables_split().index(var2)
            cd = self.get_correlation_data()
            data = np.array([cd[vi], cd[vj]])
            x = np.linspace(np.amin(data[0]), np.amax(data[0]), 100)
            pl.clf()
            pl.xlabel('%s-%d' % (var1[1], var1[2]))
            pl.ylabel('%s-%d' % (var2[1], var2[2]))
            pl.plot(data[0], data[1], 'o')
            pl.plot(x, func(x), '-')
            pl.show()
        return func

    def gen_correlation(self, nsamples=50, nstrucs='default'
                        , strucs_sampling='structure_map', model_sampling='random'
                        , success_criteria='default', dump=True):
        # var = self._free_var_pair((var1,var2))
        # just fit all variables and derive correlation 
        var = self._variables
        new_pars = self.fit(nsamples=nsamples, strucs_sampling=strucs_sampling
                            , model_sampling=model_sampling, nstrucs=nstrucs
                            , success_criteria=success_criteria)
        self._correlation_data = np.array(new_pars).T
        self._fit_correlation()
        if dump:
            # dump to file
            ct = cattxt(filename='correlation.dat')
            for i in range(len(new_pars)):
                ct.add(new_pars[i])
            ct.write()
            self.get_refmodel().write(filename='models_wcorrelation.dat')

    def run(self):
        """
        Main process
        """
        # identity correlation in variables to reduce number 
        self.gen_correlation()
        new_par = self.fit(nsamples=1, strucs_sampling='structure_map'
                           , model_sampling='optimized'
                           , nstrucs=self.get_nstrucs_fit()
                           , success_criteria='default', global_opt=True)
        terr = self.test(newpar)
        #  
        return
