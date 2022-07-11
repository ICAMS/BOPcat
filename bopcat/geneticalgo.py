#!/usr/bin/env python

# Definition of the Genetic algorithm

# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import numpy as np
from copy import deepcopy
from .process_management import Process_catobjectives, Process_catobjective
from .process_management import Process_catkernels, Process_catkernel


class Individual:
    """
    Defines an individual in a gene pool.

    It has trait and corresponding error.

    :Parameters:
	
	- *trait*: list

	    list of parameters specifying the individual character
              
     - *bounds*: list 
     
         list of tuples each corresponding to max and min value on parameters

     - *mutation*: float 
     
         probablity that it will mutate, i.e. a random entry trait changes

    """

    def __init__(self, **kwargs):
        self.trait = None
        self.bounds = None
        self.mutation = 0.01
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key.lower() == 'trait':
                self.trait = np.array(val)
            elif key.lower() == 'bounds':
                self.bounds = val
            elif key.lower() == 'mutation':
                self.mutation = val
        if self.bounds is None:
            self.bounds = [(0, 5.) for i in range(len(self.trait))]

    def set_error(self, err):
        self.error = err

    def mutate(self):
        if self.mutation > np.random.random():
            index = np.random.randint(len(self.trait))
            self.trait[index] *= np.random.uniform(self.bounds[index][0]
                                                   , self.bounds[index][1])

    def copy(self):
        return deepcopy(self)


class Geneticalgo:
    """
    Implementation of the genetic algorithm for BOPcat.

    :Parameters:
	
	- *objective*: callable object/function 

	    returns list of error corresponding to list of parameter sets

	- *x0*: list
   	    
	    initial starting position in parameter space

     - *size*: int
     
          number of individuals in population

	- *maxiter*: int

         maximum iterations to terminate evolution        
         
     - *retention*: float
     
         fraction of population to retain in next generation based on fitness 
         
     - *random_selection*: float
     
         probability that not ideal individual is selected in next generation 
         
     - *mutation*: float
     
         probability that individual mutates
         
     - *bounds*: list 
     
         list of tuples each corresponding to max and min value on parameters

     - *queue*: str/instance of Queue 
     
         mode to run processes

     - *directives*: list 
     
         if queue is string, directives to pass to Queue

     - *nproc*: int 
     
         if queue is string, number of cores to pass to Queue

     - *catprocess*: instance of CATProcess 
     
         will evaluate errors from catprocess.test

     - *catkernel*: instance of CATKernel 
     
         execute catkernel to do local optimization for each particle
    """

    def __init__(self, **kwargs):
        self.objective = None
        self.x0 = None
        self.size = 100
        self.retention = 0.2
        self.random_selection = 0.05
        self.mutation = 0.01
        self.bounds = None
        self.maxiter = 100
        self.queue = 'serial'
        self.directives = ['source ~/.bashrc']
        self.nproc = 1
        self.catprocess = None
        self.catkernel = None
        self.xerror = -1
        self.x = None
        self.niter = 0
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key.lower() == 'objective':
                self.objective = val
            elif key.lower() == 'x0':
                self.x0 = val
            elif key.lower() == 'size':
                self.size = val
            elif key.lower() == 'retention':
                self.retention = val
            elif key.lower() == 'random_selection':
                self.random_selection = val
            elif key.lower() == 'mutation':
                self.mutation = val
            elif key.lower() == 'bounds':
                self.bounds = val
            elif key.lower() == 'size':
                self.size = val
            elif key.lower() == 'maxiter':
                self.maxiter = val
            elif key.lower() == 'queue':
                self.queue = val
            elif key.lower() == 'directives':
                self.directives = val
            elif key.lower() == 'nproc':
                self.nproc = val
            elif key.lower() == 'catkernel':
                self.catkernel = val
            else:
                raise ValueError('No option for key %s' % key)
        if self.bounds is None:
            self.bounds = [(0, 5.) for i in range(len(self.x0))]

    def gen_parents(self):
        parents = []
        # add best performing individuals first
        errors = np.array([ind.error for ind in self._population])
        sorted_error = list(np.argsort(errors))
        retain = list(range(0, int(self.retention * self.size)))
        for i in retain:
            parents.append(self._population[sorted_error[i]])
        # randomly add individuals from rest of population
        for i in range(len(retain), self.size):
            if self.random_selection > np.random.random():
                parents.append(self._population[sorted_error[i]])
                # mutate individuals
        for i in range(len(parents)):
            parents[i].mutate()
        return parents

    def evolve(self):
        new_population = self.gen_parents()
        # extend population to current size by mating
        while len(new_population) < self.size:
            father = np.random.randint(len(new_population))
            mother = np.random.randint(len(new_population))
            if father != mother:
                father = new_population[father]
                mother = new_population[mother]
                child = father.copy()
                # randomize which trait to get from mother/father
                index = np.random.permutation(len(child.trait))
                for i in range(int(len(index) / 2)):
                    child.trait[index[i]] = father.trait[index[i]]
                for i in range(int(len(index) / 2), len(index)):
                    child.trait[index[i]] = mother.trait[index[i]]
                new_population.append(child)
        self._population = new_population

    def _randomize(self):
        trait = []
        for i in range(len(self.x0)):
            trait.append(self.x0[i] * np.random.uniform(self.bounds[i][0]
                                                        , self.bounds[i][1]))
        return trait

    def _optimize_local(self):
        procs = []
        for p in self._population:
            ckern = self.catkernel.copy()
            ckern.set(init_param_weights=p.trait)
            procs.append(Process_catkernel(catkernel=ckern, queue=self.queue
                                           , cores=self.nproc, directives=self.directives))
        proc = Process_catkernels(procs=procs)
        proc.run()
        opt_param = proc.get_results(wait=True)['optimized_parameters']
        for i in range(len(self._population)):
            self._population[i].trait = \
                np.array(opt_param[i]) / np.array(self.objective.p0)
        proc.clean()

    def optimize(self):
        # can only propagate population if there are parents
        if int(self.size * self.retention) < 2:
            raise ValueError("Increase population size or increase retention.")
        # generate initial population
        self._population = []
        for i in range(self.size):
            # randomize initial trait
            trait = self._randomize()
            # trait = list(self.x0)
            self._population.append(Individual(trait=trait, bounds=self.bounds
                                               , mutation=self.mutation))
        derr = float('inf')
        err0 = float('inf')
        while self.niter < self.maxiter:
            if self.catkernel is not None:
                # search local minimum about current position
                self._optimize_local()
            traits = [ind.trait for ind in self._population]
            if self.queue != 'serial':
                # get results from parallel process
                procs = [Process_catobjective(catobjective=self.objective
                                              , x0=traits[i], queue=self.queue, cores=self.nproc
                                              , directives=self.directives) for i in range(len(traits))]
                proc = Process_catobjectives(procs=procs)
                proc.run()
                errors = proc.get_results(wait=True)['function_value']
                proc.clean()
                if self.objective.verbose:
                    for i in range(len(errors)):
                        self.objective.rms.append(errors[i])
                        self.objective.par.append(traits[i])
                        self.objective.write_to_screen()
            elif self.objective is not None:
                # evaluate error from objective
                obj = self.objective
                # verbose = self.objective.verbose
                # obj.verbose=False
                errors = [obj(t) for t in traits]
            elif self.catprocess is not None:
                errors = self.catprocess.test(traits)
            for i in range(len(self._population)):
                # set error to individual
                self._population[i].set_error(errors[i])
                if self._population[i].error < self.xerror \
                        or self.xerror == -1:
                    self.x = np.array(self._population[i].trait)
                    self.xerror = self._population[i].error
            # evolve population
            self.evolve()
            self.niter += 1
            derr = abs(err0 - self.xerror)
            err0 = self.xerror
        return self
