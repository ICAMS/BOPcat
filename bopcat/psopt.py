#!/usr/bin/env python

# Implementation of the Particle Swarm optimizer

# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import random
import numpy as np
from .process_management import Process_catobjectives, Process_catobjective
from .process_management import Process_catkernels, Process_catkernel


###########################################################################
class Particle:
    """
    Defines a particle in a swarm.

    It has both a position and velocity in parameter space.

    :Parameters:
	
	- *position*: list

	    list of parameters specifying its position in parameter space
     
     - *alpha*: float
     
         mixing parameter for new velocity
         
     - *c1*: float
     
         acceleration coefficient for particle
         
     - *c2*: float
     
         acceleration coefficient for swarm
         
     - *bounds*: list 
     
         list of tuples each corresponding to max and min value on parameters
    """

    def __init__(self, **kwargs):
        self.position = None
        self.alpha = 0.5
        self.c1 = 1.0
        self.c2 = 2.0
        self.bounds = None
        # list of velocity components in parameter space
        self.velocity = None
        # error corresponding to current position
        self.error = -1
        # best position in parameter space (local attractor) 
        self.position_la = None
        # error corresponding error to local attractor
        self.error_la = -1
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key.lower() == 'x0':
                self.position = np.array(val)
                self.velocity = np.array(
                    [random.uniform(-1, 1) for i in range(len(self.position))])
            elif key.lower() == 'alpha':
                self.alpha = val
            elif key.lower() == 'bounds':
                self.bounds = val
            elif key.lower() == 'c1':
                self.c1 = val
            elif key.lower() == 'c2':
                self.c2 = val
            else:
                raise ValueError("No options for key " % key)
        if self.position is None:
            raise ValueError("Initial positions should be set.")
        if self.bounds is None:
            self.bounds = [(0, 5.) for i in range(len(self.position))]

    def move(self, x_g):
        r1 = random.random()
        r2 = random.random()
        v1 = self.c1 * r1 * (self.position_la - self.position)
        v2 = self.c2 * r2 * (x_g - self.position)
        self.velocity = self.alpha * self.velocity + v1 + v2
        self.position += self.velocity
        # print "MOVE", x_g, self.c1,r1,self.position_la, self.position
        # apply bounds
        for i in range(len(self.position)):
            if self.position[i] < self.bounds[i][0]:
                self.position[i] = self.bounds[i][0]
            if self.position[i] > self.bounds[i][1]:
                self.position[i] = self.bounds[i][1]

    def set_error(self, error):
        self.error = error
        if self.error < self.error_la or self.error_la == -1:
            self.position_la = list(self.position)
            self.error_la = self.error

    # def


class PSopt:
    """
    Implementation of the particle swarm optimizer for BOPcat.

    :Parameters:
	
	- *objective*: callable object/function 

	    returns list of error corresponding to list of parameter sets

	- *x0*: list
   	    
	    initial starting position in parameter space

     - *nparticle*: int
     
          number of particles in swarm

	- *maxiter*: int

         maximum iterations to terminate search        
         
     - *alpha*: float
     
         mixing parameter for new velocity
         
     - *c1*: float
     
         acceleration coefficient for particle
         
     - *c2*: float
     
         acceleration coefficient for swarm
         
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
        self.nparticles = 50
        self.maxiter = 100
        self.xtol = 0.0001
        self.alpha = 0.5
        self.c1 = 1.0
        self.c2 = 2.0
        self.bounds = None
        self.queue = 'serial'
        self.directives = ['source ~/.bashrc']
        self.nproc = 1
        # self.pobjective = False
        self.catprocess = None
        self.catkernel = None
        self.x = None  # optimized parameters
        self.xerror = -1
        self.niter = 0
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key.lower() == 'objective':
                self.objective = val
            elif key.lower() == 'x0':
                self.x0 = val
            elif key.lower() == 'bounds':
                self.bounds = val
            elif key.lower() == 'nparticles':
                self.nparticles = val
            elif key.lower() == 'maxiter':
                self.maxiter = val
            elif key.lower() == 'xtol':
                self.xtol = val
            elif key.lower() == 'alpha':
                self.alpha = val
            elif key.lower() == 'c1':
                self.c1 = val
            elif key.lower() == 'c2':
                self.c2 = val
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

    def _optimize_local(self):
        procs = []
        for p in self._particles:
            ckern = self.catkernel.copy()
            ckern.set(init_param_weights=p.position)
            procs.append(Process_catkernel(catkernel=ckern, queue=self.queue
                                           , cores=self.nproc, directives=self.directives))
        proc = Process_catkernels(procs=procs)
        proc.run()
        opt_param = proc.get_results(wait=True)['optimized_parameters']
        for i in range(len(self._particles)):
            self._particles[i].position = \
                np.array(opt_param[i]) / np.array(self.objective.p0)
        proc.clean()

    def optimize(self):
        # create swarm
        particles = []
        # procs = []
        for i in range(self.nparticles):
            # procs.append(Process_catobjective(catobjective=self.objective
            #                           ,x0=self.x0
            #                           ,queue=self.queue,cores=self.nproc))
            particles.append(Particle(x0=self.x0, bounds=self.bounds
                                      , alpha=self.alpha
                                      , c1=self.c1, c2=self.c2))
        self._particles = particles
        derr = float('inf')
        err0 = float('inf')
        # optimization loop
        while self.niter < self.maxiter:
            if self.catkernel is not None:
                # search local minimum about current position
                self._optimize_local()
            # get current positions of particles in swarm
            positions = [p.position for p in particles]
            # print 'positions',positions
            if self.queue != 'serial':
                # get results from parallel process
                procs = [Process_catobjective(catobjective=self.objective
                                              , x0=positions[i], queue=self.queue, cores=self.nproc
                                              , directives=self.directives) \
                         for i in range(len(positions))]
                proc = Process_catobjectives(procs=procs)
                proc.run()
                errors = proc.get_results(wait=True)['function_value']
                proc.clean()
                if self.objective.verbose:
                    for i in range(len(errors)):
                        self.objective.rms.append(errors[i])
                        self.objective.par.append(positions[i])
                        self.objective.write_to_screen()
            elif self.objective is not None:
                # evaluate error from objective
                obj = self.objective
                # verbose = self.objective.verbose
                # obj.verbose=False
                errors = [obj(p) for p in positions]
            elif self.catprocess is not None:
                errors = self.catprocess.test(positions)
            # identify global attractor
            for i in range(len(particles)):
                # set error to particle
                particles[i].set_error(errors[i])
                if particles[i].error < self.xerror or self.xerror == -1:
                    self.x = np.array(particles[i].position)
                    self.xerror = particles[i].error
            self.niter += 1
            # move particles
            for p in particles:
                p.move(self.x)
            derr = abs(err0 - self.xerror)
            err0 = self.xerror
        return self
