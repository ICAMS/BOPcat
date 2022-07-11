# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

from multiprocessing import Process
import time


class PBOPProc(Process):
    """
    sub-class for Process for parallelizing BOPfox processes in BOPcat
    """

    def __init__(self, target, pipe, atoms, calcs, kwargs):
        # Process.__init__(self,target=target,args=(atoms,pipes),kwargs=kwargs)
        Process.__init__(self)
        self._pipe = pipe
        self._atoms = atoms
        self._target = target
        self._calcs = calcs
        self._kwargs = kwargs

    def run(self):
        out = []
        s = time.time()
        for i in range(len(self._atoms)):
            if self._atoms[i] is None:
                continue
            self._atoms[i].set_calculator(self._calcs[i])
            out.append(self._target(self._atoms[i], self._kwargs))
            # out[-1]._calc = None
        dt = time.time() - s
        self._pipe[1].send((out, dt))
        self._pipe[1].close()


class _PBOPProc(Process):
    """
    sub-class for Process for parallelizing BOPfox processes in BOPcat
    """

    def __init__(self, target, pipe, atoms, calcs, kwargs):
        # Process.__init__(self,target=target,args=(atoms,pipes),kwargs=kwargs)
        Process.__init__(self)
        self._pipe = pipe
        self._atoms = atoms
        self._target = target
        self._calcs = calcs
        self._kwargs = kwargs
        self._status = 'free'

    def run(self):
        out = []
        s = time.time()
        for i in range(len(self._atoms)):
            if self._atoms[i] is None:
                continue
            self._atoms[i].set_calculator(self._calcs[i])
            out.append(self._target(self._atoms[i], self._kwargs))
            # out[-1]._calc = None
        self._pipe[1].send(out)
        self._pipe[1].close()


if __name__ == "__main__":
    from .catcalc import CATCalc
    from ase.calculators import bopfox
    from ase.lattice import bulk
    import time
    import numpy as np
    from .bopmodel import read_modelsbx
    from .calc_bopfox import initialize_bopfox
    from multiprocessing import Queue, Pool, cpu_count, JoinableQueue, Pipe


    def run_bopfox(atoms, calcs, pipes):
        s = time.time()
        for i in range(len(pipes)):
            if atoms[i] is None:
                continue
            atoms[i].set_calculator(calcs[i])
            pipes[i][0].send(atoms[i].get_potential_energy())
            # print atoms[i].get_potential_energy()
        print(('run time:', time.time() - s, len([at for at in atoms if at is not None])))


    def _parcat(atoms, Nproc='default'):
        if Nproc == 'default':
            Nproc = cpu_count()

        iq = JoinableQueue(len(atoms))
        oq = Queue(len(atoms))
        pipes = [Pipe() for N in range(Nproc)]
        procs = [PBOPProc(iq, oq, p, CATCalc().calculate) for (p, c) in pipes]

        for p in procs:
            p.start()

        modelsbx = read_modelsbx(filename='models.bx', model='Madsen-2011')[0]
        calcs = [initialize_bopfox({'task': 'energy'}, modelsbx, atom) for atom in atoms]

        for i in range(len(atoms)):
            atoms[i].info['qID'] = i
            atoms[i].info['required_property'] = 'energy'
            atoms[i].set_calculator(calcs[i])
            iq.put({'atom': atoms[i], 'kwargs': {}})

        for i in range(Nproc):
            iq.put(None)

        res = {}
        iq.join()

        while len(res) < len(atoms):
            atom = oq.get()
            key = atom.info['qID']
            ene = atom.get_potential_energy()
            res[key] = ene
        return res


    def parcat(atoms):
        modelsbx = read_modelsbx(filename='models.bx', model='Madsen-2011')[0]
        # calcs = [initialize_bopfox({'task':'energy'},modelsbx,atom) for atom in atoms]
        calcs = [bopfox.BOPfox(modelsbx=modelsbx, task='energy') for i in range(len(atoms))]
        # [atom.set_calculator(calc) for calc in calcs]

        Nproc = 4
        div = int(np.ceil(len(atoms) / float(Nproc)))

        pipes = [Pipe() for N in range(len(atoms))]

        procs = []
        for N in range(Nproc):
            todo = list(range(N * div, (N + 1) * div))
            todo_atoms = [None] * len(atoms)
            todo_calcs = [None] * len(atoms)
            for i in todo:
                if i < len(atoms):
                    todo_atoms[i] = atoms[i]
                    todo_calcs[i] = calcs[i]
            procs.append(Process(target=run_bopfox, args=(todo_atoms, todo_calcs, pipes)))

        for p in procs:
            p.start()

        for p in procs:
            p.join()

        # out = []
        out = [op.recv() for (ip, op) in pipes]
        # out = [at.get_potential_energy() for at in atoms]
        # print out
        return out


    def _parcat(atoms):
        for i in range(len(atoms)):
            atoms[i].info['qID'] = i

        modelsbx = read_modelsbx(filename='models.bx', model='Madsen-2011')[0]
        calcs = [initialize_bopfox({'task': 'energy'}, modelsbx, atom) for atom in atoms]

        for i in range(len(atoms)):
            atoms[i].set_calculator(calcs[i])

        pool = Pool(5)
        pool.map(run_bopfox, atoms)
        pool.close()
        pool.join()
        res = {}
        for i in range(len(atoms)):
            key = atoms[i].info['qID']
            ene = atoms[i].get_potential_energy()
            res[key] = ene
        return res


    def serial(atoms):
        res = {}
        modelsbx = read_modelsbx(filename='models.bx', model='Madsen-2011')[0]
        calcs = [initialize_bopfox({'task': 'energy'}, modelsbx, atom) for atom in atoms]

        for i in range(len(atoms)):
            atoms[i].set_calculator(calcs[i])
            ene = atoms[i].get_potential_energy()
            res[i] = ene
        print(res)
        return res


    def build():
        aarr = list(range(1, 80))
        atoms = []
        Fe = bulk('Fe')
        for i in aarr:
            # atom = Fe.repeat((i%8+1,i%4+1,i%2+1))
            atom = Fe.copy()
            atom.info['strucname'] = '%s' % i
            atoms.append(atom)
        return atoms


    atoms = build()
    start = time.time()
    res = parcat(atoms)
    end = time.time()
    print(('time: ', end - start))

    atoms = build()
    start = time.time()
    res = serial(atoms)
    end = time.time()
    print(('time: ', end - start))
