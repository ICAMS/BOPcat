#!/usr/bin/env python

# Definition of the CATCalc object

# This module is part of the BOPcat package

from .calc_bopfox import calc_ebs_bopfox, calc_efs_bopfox
from .calc_bopfox import initialize_bopfox
from .output import print_format
from multiprocessing import cpu_count, Pipe, Process
from .parallel import PBOPProc
import time
import sys
from .variables import homedirectory


# import mpi4py
###########################################################################
class CATCalc:
    """
    Defines a set of calculators to determine required properties.

    The calculators are stored as list of instances of the calculator.

    :Parameters:
	
	- *calculator*: str

	    name of the calculator

	- *calculator_settings*: dict
   	    
	    dictionary of calculator-specific parameters

	- *nproc*: int

	    number of parallel processes

	    ``None``: will not parallelize

            ``default``: number of cpu * 2

	- *controls*: instance of CATControls   
        
            CATControls object to initialize parameters      

        - *parallel*: string
            
            serial, multiprocessing, mpi 
    """

    def __init__(self, **kwargs):
        # self._init_msg()
        self.calculator = 'bopfox'
        self.calculator_settings = None
        self.nproc = 1
        self.controls = None
        self.docalculate = True
        self.structures = None
        self._atoms = None
        self._model = None
        self._calcs = None
        self.results = None
        self.parallel = 'serial'
        self.mpifile = None
        self.ini_magmoms = {}
        self.set(**kwargs)

    def _init_msg(self):
        print_format('Generating calculator', level=1)

    def set(self, **kwargs):
        if 'controls' in kwargs:
            self.controls = kwargs['controls']
            if self.controls is not None:
                self._unfold_controls()
        for key in kwargs:
            if key.lower() == 'calculator':
                self.calculator = kwargs[key]
            elif key.lower() == 'calculator_settings':
                self.calculator_settings = kwargs[key]
            elif key.lower() == 'model':
                self._model = kwargs[key]
            elif key.lower() == 'atoms':
                self.set_atoms(kwargs[key])
            elif key.lower() == 'parallel':
                self.parallel = kwargs[key]
            elif key.lower() == 'ini_magmoms':
                self.set_ini_magmoms(kwargs[key])
            elif key.lower() == 'controls':
                pass
            else:
                raise ValueError('Unrecognized key %s' % key)
        self.docalculate = True
        if self.nproc == 'default':
            self.nproc = cpu_count() * 2
        if self.nproc is None:
            self.nproc = 1

    def _unfold_controls(self):
        self.calculator = self.controls.calculator
        self.calculator_settings = self.controls.calculator_settings
        self.nproc = self.controls.calculator_nproc
        self.ini_magmoms = self.controls.data_ini_magmoms
        self.parallel = self.controls.calculator_parallel

    @staticmethod
    def calc_ebs(atom, kwargs):
        """
   	Returns the Atoms object with the calculated eigenvalues.

        :Parameters:
        
            - *atom*: instance of ASE Atoms object
        
                structure to calculate

            - *kwargs*: dict
            
                directives for calculator
	"""
        calculator = 'bopfox'
        if 'calculator' in kwargs:
            calculator = kwargs['calculator']
        if calculator.lower() == "bopfox":
            shift_Fermi = False
            for key, val in list(kwargs.items()):
                if key.lower() == "shift_fermi":
                    shift_Fermi = val
            try:
                k_points = atom.info['k_points'][0]
                coord_k = atom.info['coord_k']
            except:
                print_format("Needs k-point data to continue", level=3)
                return None
            cartesian = True
            if coord_k[0].lower() == 'd':
                cartesian = False
            out = calc_ebs_bopfox(atom, atom.get_calculator(), k_points=k_points
                                  , cartesian=cartesian
                                  , shift_Fermi=shift_Fermi)
        else:
            raise NotImplementedError("No options for %s" \
                                      % calculator)
        return out

    @staticmethod
    def calc_efs(atom, kwargs):
        """
        Returns the Atoms object with the calculated energy, forces and 
        stresses.

        :Parameters:
        
            - *atom*: instance of ASE Atoms object
        
                structure to calculate

            - *kwargs*: dict
            
                directives for calculator
	"""
        calculator = 'bopfox'
        if 'calculator' in kwargs:
            calculator = kwargs['calculator']
        if calculator.lower() == "bopfox":
            contribution = 'binding'
            atom_index = None
            required_property = atom.info['required_property']
            for key, val in list(kwargs.items()):
                if key.lower() == 'contribution':
                    contribution = val
                elif key.lower() == 'atom_index':
                    atom_index = val
                elif key.lower() == 'required_property':
                    required_property = val
            if atom_index is None:
                if required_property in ['energy', 'stress']:
                    atom_index = 'total'
                elif required_property in ['forces', 'stresses']:
                    atom_index = 'all'
            out = calc_efs_bopfox(atom, atom.get_calculator()
                                  , required_property=required_property
                                  , contribution=contribution
                                  , atom_index=atom_index)
        else:
            raise NotImplementedError("No options for %s" \
                                      % calculator)
        return out

    @staticmethod
    def calc_def_ene(atom, kwargs):
        """
        Returns the Atoms object with the calculated defect energy.
        
        The reference bulk structures are in atom.info['reference_atoms']. 

        :Parameters:
        
            - *atom*: instance of ASE Atoms object
        
                structure to calculate

            - *kwargs*: dict
            
                directives for calculator
	"""
        bulk_ene = {}
        bulk_atoms = atom.info['reference_atoms']
        calculator = 'bopfox'
        if 'calculator' in kwargs:
            calculator = kwargs['calculator']
        if calculator.lower() == 'bopfox':
            contribution = 'binding'
            atom_index = 'total'
            for key, val in list(kwargs.items()):
                if key.lower() == 'contribution':
                    contribution = val
                elif key.lower() == 'atom_index':
                    atom_index = val
                elif key.lower() == 'required_property':
                    required_property = val
            out = calc_efs_bopfox(atom, atom.get_calculator()
                                  , required_property='energy'
                                  , contribution=contribution
                                  , atom_index=atom_index)
            def_ene = out.get_potential_energy()
            for i in range(len(bulk_atoms)):
                out = calc_efs_bopfox(bulk_atoms[i]
                                      , bulk_atoms[i].get_calculator()
                                      , required_property='energy'
                                      , contribution=contribution
                                      , atom_index=atom_index)
                ene = out.get_potential_energy() / len(bulk_atoms[i])
                # reference atoms are elemental               
                bulk_ene[bulk_atoms[i].get_chemical_symbols()[0]] = ene
        else:
            raise NotImplementedError("No options for %s" \
                                      % calculator)

        sym = atom.get_chemical_symbols()
        for s in sym:
            if s not in bulk_ene:
                raise ValueError("No reference energy for %s" % s)
            def_ene -= bulk_ene[s]
        out = atom.copy()
        out.info[atom.info['required_property']] = def_ene
        return out

    @staticmethod
    def calculate(atom, kwargs):
        """
        Returns the Atoms object with the calculated property.
        
        The required property is defined by atom.info['required_property']

        Calls :func:`calc_ebs`, :func:`calc_efs`, :func:`calc_def_ene`

        :Parameters:
        
            - *atom*: instance of ASE Atoms object
        
                structure to calculate

            - *kwargs*: dict
            
                directives for calculator
	"""
        required_property = atom.info['required_property']
        if not isinstance(required_property, str):
            required_property = required_property[-1]
        if required_property.lower() == "eigenvalues":
            out = CATCalc.calc_ebs(atom, kwargs)
        elif required_property.lower() in ["energy", "forces", "stress"
            , "stresses"]:
            out = CATCalc.calc_efs(atom, kwargs)
        elif required_property.lower() in ["vacancy_energy"]:
            out = CATCalc.calc_def_ene(atom, kwargs)
        else:
            print_format("Cannot calculate %s" % required_property, level=3)
            out = None
        return out

    def clean(self):
        for i in range(len(self._calcs)):
            self._calcs[i].clean()

    def clear_atom(self, atom):
        delete = ['orbital_character']
        info = atom.info
        for i in delete:
            if i in info:
                info.pop(i)
        atom.info = info
        return atom

    def _is_same(self, at1, at2):
        out = True
        if at1 != at2:
            out = False
        # strangely comparison of ASE atoms is not enough
        if at1.info['strucname'] != at2.info['strucname']:
            out = False
        # only e,f,s can be calculated simulataneously
        if at1.info['required_property'] not in \
                ['energy', 'forces', 'stress', 'stresses']:
            out = False
        if at2.info['required_property'] not in \
                ['energy', 'forces', 'stress', 'stresses']:
            out = False
        return out

    def pack_atoms(self):
        """
        Pack all properties for the same structure so do only one calculation
        of all properties for one structure. 
        """
        self._atoms_packed = []
        self._index_packed = []
        done = []
        for i in range(len(self._atoms)):
            if i in done:
                continue
            prop = self._atoms[i].info['required_property']
            if isinstance(prop, str):
                prop = [prop]
            index = [i]
            for j in range(len(self._atoms)):
                if i >= j:
                    continue
                if self._is_same(self._atoms[i], self._atoms[j]):
                    prop += [self._atoms[j].info['required_property']]
                    index += [j]
            if True:
                self._atoms_packed.append(self._atoms[i].copy())
                self._atoms_packed[-1].info['required_property'] = prop
                self._index_packed.append(index)
                done += index

    def get_property(self, **kwargs):
        """
        Returns list of calculated properties for all structures.

        Calls :func:`calculate`
        
        :Parameters:
        
            - *kwargs*: dict
            
                directives for calculator
        """
        if self.docalculate:
            if self._model is None:
                raise ValueError("Needs model to continue.")
            if self._atoms is None or self._atoms == []:
                raise ValueError("Needs atoms to continue.")
            else:
                atoms = list(self._atoms_packed)
                calcs = [self._calcs[k[0]] for k in self._index_packed]
            start = time.time()
            if self.nproc in [1, None] or self.parallel in ['serial', False]:
                data, atoms = self._calc_serial(atoms, calcs, kwargs)
            elif self.parallel.lower() == 'multiprocessing':
                data, atoms = self._calc_multiprocessing(atoms, calcs, kwargs)
            elif self.parallel.lower() == 'mpi':
                data, atoms = self._calc_mpi(atoms, calcs, kwargs)
            else:
                raise NotImplementedError('No options for %s' % self.parallel)
            # map calculated atoms to original list
            for i in range(len(self._index_packed)):
                for k in range(len(self._index_packed[i])):
                    self._atoms[self._index_packed[i][k]] = atoms[i]
            self.results = data
            end = time.time()
            # print 'time: ',end-start
        return self.results

    def _calc_serial(self, atoms, calcs, kwargs):
        data = [None] * len(self._atoms)
        assigned = False
        if 'required_property' in kwargs:
            required_property = kwargs['required_property']
            assigned = True
        for i in range(len(atoms)):
            if not assigned:
                required_property = atoms[i].info['required_property']
            if isinstance(required_property, str):
                required_property = [required_property]
            atom = atoms[i]
            atom.set_calculator(calcs[i])
            atom.info['required_property'] = required_property
            temp = CATCalc.calculate(atom, kwargs)
            for k in range(len(required_property)):
                data[self._index_packed[i][k]] = \
                    temp.info[required_property[k]]
            atoms[i] = temp
        return data, atoms

    def _check_loading(self, times):
        """
        modify todo based on processing times of previous iteration
        """
        maxdt = 0.1 * sum(times) / len(times)
        if (max(times) - min(times)) < maxdt:
            return
        maxi = times.index(max(times))
        mini = times.index(min(times))
        # transfer last item in proc with max time to proc with min time
        totransfer = self.todo[maxi].pop(-1)
        self.todo[mini].append(totransfer)

    def _calc_multiprocessing(self, atoms, calcs, kwargs):
        data = [None] * len(atoms)
        assigned = False
        if 'required_property' in kwargs:
            required_property = kwargs['required_property']
            assigned = True
        # prepare atoms and processes
        procs = []
        pipes = []
        for N in range(self.nproc):
            todo_atoms = []
            todo_calcs = []
            pipes.append(Pipe(duplex=False))
            for k in self.todo[N]:
                if k < len(atoms):
                    todo_atoms.append(atoms[k])
                    todo_calcs.append(calcs[k])
            procs.append(PBOPProc(CATCalc.calculate, pipes[N]
                                  , todo_atoms, todo_calcs, kwargs))
        try:
            # run processes
            for p in procs:
                p.start()

            # get output atoms
            out = [op.recv() for (op, ip) in pipes]
            out_atoms = []
            ptimes = []
            for N in range(len(out)):
                out_atoms.append(out[N][0])
                ptimes.append(out[N][1])
            # get calculated properties
            for N in range(len(out_atoms)):
                for j in range(len(out_atoms[N])):
                    if not assigned:
                        required_property = \
                            out_atoms[N][j].info['required_property']
                    if isinstance(required_property, str):
                        required_property = [required_property]
                    props = [out_atoms[N][j].info[rprop] for rprop \
                             in required_property]
                    data[self.todo[N][j]] = props
                    atoms[self.todo[N][j]] = out_atoms[N][j]

            # close communicators and processes
            for j in range(len(pipes)):
                pipes[j][0].close()
            for p in procs:
                p.join()

        except KeyboardInterrupt:
            for p in procs:
                p.terminate()
                exit()
        except:
            raise
        # map and reshape data
        temp = [None] * len(self._atoms)
        for i in range(len(self._index_packed)):
            for k in range(len(self._index_packed[i])):
                temp[self._index_packed[i][k]] = data[i][k]
        data = list(temp)

        # check load balancing
        self._check_loading(ptimes)
        return data, atoms

    def _calc_mpi(self, atoms, calcs, kwargs):
        if self.mpifile is None:
            from mpi4py import MPI
            self.mpifile = '%s/bopcat/mpiproc.py' % homedirectory()
        data = [None] * len(atoms)
        assigned = False
        if 'required_property' in kwargs:
            required_property = kwargs['required_property']
            assigned = True
        # spawning a processing is more straightforward compared to 
        # making the code MPI compatible
        # spawn also every call of get_property because MPI is not
        # picklable and is problematic for process management
        mpicomm = MPI.COMM_SELF.Spawn(sys.executable, args=[self.mpifile]
                                      , maxprocs=self.nproc)

        status = MPI.Status()
        sent = [False] * self.nproc
        s = time.time()
        while False in sent:
            # wait for procs to be ready
            res = mpicomm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
            tag = status.Get_tag()
            source = status.Get_source()
            if sent[source]:
                continue
            if tag != 0:
                continue
            # send jobs to proc
            todo_atoms = []
            todo_calcs = []
            for k in self.todo[source]:
                if k < len(atoms):
                    todo_atoms.append(atoms[k])
                    todo_calcs.append(calcs[k])
            # send jobs to workers
            mpicomm.send((todo_atoms, todo_calcs, kwargs), dest=source, tag=1)
            sent[source] = True
        s = time.time()
        # get results
        out_atoms = [None] * self.nproc
        ptimes = [None] * self.nproc
        status = MPI.Status()
        while None in out_atoms:
            res = mpicomm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
            source = status.Get_source()
            if out_atoms[source] is not None:
                continue
            out_atoms[source] = res[0]
            ptimes[source] = res[1]
        # disconnect communicator
        # mpicomm.Free()
        mpicomm.Disconnect()

        # get calculated properties
        for N in range(len(out_atoms)):
            for j in range(len(out_atoms[N])):
                if not assigned:
                    required_property = \
                        out_atoms[N][j].info['required_property']
                if isinstance(required_property, str):
                    required_property = [required_property]
                props = [out_atoms[N][j].info[rprop] for rprop \
                         in required_property]
                data[self.todo[N][j]] = props
                atoms[self.todo[N][j]] = out_atoms[N][j]

        # map and reshape data
        temp = [None] * len(self._atoms)
        for i in range(len(self._index_packed)):
            for k in range(len(self._index_packed[i])):
                temp[self._index_packed[i][k]] = data[i][k]
        data = list(temp)

        # check load balancing
        self._check_loading(ptimes)
        return data, atoms

    def __calc_mpi(self, atoms, calcs, kwargs):
        sortd = [(len(atoms[i]), i) for i in range(len(atoms))]
        sortd.sort()
        sortd.reverse()
        sortd = [s[1] for s in sortd]
        assigned = False
        if 'required_property' in kwargs:
            required_property = kwargs['required_property']
            assigned = True

        mpicomm = MPI.COMM_SELF.Spawn(sys.executable, args=[self.mpifile]
                                      , maxprocs=self.nproc)
        # mpi message tags
        # 0: READY 1:DONE 2:EXIT 3:START
        s = time.time()

        out_atoms = [None] * len(atoms)
        N = 0
        status = MPI.Status()
        while None in out_atoms:
            # ask which worker is ready or contain data
            res = mpicomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG
                               , status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == 0:
                # send task
                if N < len(atoms):
                    # get atom
                    todo_atom = atoms[sortd[N]]
                    todo_calc = calcs[sortd[N]]
                    todo_atom.info['pid'] = sortd[N]
                    mpicomm.send(([todo_atom], [todo_calc], kwargs)
                                 , dest=source, tag=3)
                    N += 1
                else:
                    mpicomm.send((None, None, None), dest=source, tag=2)
            elif tag == 1:
                # get calculated atom
                out_atoms[res[-1].info['pid']] = res[-1]

                # get calculated properties
        data = [None] * len(atoms)
        for N in range(len(out_atoms)):
            if not assigned:
                required_property = out_atoms[N].info['required_property']
            if isinstance(required_property, str):
                required_property = [required_property]
            props = [out_atoms[N].info[rprop] for rprop in required_property]
            data[N] = props
            atoms[N] = out_atoms[N]
        # map and reshape data
        temp = [None] * len(self._atoms)
        for i in range(len(self._index_packed)):
            for k in range(len(self._index_packed[i])):
                temp[self._index_packed[i][k]] = data[i][k]
        data = list(temp)
        return data, atoms

    def _atoms_to_procs(self):
        # returns list of atoms indices corresponding to each proc
        if self.nproc in [None, 1]:
            return
            # sort atoms according to size:
        sortd = [(len(self._atoms_packed[i]), i) for i in \
                 range(len(self._atoms_packed))]
        sortd.sort()
        sortd.reverse()
        self.todo = [[] for i in range(self.nproc)]
        nats = [0] * self.nproc
        maxatoms_proc = sum([len(at) for at in self._atoms_packed]) / self.nproc
        for i in range(len(sortd)):
            l = i % self.nproc
            # if nats[l] > maxatoms_proc:
            #    for j in range(len(nats)):
            #        if nats[j] < maxatoms_proc:
            #            l = j
            #            break
            if (i / self.nproc) % 2 == 1: l = -(l + 1)
            nats[l] += len(self._atoms_packed[sortd[i][1]])
            self.todo[l].append(sortd[i][1])
        for i in range(len(self.todo)):
            nats = [len(self._atoms_packed[k]) for k in self.todo[i]]

    def __atoms_to_procs(self):
        if self.nproc in [None, 1]:
            return
        # sort atoms according to size:
        sortd = [(len(self._atoms_packed[i]), i) for i in \
                 range(len(self._atoms_packed))]
        sortd.sort()
        # sortd.reverse()
        sortd = [s[1] for s in sortd]
        print(sortd)
        self.todo = []
        # nats = [len(at) for at in self._atoms_packed]
        nats = [len(self._atoms_packed[k]) for k in sortd]
        maxatoms_proc = sum(nats) / self.nproc
        print((sum(nats), maxatoms_proc))
        s = 0
        for N in range(self.nproc):
            todo = []
            nat = 0
            for i in range(s, len(nats)):
                nat += nats[i]
                todo += [sortd[i]]
                if nat > maxatoms_proc:
                    s = i + 1
                    break
            self.todo.append(todo)
        for i in range(len(self.todo)):
            nats = [len(self._atoms_packed[k]) for k in self.todo[i]]

    def add_initial_moments(self, atom):
        mag = 1
        if 'spin' in atom.info:
            mag = atom.info['spin']
        if mag == 1:
            return atom
        if 'initial_magnetic_moments' in atom.info:
            ini_mom = atom.info['initial_magnetic_moments'][0]
            atom.set_initial_magnetic_moments(ini_mom)
            return atom
        mom = []
        sym = atom.get_chemical_symbols()
        ini_mom = self.ini_magmoms
        for i in sym:
            if i in ini_mom:
                mom.append(ini_mom[i])
            else:
                # raise ValueError('Provide initial magnetic moments for %s.'%i)
                mom.append(3.0)
        atom.set_initial_magnetic_moments(mom)
        return atom

    def set_atoms(self, atoms):
        """
        Set the structures for calculation. The calculators
        and results are reset.

        :Parameters:
        
            - *atoms*: list
            
                list of ASE Atoms objects
        """
        # self._atoms = atoms
        self._atoms = [at.copy() for at in atoms]
        self._atoms = [self.add_initial_moments(at) for at in self._atoms]
        # reset calculators
        if self._calcs is not None:
            self.clean()
        self._calcs = None
        self.get_calculators()
        self.structures = self.get_structures()
        # reset results
        self.results = None
        self.docalculate = True
        self.pack_atoms()
        self._atoms_to_procs()

    def set_model(self, model):
        """
        Set the model for the calcuator. The calculators
        and results are reset.

        :Parameters:
        
            - *model*: instance of calculator-specific model
            
                model used for the calculator
        """
        self._model = model
        # reset results
        self.results = None
        self.docalculate = True
        # update calculators
        self.get_calculators()

    def get_atoms(self):
        """
        Returns a list of the structures each an ASE Atoms object
        assigned for calculation.
        """
        return self._atoms

    def get_calculator(self, i, update):
        """
        Returns the calculator. If None will initialize 
        the calculator. 

	:Parameters:

	    - *update*: bool
                
                True: will update the model used by the calculator
         """
        if self.calculator.lower() == 'bopfox':
            if update:
                assert (self._calcs is not None)
                calc = self._calcs[i]
                calc.set_modelsbx(self._model)
            else:
                calc = initialize_bopfox(self.calculator_settings
                                         , self._model, self._atoms[i])
        else:
            raise NotImplementedError('No options for %s' % self.calculator)
        return calc

    def get_calculators(self):
        """
	Returns a list of calculators corresponding to each structure

        calls :func:`get_calculator`
        """
        # if self._calcs is None:
        #    calc = self.get_calculator()
        #    self._calcs = [None]*len(self._atoms)
        # else:
        #    calc = self.get_calculator(update=True)
        # for i in range(len(self._atoms)):
        #    self._calcs[i] = calc.copy()  
        if self._calcs is None:
            self._calcs = [None] * len(self._atoms)
            update = False
        else:
            update = True
            # parallelization does not work for library call, so do
        # not parallelize
        # if self.nproc == 1:
        if True:
            for i in range(len(self._atoms)):
                self._calcs[i] = self.get_calculator(i, update=update)
        else:
            div = int(len(self._atoms) / self.nproc) + \
                  (len(self._atoms) % self.nproc > 0)
            for i in range(div):
                procs = []
                pipes = [None] * len(self._atoms)
                todo = list(range(i * self.nproc, i * self.nproc + self.nproc))
                for N in todo:
                    if N < len(self._atoms):
                        pipes[N] = Pipe(duplex=False)
                        procs.append(cProcess(self.get_calculator, pipes, N, update))
                try:
                    for p in procs:
                        p.start()
                except KeyboardInterrupt:
                    for p in procs:
                        p.terminate()
                    exit()
                for N in todo:
                    if N < len(self._atoms):
                        self._calcs[N] = pipes[N][0].recv()
                        pipes[N][0].close()
                for p in procs:
                    p.join()
        return self._calcs

    def get_structures(self):
        """
	Returns a list of the system ID of all the structures
	"""
        if self.structures is None:
            atoms = self._atoms
            strucs = []
            if atoms is not None:
                for i in range(len(atoms)):
                    sID = ''
                    IDs = atoms[i].info['system_ID']
                    if IDs is not None:
                        sID = IDs
                    strucs.append(sID)
            self.structures = list(strucs)
        return self.structures

    def get_model(self):
        return self._model


class cProcess(Process):
    def __init__(self, target, pipes, i, update):
        Process.__init__(self)
        self._target = target
        self._pipes = pipes
        self._i = i
        self._update = update

    def run(self):
        calc = self._target(self._i, self._update)
        self._pipes[self._i][1].send(calc)
        self._pipes[self._i][1].close()


if __name__ == "__main__":
    from ase.lattice import bulk
    from .bopmodel import read_modelsbx
    import numpy as np

    atoms = []
    for a in np.linspace(2.7, 3.1, 5):
        atoms.append(bulk('Fe', a=a))
    calc = CATCalc()
    calc.set_atoms(atoms)
    model = read_modelsbx(filename= 'models.bx')[0]
    calc.set_model(model)
    print(("Energies: ", calc.get_property(required_property='energy')))
    calc.docalculate = True
