import pickle as pickle
import json
from .output import cattxt, print_format
import os
import time
import subprocess
# import deepcopy
# import tempfile
from numpy.random import random as nrandom
from multiprocessing import Queue, JoinableQueue
from multiprocessing import Process as mProcess


class Process(object):
    """
    wraps and manages a generic process
    
    Defines a generic process in bopcat.

    Prepares and manages a job specified by the kernel.

    :Parameters:

    - *pid*: int

        unique id of the process, process info are stored in .bopcat.json

    - *queue*: str, instance of queue
       
        name of the queue, one of serial, subprocess, cluster 

    - *nproc*: int

        number of parallel processes

            ``None``: will not parallelize

            ``default``: number of cpu * 2

    - *controls*: instance of CATControls   
        
        CATControls object to initialize parameters  
    """

    def __init__(self, **kwargs):
        self._pid = None  # process id
        self._ppid = None  # parent process id
        self._pdir = None  # directory where process is executed
        self._kernel = None  # kernel is a python object
        self._pidprefix = 'bopcat'
        self._script = 'run.py'
        self._queue = 'serial'
        self._name = 'cat'
        self._python = 'python'
        self._log = 'log.cat'
        self._cwd = os.getcwd()
        self._cores = 1
        self._runtime = 129600
        self._directives = []
        self._timeid = time.time()
        self._qid = None
        self._done = False
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key == 'kernel':
                self._kernel = val
            elif key == 'queue':
                self._queue = val
            elif key == 'python':
                self._python = val
            elif key == 'log':
                self._log = val
            elif key == 'cores':
                self._cores = val
            elif key == 'runtime':
                self._runtime = val
            elif key == 'directives':
                self._directives = val
            elif key == 'pid':
                self._pid = val
        if self._pid is not None:
            self.load()

    def __get_pid(self):
        out = None
        fn = '%s/.bopcat.pckl' % (self._cwd)
        with open(fn, 'r') as f:
            procs = pickle.load(f)
        for pid, val in list(procs.items()):
            if val['time'] == self._timeid:
                self.pdir = val['working_directory']
                out = pid
                break
        return out

    def _get_pid(self):
        out = None
        while True:
            try:
                fn = '%s/.bopcat.json' % (self._cwd)
                with open(fn, 'r') as f:
                    procs = json.load(f)
            except:
                continue
            if procs != 'hold':
                break
        for pid, val in list(procs.items()):
            if val['time'] == self._timeid:
                self.pdir = val['working_directory']
                out = int(pid)
                break
        return out

    @property
    def pdir(self):
        return self._pdir

    @pdir.setter
    def pdir(self, pdir):
        self._pdir = pdir

    @property
    def pid(self):
        return self._pid

    @pid.setter
    def pid(self, pid):
        self._pid = pid

    def __get_log(self):
        fn = os.path.abspath('.bopcat.pckl')
        if os.path.isfile(fn):
            with open(fn, 'r') as f:
                procs = pickle.load(f)
            while procs == 'hold':
                time.sleep(nrandom() * 0.001)
                with open(fn, 'r') as f:
                    procs = pickle.load(f)
            with open(fn, 'w') as f:
                pickle.dump('hold', f)
        else:
            procs = {}
        # try:
        #    fn = os.path.abspath('.bopcat.pckl')
        #    with open(fn,'r') as f:
        #        procs = pickle.load(f)
        # except:
        #    procs = {}
        return procs

    def _get_log(self):
        fn = os.path.abspath('.bopcat.json')
        if os.path.isfile(fn):
            with open(fn, 'r') as f:
                procs = json.load(f)
            while procs == 'hold':
                time.sleep(nrandom() * 0.001)
                with open(fn, 'r') as f:
                    procs = json.load(f)
            with open(fn, 'w') as f:
                json.dump('hold', f)
        else:
            procs = {}
        return procs

    def is_done(self):
        if self._done:
            return self._done
        if self._pid is None:
            self._kernel = self.get_kernel()
        fn = '%s/%s' % (self._pdir, self._log)
        if not os.path.isfile(fn):
            return self._done
        l = open(fn).readlines()
        for i in range(len(l) - 1, -1, -1):
            if 'process done' in l[i]:
                self._done = True
                break
        if not self._done and self._queue not in ['serial', 'subprocess']:
            if not self.is_in_queue():
                self._done = True
        return self._done

    def clean(self):
        kernel = self.get_kernel()
        if self._queue != 'serial' and kernel is not None:
            fname = '%s_%d.sh' % (self._pidprefix, self._pid)
            os.system('rm -rf %s' % fname)
        os.system('rm -rf %s' % self._pdir)

    def _prepare_directories(self):
        self._cwd = os.getcwd()
        procs = self._get_log()
        pid = 1
        if len(procs) > 0:
            # pckl file contains info pertaining to the job
            # like job name etc. from self._kernel.info
            pid = len(procs) + 1
        procs[pid] = {}
        procs[pid]['info'] = self._kernel._info
        procs[pid]['time'] = self._timeid
        self._pid = pid
        pdir = '%s_%d' % (self._pidprefix, self._pid)
        self.pdir = os.path.abspath(pdir)
        procs[pid]['working_directory'] = self.pdir
        fn = os.path.abspath('.bopcat.pckl')
        with open(fn, 'wb') as f:
            pickle.dump(procs, f)
        os.mkdir(self.pdir)
        self.write_process_script()
        with open('%s/%s.pckl' % (self.pdir, self._name), 'wb') as f:
            pickle.dump(self._kernel, f)

    def prepare_directories(self):
        self._cwd = os.getcwd()
        procs = self._get_log()
        pid = 1
        if len(procs) > 0:
            # pckl file contains info pertaining to the job
            # like job name etc. from self._kernel.info
            pid = len(procs) + 1
        procs[pid] = {}
        procs[pid]['info'] = self._kernel._info
        procs[pid]['time'] = self._timeid
        self._pid = pid
        pdir = '%s_%d' % (self._pidprefix, self._pid)
        self.pdir = os.path.abspath(pdir)
        procs[pid]['working_directory'] = self.pdir
        fn = os.path.abspath('.bopcat.json')
        with open(fn, 'w') as f:
            json.dump(procs, f)
        os.mkdir(self.pdir)
        self.write_process_script()
        with open('%s/%s.pckl' % (self.pdir, self._name), 'wb') as f:
            pickle.dump(self._kernel, f)

    def _load(self):
        procs = self._get_log()
        fn = os.path.abspath('.bopcat.pckl')
        with open(fn, 'w') as f:
            pickle.dump(procs, f)
        if self._pid not in procs:
            raise ValueError('process %s not found!' % self._pid)
        self._pdir = procs[self._pid]['working_directory']
        self._kernel = self.get_kernel()
        if self._kernel is None:
            raise RuntimeError("kernel is corrupted.")
        self._kernel._info = procs[self._pid]['info']
        self._timeid = procs[self._pid]['time']

    def load(self):
        procs = self._get_log()
        fn = os.path.abspath('.bopcat.json')
        with open(fn, 'w') as f:
            json.dump(procs, f)
        if '%s' % self._pid not in procs:
            raise ValueError('process %s not found!' % self._pid)
        self._pdir = procs['%s' % self._pid]['working_directory']
        self._kernel = self.get_kernel()
        if self._kernel is None:
            raise RuntimeError("kernel is corrupted.")
        self._kernel._info = procs['%s' % self._pid]['info']
        self._timeid = procs['%s' % self._pid]['time']

    def write_run_script(self):
        script = cattxt(filename='%s/__run.py' % self.pdir)
        stream = []
        stream.append("with open('%s/%s') as f:" % (self.pdir, self._script))
        stream.append("exec(f)")
        script.add(stream[0], length=len(stream[0]))
        script.add(stream[1], length=len(stream[1]) + 4)
        script.write()

    def write_process_script(self):
        """
        wrapper for process
        """
        script = cattxt(filename='%s/%s' % (self.pdir, self._script))
        stream = []
        stream.append('import pickle')
        stream.append('import time')
        stream.append("success=False")
        stream.append("count=0")
        stream.append("while not success:")
        stream.append("    try:")
        stream.append("        %s = pickle.load(open('%s/%s.pckl','rb'))" % ( \
            self._name, self.pdir, self._name))
        stream.append("        %s.run()" % (self._name))
        stream.append("        success=True")
        stream.append("    except:")
        # stream.append("        raise")
        stream.append("        count+=1")
        stream.append("        time.sleep(%s)" % (nrandom() * 10))
        # stream.append("    except:")
        # stream.append("        raise")
        stream.append("    if count > -1:")
        stream.append("        break")
        stream.append("if success:")
        stream.append("    pickle.dump(%s,open('%s/%s.pckl','wb'))" % ( \
            self._name, self.pdir, self._name))
        stream.append("print('process done')")
        for s in stream:
            script.add(s, length=len(s))
        script.write()

    def is_in_queue(self):
        script = '%s_%d.sh' % (self._pidprefix, self._pid)
        try:
            sout = subprocess.check_output(["qstat", "-j", script]
                                           , stderr=subprocess.STDOUT).split('\n')
        except:
            return False
        self._qid = int(sout[1].split()[-1])
        return True

    def build_queue(self):
        run = [
            'cd %s' % self._pdir
            , '%s %s > %s' % (self._python, self._script, self._log)
            , 'cd %s' % self._cwd]
        fname = '%s_%d.sh' % (self._pidprefix, int(self._pid))
        if isinstance(self._queue, queue):
            q = self._queue
            q.set(directives=self._directives + run, filename=fname)
        elif self._queue.lower() == 'cluster_serial':
            q = cluster_serial(cores=self._cores, runtime=self._runtime
                       , directives=self._directives + run
                       , filename=fname)
        elif self._queue.lower() == 'cluster_parallel':
            q = cluster_parallel(cores=self._cores, runtime=self._runtime
                                  , directives=self._directives + run
                                  , filename=fname)
        else:
            raise NotImplementedError('No options for %s' % self._queue)
        return q

    def submit_to_queue(self):
        q = self.build_queue()
        q.write_submit_script()
        q.submit()
        self._qid = q.get_id()

    def run_serial(self):
        cmd = '%s %s/%s' % (self._python, self.pdir, self._script)
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            output = process.communicate()[0]
            with open('%s/%s' % (self.pdir, self._log), 'w') as f:
                print(output, file=f)
        except:
            del process
            exit()

    def run_subprocess(self):
        # will only run if kernel contains more than one process 
        kernel = self.get_kernel()
        if not isinstance(kernel, Kernel_genericlist):
            raise ValueError('Cannot run subprocess on %s' % type(kernel))
        procs = kernel._procs
        inQ = JoinableQueue(len(procs))
        outQ = Queue(len(procs))

        sprocs = [sProcess(inQ, outQ) for i in range(self._cores)]
        # Nprocess should be same as self._cores
        # for i in range(len(procs)):
        #    if procs[i]._queue == 'subprocess':
        #        raise RuntimeError("Cannot run subprocess.")
        #    procs[i].set(inq=inQ, outq= outQ)    
        for p in sprocs:
            p.start()
        msg = 'Warning: spawning subprocesses of subproces job'
        for i in range(len(procs)):
            if procs[i]._queue == 'subprocess':
                print_format(msg, level=3)
            procs[i]._spid = i
            inQ.put(procs[i])
            time.sleep(0.2)
        for i in range(len(sprocs)):
            inQ.put(None)
        inQ.join()
        res = {}
        # count = 0
        while len(res) < len(procs):
            temp = outQ.get()
            res[temp._spid] = temp
            # if (len(procs) - len(res)) < len(sprocs):
            #    count += 1
            #    if count > 100:
            #        break
        with open('%s/%s' % (self.pdir, self._log), 'w') as f:
            f.write('process done')
        # for i in range(len(atoms)):
        #          if not assigned: 
        #              required_property = atoms[i].get_required_property()
        #          data.append(res[i].get_property(required_property))
        #          atoms[i] = res[i]

    def run(self):
        if self._pid is None:
            # fresh run
            self.prepare_directories()
        print(self._queue, '<<<<')
        if not self.is_done():
            # restart
            if self._queue == 'serial':
                self.run_serial()
            elif self._queue == 'subprocess':
                self.run_subprocess()
            else:
                self.submit_to_queue()
        else:
            self._kernel = self.get_kernel(update=True)

    def get_kernel(self, update=True):
        if update:
            if self._pid is None:
                self.pid = self._get_pid()
            # do this to avoid simulataneous access of pickle
            read = False
            count = 0
            while not read:
                try:
                    f = open('%s/%s.pckl' % (self.pdir, self._name), 'rb')
                    self._kernel = pickle.load(f)
                    read = True
                except:
                    count += 1
                    time.sleep(nrandom() * 2)
                if count > 5:
                    break
        return self._kernel


class sProcess(mProcess):
    def __init__(self, inq, outq):
        mProcess.__init__(self)
        self._inq = inq
        self._outq = outq
        # self._proc = proc

    def run(self):
        while True:
            indata = self._inq.get()
            if indata is None:
                self._inq.task_done()
                break
            indata.run()
            indata.get_kernel()
            self._outq.put(indata)
            self._inq.task_done()
        return


class Kernel_catkernel:
    """
    Wrapper for single catkernel job
    """

    def __init__(self, catkernel):
        self._catkernel = catkernel
        self._mode = 'test'
        if self._catkernel.variables is not None:
            self._mode = 'optimize'
        self._info = {'variables': self._catkernel.variables
                      # ,'model':self._catkernel.calc.get_model()
            , 'nstrucs': len(self._catkernel.calc.get_atoms())
                      }

    def run(self):
        if self._mode == 'test':
            self._catkernel.test()
        elif self._mode == 'optimize':
            self._catkernel.optimize()


class Kernel_catobjective:
    """
    Wrapper for single catobjective job
    """

    def __init__(self, catobjective, x0):
        self._catobjective = catobjective
        self._x0 = x0
        self._info = {'variables': self._catobjective.variables
            , 'name': self._catobjective.name
                      }

    def run(self):
        self._catobjective.verbose = True
        self._catobjective(self._x0)


class Kernel_genericlist(object):
    """
    Wrapper for a list of generic processes
    """

    def __init__(self, procs):
        self._procs = procs
        self._info = {'nprocs': len(procs)}

    def run(self):
        for i in range(len(self._procs)):
            if self._procs[i].is_done():
                continue
            self._procs[i].run()


class Kernel_catkernels(Kernel_genericlist):
    """
    Wrapper for list of catkernel jobs
    """

    def __init__(self, procs):
        self._procs = procs
        self._info = {'nprocs': len(procs)}


class Kernel_catprocess:
    """
    Wrapper for a single catprocess job
    """

    def __init__(self, catprocess):
        self._catprocess = catprocess
        self._info = {}

    def run(self):
        self._catprocess.run()


class Kernel_catprocesses(Kernel_genericlist):
    """
    Wrapper for a list of catprocess jobs
    """

    def __init__(self, catprocesses):
        self._catprocesses = catprocesses
        self._info = {'nprocs': len(catprocesses)}


class Process_catobjective(Process):
    """
    Defines a single catobjective job 
    """

    def __init__(self, **kwargs):
        self._catobjective = None
        self._x0 = None
        self._err = None
        self._success = None
        self._done = False
        self._results = None
        Process.__init__(self, **kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key == 'catobjective':
                self._catobjective = val
                self._catobjective.verbose = True
                self._kernel = Kernel_catobjective(self._catobjective, self._x0)
            elif key == 'x0':
                self._x0 = val
        Process.set(self, **kwargs)

    def get_results(self, wait=True):
        if wait:
            while not self.is_done():
                time.sleep(50)
        if not self.is_done():
            return
        if self._results is None:
            self.gen_summary()
            self._results = {}
            self._results['function_value'] = self._err
        return self._results

    def gen_summary(self):
        """
        returns the error as evaluated by catobjective
        """
        err = None
        fn = '%s/%s' % (self._pdir, self._log)
        try:
            l = open(fn, 'r').readlines()
        except:
            return
        for i in range(len(l)):
            if "rms:" in l[i]:
                s = l[i].split()
                err = float(s[s.index('rms:') + 1])
        if err is None:
            return
        self._err = err


class Process_catobjectives(Process):
    """
    Defines a list catobjective jobs 
    """

    def __init__(self, **kwargs):
        self._catobjectives = None
        self._x0 = None
        self._err = None
        self._success = None
        self._done = False
        self._results = None
        self._procs = None
        Process.__init__(self, **kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key == 'catobjectives':
                self._catobjectives = val
            elif key == 'x0':
                self._x0 = val
            elif key == 'procs':
                self._procs = val
                self._kernel = Kernel_genericlist(procs=self._procs)
        if self._catobjectives is not None:
            self._procs = []
            for i in range(len(self._catobjectives)):
                self._procs.append(Process_catobjective(
                    catobjective=self._catobjectives[i], x0=self._x0[i]))
            self._kernel = Kernel_genericlist(procs=self._procs)
        Process.set(self, **kwargs)
        self._procs = self._kernel._procs

    def is_done(self):
        if self._done:
            return self._done
        self._done = True
        subprocs = self.get_kernel()._procs
        subprocs = [sp.is_done() for sp in subprocs]
        if False in subprocs:
            self._done = False
        return self._done

    def get_results(self, wait=False):
        if wait:
            while not self.is_done():
                time.sleep(50)
        if not self.is_done():
            return
        if self._results is None:
            self.gen_summary()
            self._results = {}
            self._results['function_value'] = self._err
        return self._results

    def gen_summary(self):
        """
        gets results from procs 
        """
        self._err = []
        for i in range(len(self._procs)):
            resi = self._procs[i].get_results()
            self._err.append(resi['function_value'])

    def clean(self):
        kernel = self.get_kernel()
        for proc in self._procs:
            proc.clean()
        Process.clean(self)


class Process_catkernels(Process):
    def __init__(self, **kwargs):
        self._catkernels = None
        self._success_criteria = {}
        self._strucs = []
        self._err = []
        self._newpar = []
        self._success = None
        self._func_calls = []
        self._done = False
        self._procs = None
        self._results = None
        Process.__init__(self, **kwargs)
        # self.set(**kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key == 'catkernels':
                self._catkernels = val
                self._procs = [Process_catkernel(catkernel=kern, **kwargs) \
                               for kern in self._catkernels]
                self._kernel = Kernel_catkernels(self._procs)
            elif key == 'success_criteria':
                self._success_criteria = val
            elif key == 'procs':
                self._procs = val
                self._kernel = Kernel_catkernels(self._procs)
        # if loading 
        Process.set(self, **kwargs)
        self._procs = self._kernel._procs

    def is_done(self):
        if self._done:
            return self._done
        self._done = True
        subprocs = self.get_kernel()._procs
        subprocs = [sp.is_done() for sp in subprocs]
        if False in subprocs:
            self._done = False
        return self._done

    def is_success(self):
        if self._success is not None:
            return self._success
        if not self.is_done():
            return self._success
        self.gen_summary()
        if self._err == [] or self._newpar == [] or self._func_calls == []:
            self._success = False
        else:
            self._success = True
            procs = []
            for i in range(len(self._procs)):
                procs.append(self._procs[i].is_success())
            nsuccess = procs.count(True)
            for key, val in list(self._success_criteria.items()):
                if key == 'nprocs':
                    if nsuccess < val:
                        self._success = False
                        break
        return self._success

    def get_results(self, wait=False):
        if wait:
            while not self.is_done():
                time.sleep(50)
        if not self.is_done():
            return
        if self._results is None:
            self.gen_summary()
            self._results = {}
            self._results['function_value'] = self._err
            self._results['N_function_calls'] = self._func_calls
            self._results['optimized_parameters'] = self._newpar
        return self._results

    def gen_summary(self):
        """
        returns the model parameters used, test/fit structures, error
        """
        self._err = []
        self._func_calls = []
        self._newpar = []
        for i in range(len(self._procs)):
            resi = self._procs[i].get_results()
            if resi is not None and self._procs[i].is_success():
                self._err.append(resi['function_value'])
                self._func_calls.append(resi['N_function_calls'])
                self._newpar.append(resi['optimized_parameters'])
            elif resi is None and self._success_criteria == {}:
                self._err.append(None)
                self._func_calls.append(None)
                self._newpar.append(None)

    def clean(self):
        kernel = self.get_kernel()
        for proc in self._procs:
            proc.clean()
        Process.clean(self)


class Process_catkernel(Process):
    """
    Defines a single job 
    """

    def __init__(self, **kwargs):
        self._catkernel = None
        self._success_criteria = {}
        self._strucs = None
        self._err = None
        self._newpar = None
        self._func_calls = None
        self._success = None
        self._done = False
        self._mode = None
        self._results = None
        Process.__init__(self, **kwargs)
        # self.set(**kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key == 'catkernel':
                self._catkernel = val
                self._kernel = Kernel_catkernel(self._catkernel)
            elif key == 'success_criteria':
                self._success_criteria = val
        Process.set(self, **kwargs)

    def get_mode(self):
        if self._mode is None:
            if self._catkernel.variables is None:
                self._mode = 'test'
            elif len(self._catkernel.variables) == 0:
                self._mode = 'test'
            else:
                self._mode = 'optimize'
        return self._mode

    def is_success(self):
        if self._success is not None:
            return self._success
        if not self.is_done():
            return self._success
        self.gen_summary()
        if self._err is None or self._func_calls is None:
            self._success = False
        else:
            self._success = True
            for key, val in list(self._success_criteria.items()):
                if key == 'func_calls':
                    if self._func_calls < val:
                        self._success = False
                        break
                elif key == 'err':
                    if self._err > val:
                        self._success = False
                        break
        return self._success

    def get_results(self, wait=True):
        if wait:
            while not self.is_done():
                time.sleep(50)
        if not self.is_done():
            return
        if self._results is None:
            self.gen_summary()
            self._results = {}
            self._results['function_value'] = self._err
            self._results['N_function_calls'] = self._func_calls
            self._results['optimized_parameters'] = self._newpar
        return self._results

    def gen_summary(self):
        """
        returns the model parameters used, test/fit structures, error
        TODO: get results directly from kernel
        """
        newpar = []
        strucs = []
        err = None
        func_calls = None
        fn = '%s/%s' % (self._pdir, self._log)
        try:
            l = open(fn, 'r').readlines()
        except:
            return
        for i in range(len(l)):
            if "function value" in l[i]:
                err = float(l[i].split()[-1])
            elif "N function calls" in l[i]:
                func_calls = int(l[i].split()[-1])
            elif "Start" in l[i] and 'End' in l[i]:
                for j in range(i + 1, len(l)):
                    if len(l[j].split()) != 3:
                        break
                    newpar.append(float(l[j].split()[-1]))
            elif "Structures in set" in l[i]:
                nstrucs = int(l[i].split()[-1])
                for j in range(nstrucs):
                    strucs.append(l[i + 1 + j].split()[-1])
        if err is None:
            return
        self._err = err
        self._strucs = list(strucs)
        self._func_calls = func_calls
        self._newpar = list(newpar)


class Process_catprocess(Process):
    def __init__(self, **kwargs):
        self._catprocess = None
        self._done = False
        self._results = None
        self._optimized_model = None
        self._score = None
        self._fit_variables = None
        self._correlations = None
        self._nstrucs_fit = None
        self._success = None
        # self.set(**kwargs)
        Process.__init__(self, **kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key == 'catprocess':
                self._catprocess = val
                self._kernel = Kernel_catprocess(self._catprocess)
        # if loading 
        Process.set(self, **kwargs)

    def is_success(self):
        if self._success is not None:
            return self._success
        if not self.is_done():
            return self._success
        self.gen_summary()
        if self._optimized_model is None or self._score is None:
            self._success = False
        else:
            self._success = True
            for key, val in list(self._success_criteria.items()):
                if key == 'score':
                    if self._score < val:
                        self._success = False
                        break
        return self._success

    def get_results(self, wait=False):
        if wait:
            while not self.is_done():
                time.sleep(50)
        if not self.is_done():
            return
        if self._results is None:
            self.gen_summary()
            self._results = {}
            self._results['optimized_model'] = self._optimized_model
            self._results['score'] = self._score
        return self._results

    def gen_summary(self):
        """
        
        """
        self._optimized_model, self._score = \
            self._catprocess.get_fit_model_optimized(return_score=True)


class Process_catprocesses(Process):
    def __init__(self, **kwargs):
        self._catprocesses = None
        self._procs = None
        self._success_criteria = {}
        self._success = None
        self._done = False
        self._results = None
        self._score = None
        self._optimized_model = None
        # self.set(**kwargs)
        Process.__init__(self, **kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key == 'catprocesses':
                self._catprocesses = val
                self._procs = [Process_catprocess(catprocess=proc, **kwargs) \
                               for proc in self._catprocesses]
                self._kernel = Kernel_catprocesses(self._procs)
            elif key == 'success_criteria':
                self._success_criteria = val
            elif key == 'procs':
                self._procs = val
                self._kernel = Kernel_catprocesses(self._procs)
        # if loading 
        Process.set(self, **kwargs)
        self._procs = self._kernel._procs

    def is_done(self):
        subprocs = self.get_kernel()._procs
        subprocs = [sp.is_done() for sp in subprocs]
        if False in subprocs:
            self._done = False
        return self._done

    def is_success(self):
        if self._success is not None:
            return self._success
        if not self.is_done():
            return self._success
        self.gen_summary()
        if self._score == [] or self._optimized_model == []:
            self._success = False
        else:
            self._success = True
            procs = []
            for i in range(len(self._procs)):
                procs.append(self._procs[i].is_success())
            nsuccess = procs.count(True)
            for key, val in list(self._success_criteria.items()):
                if key == 'nprocs':
                    if nsuccess < val:
                        self._success = False
                        break
        return self._success

    def get_results(self, wait=False):
        if wait:
            while not self.is_done():
                time.sleep(50)
        if not self.is_done():
            return
        if self._results is None:
            self.gen_summary()
            self._results = {}
            self._results['function_value'] = self._err
            self._results['N_function_calls'] = self._func_calls
            self._results['optimized_parameters'] = self._newpar
        return self._results

    def gen_summary(self):
        """
        returns list of optimized models and scores
        """
        self._score = []
        self._optimized_model = []
        for i in range(len(self._procs)):
            resi = self._procs[i].get_results()
            if resi is not None and self._procs[i].is_success():
                self._score.append(resi['score'])
                self._optimized_model.append(resi['optimized_model'])

    def clean(self):
        kernel = self.get_kernel()
        for proc in self._procs:
            proc.clean()
        Process.clean(self)


class queue(object):
    def __init__(self, **kwargs):
        self._cores = 1
        self._runtime = 18000
        self._options = []
        self._directives = []
        self._headers = []
        self._filename = 'bopcat.sh'
        self._sid = None
        self._qexec = 'qsub'
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key.lower() == 'cores':
                self._cores = val
            elif key.lower() == 'runtime':
                self._runtime = val
            elif key.lower() == 'headers':
                self._headers = val
            elif key.lower() == 'options':
                self._options = val
            elif key.lower() == 'directives':
                self._directives = val
            elif key.lower() == 'filename':
                self._filename = val
            elif key.lower() == 'qexec':
                self._qexec = val
            # elif key.lower() == 'qname':
            #    self._qname = val
            # elif key.lower() == 'pe':
            #    self._pe = val

    def write_submit_script(self):
        f = open(self._filename, 'w')
        for head in self._headers:
            f.write('%s\n' % head)
        # h =  int(self._runtime/3600)
        # m = int(int(self._runtime - h *3600)/60)
        # s = int(self._runtime - h *3600 - m*60)
        # f.write('#$ -l h_rt=%d:%02d:%02d\n'%(h, m, s))
        for option in self._options:
            f.write('%s\n' % option)
        for direct in self._directives:
            f.write('%s\n' % direct)
        f.close()

    def get_hms(self):
        h = int(self._runtime / 3600)
        m = int(int(self._runtime - h * 3600) / 60)
        s = int(self._runtime - h * 3600 - m * 60)
        return h, m, s

    def submit(self):
        print(self._qexec)
        print(self._filename)
        sout = subprocess.check_output([self._qexec, self._filename]).split('\n')
        for s in sout:
            try:
                self._sid = int(s)
                break
            except:
                pass

    def get_id(self):
        return self._sid


class cluster_serial(queue):
    def __init__(self, **kwargs):
        self._qname = 'serial.q'
        self._pe = 'smp'
        queue.__init__(self, **kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key.lower() == 'qname':
                self._qname = val
            elif key.lower() == 'pe':
                self._pe = val
        queue.set(self, **kwargs)
        if self._options == []:
            h, m, s = self.get_hms()
            self._options = [
                '#$ -l h_rt=%d:%02d:%02d' % (h, m, s)
                , '#$ -l qname=%s' % self._qname
                , '#$ -pe %s %d' % (self._pe, self._cores)
                , '#$ -j y'
                , '#$ -o queue.out'
                , '#$ -S /bin/bash'
                , '#$ -R y'
            ]
        if self._headers == []:
            self._headers = ['#!bin/tcsh -f', '#$ -cwd']


class cluster_parallel(queue):
    def __init__(self, **kwargs):
        self._qname = 'parallel16.q'
        self._pe = 'mpi16'
        queue.__init__(self, **kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key.lower() == 'qname':
                self._qname = val
            elif key.lower() == 'pe':
                self._pe = val
        queue.set(self, **kwargs)
        if self._options == []:
            self._options = ['#$ -l qname=%s' % self._qname
                , '#$ -pe %s %d' % (self._pe, self._cores)
                , '#$ -j y'
                , '#$ -o queue.out'
                , '#$ -S /bin/bash'
                , '#$ -R y'
                             ]

