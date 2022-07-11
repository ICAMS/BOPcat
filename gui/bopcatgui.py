#!/usr/bin/env python

# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import Tkinter as tk
from tkFileDialog import askopenfilename
#download and install pillow:
# http://www.lfd.uci.edu/~gohlke/pythonlibs/#pillow
#from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
import matplotlib.pyplot as plt

from bopcat.catcontrols import CATControls
from bopcat.catdata import CATData
from bopcat.catkernel import CATKernel
from bopcat.catcalc import CATCalc
from bopcat.catparam import CATParam
from bopcat.process_management import Process_catkernel, Process_catkernels, cluster_serial, cluster_parallel
from bopcat.bopmodel import read_modelsbx, average_models
from bopcat.variables import bopcat_version
from bopcat.output import get_bopfox_version
from bopcat.utils import gen_param_model, get_par_keys

import numpy as np
import math
from numpy import inf
import os
import sys
import time
import threading
import ase
from copy import deepcopy
from tempfile import mkstemp

#style.use("ggplot")
#style.use("bmh")
style.use("seaborn-white")


class BOPcatrun:
    """
    Defines set up and running of a BOPcat kernel.
    """
    def __init__(self):
        self.cat_controls = CATControls()
        self.cat_controls_var = CATControls()
        self.modelbx      = None
        self.modelbx0     = None
        self.kernel       = None
        self.ref_atoms    = None
        self.ref_data     = None
        self.modelbx_var  = None
        self.queue        = 'serial'
        self.directives   = ['source ~/.bashrc']
        self.runtime      = 60*60*24*7
        self.par          = None
        self.is_done      = False
        self.proc         = None
        self.iter_file    = mkstemp(suffix='.dat',prefix='iteration',dir=os.getcwd())[1]
        self.rattle       = False
        self.rattle_options = ('gaussian',0.01)
        self.nstrucmax    = 10000

    def init_cat_controls(self):
        """
        convert the current values of BOPcat control parameters in GUI to BOPcat
        """
        if self.cat_controls_var.elements is None:
            return
        self.cat_controls.elements = [unpack_value(v) for v in self.cat_controls_var.elements]
        self.cat_controls.data_free_atom_energies = {}
        for okey, val in self.cat_controls_var.data_free_atom_energies.items():
            key = self.cat_controls.elements[okey]
            val = unpack_value(val)
            self.cat_controls.data_free_atom_energies[key] = val
        self.cat_controls.data_ini_magmoms = {}    
        for okey, val in self.cat_controls_var.data_ini_magmoms.items():
            key = self.cat_controls.elements[okey]
            val = unpack_value(val)
            self.cat_controls.data_ini_magmoms[key] = val
        self.cat_controls.model = unpack_value(self.cat_controls_var.model)
        self.cat_controls.model_pathtomodels = unpack_value(self.cat_controls_var.model_pathtomodels)
        self.cat_controls.calculator_nproc = unpack_value(self.cat_controls_var.calculator_nproc)
        self.cat_controls.calculator_queue = unpack_value(self.cat_controls_var.calculator_queue)
        self.cat_controls.opt_optimizer = unpack_value(self.cat_controls_var.opt_optimizer)
        self.cat_controls.calculator_queue = unpack_value(self.cat_controls_var.calculator_queue)
        for key, val in self.cat_controls_var.opt_optimizer_options.items():
            self.cat_controls.opt_optimizer_options[key] = unpack_value(val)
        for key, val in self.cat_controls_var.calculator_settings.items():
            self.cat_controls.calculator_settings[key] = unpack_value(val)

        self.cat_controls.data_system_parameters = {}
        for key, val in self.cat_controls_var.data_system_parameters.items():
            nval = []
            for l in range(len(val)):
                v = unpack_value(val[l])  
                if v == 1:
                    nval.append(l) 
            if key == 'spin':
                nval = [vi+1 for vi in nval]
            self.cat_controls.data_system_parameters[key] = nval
        self.cat_controls.data_filename = unpack_value(self.cat_controls_var.data_filename)
        self.is_done = False

    def init_structures(self,structures,quantities,sort_by='volume'):
        """
        convert the current values regarding the reference data in GUI to BOPcat
        """
        self.refdata = CATData(controls=self.cat_controls)
        try:
            self.ref_atoms = self.refdata.get_ref_atoms(structures=structures
                                                   ,quantities=quantities
                                                   ,sort_by=sort_by)  
        except:
            return
        self.ref_data = self.refdata.get_ref_data()
        nstrucmax = int(unpack_value(self.nstrucmax))
        self.ref_atoms = self.ref_atoms[:nstrucmax]
        self.ref_data = self.ref_data[:nstrucmax]

    def init_opt_variables(self):
        """
        convert the current values for the optimization variables in GUI to BOPcat
        """
        opt_var = []
        abx_par = ['valenceelectrons','stonerintegral','onsitelevels'
                  ,'delta_onsitelevels','jii','valenceorbitals']
        abxs = self.modelbx.atomsbx
        for i in range(len(abxs)):
            var = {}
            var['atom'] = abxs[i].atom
            par = abxs[i].get_atomspar()
            for key, val in par.items():
                if key not in abx_par or val is None:
                #if val is None:
                    continue
                if isinstance(val,float) or isinstance(val,int):
                    val = [val]
                var[key] = [False]*len(val)
            opt_var.append(var) 
        bbxs = self.modelbx.bondsbx
        for i in range(len(bbxs)):
            var = {}
            var['bond']  = bbxs[i].bond
            par = bbxs[i].get_bondspar()
            par.update(bbxs[i].get_overlappar())
            par.update(bbxs[i].get_repetal())
            for key, val in par.items():
                if val is None:
                    continue
                var[key] = [False]*len(val)
            opt_var.append(var)
        self.cat_controls.opt_variables = opt_var    
        
    def update_modelbx(self):    
        """
        convert the current values of the model in GUI to BOPcat
        """
        if self.modelbx_var is None:
            return
        bbxs = self.modelbx_var.bondsbx
        for i in range(len(bbxs)):
            par = bbxs[i].get_bondspar()
            par.update(bbxs[i].get_overlappar())
            par.update(bbxs[i].get_repetal())
            for key, val in par.items():
                if val is None:
                    continue
                val = [unpack_value(v) for v in val]  
                self.modelbx.bondsbx[i].set_bondspar({key:val})   
        abxs = self.modelbx_var.atomsbx 
        for i in range(len(abxs)):
            par = abxs[i].get_atomspar()
            for key, val in par.items():
                if val is None:
                    continue
                if type(val) in [float,int]:
                    val = [val]
                val = [unpack_value(v) for v in val]
                self.modelbx.atomsbx[i].set_atomspar({key:val})
        infox = self.modelbx_var.infox_parameters
        for key, val in infox.items():
            if val is None:
                continue
            val = unpack_value(val)
            self.modelbx.infox_parameters.update({key:val})
            

    def update_modelbx_var(self):
        """
        convert the current values of the model to the GUI
        """
        bbxs = self.modelbx_var.bondsbx
        for i in range(len(bbxs)):
            par = bbxs[i].get_bondspar()
            par.update(bbxs[i].get_overlappar())
            par.update(bbxs[i].get_repetal())
            bpar = self.modelbx.bondsbx[i].get_bondspar()
            bpar.update(self.modelbx.bondsbx[i].get_overlappar())
            bpar.update(self.modelbx.bondsbx[i].get_repetal())
            for key, val in par.items():
                if val is None:
                    continue
                if type(bpar[key]) in [float,int]:
                    bpar[key] = [bpar[key]]
                for j in range(len(val)):
                    #if isinstance(val[j],str) or isinstance(val[j],float):
                    if not isinstance(val[j],tk.StringVar):
                        continue
                    val[j].set(str(bpar[key][j]))
        abxs = self.modelbx_var.atomsbx 
        for i in range(len(abxs)):
            par = abxs[i].get_atomspar()
            apar = self.modelbx.atomsbx[i].get_atomspar()
            for key, val in par.items():
                if val is None:
                    continue
                if type(val) in [float,int]:
                    val = [val]
                if type(apar[key]) in [float,int]:
                    apar[key] = [apar[key]]
                for j in range(len(val)):
                    #if isinstance(val[j],str) or isinstance(val[j],float):
                    if not isinstance(val[j],tk.StringVar):
                        continue
                    val[j].set(str(apar[key][j]))
        infox = self.modelbx_var.infox_parameters
        for key, val in infox.items():
            if val is None:
                continue
            #if type(val) in [str,float,int,bool]:
            if not isinstance(val,tk.StringVar):
                continue
            infox[key].set(str(self.modelbx.infox_parameters[key]))

    def rattle_model(self):
        """
        Rattle the model depending on current options in GUI
        """
        if not self.rattle:
            return
        self.rattle_options = (unpack_value(v) for v in self.rattle_options)
        var = self.cat_controls.opt_variables
        mode = self.rattle_options[0]
        fac = self.rattle_options[1]
        self.modelbx = self.modelbx.rattle(var,factor=mode,maxf=fac)

    def init_par(self):
        """
        get list of optimization parameters
        """ 
        self.par = gen_param_model(self.modelbx,self.cat_controls.opt_variables)
            
    def init_modelbx(self):
        """
        initialize the values of the model update the model from the GUI vaules
        """
        if self.modelbx_var is None:
            if self.cat_controls.model_pathtomodels is None:
                return 
            self.modelbx = read_modelsbx(filename=self.cat_controls.model_pathtomodels,model=self.cat_controls.model)[0]
            self.modelbx0 = self.modelbx.copy()
            self.init_opt_variables() 
        else:
            self.update_modelbx()
        self.init_par()
        # we also get modelbx with bopcat functions
        self.modelbx_func = self.modelbx.copy()
        self.modelbx_func.bond_parameters_to_functions() 
     
    def init_calc(self):
        """
        set up the calculator given the model and the reference structures
        """
        self.init_modelbx()
        self.calc = CATCalc(controls=self.cat_controls,model=self.modelbx)
        self.calc.set_atoms(self.ref_atoms)
        self.calc.set_model(self.modelbx)

    def init_kernel(self):
        """
        set up the BOPcat kernel
        """
        self.init_calc()
        opt_var = deepcopy(self.cat_controls.opt_variables)
        if self.cat_controls.opt_variables is not None:
          # reduce opt_variables (not necessary, just reduces loops)
          for i in range(len(self.cat_controls.opt_variables)-1,-1,-1):
            for key, val in self.cat_controls.opt_variables[i].items():
                if key.lower() in ['atom','bond']:
                    continue
                if True not in val:
                    del self.cat_controls.opt_variables[i][key]
            if len(self.cat_controls.opt_variables[i]) < 2:
                del self.cat_controls.opt_variables[i]    
        self.kernel = CATKernel(controls=self.cat_controls,calc=self.calc
                               ,ref_data=self.ref_data,gui_log=self.iter_file)
        self.cat_controls.opt_variables = opt_var
        
    def gen_queue(self):
        """
        set up the queue
        """
        qs = ['serial','cluster_serial','cluster_p16']
        Ns = [1,1,16]          
        N = Ns[qs.index(self.cat_controls.calculator_queue)] 
        Nin = self.cat_controls.calculator_nproc
        if Nin is None:
            Nin = 1
        if Nin%N != 0:
            print 'Invalid nproc: %d switching to default: %d'%(Nin,N)
            Nin = N             
        if self.cat_controls.calculator_queue == 'serial':
            q = self.cat_controls.calculator_queue 
            self.cat_controls.calculator_parallel = 'serial'
        elif self.cat_controls.calculator_queue == 'cluster_serial':
            q = cluster_serial(cores=Nin,qname='serial.q',pe='smp',runtime=self.runtime)
            self.cat_controls.calculator_parallel = 'multiprocessing'
        elif self.cat_controls.calculator_queue == 'cluster_p16':
            q = cluster_parallel(cores=Nin,qname='parallel16.q',pe='mpi16',runtime=self.runtime)
            self.cat_controls.calculator_parallel = 'multiprocessing'
        return q 
    
    def run(self):
        """
        execute the kernel
        """
        self.init_kernel()
        self.kernel.dump_min_model = os.path.join(os.getcwd(),'model_min.bx')
        #self.kernel.run()
        queue = self.gen_queue()
        self.proc = Process_catkernel(catkernel=self.kernel,queue=queue
        ,cores=self.cat_controls.calculator_nproc,directives=self.directives)
        self.proc.run()
        res = self.proc.get_results(wait=True)
        self.kernel = self.proc.get_kernel()._catkernel
        self.modelbx_opt = self.kernel.get_optimized_model()
        if self.modelbx_opt is not None:
            self.modelbx_opt.write('model_optimized.bx')
        # wait for the GUI to read iteration data
        time.sleep(10)
        self.is_done = True

    def clean(self):
        if self.proc is None:
            return 
        self.proc.clean()
        self.proc = None    
        os.system('rm %s'%self.iter_file)
         
    def stop(self):
        if self.proc is None:
            return 
        self.proc.stop()

    def copy(self):
        return deepcopy(self)

# auxilliary functions used in the GUI

def open_file(v,ftype,fext):
    """
    opening file
    """
    filename = askopenfilename(initialdir=os.getcwd(),filetypes =((ftype, fext),("All Files","*.*"))
                              ,title = "Choose a file.")
    v.set(filename)

def user_input(v):
    """
    setting user input
    """
    val = v.get()
    v.set(val)

def sconvert(s):
    """
    convert string to proper variable type
    """
    s = s.strip()
    if len(s) == 0:
        s = None
    elif s in ['-inf','inf']:
        s = float(s)
    elif s[0] in '-+0123456789.' and s[-1] in '-+0123456789.' and ('.' in s or 'e' in s.lower()):
        s = float(s)
    elif s[0] in '-+0123456789' and s[-1] in '-+0123456789':
        s = int(s)
    elif s.lower() in ['true','t','y']:
        s = True
    elif s.lower() in ['false','f','n']:
        s = False
    elif s in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~': 
        s = None
    elif s.lower() == 'none':
        s = None
    else:
        s = str(s)
    return s
        
def unpack_value(v):
    """
    convert user input (string) to proper variable type
    """
    try:
        val = v.get()
    except:
        return v
    if not isinstance(val,str):
        return val
    val = val.strip()
    if len(val) == 0:
        val = None
    elif val[0] in ['(','['] and val[-1] in [')',']']:
        #val = list(val)
        #val = val[1:len(val)-1]
        l = val.split(',')
        val  = [i.lstrip('(').lstrip('[').rstrip(')').rstrip(']') for i in l]
        val = [sconvert(s) for s in val]
        val = [s for s in val if s is not None]
    elif val[0] in ['{']  and val[-1] in ['}']:
        d = {}
        e = filter(str.isalnum,val.split("'"))
        vs = e[1::2]
        ks = e[0:2]
        d.update(zip(ks,vs))
        val= dict(d)
    elif val[0] == '=':
        exec("val %s"%val)
    else:
        val = sconvert(val) 
    return val
 
def init_entry_file(master,caption,place,ftype,fext,v=None):
    """
    creates entry fields on a master frame for opening files
    """
    if v is None:
        v = tk.StringVar()
    tk.Label(master, text = caption, width=20, anchor='e').grid(row=place[0],column=place[1],sticky='news')
    num1 = tk.Entry(master, textvariable=v, width=20)
    num1.grid(row=place[0],column=place[1]+1, sticky='news', columnspan=1)
    num1.bind('<Button>',lambda event, v=v:open_file(v,ftype,fext)) 
    return v

def init_entry_input(master,caption,place,v=None,default=None):
    """
    creates entry fields on a master frame for user inputs
    """
    if v is None:
        v = tk.StringVar()
    if default is not None:
        v.set(str(default))
    tk.Label(master, text = caption, width=20, anchor='e').grid(row=place[0],column=place[1],sticky='news')
    num1 = tk.Entry(master, textvariable=v)
    num1.grid(row=place[0],column=place[1]+1,sticky='news',columnspan=1)
    num1.bind('<KeyPress-Return>',lambda event, v=v:user_input(v)) 
    return v

def init_checkbutton(master,caption,place,v=None):  
    """
    creates entry fields on a master frame for check buttons
    """
    if v is None:
        v = tk.IntVar()
    tk.Checkbutton(master, text=caption, variable=v).grid(row=place[0], column=place[1], sticky = 'w')
    return v
        
def resize_frame(frame):
    """
    resize the frame app, !does not work
    """
    frames = frame.winfo_children()
    N = [f for f in frames if isinstance(f,tk.Frame)]
    N = len(N)
    for i in range(N):
        frame.grid_columnconfigure(i,weight=1)
        frame.grid_rowconfigure(i,weight=1)
    for i in range(len(frames)):
        if isinstance(frames[i], list):
            resize_frames(frames[i])
        else:
            frames[i].grid_columnconfigure(0,weight=1)
            frames[i].grid_rowconfigure(0,weight=1)

def pathtologo():
     from bopcat.variables import homedirectory
     #from variables import homedirectory
     return os.path.join(homedirectory(),'gui/logo.gif')

HEAD1_FONT = ("Helvetica", 20, 'bold')
HEAD2_FONT = ("Helvetica", 16, 'bold')
#Fig = Figure(figsize=(5,5), dpi=100)
#Ax = Fig.add_subplot(111)

class BOPcatapp(tk.Tk):
    """
    The BOPcat application. It contains five pages so far.
    HomeFrame             : Set up general parameters and visualize single
                            optimization
    ReferencedataFrame    : Choose and customize fit/test structures 
    ModelFrame            : Set up optimization variables, visualize model, etc.
    OptCalcFrame          : Input optimizer options and BOPfox settings
    ParameterizationFrame : Tool for defining and running optimization protocol
    """
    def __init__(self,*args, **kwargs):
        tk.Tk.__init__(self,*args,**kwargs)

        tk.Tk.wm_title(self,"BOPcat ver. 0.1 @ %s"%os.uname()[1])

        #self.geometry("1000x800")

        self.appframe = tk.Frame(self)
        self.frames = [HomeFrame,ReferencedataFrame,ModelFrame,OptCalcFrame,ParameterizationFrame]
        
        for i in range(len(self.frames)):
            frame = self.frames[i](self.appframe,self)
            self.frames[i] = frame
            frame.grid(row=0, column=0, sticky= "nsew")

        self.init_window()

    def init_window(self):
        self.showLogo()
        self.showHeader()
        self.appframe.grid(row=0,column=0,sticky='nsew')

    def showLogo(self):
        self.startframe = tk.Frame(self.appframe)
        image = tk.PhotoImage(file=pathtologo())
        label = tk.Label(self.startframe, image=image, width=1000)
        label.image = image
        label.grid(row=0,column=0, sticky='nsew',padx=0,pady=30)

    def showHeader(self):
        try:
            catv = bopcat_version()
        except:
            catv = 'unknown'
        try:
            bopv = get_bopfox_version()
        except:
            bopv = 'unknown'
        try:
            asev = ase.__version__ + ' (%s)'%os.path.dirname(ase.__file__)
        except:
            
            asev = 'unknown'
        
        head = """

                   Bond-Order Potential construction, assessment and testing
                   version: %s

                   \xc2\xa9 2018 ICAMS 
                   Alvin Noe Ladines (ladinesalvinnoe@gmail.com)
               
                   BOPfox version : %s
                   ASE version    : %s


                   """%(catv,bopv,asev)
        label = tk.Label(self.startframe,text=head,font=('Helvetica',11),justify='left',anchor='w') 
        label.grid(row=1,column=0,sticky='ew')
        if 'unknown' in (catv.lower(),bopv.lower(),asev.lower()):
            if catv.lower() == 'unknown':
                label = tk.Label(self.startframe,text='BOPcat not found.',font=('Helvetica',11),justify=tk.LEFT) 
            elif bopv.lower() == 'unknown':
                label = tk.Label(self.startframe,text='BOPfox not found.',font=('Helvetica',11),justify=tk.LEFT) 
            elif asev.lower() == 'unknown':
                label = tk.Label(self.startframe,text='ASE not found.',font=('Helvetica',11),justify=tk.LEFT) 
            label.grid(row=2,column=0,sticky='ew')
            button = tk.Button(self.startframe,text='Exit',command=self.client_exit)   
            button.grid(row=3,column=0)
        else:
            tk.Button(self.startframe,text='Continue',font=HEAD2_FONT,command=lambda i=0: self.show_frame(i)).grid(row=4,column=0)
            label = tk.Label(self.startframe,text='By continuing, you agree to the terms and conditions.', fg='red')
            label.grid(row=5,column=0,sticky='ew')
            label.bind('<Button-1>',lambda event : self.show_license())
        self.startframe.grid(row=0,column=0,sticky='news')

    def show_license(self):
        w = tk.Toplevel(self)
        w.wm_title("Terms and conditions")
        txt = '''
                 Copyright 2018 ICAMS

                 BOPcat is a free software. Distribution and/or modifications are
                 governed by the terms of the GNU General Public License as published
                 by the Free Software Foundation.

                 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
                 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
                 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
                 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
                 ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
                 CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
                 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'''
        tk.Label(w,text=txt,justify='left').grid(row=0,column=0,sticky='nsew')
        tk.Button(w,text='Agree',command=w.destroy).grid(row=1,column=0)
        tk.Button(w,text='Exit',command=sys.exit).grid(row=1,column=1) 
        
    def show_frame(self,i):
        frame = self.frames[i]
        frame.tkraise()

    def client_exit(self):
        self.kill_client = True
        exit()

class HomeFrame(tk.Frame):
    """
    Main Frame for defining general options
    masterframe is the application frame and master is the BOPcat app
    """
    def __init__(self, masterframe, master):
        
        tk.Frame.__init__(self, masterframe)   

        self.master = master

        # define interface to BOPcat
        self.BOPCAT = BOPcatrun()
        # define worker for threading
        self.worker_runbopcat = None

        self.init_window()
       

    def init_window(self):
        # set up menu
        menu = tk.Menu(self.master)

        # file option
        ffile = tk.Menu(menu)
        # exit and open(!does not exist)
        ffile.add_command(label="Exit", command=self.client_exit)
        ffile.add_command(label="Open", command=self.open_old)
        menu.add_cascade(label="File", menu=ffile)

        # view option
        fview = tk.Menu(menu)
        fview.add_command(label="Home", command=lambda arg=0: self.master.show_frame(arg))
        fview.add_command(label="Reference data", command=lambda arg=1: self.master.show_frame(arg))
        fview.add_command(label="Model", command=lambda arg=2: self.master.show_frame(arg))
        fview.add_command(label="Optimizer", command=lambda arg=3: self.master.show_frame(arg))
        fview.add_command(label="Calculator", command=lambda arg=3: self.master.show_frame(arg))
        menu.add_cascade(label="View", menu=fview)

        # run option 
        frun = tk.Menu(menu)
        frun.add_command(label="Run", command=self.runBOPcat)
        frun.add_command(label="Stop", command=self.stopBOPcat)
        menu.add_cascade(label="Run", menu=frun)

        # tools option
        ftools = tk.Menu(menu)
        ftools.add_command(label='Parameterization',command=lambda arg=4:self.master.show_frame(arg))
        menu.add_cascade(label='Tools',menu=ftools)
        
        # put everything together
        self.master.config(menu=menu)
      
        # set up the sub frames
        self.init_frames()

        self.kill_client = False
 
        # iteration data for plotting
        self.iteration_data = []
        self.iteration_vars = [] 
        self.itergraph = GraphFrame(self,('Iteration','rms'))
        self.itergraph.grid(row=3,column=0,columnspan=1,sticky='ew')

        self.start_time = 0.0
        
    def open_old(self):
        """
        Not yet implemented. will have to think about format and what to save.
        """
        return

    def destroy_window_element(self,w):
        """
        defines command to destroy pop up window for defining element and subsequently
        creates button for element for editing
        """
        w.destroy()
        # upon destroying window, create button in Home to display/edit element
        N = len(self.BOPCAT.cat_controls_var.elements)-1
        elem = unpack_value(self.BOPCAT.cat_controls_var.elements[N])
        b = tk.Button(self.elemframe,text=elem,command=lambda arg=N:self.edit_element(arg)).grid(row=0,column=2+N)

    def create_window_element(self):
        """
        creates a window for defining element properties
        """
        w = tk.Toplevel(self)
        w.wm_title("Add element")
        ve = tk.StringVar()
        ve.set('Nb')
        vr = tk.StringVar()
        vr.set('-0.593')
        vm = tk.StringVar()
        vm.set('1')
        elem = init_entry_input(w,'Element',(0,1),v=ve)
        if self.BOPCAT.cat_controls_var.elements is None:
            self.BOPCAT.cat_controls_var.elements = [elem]
        else:
            self.BOPCAT.cat_controls_var.elements.append(elem)
        N = max(0,len(self.BOPCAT.cat_controls_var.elements)-1)
        ref_ene = init_entry_input(w,'Reference energy',(1,1),v=vr)
        if self.BOPCAT.cat_controls_var.data_free_atom_energies is None:
            self.BOPCAT.cat_controls_var.data_free_atom_energies = {N:ref_ene}
        else:
            self.BOPCAT.cat_controls_var.data_free_atom_energies[N] = ref_ene
        ini_mag = init_entry_input(w,'Initial mag. mom.',(2,1),v=vm)
        if self.BOPCAT.cat_controls_var.data_ini_magmoms is None:
            self.BOPCAT.cat_controls_var.data_ini_magmoms = {N:ini_mag}
        else:
            self.BOPCAT.cat_controls_var.data_ini_magmoms[N] = ini_mag
        tk.Button(w,text='Done',command=lambda arg=w:self.destroy_window_element(arg)).grid(row=3,column=1)
        
    def edit_element(self,i):
        """
        edit element properties
        """
        w = tk.Toplevel(self)
        w.wm_title("Edit element")
        elem = init_entry_input(w,'Element',(0,1),self.BOPCAT.cat_controls_var.elements[i])
        self.BOPCAT.cat_controls_var.elements[i] = elem
        ref_ene = init_entry_input(w,'Reference energy',(1,1),self.BOPCAT.cat_controls_var.data_free_atom_energies[i])
        self.BOPCAT.cat_controls_var.data_free_atom_energies[i] = ref_ene
        ini_mag = init_entry_input(w,'Initial mag. mom.',(2,1),self.BOPCAT.cat_controls_var.data_ini_magmoms[i])
        self.BOPCAT.cat_controls_var.data_ini_magmoms[i] = ini_mag
        tk.Button(w,text='Done',command=w.destroy).grid(row=3,column=1)

    def init_elemframe(self):
        """
        set up Elements frame
        """
        self.elemframe = tk.Frame(self,bg='dark green')
        tk.Label(self.elemframe,text='Elements',fg='dark green',font=HEAD2_FONT,width=20).grid(row=0,column=0,sticky='we')
        tk.Button(self.elemframe,text='Add', command=self.create_window_element).grid(row=0,column=1)
        self.elemframe.grid(row=0,column=0,columnspan=2,sticky='we') 

    def init_data_filename(self,place):
        """
        create field for filename of fit file
        """
        value = init_entry_file(self.refframe,'Data filename', place, 'fit file', '*.fit')
        self.BOPCAT.cat_controls_var.data_filename = value

    def init_refframe(self):
        """
        set up Reference data frame
        """
        self.refframe = tk.Frame(self,bg='dark blue')
        tk.Label(self.refframe,text='Reference data',fg='dark blue',font=HEAD2_FONT, width=42).grid(row=0,column=0,columnspan=3,sticky='ew')
        self.init_data_filename((1,0))
        tk.Button(self.refframe,text='More',command=lambda arg=1:self.master.show_frame(arg)).grid(row=2,column=2,sticky='nsew')
        self.refframe.grid(row=1,column=0,columnspan=1,sticky='ew')

    def init_model_pathtomodels(self,place):
        """
        create field for model file
        """ 
        value = init_entry_file(self.modframe,'Modelsbx filename', place, 'modelsbx file', '*.bx')
        self.BOPCAT.cat_controls_var.model_pathtomodels = value 

    def init_model(self,place):
        """
        create field for model
        """
        value = init_entry_input(self.modframe,'Model', place, default='orthogonal_exponential-3_gaussian-3')
        self.BOPCAT.cat_controls_var.model = value

    def init_modframe(self):
        """
        set up Model frame
        """ 
        self.modframe = tk.Frame(self,bg='dark red')
        tk.Label(self.modframe, text = 'Model',fg='dark red',font=HEAD2_FONT, width=42).grid(row=0,column=0,columnspan=3,sticky='ew')
        self.init_model_pathtomodels((1,0))
        self.init_model((2,0))
        tk.Button(self.modframe,text='More',command=lambda arg=2:self.master.show_frame(arg)).grid(row=3,column=2,sticky='nsew')
        self.modframe.grid(row=2,column=0,columnspan=1,sticky='ew')

    def init_optimizer(self,place):
        """
        create field for optimizer
        """
        choices = {'leastsq','least_squares','nelder-mead','powell','cg','bfgs','ncg','l-bfgs-b'
                  ,'tnc','slsqp','differential_evolution','basinhopping','brute'}
        v = tk.StringVar()
        v.set('leastsq')
        popup = tk.OptionMenu(self.optframe,v,*choices)
        popup.config(width=20)
        tk.Label(self.optframe,text='optimizer',width=20, anchor='e').grid(row=place[0],column=place[1],sticky='nsew')
        popup.grid(row=place[0],column=place[1]+1,sticky='we')          
        self.BOPCAT.cat_controls_var.opt_optimizer = v

    def init_optframe(self):
        """
        set up Optimizer frame
        """
        self.optframe = tk.Frame(self,bg='slate gray')
        tk.Label(self.optframe,text='Optimizer',fg='slate gray',font=HEAD2_FONT, width=42).grid(row=0,column=0,columnspan=3,sticky='ew')
        self.init_optimizer((1,0))
        tk.Button(self.optframe,text='More',command=lambda arg=3:self.master.show_frame(arg)).grid(row=3,column=2,sticky='nsew')
        self.optframe.grid(row=1,column=1,columnspan=1,sticky='ew')

    def init_calculator_nproc(self,place):
        """
        create field for number of processes
        """
        value = init_entry_input(self.calcframe,'nproc', place)
        self.BOPCAT.cat_controls_var.calculator_nproc = value

    def init_calculator_queue(self,place):
        """
        create field for queue
        """
        choices = {'serial','cluster_serial','cluster_p16'}
        v = tk.StringVar()
        v.set('serial')
        popup = tk.OptionMenu(self.calcframe,v,*choices)
        popup.config(width=20)
        tk.Label(self.calcframe,text='queue',width=20, anchor='e').grid(row=place[0],column=place[1],sticky='nsew')
        popup.grid(row=place[0],column=place[1]+1,sticky='we')          
        self.BOPCAT.cat_controls_var.calculator_queue = v

    def init_calcframe(self):
        """
        set up up Calculator frame
        """
        self.calcframe = tk.Frame(self,bg='dark orange')
        tk.Label(self.calcframe, text = 'Calculator',fg='dark orange',font=HEAD2_FONT, width=42).grid(row=0,column=0,columnspan=3,sticky='ew')
        self.init_calculator_nproc((1,0))
        self.init_calculator_queue((2,0))
        tk.Button(self.calcframe,text='More',command=lambda arg=3:self.master.show_frame(arg)).grid(row=3,column=2,sticky='nsew')
        self.calcframe.grid(row=2,column=1,columnspan=1,sticky='ew')

    def toggle_run(self):
        """
        command to start or stop BOPcat run
        """
        if self.stat_button['text'] == 'Run':
            self.stat_button.config(bg='red')
            self.stat_button.config(text='Stop')
            self.runBOPcat()
        else:
            self.stat_button.config(bg='green')
            self.stat_button.config(text='Run')
            self.stopBOPcat() 
        
    def init_iterframe(self):
        """
        set up the iteration frame to show parameters during optimization
        """
        self.iterframe = tk.Frame(self,bg='lime green')
        self.stat_button = tk.Button(self.iterframe,text='Run',font=HEAD2_FONT,bg='green', command=self.toggle_run)
        self.stat_button.grid(row=0,column=0,columnspan=1,sticky='w')
        self.timer = tk.Label(self.iterframe,text='Runtime: 0.0',width=20)
        self.viewrms = tk.Label(self.iterframe,text='rms: 0.0',width=20)
        self.timer.grid(row=0,column=1,sticky='ew')
        self.viewrms.grid(row=0,column=2,sticky='ew')
        self.iterframe.grid(row=3,column=1,sticky='ew')

    def init_frames(self):
        """
        build all frames
        """
        self.init_elemframe()
        self.init_refframe()
        self.init_modframe()
        self.init_optframe()
        self.init_calcframe()
        self.init_iterframe()
        for i in range(100):
            self.grid_columnconfigure(i,weight=1)
            
    def update_status(self):
        """
        update the status of BOPcat run
        """
        if self.start_time > 0.0:
            self.timer.config(text='Runtime: %8.2f'%(time.time()-self.start_time))
            if len(self.iteration_data) > 0:
                self.viewrms.config(text='rms :%12.8f'%self.iteration_data[-1][1])
            if self.BOPCAT.is_done:
                self.toggle_run()
                self.start_time = 0.0
        self.master.after(5000,self.update_status)

    def init_par(self):
        """
        initialize the parameter display
        """
        p0 = self.BOPCAT.par
        self.iteration_vars = [tk.StringVar() for i in range(len(p0))]
        par_keys = get_par_keys(self.BOPCAT.cat_controls.opt_variables)
        tk.Label(self.iterframe,text='Parameters',width=20).grid(row=1,column=0)
        tk.Label(self.iterframe,text='Start',width=20).grid(row=1,column=1)
        tk.Label(self.iterframe,text='Current',width=20).grid(row=1,column=2)
        pmax = (max(20,len(p0)))
        for i in range(pmax):
          if i < len(p0):
            self.iteration_vars[i].set(str(p0[i]))
            tk.Label(self.iterframe,text=par_keys[i],width=20).grid(row=2+i,column=0)
            tk.Label(self.iterframe,text=str(p0[i]),width=20).grid(row=2+i,column=1)
            tk.Label(self.iterframe,textvariable=self.iteration_vars[i],width=20).grid(row=2+i,column=2)
          else:
            tk.Label(self.iterframe,text=' ',width=20).grid(row=2+i,column=0)
            tk.Label(self.iterframe,text=' ',width=20).grid(row=2+i,column=1)
            tk.Label(self.iterframe,text=' ',width=20).grid(row=2+i,column=2)

        t = threading.Thread(target=self.iter_par)
        t.start()

    def runBOPcat(self):
        """
        procedure for running the BOPcat
        """
        # initialization of catcontrols does not work in thread
        self.BOPCAT.init_cat_controls()
        self.BOPCAT.init_modelbx()
        self.init_par()
        self.worker_runbopcat = threading.Thread(target=self.BOPCAT.run)
        self.start_time = time.time()
        self.worker_runbopcat.start()
        self.read_iteration()

    def iter_par(self):
        """
        update values of the parameters
        """
        while not self.BOPCAT.is_done:
            time.sleep(2)
            p1 = self.iteration_data
            if len(p1) < 1:
                continue
            p1 = p1[-1][2:]
            for i in range(len(p1)):
                self.iteration_vars[i].set(str(p1[i]))

    def stopBOPcat(self):
        """
        terminate BOPcat run
        """
        if self.worker_runbopcat is None:
            return 
        self.BOPCAT.stop()
        self.worker_runbopcat._stop_req = True  
        self.worker_runbopcat.join(timeout=1)  

    def client_exit(self):
        """
        exit application
        """
        self.kill_client = True
        exit()

    def read_iteration(self):
        """
        run reading of iteration data on separate thread
        """
        t = threading.Thread(target=self._iteration)
        t.start()

    def _iteration(self):
        """
        read iteration data from iter_file written by catkernel
        """
        self.iteration_data = [] 
        while not self.BOPCAT.is_done:
          try:
            with open(self.BOPCAT.iter_file) as f:
                lines = f.readlines()
                for i in range(len(self.iteration_data),len(lines)):
                    line = lines[i].split()
                    line = [float(l) for l in line] 
                    self.iteration_data.append(line)
          except:
            continue
          # read only every 5 s
          time.sleep(5)
        # clean up
        self.BOPCAT.clean()

class ReferencedataFrame(tk.Frame):
    """
    Frame to handle reference data selection     
    """
    def __init__(self,masterframe,master):
        tk.Frame.__init__(self,masterframe)  
        self.master = master
        self.BOPCAT = master.frames[0].BOPCAT
        label = tk.Label(self, text="Reference Data", font=HEAD1_FONT, fg='dark green')
        label.grid(row=0,column=0,columnspan=2,sticky='nsew')
        #button = tk.Button(self,text='Home',command=lambda: master.show_frame(0))

        self.structures = []
        self.quantities = []
        self.filters = ['quantities','spin','calculation_type']
        self.qlist = ['energy','forces','stress']
        self.init_frames()
        self.lframe = None
        self.weight_vars = []

    def create_grid_structure(self):
        """
        create field for input structures
        """
        N = len(self.structures)
        value = init_entry_input(self.strucsframe,'Structure %d'%N, (1+N,0))
        self.structures.append(value)

    def init_strucsort(self):
        """
        create field for sorting of structures
        supports sorting by energy vollume formation energy and random
        """
        choices = {'energy','volume','formation energy','random'}
        v = tk.StringVar()
        v.set('energy')
        popup = tk.OptionMenu(self.strucsframe,v,*choices)
        #popup.config(width=16)
        tk.Label(self.strucsframe,text='sort by',width=20, anchor='e').grid(row=30,column=2,sticky='ew')
        popup.grid(row=30,column=3,sticky='ew')          
        self.sort_by = v

    def init_strucsframe(self):
        """
        set up frame for structure selection
        """
        self.strucsframe = tk.Frame(self,bg='dark green')
        tk.Label(self.strucsframe,text = 'Fit Structures', font=HEAD2_FONT).grid(row=0,column=0)
        tk.Button(self.strucsframe,text='Add', command=self.create_grid_structure).grid(row=0,column=1,sticky='ew')
        tk.Button(self.strucsframe,text='Load structures', command=self.show_structures).grid(row=25,column=0,columnspan=2,sticky='ew')
        self.init_strucsort()
        self.BOPCAT.nstrucmax = init_entry_input(self.strucsframe,'max nstrucs', (31,2), default=10000)    
        self.strucsframe.grid(row=1,column=0,sticky='nsew')

    def create_filter(self,name,options):
        """
        create field for filter
        """
        row = 1 + self.filters.index(name)
        tk.Label(self.filtersframe, text = name, width=20, anchor='e').grid(row=row,column=0,sticky='ew')
        var = []
        for i in range(len(options)):
            v = init_checkbutton(self.filtersframe,options[i],(row,1+i))
            var.append(v)
            if i == 0:
                v.set(1)
        return var

    def init_filtersframe(self):
        """
        set up frame for structure filtering options
        """
        self.filtersframe = tk.Frame(self,bg='dark green')
        tk.Label(self.filtersframe,text = 'Filters',font=HEAD2_FONT).grid(row=0,column=0,sticky='ew')
        var = self.create_filter('quantities',self.qlist)
        self.quantities = var 
        var = self.create_filter('spin',[1,2])
        self.BOPCAT.cat_controls_var.data_system_parameters['spin'] = var
        self.filtersframe.grid(row=1,column=1,sticky='nsew')

    def init_graphframe(self):
        """
        set up frame for bond lengths histogram
        """
        self.hist = GraphFrame(self,('$R_{ij}$','Count'))
        self.hist.grid(row=2,column=0)

    def init_tableframe(self):
        """
        set up frame for list of structures in table
        """
        self.tableframe = tk.Frame(self)
        self.tableframe.grid(row=2,column=1,sticky='e')
        #self.tableframe.grid_propagate(False)  
        tk.Button(self.tableframe,text='Show details',command=self.list_structures,width=65).grid(row=0,column=0,sticky='ew') 

        self.table = tk.Canvas(self.tableframe,width=500,height=600)
        self.table.grid(row=1,column=0,sticky='nw')
 
        vscroll = tk.Scrollbar(self.tableframe,orient=tk.VERTICAL,command=self.table.yview)
        vscroll.grid(row=1,column=1,sticky='nsew')
 
        hscroll = tk.Scrollbar(self.tableframe,orient=tk.HORIZONTAL,command=self.table.xview)
        hscroll.grid(row=2,column=0,sticky='nsew')


        self.table.config(yscrollcommand = vscroll.set)
        self.table.config(xscrollcommand = hscroll.set)
        #self.table.config(scrollregion=(400,400,800,800))
 
    def init_frames(self):
        """
        set up all frames
        """
        self.init_strucsframe()
        self.init_filtersframe()
        self.init_graphframe()
        self.init_tableframe()

    def set_weight(self,iat,val):
        """
        set weight to structure. can be float, None or formula
        which is executed. accepted variables are ENERGY, NATOM
        MAXFORCE, MAXTRESS, VOLUME. Use np or math for math operation.

        example : =np.exp(-0.5*(ENERGY[i]/NATOM[i]-ENERGY[0]/NATOM[0]))
        
        """
        if val[0] != '=':
            print "Cannot set weight. String should start with '='"
            return
        atoms = self.BOPCAT.ref_atoms
        if 'ENERGY' in val:
            ENERGY = np.array([at.get_potential_energy() for at in atoms])
        if 'NATOM' in val:
            NATOM = np.array([len(at) for at in atoms])
        if 'MAXFORCE' in val:
            MAXFORCE = np.array([np.amax(np.abs(at.get_forces())) for at in atoms])
        if 'MAXSTRESS' in val:
            MAXSTRESS = np.array([np.amax(np.abs(at.get_stress())) for at in atoms])
        if 'VOLUME' in val:
            VOLUME = np.array([at.get_volume() for at in atoms])
        STRUC_WEIGHTS = [0]*len(atoms)
        for i in range(len(STRUC_WEIGHTS)):
            exec("STRUC_WEIGHTS[i]%s"%val)
        for i in range(iat,len(atoms)):
            self.weight_vars[i].set(str(STRUC_WEIGHTS[i]))
            atoms[i].info['weight'] = STRUC_WEIGHTS[i]

    def update_atom_info(self,v,i,key):
        """
        update atom.info[key] with user input
        """
        val = unpack_value(v)
        if key == 'Structure':
            self.BOPCAT.ref_atoms[i].info['strucname'] = val
        elif key == 'Weight':
            if isinstance(val,str):
                self.set_weight(i,val)
            else:
                self.BOPCAT.ref_atoms[i].info['weight'] = val

    def get_atom_info(self,atom,key):
        """
        extract current atom.info[key]
        """
        if key == 'Structure':
            out = atom.info['strucname']
        elif key == 'N atoms':
            out = len(atom)
        elif key == 'Weight':
            if atom.info.has_key('weight'):
                out = atom.info['weight']
            else:
                out = None
        elif key == 'Property':
            out = atom.info['required_property']
        elif key == 'Volume':
            out = atom.get_volume()/len(atom)
        elif key == 'Value':
            p = atom.info['required_property']
            if p == 'energy':
                out = atom.get_potential_energy()
            elif p == 'forces':
                out = np.amax(np.abs(atom.get_forces())) 
            elif p == 'stress':
                out = np.amax(np.abs(atom.get_stress()))
            else:
                out = None
        else:
            out = None
        return out    

    def list_structures(self):
        """
        generate entries of structure properties for the table
        currently includes only structure name, volume, N atoms, 
        required property, value(maximum value for forces and stress),
        and optimization weight
        """
        atoms = self.BOPCAT.ref_atoms
        if self.lframe is not None:
            for w in self.lframe.children.values():
                w.grid_forget()
            self.table.delete(self.lframe)
        self.lframe = tk.Frame(self.table)
        self.table.create_window((0,0),window=self.lframe,anchor='w')
        table_infos = ['Structure','Volume', 'N atoms', 'Property', 'Value', 'Weight']
        table_widths= [15,8,8,8,8,8]
        for i in range(len(table_infos)):
            tk.Label(self.lframe,text=table_infos[i],width=table_widths[i]).grid(row=0,column=i)

        labels = [[None for j in range(len(table_infos))] for i in range(len(atoms))]
        #self.table_vars = [[tk.StringVar() for j in range(len(table_infos))] for i in range(len(atoms))]
        self.weight_vars = [] 
        for i in range(len(atoms)):
            for j in range(len(table_infos)):
                info = self.get_atom_info(atoms[i],table_infos[j])
                v = tk.StringVar()
                v.set(str(info))   
                labels[i][j] = tk.Entry(self.lframe,textvariable=v,width=table_widths[j])
                labels[i][j].bind('<KeyPress-Return>',lambda event, v =v, i=i,key=table_infos[j]:self.update_atom_info(v,i,key))    
                if table_infos[j] == 'Weight':
                     self.weight_vars.append(v)
                labels[i][j].grid(row=1+i,column=j,sticky='nsew')
        self.lframe.update_idletasks()        
        #minh = min(30,len(atoms))
        #wm = sum([labels[0][j].winfo_width() for j in range(6)])
        #hm = sum([labels[i][0].winfo_height() for i in range(minh)])
        #self.tableframe.config(width=wm,height=hm)
        #self.tableframe.config(width=800,height=1000)
        self.table.config(scrollregion=self.table.bbox('all'))

    def get_rijs(self,atoms):
        """
        calculate bond lengths in list of atoms for histogram
        distance is calculated between unit cell and repeated cell
        """
        dist = []
        maxcut = 6.0
        for at in atoms:
            at = at.copy()
            rep = int((200/len(at))**(1/3.))
            atr = at.repeat(rep)
            for i in range(len(at)):
                for j in range(len(atr)):
                    if i >= j:
                        continue
                    d = atr.get_distance(i,j,mic=False)
                    if d < maxcut:
                        dist.append(d)
        return dist

    def show_structures(self):
        """
        generate reference atoms and data together with histogram
        """
        structures = [unpack_value(v) for v in self.structures]
        structures = [v for v in structures if v is not None]
        quantities = [int(unpack_value(v)) for v in self.quantities]
        quantities = [self.qlist[i] for i in range(len(quantities)) if quantities[i] == 1]
        sort_by = unpack_value(self.sort_by)
        self.BOPCAT.init_cat_controls()
        self.BOPCAT.init_structures(structures,quantities,sort_by)
        ref_atoms = self.BOPCAT.ref_atoms  
        tk.Label(self.strucsframe,text = 'Found %d structures.'%len(ref_atoms)).grid(row=31,column=1)
        rijs = self.get_rijs(ref_atoms)  
        self.show_histogram(rijs)       
        # put in table but too slow so ask insted user to display
        #self.list_structures() 

    def show_histogram(self,data):
        """
        generate histogram of bond lengths
        """
        self.hist.figsubplot.hist(data,alpha=0.8)
        self.hist.canvas.draw()
        
class ModelFrame(tk.Frame):
    """
    Frame to handle model
    """
    def __init__(self,masterframe,master):
        tk.Frame.__init__(self,masterframe)  
        self.master = master
        self.BOPCAT = master.frames[0].BOPCAT
        label = tk.Label(self, text="Model", font=HEAD1_FONT, fg='dark red')
        label.grid(row=0,column=0,columnspan=2,sticky='ew')
        #label.pack(pady=10,padx=10) 
        #button = tk.Button(self,text='Home',command=lambda: self.master.show_frame(0))
        #button.grid(row=1,column=0)

        self.status_plot = {}
        self.lines_plot = {}
        self.lines_test = []
        self.cur_bi = 0
        self.lframe = None    
        self.init_window()
        #self.init_graph()


    def init_window(self):
        """
        initialize main frames
        """
        # frame1 for main buttons
        self.frame1 = tk.Frame(self,bg='dark red')
        self.frame1.grid(row=1,column=0,sticky='ew',columnspan=2)
        self.loadvar = tk.StringVar()
        self.loadvar.set("Load")
        tk.Button(self.frame1,textvariable=self.loadvar,command=self.load_model,font=HEAD2_FONT).grid(row=0,column=0,sticky='w')        
        tk.Button(self.frame1,text='Restore',command=self.restore_model,font=HEAD2_FONT).grid(row=0,column=1,sticky='w')
        self.init_rattle()
        # subframe for atoms and bonds button 
        self.subframe = tk.Frame(self)
        self.subframe.grid(row=2,column=0,sticky='ew',columnspan=2,rowspan=1)
        # table frame for parameters table
        self.tableframe = tk.Frame(self)
        self.tableframe.grid(row=3,column=0,sticky='nsew',pady=(10,10),columnspan=2)   
        self.init_tableframe()
        # plot model functions
        self.mbxgraph = GraphFrame(self,('$R_{ij}$ $(\AA)$','$\\beta/O/\Phi$ (eV)'))
        tk.Button(self.mbxgraph,text='Plot',font=HEAD2_FONT,command=lambda: self.plot_bond()).pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        self.mbxgraph.grid(row=4,column=0,sticky='ew')
        # plot energy-volume 
        self.evgraph = GraphFrame(self,('$V$ $(\AA^3)$','$E$/at (eV)'))
        test_button = tk.Button(self.evgraph,text='Test',font=HEAD2_FONT,command=lambda: self.test_model())
        test_button.pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        test_button.bind('<Button-3>',lambda event: self.reset_ev())
        self.evgraph.grid(row=4,column=1,sticky='ew')

    def init_rattle(self):
        """
        initialize Rattle button
        to enable rattling, button should be armed (red) by double left click
        """
        self.rattle_button = tk.Button(self.frame1,text='Rattle',bg='green',command=self.rattle_model,font=HEAD2_FONT)

        self.rattle_button.bind('<Double-Button-1>', lambda event: self.toggle_rattle())
        self.rattle_button.grid(row=0,column=2,sticky='w')
        
        choices = {'gaussian','uniform'}
        v1 = tk.StringVar()
        v1.set('gaussian')
        popup = tk.OptionMenu(self.frame1,v1,*choices)
        tk.Label(self.frame1,text='mode',width=20,anchor='e').grid(row=0,column=3,sticky='nsew')
        popup.grid(row=0,column=4,sticky='nsew')          

        v2 = init_entry_input(self.frame1,'factor',(0,5),default=0.1)

        self.BOPCAT.rattle_options = (v1,v2)

    def toggle_rattle(self):
        cur = self.rattle_button['bg']
        if cur == 'green':
            self.rattle_button['bg']  = 'red'
            self.BOPCAT.rattle = True
        elif cur == 'red':
            self.rattle_button['bg'] = 'green'
            self.BOPCAT.rattle = False

    def load_model(self):
        """
        command to load model parameters
        """
        if self.loadvar.get() == "Load":
            self.BOPCAT.init_cat_controls()
            self.BOPCAT.init_modelbx()
            self.BOPCAT.modelbx_var = self.BOPCAT.modelbx.copy()
            mbx = self.BOPCAT.modelbx
            abxs = mbx.atomsbx
            for i in range(len(abxs)):
                a = abxs[i].atom
                if isinstance(a,str):
                    a = [a]
                v = tk.StringVar()
                v.set(a[0])
                button = tk.Button(self.subframe,text=a,width=3,command=lambda arg=i:self.show_atompar(arg))
                button.grid(row=1,column=i)
            self.lframes = [None]*len(abxs)
            bbxs = mbx.bondsbx
            for i in range(len(bbxs)):
                b = bbxs[i].bond   
                b = '%s-%s'%(b[0],b[1])
                v = tk.StringVar()
                v.set(b)        
                button = tk.Button(self.subframe,text=b,width=3,command=lambda arg=i+len(abxs):self.show_bondpar(arg))
                button.grid(row=1,column=i+len(abxs))
            self.lframes += [None]*len(bbxs)
            button = tk.Button(self.subframe,text='infox',width=3,command=lambda :self.show_infoxpar()) 
            button.grid(row=1,column=len(abxs)+len(bbxs))
            self.loadvar.set('Clear')
        elif self.loadvar.get() == "Clear":
            self.clear_model()

    def clear_model(self):
        """
        Clear button to enable reloading from model file
        """        
        self.BOPCAT.modelbx_var = None
        for w in self.subframe.children.values():
            w.destroy()
        self.BOPCAT.init_cat_controls()        
        self.loadvar.set("Load")

    def rattle_model(self):
        """
        calls BOPcat to rattle model
        """
        if not self.BOPCAT.rattle:
            return
        if self.BOPCAT.modelbx_var is None:
            return
        factor = unpack_value(self.BOPCAT.rattle_options[0])
        maxf = unpack_value(self.BOPCAT.rattle_options[1])
        self.BOPCAT.modelbx = self.BOPCAT.modelbx0.rattle(var=self.BOPCAT.cat_controls.opt_variables,factor=factor,maxf=maxf)
        self.BOPCAT.update_modelbx_var()

    def restore_model(self):
        """
        command to restore model parameters 
        """
        if self.BOPCAT.modelbx0 is None:
            return
        self.BOPCAT.modelbx = self.BOPCAT.modelbx0
        self.BOPCAT.update_modelbx_var() 
                  
    def init_tableframe(self):
        """
        initialize parameters table
        """
        self.table = tk.Canvas(self.tableframe,width=1000)

        self.table.grid(row=0,column=0,columnspan=2,sticky='nsew')
 
        hscroll = tk.Scrollbar(self.tableframe,orient=tk.HORIZONTAL,command=self.table.xview)
        hscroll.grid(row=1,column=0,columnspan=2,sticky='ew')

        vscroll = tk.Scrollbar(self.tableframe,orient=tk.VERTICAL,command=self.table.yview)
        vscroll.grid(row=0,column=2,sticky='ns')
 
        self.table.config(yscrollcommand = vscroll.set)
        self.table.config(xscrollcommand = hscroll.set)
        self.table.config(scrollregion=(0,0,1000,400))

    def show_infoxpar(self):
        """
        generate table of infor parameters
        """
        if self.lframe is not None:
            for w in self.lframe.children.values():
                w.grid_forget()
            self.table.delete(self.lframe)
        self.lframe = tk.Frame(self.table)
        self.table.create_window((0,0),window=self.lframe,anchor='nw')

        #update first modelsbx
        self.BOPCAT.update_modelbx()

        infox = self.BOPCAT.modelbx.get_infox()

        maxnpar = 30
        maxnkey = 20
        labels = [[None for m in range(maxnpar)] for l in range(maxnkey)] 

        countk = 0
        for key, val in infox.items():
            if val is None:
                continue
            labels[countk][0] = tk.Label(self.lframe,text=key,width=14)
            labels[countk][0].grid(row=countk,column=0)
            v = tk.StringVar()
            v.set(str(val))
            labels[countk][1] = tk.Entry(self.lframe,textvariable=v,width=14,bg='gray')
            labels[countk][1].grid(row=countk,column=1,sticky='nsew')
            labels[countk][1].bind('<KeyPress-Return>',lambda event, v=v:user_input(v))
            countk += 1
            self.BOPCAT.modelbx_var.infox_parameters.update({key:v})
        self.lframe.update_idletasks()
        self.tableframe.config(width=1000,height=300)
        self.table.config(scrollregion=self.table.bbox('all'))

    def show_atompar(self,i):
        """
        generate table of atomsbx parameters
        """
        if self.lframe is not None:
            for w in self.lframe.children.values():
                w.grid_forget()
            self.table.delete(self.lframe)
        self.lframe = tk.Frame(self.table)
        self.table.create_window((0,0),window=self.lframe,anchor='nw')

        #update first modelsbx
        self.BOPCAT.update_modelbx()
 
        abx = self.BOPCAT.modelbx.atomsbx[i]  
        atom = abx.atom
       
        maxnpar = 30
        maxnkey = 20
        labels = [[None for m in range(maxnpar)] for l in range(maxnkey)] 
 
        pars = [abx.get_atomspar()]
        countk = 0
        for p in range(len(pars)):
            for key,val in pars[p].items():        
                if val is None:
                    continue    
                if isinstance(val,float) or isinstance(val,int):
                    val = [val]                        
                labels[countk][0] = tk.Label(self.lframe,text=key,width=14)
                labels[countk][0].grid(row=countk,column=0)
                val_var = []
                for j in range(len(val)):
                    v = tk.StringVar()
                    v.set(str(val[j]))
                    bg = self.get_bgcolor(i,key,j) 
                    labels[countk][j+1] = tk.Entry(self.lframe,textvariable=v,width=8,bg=bg)
                    labels[countk][j+1].grid(row=countk,column=j+1,sticky='nsew')
                    labels[countk][j+1].bind('<KeyPress-Return>',lambda event, v=v:user_input(v)) 
                    labels[countk][j+1].bind('<Double-Button-1>',lambda event
                    ,entry=labels[countk][j+1], ai=i
                    , key=key, vj = j:self.set_constraint(entry,ai,key,vj)) 
                    val_var.append(v)
                countk += 1
                self.BOPCAT.modelbx_var.atomsbx[i].set_atomspar({key:val_var})
                # onsitelevels problematic will not add delta_onsites! 
        self.lframe.update_idletasks()
        #wm = 0
        #for m in range(20):
        #    if labels[0][m] is not None:
        #        wm += labels[0][m].winfo_width()
        #hm = 0
        #for m in range(5):
        #    if labels[m][0] is not None:
        #        hm += labels[m][0].winfo_height()

        #wm = sum([labels[0][m].winfo_width() for m in range(20)])
        #hm = sum([labels[l][0].winfo_height() for l in range(5)])
        #self.tableframe.config(width=wm,height=hm)
        self.tableframe.config(width=1000,height=300)
        self.table.config(scrollregion=self.table.bbox('all'))
        
        #self.lframes[i] = self.lframe

    def show_bondpar(self,i):
        """
        generate table of bondsbx parameters
        """
        if self.lframe is not None:
            for w in self.lframe.children.values():
                w.grid_forget()
            self.table.delete(self.lframe)
        self.lframe = tk.Frame(self.table)
        self.table.create_window((0,0),window=self.lframe,anchor='nw')

        i -= len(self.BOPCAT.modelbx.atomsbx)

        #update first modelsbx
        self.BOPCAT.update_modelbx()

        bbx = self.BOPCAT.modelbx.bondsbx[i]  
        bond = bbx.bond
       
        maxnpar = 30
        maxnkey = 20
        labels = [[None for m in range(maxnpar)] for l in range(maxnkey)]
 
        pars = [bbx.get_bondspar(),bbx.get_overlappar(),bbx.get_repetal()]
        funcs = bbx.get_functions()
        countk = 0
        for p in range(len(pars)):
            for key,val in pars[p].items():        
                if val is None:
                    continue                        
                labels[countk][0] = tk.Button(self.lframe,text=key,width=14,command=lambda bi=i,key=key:self.plot_bondfunc(bi,key))
                labels[countk][0].grid(row=countk,column=0,sticky='we')              
                labels[countk][1] = tk.Label(self.lframe,text=funcs[key],width=28)
                labels[countk][1].grid(row=countk,column=1,sticky='we')
                val_var = []
                for j in range(len(val)):
                    v = tk.StringVar()
                    v.set(str(val[j]))
                    bg = self.get_bgcolor(i+len(self.BOPCAT.modelbx.atomsbx),key,j) 
                    labels[countk][j+2] = tk.Entry(self.lframe,textvariable=v,width=8,bg=bg)
                    labels[countk][j+2].grid(row=countk,column=j+2,sticky='nsew')
                    labels[countk][j+2].bind('<KeyPress-Return>',lambda event, v=v:user_input(v)) 
                    labels[countk][j+2].bind('<Double-Button-1>',lambda event
                    ,entry=labels[countk][j+2], bi=i+len(self.BOPCAT.modelbx.atomsbx)
                    , key=key, vj = j:self.set_constraint(entry,bi,key,vj)) 
                    val_var.append(v)    
                countk += 1
                self.BOPCAT.modelbx_var.bondsbx[i].set_bondspar({key:val_var})
        self.lframe.update_idletasks()        
        #wm = 0
        #for m in range(20):
        #    if labels[0][m] is not None:
        #        wm += labels[0][m].winfo_width()
        #hm = 0
        #for m in range(5):
        #    if labels[m][0] is not None:
        #        hm += labels[m][0].winfo_height()

        #wm = sum([labels[0][m].winfo_width() for m in range(20)])
        #hm = sum([labels[l][0].winfo_height() for l in range(5)])
        #self.tableframe.config(width=wm,height=hm)
        self.tableframe.config(width=1000,height=400)
        self.table.config(scrollregion=self.table.bbox('all'))


    def plot_bond(self):
        """
        command to plot all bond functions
        """
        self.BOPCAT.init_modelbx()
        bbx = self.BOPCAT.modelbx.bondsbx[self.cur_bi]
        bond = bbx.bond
        par = bbx.get_bondspar() 
        par.update(bbx.get_overlappar())
        par.update(bbx.get_repetal())
        for key, val in par.items():
            if val is None:
                continue
            self.plot_bondfunc(self.cur_bi,key)

    def plot_bondfunc(self,i,key):
        """
        plot bond function. can toggle plot by clicking on button
        with key
        """
        xarr = np.linspace(1.5,6.5,100)
        skey = '%d-%s'% (i,key)
        if not self.status_plot.has_key(skey):
            self.status_plot[skey] = 0
        stat = self.status_plot[skey] 
        self.BOPCAT.init_modelbx()
        func = self.BOPCAT.modelbx_func.bondsbx[i].get_bondspar()
        func.update(self.BOPCAT.modelbx_func.bondsbx[i].get_overlappar())
        func.update(self.BOPCAT.modelbx_func.bondsbx[i].get_repetal())
        func = func[key] 
        if stat == 0:
            # turn on plot
            line, = self.mbxgraph.figsubplot.plot(xarr,func(xarr),'-',label=key, lw=3)  
            self.status_plot[skey] = 1
            self.lines_plot[skey] = line
        else:
            line = self.lines_plot[skey]   
            self.mbxgraph.figsubplot.lines.remove(line)
            self.status_plot[skey] = 0
        self.mbxgraph.figsubplot.autoscale()
        self.mbxgraph.figsubplot.legend(loc=1)    
        self.set_alabels()
        self.mbxgraph.canvas.draw()

    def set_alabels(self):
        if len(self.mbxgraph.figsubplot.lines) == 0:
            return
        legends = [line.get_label() for line in self.mbxgraph.figsubplot.lines]
        temp = []
        O = 'O'
        B = '$\\beta$'
        P = '$\Phi$'
        for legend in legends:
            if 'sigma' in legend or 'pi' in legend  or 'delta' in legend:
                if 'overlap' in legend and O not in temp:
                    temp.append(O)
                else:
                    if B not in temp:
                        temp.append(B)     
            else:
                if P not in temp:
                    temp.append('$\Phi$')
        ylabel = ''
        for i in range(len(temp)-1):
            ylabel += '%s/'%temp[i]
        ylabel += '%s (eV)'%temp[-1]
        self.mbxgraph.set_xylabels(('$R_{ij}$ $(\AA)$',ylabel))

    def get_bgcolor(self,i,key,vj):
        """
        get current bg color for parameter
        red : fit
        green : not fit
        """
        if not self.BOPCAT.cat_controls.opt_variables[i].has_key(key):
            return None  
        if self.BOPCAT.cat_controls.opt_variables[i][key][vj]:
            bg = 'red'
        else:
            bg = 'green'
        return bg        
      
    def set_constraint(self,entry,i,key,vj):
        """
        setting parameter constraint depending on bg color
        """
        old = entry['bg']
        bg = ['red', 'green']
        new = bg[(bg.index(old)+1)%2]
        entry.config(bg=new)
        if new == 'red':
            self.BOPCAT.cat_controls.opt_variables[i][key][vj] = True
        elif new == 'green':     
            self.BOPCAT.cat_controls.opt_variables[i][key][vj] = False

    def reset_ev(self):
        self.evgraph.figsubplot.cla()
        self.evgraph.figsubplot.set_xlabel('$V (\AA^3)$')
        self.evgraph.figsubplot.set_ylabel('$E/at (eV)$')
        self.evgraph.canvas.draw()

    def plot_ev(self):
        """
        plot energy volume-curves
        """
        atoms = self.BOPCAT.kernel.calc.get_atoms()
        strucs = []
        seps = ['_','-']
        for at in atoms:
            s = at.info['strucname']
            for sep in seps:
                s = s.split(sep)[0] 
            if s not in strucs:
                strucs.append(s)
        markers = ['o','v','d','s','p','h','>','<','^']
        colors = ['r','g','b','c','m','y','k']
        # decrease alpha for previous lines
        alphas = np.linspace(0.2,1.0,len(self.lines_test)+1)
        for i in range(len(self.lines_test)):
            for j in range(len(self.lines_test[i])):
                self.lines_test[i][j].set_alpha(alphas[i])
        add_lines = []
        for i in range(len(strucs)):
            vol = []
            ene = []
            for at in atoms:
                if strucs[i] not in at.info['strucname']:
                    continue
                vol.append(at.get_volume()/len(at))
                ene.append(at.get_potential_energy()/len(at))
            #line, = self.evgraph.figsubplot.plot(vol,ene,'%s-'%colors[i%len(colors)],label=strucs[i],lw=3)         
            line, = self.evgraph.figsubplot.plot(vol,ene,'%s%s'%(colors[i%len(colors)],markers[i%len(markers)]),label=strucs[i],fillstyle='full')         
            add_lines.append(line)
            vol = []
            ene = []
            for at in self.BOPCAT.ref_atoms:
                if strucs[i] not in at.info['strucname']:
                    continue
                vol.append(at.get_volume()/len(at))
                e = at.get_potential_energy()
                for s in at.get_chemical_symbols():
                    e -= self.BOPCAT.cat_controls.data_free_atom_energies[s]
                ene.append(e/len(at))
            self.evgraph.figsubplot.plot(vol,ene,'%s%s'%(colors[i%len(colors)],markers[i%len(markers)]),fillstyle='none')
        self.lines_test.append(add_lines)
        self.evgraph.figsubplot.legend(loc=1)    
        self.evgraph.canvas.draw()
    
    def test_model(self):
        """
        run BOPcat for energy-volume curve
        """
        self.BOPCAT.init_cat_controls()
        self.BOPCAT.init_modelbx()
        opt_var = deepcopy(self.BOPCAT.cat_controls.opt_variables)
        self.BOPCAT.cat_controls.opt_variables = None
        self.worker_runbopcat = threading.Thread(target=self.BOPCAT.run)
        self.worker_runbopcat.start()
        self.worker_runbopcat.join()
        self.plot_ev()
        self.BOPCAT.cat_controls.opt_variables = opt_var 
        self.BOPCAT.clean()

class GraphFrame(tk.Frame):
    """
    generic frame for graphs in GUI
    """
    def __init__(self,masterframe,alabels,size=(5,4)):
        tk.Frame.__init__(self,masterframe)
        self.canvasFig =  plt.figure(1)
        self.fig = Figure(figsize=size,dpi=100)  
        self.figsubplot = self.fig.add_subplot(111)
        self.set_xylabels(alabels)
        self.canvas = FigureCanvasTkAgg(self.fig,self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)

        toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True)

        self.fig.tight_layout()

    def set_xylabels(self,alabels):
        self.figsubplot.set_xlabel(alabels[0])  
        self.figsubplot.set_ylabel(alabels[1])

class OptCalcFrame(tk.Frame):
    """
    Frame for optimizer and BOPfox calculator
    TODO: update settings depending on versions of scipy and BOPfox 
    """
    def __init__(self,masterframe,master):
        tk.Frame.__init__(self,masterframe)  
        self.master = master
        self.BOPCAT = master.frames[0].BOPCAT
        self.init_window()
        label = tk.Label(self.opt_frame, text="Optimizer", font=HEAD1_FONT, fg='slate gray',width=20)
        label.grid(row=0,column=0,columnspan=1,sticky='w')
        label = tk.Label(self.calc_frame, text="Calculator", font=HEAD1_FONT, fg='dark orange',width=20)
        label.grid(row=0,column=0,columnspan=1,sticky='w')

    def init_opt_options(self):
        opt = unpack_value(self.BOPCAT.cat_controls_var.opt_optimizer)
        if opt.lower() == 'leastsq':
            self.opt_options =  [('args', None), ('Dfun', None), ('full_output', 0)
            , ('col_deriv', 0), ('ftol', 1.49012e-08), ('xtol', 1.49012e-08)
            , ('gtol', 0.0), ('maxfev', 0), ('epsfcn', 1e-6), ('factor', 100)
            , ('diag', None)]
        elif opt.lower() == 'least_squares':
            self.opt_options = [('jac', '2-point'), ('bounds', (-inf, inf))
            , ('method', 'lm'), ('ftol', 1e-08), ('xtol', 1e-08)
            , ('gtol', 1e-08), ('x_scale', 1.0), ('loss', 'linear')
            , ('f_scale', 1.0), ('diff_step', 0.005), ('tr_solver', None)
            , ('tr_options', {}), ('jac_sparsity', None), ('max_nfev', None)
            , ('verbose', 0), ('args', None), ('kwargs', {})]
        elif opt.lower() == 'nelder-mead':
            self.opt_options = [('args', None), ('xtol', 0.0001), ('ftol', 0.0001)
            , ('maxiter', None), ('maxfun', None), ('full_output', 0)
            , ('disp', 1), ('retall', 0), ('callback', None), ('initial_simplex', None)]
        elif opt.lower() == 'cg':
            self.opt_options = [('fprime', None), ('gtol', 1e-05)
            , ('norm', inf), ('epsilon', 1.4901161193847656e-08), ('maxiter', None)
            , ('full_output', 0), ('disp', 1), ('retall', 0), ('callback', None)]
        elif opt.lower() == 'bfgs':
            self.opt_options = [('fprime', None), ('args', None), ('gtol', 1e-05)
            , ('norm', inf), ('epsilon', 1.4901161193847656e-08), ('maxiter', None)
            , ('full_output', 0), ('disp', 1), ('retall', 0), ('callback', None)]
        elif opt.lower() == 'ncg':
            self.opt_options = [('fhess_p', None), ('fhess', None), ('args', None)
            , ('avextol', 1e-05), ('epsilon', 1.4901161193847656e-08)
            , ('maxiter', None), ('full_output', 0), ('disp', 1), ('retall', 0)
            , ('callback', None)]
        elif opt.lower() == 'l-bfgs-b':
            self.opt_options = [('fprime', None), ('args', None), ('approx_grad', 0)
            , ('bounds', None), ('m', 10), ('factr', 10000000.0), ('pgtol', 1e-05)
            , ('epsilon', 1e-08), ('iprint', -1), ('maxfun', 15000)
            , ('maxiter', 15000), ('disp', None), ('callback', None), ('maxls', 20)]
        elif opt.lower() == 'tnc':
            self.opt_options = [('fprime', None), ('args', None), ('approx_grad', 0)
            , ('bounds', None), ('epsilon', 1e-08), ('scale', None), ('offset', None)
            , ('messages', 15), ('maxCGit', -1), ('maxfun', None), ('eta', -1)
            , ('stepmx', 0), ('accuracy', 0), ('fmin', 0), ('ftol', -1), ('xtol', -1)
            , ('pgtol', -1), ('rescale', -1), ('disp', None), ('callback', None)]
        elif opt.lower() == 'slsqp':
            self.opt_options = [('eqcons', ()), ('f_eqcons', None), ('ieqcons', ())
            , ('f_ieqcons', None), ('bounds', ()), ('fprime', None)
            , ('fprime_eqcons', None), ('fprime_ieqcons', None), ('args', None)
            , ('iter', 100), ('acc', 1e-06), ('iprint', 1), ('disp', None)
            , ('full_output', 0), ('epsilon', 1.4901161193847656e-08)
            , ('callback', None)]
        elif opt.lower() == 'differential_evolution':
            self.opt_options = [('args', None), ('strategy', 'best1bin')
            , ('maxiter', 1000), ('popsize', 15), ('tol', 0.01)
            , ('mutation', (0.5, 1)), ('recombination', 0.7), ('seed', None)
            , ('callback', None), ('disp', False), ('polish', True)
            , ('init', 'latinhypercube'), ('atol', 0)]
        elif opt.lower() == 'basinhopping':
            self.opt_options = [('niter', 100), ('T', 1.0), ('stepsize', 0.5)
            , ('minimizer_kwargs', None), ('take_step', None), ('accept_test', None)
            , ('callback', None), ('interval', 50), ('disp', False)
            , ('niter_success', None), ('seed', None)]
        elif opt.lower() == 'brute':
            self.opt_options = [('args', None), ('Ns', 20), ('full_output', 0)
            , ('finish', None), ('disp', False)]
        self.BOPCAT.cat_controls_var.opt_optimizer_options = {}

    def init_calc_settings(self):
        self.calc_settings = [('forces','analytic'),('numfdisp',2),('numfinc',2e-4)
                            ,('rskin',0.01),('rthickskin',1.0),('maxpathradius',0.)
                            ,('efermimixer','bisection'),('efermitol',1e-8)
                            ,('efermisteps',100),('scfmixer','broyden'),('scfsteps',200)
                            ,('scftol',1e-3),('scflinearmixpara',1e-1),('scfbroydenmixpara',1e-1)
                            ,('scfonsitemddtsq',1e-3),('scfonsitemdmass',1.0)
                            ,('scfonsitemddamping',0.9),('scffirenmin',5),('scffirefinc',1.1)
                            ,('scffirefdec',0.1),('scffirefalpha',0.99),('scffirealphastart',0.1)
                            ,('scffiredtinitial',0.1),('scffiredtmax',1.0),('scfreusehii',True)
                            ,('scfrigidhiishift',True),('scffixmagcom',False)
                            ,('scffixmagdiff',False),('scfdamping',False),('scfdampinglimit',0.1)
                            ,('scfsaveonsite',False),('scfrestart',True),('tbnbandstotal',20)
                            ,('tbkpointmesh','gamma-centered'),('tbnsmear',-5),('tbdsmear',0.2)
                            ,('tbintegration','tetrahedron'),('tbkpointfolding',[10,10,10]),('tbsolver','lapack')
                            ,('fastbop',True),('openmp_nthreads',0)
                            ]
        self.BOPCAT.cat_controls_var.calculator_settings = {}
            
    def init_window(self):
        self.opt_frame = tk.Frame(self)
        self.opt_frame.grid(row=0,column=0,sticky='nsew')
        #optimizer = unpack_value(self.BOPCAT.cat_controls_var.opt_optimizer)
        button = tk.Button(self.opt_frame,textvariable=self.BOPCAT.cat_controls_var.opt_optimizer,command=self.set_opt_options)
        button.grid(row=1,column=0,sticky='we')

        self.calc_frame = tk.Frame(self)
        self.calc_frame.grid(row=0,column=1,sticky='nsew')
        button = tk.Button(self.calc_frame,text='BOPfox settings',command=self.set_calc_settings)
        button.grid(row=1,column=0,sticky='we')
        
    def set_opt_options(self):
        self.init_opt_options()
        for i in range(len(self.opt_options)):
            key = self.opt_options[i][0]
            val = self.opt_options[i][1]
            v = init_entry_input(self.opt_frame,key,(2+i,0),default=val)
            self.BOPCAT.cat_controls_var.opt_optimizer_options[key] = v

    def set_calc_settings(self):
        self.init_calc_settings()
        for i in range(len(self.calc_settings)):
            key = self.calc_settings[i][0]
            val = self.calc_settings[i][1]
            v = init_entry_input(self.calc_frame,key,(2+i,0),default=val)
            self.BOPCAT.cat_controls_var.calculator_settings[key] = v
            

class ParameterizationFrame(tk.Frame):
    """
    Frame to handle sequential parameterization
    """
    def __init__(self,masterframe,master):
        tk.Frame.__init__(self,masterframe)  
        self.master = master
        self.nmaxlayer = 10
        label = tk.Label(self, text="Parameterization", font=HEAD1_FONT, fg='dark slate blue')
        label.grid(row=0,column=0,columnspan=self.nmaxlayer,sticky='ew')
        self.layers = []
        self.kernels = []
        self.buttons = []
        self.options = []
        self.rattle_options = []
        self.opt_models = []
        self.status_bar = []
        self.status = []
        self.model_manager_options = {'Best','Average'}
        self.init_window()
        self.start_time = 0.0
        
    def init_window(self):
        tk.Button(self,text='Add layer',command=self.add_layer).grid(row=1,column=0,sticky='ew',columnspan=self.nmaxlayer/2)         
        tk.Button(self,text='Run',command=self.run).grid(row=1,column=self.nmaxlayer/2,sticky='ew',columnspan=self.nmaxlayer/2)         

    def gen_options(self):
        """
        generate options for process
        """
        bopcat = self.master.frames[0].BOPCAT
        bopcat.init_cat_controls()     
        q = bopcat.gen_queue()
        options = {'queue':q,'cores':bopcat.cat_controls.calculator_nproc
                  ,'directives':bopcat.directives}   
        return options

    def gen_rattle_options(self):
        """
        generate options for rattling model
        """
        bopcat = self.master.frames[0].BOPCAT
        rattle = bopcat.rattle
        options = bopcat.rattle_options
        options = [unpack_value(v) for v in options] 
        return [rattle,options[0],options[1]]

    def delete_kernel(self,nl,nm,nk):
        """
        remove kernel from list
        """
        self.kernels[nl][nm][nk] = None
        self.buttons[nl][nm][nk].destroy()
        self.buttons[nl][nm][nk] = None
        self.rattle_options[nl][nm][nk] = None
        if len(self.options[nl][nm]) > nk: 
            self.options[nl][nm][nk] = None
        self.status[nl][nm][nk] = 3

    def add_opt_kernel(self,nl):
        """
        add kernel for optimization
        """
        nk = len(self.kernels[nl][0])
        button = tk.Button(self.layers[nl],text='opt kernel %d'%nk,bg='red',command=lambda nl=nl, nk=nk, nm=0:self.edit_kernel(nl,nk,nm))
        button.bind("<Double-Button-3>",lambda event, nl=nl, nm=0, nk=nk:self.delete_kernel(nl,nm,nk))
        self.buttons[nl][0].append(button)
        kernel = self.master.frames[0].BOPCAT
        self.kernels[nl][0].append(kernel)
        button.grid(row=nk+2,column=0)
        self.options[nl][0].append(self.gen_options())
        self.status[nl][0].append(0)
        self.rattle_options[nl].append(self.gen_rattle_options())
                
    def add_test_kernel(self,nl):
        """
        add kernel for testing
        """
        nk = len(self.kernels[nl][1])
        button = tk.Button(self.layers[nl],text='test kernel %d'%nk,bg='red',command=lambda nl=nl, nk=nk, nm=1:self.edit_kernel(nl,nk,nm))
        button.bind("<Double-Button-3>",lambda event, nl=nl, nm=1, nk=nk:self.delete_kernel(nl,nm,nk))
        self.buttons[nl][1].append(button)
        kernel = self.master.frames[0].BOPCAT
        self.kernels[nl][1].append(kernel)
        button.grid(row=nk+2,column=1)
        self.options[nl][1].append(self.gen_options())
        self.status[nl][1].append(0)

    def add_model_manager(self,nl):
        """
        add option to manage optimized models 
        """
        v = tk.StringVar()
        v.set('Best')
        popup = tk.OptionMenu(self.layers[nl],v,*self.model_manager_options)
        #height = sum([button.winfo_height() for button in self.buttons[nl][0]])
        popup.config(width=15)
        #tk.Label(self.layers[nl],text='mode',width=5).grid(row=1,column=2)
        popup.grid(row=2,column=2,rowspan=max(1,len(self.kernels[nl][0])))          
        self.buttons[nl][2] = v
        self.status[nl][2].append(0)

    def edit_kernel(self,nl,nk,nm):
        """
        edit BOPcat kernel from user input
        """
        if self.buttons[nl][nm][nk]['bg'] == 'red':
            self.kernels[nl][nm][nk].init_cat_controls()
            self.kernels[nl][nm][nk].init_kernel()
            if nm == 1:
                self.kernels[nl][nm][nk].kernel.variables = None
            kernel = self.kernels[nl][nm][nk].kernel
            self.kernels[nl][nm][nk] = kernel
            self.buttons[nl][nm][nk].config(bg='green')
            options = self.gen_options()
            self.options[nl][nm][nk] = options
            if nm == 0:
                rattle_options = self.gen_rattle_options()
                self.rattle_options[nl][nm] = rattle_options
        elif self.buttons[nl][nm][nk]['bg'] == 'green':
            kernel = self.master.frames[0].BOPCAT
            self.kernels[nl][nm][nk] = kernel
            self.buttons[nl][nm][nk].config(bg='red')

    def rattle_model(self,model,nl,nm,nk):
        """
        rattle model if rattle is engaged
        """
        rattle = self.rattle_options[nl][nk][0]
        if not rattle:
            return model
        mode = self.rattle_options[nl][nk][1]
        fac = self.rattle_options[nl][nk][2]
        var = self.kernels[nl][nm][nk].variables
        model = model.rattle(var,factor=mode,maxf=fac)
        return model

    def set_model(self,nl,nm,nk):
        """
        set model on succeeding layer from previous
        """
        if nl == 0 and nm == 0:
            # no previous layer to get model from
            return
        kernel = self.kernels[nl][nm][nk]
        if nm == 0:
            model = self.managed_models[nl-1]
            if model is None:
                model = self.opt_models[nl-1][nk]
            # rattle model if required
            model = self.rattle_model(model,nl,nm,nk)
        elif nm == 1:
            model = self.opt_models[nl][nk]
        if model is not None:
            kernel.calc.set_model(model)

    def _update_status_proc(self,nl,nm):
        """
        set status for status bar 0=set 1=running 2=done
        """
        kernels = self.kernels[nl][nm]
        res = self.proc.get_results(wait=False)
        while True:
            res = self.proc.get_results(wait=False)
            countk = 0 
            for i in range(len(kernels)):
                if kernels[i] is None:
                    continue
                if self.proc._procs[countk].is_done():
                    self.status[nl][nm][i] = 2
                else:
                    self.status[nl][nm][i] = 1
                countk += 1
            if res is not None:
                break
            time.sleep(10)
        # for serial calculations does not go to loop
        for i in range(len(self.status[nl][nm])):
            if kernels[i] is None:
                continue
            self.status[nl][nm][i] = 2

    def run_kernels(self,nl,nm):
        """
        run kernels on layer
        kernels are packed on single process
        """
        kernels = self.kernels[nl][nm]
        options = self.options[nl][nm]
        subprocs = []
        if len(kernels) == 0:
            return
        for i in range(len(kernels)):
            if kernels[i] is None:
                continue
            queue = options[i]['queue']
            cores = options[i]['cores'] 
            directives = options[i]['directives']
            # modify model from previous layer
            self.set_model(nl,nm,i)
            kernels[i].gui_log = None
            subprocs.append(Process_catkernel(catkernel=kernels[i],queue=queue,cores=cores,directives=directives))
        self.proc = Process_catkernels(procs=subprocs)
        self.proc.run()
        self._update_status_proc(nl,nm)
        res = self.proc.get_results(wait=True)
        countk = 0
        if nm == 0:
            out = [] 
            scores = res['function_value']  
            for i in range(len(kernels)):
                if kernels[i] is None:
                    out.append(None)
                    continue
                mbx = self.proc._procs[countk].get_kernel()._catkernel.get_optimized_model()
                atoms = self.proc._procs[countk].get_kernel()._catkernel.calc.get_atoms()
                string = ''
                for at  in atoms:
                    string += '%s '%at.info['strucname']
                mbx.annotations = ['fit structures: %s'%string, 'fit rms: %s'%scores[countk]]
                out.append(mbx)
                countk += 1
            self.opt_models.append(out)
        elif nm == 1:
            scores = res['function_value']  
            for i in range(len(kernels)):
                if kernels[i] is None:
                    continue
                self.opt_models[nl][i].annotations += ['test rms: %s'%scores[countk]]
                countk += 1
        for i in range(len(self.opt_models[nl])):
            if self.opt_models[nl][i] is None:
                continue
            self.opt_models[nl][i].write(filename='model_optimized_%d_%d_%d.bx'%(nl,nm,i))
        # clean up
        self.proc.clean()
 
    def get_best_model(self,nl):
        """
        option for model manager, best model is min rms
        """
        min_rms = 1000
        for i in range(len(self.opt_models[nl])):
            if self.opt_models[nl][i] is None:
                continue
            ann = self.opt_models[nl][i].annotations
            for j in range(len(ann)):
                if 'rms:' in ann[j]:
                    rms = float(ann[j].split()[-1])
            if rms < min_rms:
                min_rms = rms
                min_i = i 
        self.opt_models[nl][min_i].write(filename='model_best_%d.bx'%nl)
        return self.opt_models[nl][min_i]
        

    def manage_model(self,nl):
        """
        command to manage models
        """
        mode = self.buttons[nl][2]
        if mode is None:
            return
        mode = mode.get()
        if mode == 'Best':
            model = self.get_best_model(nl)
        elif mode == 'Average':
            model = average_models([mbx for mbx in self.opt_models[nl] if mbx is not None]) 
        else:
            model = None
        self.managed_models[nl] = model

    def _run(self):
        """
        run all layers
        """
        for i in range(len(self.layers)):
            for j in range(2):
                self.run_kernels(i,j)
                self.manage_model(i)
            for k in range(len(self.status[i][2])):
                self.status[i][2][k] = 2
            
    def _update_status(self,init=False):
        """
        change color of status bar depending on status of kernel
        blue=set yellow=running green=done
        """
        stat_color = ['blue','yellow','green','gray']
        sublayers = []
        for nl in range(len(self.status)):
            self.status_bar.append([[],[],[]])
            sublayers.append([[],[],[]])
            for nm in range(2):
                if init:
                    sublayer = tk.Frame(self.layers[nl])
                    #sublayer.grid(row=1,column=nm,sticky='ew')
                    sublayers[nl][nm] = sublayer
                for nk in range(len(self.status[nl][nm])):
                    if self.buttons[nl][nm][nk] is None:
                        continue
                    if init:
                        #label = tk.Label(sublayers[nl][nm],text=str(nk),bg='blue')
                        #label.grid(row=0,column=nk,sticky='nsew') 
                        #self.status_bar[nl][nm].append(label)
                        self.buttons[nl][nm][nk].config(bg='blue')
                    else:
                        color = stat_color[self.status[nl][nm][nk]]
                        #self.status_bar[nl][nm][nk].config(bg=color)
                        self.buttons[nl][nm][nk].config(bg=color)

    def run(self):
        """
        run on separate thread
        """
        self.start_time = time.time()
        self._update_status(init=True)        
        self.managed_models = [None]*(len(self.layers))
        t = threading.Thread(target=self._run)
        t.start()

    def add_layer(self):
        """
        add optimization layer
        each layer may contain optimization, testing and model management
        """
        layer= tk.Frame(self)
        self.layers.append(layer)
        tk.Button(self.layers[-1],text='Add opt.',command=lambda l=len(self.layers)-1:self.add_opt_kernel(l)).grid(row=0,column=0,sticky='nsew')
        tk.Button(self.layers[-1],text='Add test',command=lambda l=len(self.layers)-1:self.add_test_kernel(l)).grid(row=0,column=1,sticky='nsew')
        tk.Button(self.layers[-1],text='model manager',command=lambda l=len(self.layers)-1:self.add_model_manager(l)).grid(row=0,column=2,sticky='nsew',columnspan=2)
        layer.grid(row=2,column=len(self.layers)-1)
        self.kernels.append([[],[]])
        self.buttons.append([[],[],None])
        self.options.append([[],[]])
        self.rattle_options.append([])
        self.status.append([[],[],[]])

    def update_status(self):
        """
        interface to app for updating status bar
        """
        if self.start_time > 0.0:
            self._update_status()
        self.master.after(10000,self.update_status)
                    
    
def animate(i,fr,l):
    #fr.read_iteration()
    data = fr.iteration_data
    if len(data) < 1:
        fr.itergraph.figsubplot.semilogy([],[],'ro-')
        return
    xarr = [data[k][0] for k in range(len(data))]
    data = [data[k][l] for k in range(len(data))]    
    fr.itergraph.figsubplot.lines[0].set_data(xarr,data)
    ax = fr.itergraph.canvas.figure.axes[0]
    ax.set_xlim(min(xarr),max(xarr))
    ax.set_ylim(min(data),max(data))
    fr.itergraph.canvas.draw()

app = BOPcatapp()
# update status for Home frame
app.after(0,app.frames[0].update_status) 
# update status for Parameterization frame
app.after(0,app.frames[4].update_status)
# animate iteration plot
ani = animation.FuncAnimation(app.frames[0].itergraph.fig, animate, fargs=(app.frames[0],1),interval=5000)   
app.mainloop()  
