from bopcat.catcontrols import CATControls
import bopcat.functions as Funcs
import numpy as np

cat_controls = CATControls()

###################MODEL##################################################
# required, list of elements
cat_controls.elements = ['Fe','Nb']

# optional, if not None, will read from models files
cat_controls.model = None 
# 
cat_controls.model_pathtomodels = None 

# required, dictionary of keyword-values used by calculator 
cat_controls.calculator_settings = {
                                    'bandwidth':'findeminemax'
                                   ,'terminator':'constantabn'
                                   ,'bopkernel':'jackson'
                                   ,'nexpmoments':100
                                   ,'scfsteps':500
                                   ,'moments':9
                                   ,'version':'bop'
                                   }

# cut_offs
cat_controls.model_cutoff = {
                             'rcut' : {'Nb-Nb':4.5,'Fe-Fe':4.5,'Fe-Nb':4.5}
                            ,'r2cut': {'Nb-Nb':6.0,'Fe-Fe':6.0,'Fe-Nb':6.0}
                            ,'dcut' : {'Nb-Nb':0.5,'Fe-Fe':0.5,'Fe-Nb':0.5}
                            ,'d2cut' : {'Nb-Nb':0.5,'Fe-Fe':0.5,'Fe-Nb':0.5}
                            #,'decut' :0.0
                            #,'ecut':0.0
                            }

cat_controls.model_orthogonal = True
cat_controls.model_valences={'Fe':'d','Nb':'d'}
cat_controls.model_basis='tz0'
cat_controls.model_betatype='loewdin'
cat_controls.model_functions = {
                                'ddsigma':{'Nb-Nb':Funcs.exponential()
                                          ,'Fe-Nb':Funcs.exponential()
                                          ,'Fe-Fe':Funcs.exponential()}
                               ,'ddpi':{'Nb-Nb':Funcs.exponential()
                                       ,'Fe-Nb':Funcs.exponential()
                                       ,'Fe-Fe':Funcs.exponential()}
                               ,'dddelta':{'Nb-Nb':Funcs.exponential()
                                          ,'Fe-Nb':Funcs.exponential()
                                          ,'Fe-Fe':Funcs.exponential()}
                               ,'rep1':{'Fe-Fe':Funcs.exponential()
                                       ,'Nb-Nb':Funcs.exponential()
                                       ,'Fe-Nb':Funcs.exponential()}
                               ,'rep2':{'Fe-Fe':Funcs.sqrt_gaussian()
                                       ,'Nb-Nb':Funcs.sqrt_gaussian()
                                       ,'Fe-Nb':Funcs.sqrt_gaussian()}
                               }

cat_controls.set(model_pathtobetas='../betas')
cat_controls.set(model_pathtoonsites='../onsites')
###################DATA##################################################
# database parameters
cat_controls.data_parameters ={'xc_functional':201,'encut':450}

# reference data
cat_controls.data_filename = '../FeNb.fit'

cat_controls.data_system_parameters = {
                                  'spin':[1]
                                  #'system_type':[0]
                                  #,'calculation_type':[0]
                                  #'stoichiometry':['Fe','Fe2']
                                  }

###################FITTING##################################################
cat_controls.opt_optimizer_options = {'maxiter':100,'maxfun':100}
cat_controls.opt_optimizer = 'powell'
###################OTHERS############################################
# verbosity
cat_controls.set(verbose = 2)
