from bopcat.catcontrols import CATControls
import bopcat.functions as Funcs

cat_controls = CATControls()

# required, list of elements
cat_controls.set(elements = ['Fe'])

###################MODEL##############################################
# cut_offs
cat_controls.set(model_cutoff = {
                                 'rcut' : {'Fe-Fe':4.0}
                                ,'r2cut': {'Fe-Fe':6.0}
                                ,'dcut' : {'Fe-Fe':0.5}
                                ,'d2cut': {'Fe-Fe':0.5}
                                })

cat_controls.set(model_orthogonal = True)
cat_controls.set(model_valences={'Fe':'d'})
cat_controls.set(model_betabasis='tz0')
cat_controls.set(model_betatype='loewdin')
cat_controls.set(model_pathtobetas='../betas')
cat_controls.set(model_pathtoonsites='../onsites')
cat_controls.set(model_functions = {
                                    'ddsigma':{'Fe-Fe':Funcs.exponential()}
                                   ,'ddpi':{'Fe-Fe':Funcs.exponential()}
                                   ,'dddelta':{'Fe-Fe':Funcs.exponential()}
                                   ,'rep1':{'Fe-Fe':Funcs.exponential()}
                                   ,'rep2':{'Fe-Fe':Funcs.sqrt_gaussian()}
                               })
###################CALCULATOR#########################################
# dictionary of keyword-values used by calculator 
cat_controls.set(calculator_settings = {
                                        'bandwidth':'findeminemax'
                                       ,'terminator':'constantabn'
                                       ,'bopkernel':'jackson'
                                       ,'nexpmoments':100
                                       ,'moments':9
                                       ,'version':'bop'
                                       })

###################DATA##################################################
# database parameters
cat_controls.set(data_parameters ={'xc_functional':201,'encut':450})

# reference data
cat_controls.set(data_filename = '../Fe.fit')

cat_controls.set(data_system_parameters = {'spin':1})

###################FITTING############################################
cat_controls.set(opt_optimizer_options = {'max_nfev':100,'diff_step':0.01})
cat_controls.set(opt_optimizer = 'least_squares')
###################OTHERS############################################
# verbosity
cat_controls.set(verbose = 2)
