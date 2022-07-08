from bopcat.catcontrols import CATControls

cat_controls = CATControls()

# required, list of elements
cat_controls.set(elements = ['Fe'])

###################MODEL##############################################
# optional, if not None, will read from models files
cat_controls.set(model = "Madsen-2011")
# 
cat_controls.set(model_pathtomodels = 'models_Madsen-2011.bx')

###################CALCULATOR#########################################
# dictionary of keyword-values used by calculator 
cat_controls.set(calculator_settings = {'scfsteps':100})

###################DATA###############################################
cat_controls.set(data_parameters = {'spin':1})
# required, reference data
cat_controls.set(data_filename = '../Fe.fit')

###################FITTING############################################
# optimization parameters
cat_controls.set(opt_optimizer_options = {'epsfcn':1e-5,'method':'lm','max_nfev':100})
cat_controls.set(opt_optimizer = 'least_squares')

###################OTHERS############################################
# verbosity
cat_controls.set(verbose = 2)
#cat_controls.set(calculator_nproc = 1)
