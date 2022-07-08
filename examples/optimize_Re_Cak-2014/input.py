from bopcat.catcontrols import CATControls

cat_controls = CATControls()

# required, list of elements
cat_controls.set(elements = ['Re'])

###################MODEL##############################################
# optional, if not None, will read from models files
cat_controls.set(model = "new")
# 
cat_controls.set(model_pathtomodels = 'models.bx')

###################CALCULATOR#########################################
# dictionary of keyword-values used by calculator 
cat_controls.set(calculator_settings = {'scfsteps':100})

###################DATA###############################################
cat_controls.set(data_parameters = {'spin':1})
# required, reference data
cat_controls.set(data_filename = 'Re.fit')
# free atom energies, should be set otherwise will get from dimer data 
cat_controls.set(data_free_atom_energies = {'Re':0.0})

###################FITTING############################################
# optimization parameters
cat_controls.set(opt_optimizer_options = {'maxiter':100})
cat_controls.set(opt_optimizer = 'nelder-mead')

###################OTHERS############################################
# verbosity
cat_controls.set(verbose = 2)
