from bopcat.catcontrols import CATControls

cat_controls = CATControls()

# required, list of elements
cat_controls.set(elements = ['Fe'])

###################MODEL##############################################
# optional, if not None, will read from models files
# Create a models file with all parameters from Madsen 2011 and optimized
# repulsion. Change bondintegrals to dimer bondintegrals
from bopcat.bopmodel import read_modelsbx

model_dimer = read_modelsbx(model=['orthogonal_TB'], system=['Fe'],
                            filename='../../bopcat/dimers/models.bx')

model_madsen = read_modelsbx(model=['Madsen-2011'], system=['Fe'],
                             filename='models_optimized_rep.bx')

# stopdists
# ddsigma [ 3.    1.7   1.35  1.35  1.35  1.35]
# ddpi [ 3.   1.4  1.2  1.2  1.2  1.2]
# dddelta [ 2.65  1.2   1.2   1.2   1.2   1.2 ]

keys = model_madsen[0].get_keys()
model_madsen[0].bondsbx[0].functions['ddsigma'] = \
    model_dimer[0].bondsbx[0].functions['ddsigma']
model_madsen[0].bondsbx[0].functions['ddpi'] = \
    model_dimer[0].bondsbx[0].functions['ddpi']
model_madsen[0].bondsbx[0].functions['dddelta'] = \
    model_dimer[0].bondsbx[0].functions['dddelta']
ddsigma = model_dimer[0].bondsbx[0].get_bondspar()['ddsigma']
ddpi = model_dimer[0].bondsbx[0].get_bondspar()['ddpi']
dddelta = model_dimer[0].bondsbx[0].get_bondspar()['dddelta']
model_madsen[0].bondsbx[0].set_bondspar(
    bondspar={'ddsigma':ddsigma, 'ddpi':ddpi, 'dddelta':dddelta})

model_madsen[0].write('model_Fe.bx')


cat_controls.set(model = "Madsen-2011")
# 
# cat_controls.set(model_pathtomodels = 'models_optimized_rep.bx')
cat_controls.set(model_pathtomodels = 'model_Fe.bx')

###################CALCULATOR#########################################
# dictionary of keyword-values used by calculator 
cat_controls.set(calculator_settings = {'scfsteps':100})

###################DATA###############################################
cat_controls.set(data_parameters = {'spin':1})
# required, reference data
cat_controls.set(data_filename = '../Fe.fit')
cat_controls.set(data_free_atom_energies = None)

###################FITTING############################################
# optimization parameters
cat_controls.set(opt_optimizer_options = {'epsfcn':1e-8, 'maxfev':1000})
cat_controls.set(opt_optimizer = 'leastsq')

###################OTHERS############################################
# verbosity
cat_controls.set(verbose = 2)
