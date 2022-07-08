from bopcat.catdata import CATData
from bopcat.catparam import CATParam
from bopcat.catcalc import CATCalc
from bopcat.catkernel import CATKernel
import sys
from bopcat.output import bopcat_logo, print_format
import os
import numpy as np

print_format(bopcat_logo(),level=1)

# execute input file and initialize 
execfile(sys.argv[-1])
cat_controls.initialize()

# generate reference data
cat_data = CATData(controls=cat_controls)

# generate initial parameters
cat_param = CATParam(controls=cat_controls,data=cat_data)

# set up calculator
cat_calc = CATCalc(controls=cat_controls,model=cat_param.models[-1])

# generate atoms and data for fitting
ref_atoms = cat_data.get_ref_atoms(structures=['Fe/229/0/1/*'],quantities=['energy'])
ref_data = cat_data.get_ref_data()
cat_calc.set_atoms(ref_atoms)

# optimization variables
var = [{"bond":["Fe","Fe"],'rep1':[True,True],'rep2':[True,True]}]

# set up optimization 
optfunc = CATKernel(calc=cat_calc,ref_data=ref_data,variables=var,log='log.cat'
                   ,controls=cat_controls)

# run optimization
optfunc.optimize()

# update models
new_model = optfunc.get_optimized_model()
cat_param.models.append(new_model)

# write out new model
new_model.write(filename='models_optimized.bx')

# check if parameters are far from original
repold = cat_param.models[0].bondsbx[0].get_repetal()
repnew = cat_param.models[1].bondsbx[0].get_repetal()
success = True
for key, val in repold.items():
    if repold[key] is None:
        continue
    diff = []
    for i in range(len(repold[key])):
        if repold[key][i] == 0.0:
            continue
        diff.append((repold[key][i] - repnew[key][i])/repold[key][i])
    if (np.array(diff) > 0.2).any():
        success = False

#remove files
os.system('rm -rf log*cat bopfox_keys.bx models_optimized.bx')
if success:
    print 'ok.'

