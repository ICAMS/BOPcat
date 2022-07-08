from bopcat.catdata import CATData
from bopcat.catparam import CATParam
from bopcat.catcalc import CATCalc
from bopcat.catkernel import CATKernel
from bopcat.output import bopcat_logo, print_format
from bopcat.eigs import get_relevant_bands
import numpy as np
import sys
import os

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
###########################optimize bond integrals################################################
ref_atoms = cat_data.get_ref_atoms(structures=['Fe/229/0/0/*'],quantities=['eigenvalues'])
ref_data = cat_data.get_ref_data()
orb_char = cat_data.orb_char
cat_calc.set_atoms(ref_atoms)

# select only relevant bands in dft eigs
ref_data, orb_char = get_relevant_bands(cat_calc,ref_data,orb_char)

# optimize valence electrons integrals
var  = [{'atom':'Fe','onsitelevels':[True]}]
# set up optimization 
optfunc = CATKernel(calc=cat_calc,ref_data=ref_data,variables=var,log='log1.cat'
                   ,controls=cat_controls,weights=orb_char)

# run optimization
optfunc.optimize()
# update models
new_model = optfunc.optimized_model
cat_param.models.append(new_model)
# optimize bond integrals
# start from previous model
cat_calc.set_model(new_model)
var  = [{"bond":["Fe","Fe"],"ddsigma":[True,True]
        ,"ddpi":[True,True], "dddelta":[True,True]
        ,'atom':'Fe','onsitelevels':[True]
        }]
# set up optimization 
optfunc = CATKernel(calc=cat_calc,ref_data=ref_data,variables=var,log='log2.cat'
                   ,controls=cat_controls,weights=orb_char)
# run optimization
optfunc.optimize()
# update models
new_model = optfunc.optimized_model
cat_param.models.append(new_model)
###########################optimize repulsion################################################
# generate atoms and data for fitting
new_model = cat_param.models[-1]
ref_atoms = cat_data.get_ref_atoms(structures=['Fe/229/0/1/*'],quantities=['energy','vacancy_energy'])
ref_data = cat_data.get_ref_data()
cat_calc.set_atoms(ref_atoms)

# start from previous model
cat_calc.set_model(new_model)
# optimization variables
var = [{"bond":["Fe","Fe"],'rep1':[True,True],'rep2':[True,True]}]

# set up optimization 
optfunc = CATKernel(calc=cat_calc,ref_data=ref_data,variables=var,log='log3.cat'
                   ,controls=cat_controls)

# run optimization
optfunc.optimize()

# update models
new_model = optfunc.get_optimized_model()
cat_param.models.append(new_model)

# write out new model
new_model.write(filename='models_optimized.bx')

#remove files
os.system('rm -rf log*cat bopfox_keys.bx models_optimized.bx') 
print 'ok.'   
