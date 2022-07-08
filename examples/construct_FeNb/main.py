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

# optimize Fe
ref_atoms = cat_data.get_ref_atoms(structures=['Fe/229/0/1/*'],quantities=['energy'])
ref_data = cat_data.get_ref_data()
cat_calc.set_atoms(ref_atoms)

# optimization variables
var = [{"bond":["Fe","Fe"],'rep1':[True,True],'rep2':[True,True]}
      ]

# set up optimization 
optfunc = CATKernel(calc=cat_calc,ref_data=ref_data,variables=var,log='log1.cat'
                   ,controls=cat_controls)

# run optimization
optfunc.optimize()

# update models
new_model = optfunc.optimized_model
cat_param.models.append(new_model)

# optimize Nb
# start from previous model
cat_calc.set_model(new_model)
ref_atoms = cat_data.get_ref_atoms(structures=['Nb/229/0/1/*'],quantities=['energy'])
ref_data = cat_data.get_ref_data()
cat_calc.set_atoms(ref_atoms)

# optimization variables
var = [{"bond":["Nb","Nb"],'rep1':[True,True],'rep2':[True,True]}
      ]

# set up optimization 
optfunc = CATKernel(calc=cat_calc,ref_data=ref_data,variables=var,log='log2.cat'
                   ,controls=cat_controls)

# run optimization
optfunc.optimize()

# update models
new_model = optfunc.optimized_model
cat_param.models.append(new_model)

# to simplify the procedure, we simply optimize repulsion
# wrt Fe2Nb C14
# generate atoms and data for fitting
# start from previous model
cat_calc.set_model(new_model)
ref_atoms = cat_data.get_ref_atoms(structures=['Fe8Nb4/194/0/1/*'],quantities=['energy'])
ref_data = cat_data.get_ref_data()
cat_calc.set_atoms(ref_atoms)

# optimization variables
var = [{"bond":["Fe","Nb"],'rep1':[True,True],'rep2':[True,True]}
      #,{"bond":["Fe","Fe"],'rep1':[True,True],'rep2':[True,True]}
      #,{"bond":["Nb","Nb"],'rep1':[True,True],'rep2':[True,True]}
      ]

# set up optimization 
optfunc = CATKernel(calc=cat_calc,ref_data=ref_data,variables=var,log='log.cat'
                   ,controls=cat_controls)

# run optimization
optfunc.optimize()

# update models
new_model = optfunc.optimized_model
cat_param.models.append(new_model)

# write out new model
new_model.write(filename='models_optimized.bx')

#remove files
os.system('rm -rf log*cat bopfox_keys.bx models_optimized.bx') 
print 'ok.'   
