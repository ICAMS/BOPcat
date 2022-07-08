from bopcat.catdata import CATData
from bopcat.catparam import CATParam
from bopcat.catcalc import CATCalc
from bopcat.catkernel import CATKernel
import sys
from bopcat.output import bopcat_logo, print_format
import numpy as np
import bopcat.plotting as plotting
from bopcat.bopmodel import read_modelsbx
import matplotlib.pyplot as pl
import os

print_format(bopcat_logo(), level=1)

# execute input file and initialize
from bopcat.catcontrols import CATControls
cat_controls = CATControls()
execfile(sys.argv[-1])
cat_controls.initialize()

def sum_exp(p, dist):
    n_terms = len(p)/3
    fit = np.zeros(len(dist))
    for i in range(n_terms):
        fit += p[i*3] * np.exp(-abs(p[i*3+1]*dist**p[i*3+2]))

    return fit

def simple_exp(p, dist):
    fit = p[0] * np.exp(-abs(p[1]*dist))

    return fit


# generate reference data
cat_data = CATData(controls=cat_controls)

# generate initial parameters
cat_param = CATParam(controls=cat_controls,data=cat_data)

# set up calculator
cat_calc = CATCalc(controls=cat_controls,model=cat_param.models[-1])

model_madsen = read_modelsbx(model=['Madsen-2011'], system=['Fe'],
                             filename='models_optimized_rep.bx')
p_ddsigma_madsen = model_madsen[0].bondsbx[0].get_bondspar()['ddsigma']
p_ddpi_madsen = model_madsen[0].bondsbx[0].get_bondspar()['ddpi']
p_dddelta_madsen = model_madsen[0].bondsbx[0].get_bondspar()['dddelta']

p_ddsigma_before = cat_param.models[-1].bondsbx[0].get_bondspar()[
    'ddsigma']
p_ddpi_before = cat_param.models[-1].bondsbx[0].get_bondspar()['ddpi']
p_dddelta_before = cat_param.models[-1].bondsbx[0].get_bondspar()[
    'dddelta']

# generate atoms and data for fitting
ref_atoms = cat_data.get_ref_atoms(structures=['Fe/229/0/1/*'],
                                   quantities=['energy'])
ref_data = cat_data.get_ref_data()
cat_calc.set_atoms(ref_atoms)

cat_calc.get_property()
bopatoms = cat_calc.get_atoms()
# plotting.plot_energy_volume(ref_atoms + bopatoms,
#                             filename='ev_before_opt.png')

# optimization variables
const = [True, True, False,
         True, True, True,
         False, False, False,
         False, False, False,
         False, False, False,
         False, False, False,
         False, False, False]

var = [{"bond": ["Fe", "Fe"],"ddsigma":const, "ddpi":const,
        "dddelta":const, 'atom':'Fe','onsitelevels':[False]
        }]

# set up optimization 
optfunc = CATKernel(calc=cat_calc, ref_data=ref_data, variables=var,
                    log='log.cat', controls=cat_controls)

# run optimization
optfunc.optimize()

# update models
new_model = optfunc.get_optimized_model()
cat_param.models.append(new_model)

cat_calc.set_model(new_model)

cat_calc.get_property()
bopatoms = cat_calc.get_atoms()
# plotting.plot_energy_volume(ref_atoms + bopatoms,
#                             filename='ev_after_opt.png')

p_ddsigma_after = new_model.bondsbx[0].get_bondspar()['ddsigma']
p_ddpi_after = new_model.bondsbx[0].get_bondspar()['ddpi']
p_dddelta_after = new_model.bondsbx[0].get_bondspar()['dddelta']

# compare all bond integals for comparison
pl.figure()
dist = np.arange(2.2, 8.0, 0.001)
ddsigma_before = sum_exp(p_ddsigma_before, dist)
ddpi_before = sum_exp(p_ddpi_before, dist)
dddelta_before = sum_exp(p_dddelta_before, dist)
pl.plot(dist, ddsigma_before, color='r', linestyle='-',
        label=r'$dd\sigma$ (before)')
pl.plot(dist, ddpi_before, color='b', linestyle='-',
        label=r'$dd\pi$ (before)')
pl.plot(dist, dddelta_before, color='g', linestyle='-',
        label=r'$dd\delta$ (before)')

ddsigma_after = sum_exp(p_ddsigma_after, dist)
ddpi_after = sum_exp(p_ddpi_after, dist)
dddelta_after = sum_exp(p_dddelta_after, dist)
pl.plot(dist, ddsigma_after, color='r', linestyle='--',
        label=r'$dd\sigma$ (after)')
pl.plot(dist, ddpi_after, color='b', linestyle='--',
        label=r'$dd\pi$ (after)')
pl.plot(dist, dddelta_after, color='g', linestyle='--',
        label=r'$dd\delta$ (after)')

ddsigma_madsen = simple_exp(p_ddsigma_madsen, dist)
ddpi_madsen = simple_exp(p_ddpi_after, dist)
dddelta_madsen = simple_exp(p_dddelta_madsen, dist)
pl.plot(dist, ddsigma_madsen, color='r', linestyle=':',
        label=r'$dd\sigma$ (madsen)')
pl.plot(dist, ddpi_after, color='b', linestyle=':',
        label=r'$dd\pi$ (madsen)')
pl.plot(dist, dddelta_after, color='g', linestyle=':',
        label=r'$dd\delta$ (madsen)')
pl.legend()
pl.xlabel(r'$R$ (A)')
pl.ylabel(r'$\beta$ (eV)')
# pl.savefig('betas.png')


# write out new model
new_model.write(filename='models_optimized.bx')

#remove files
os.system('rm -rf log*cat bopfox_keys.bx models_optimized.bx model_Fe.bx')
print 'ok.'    
