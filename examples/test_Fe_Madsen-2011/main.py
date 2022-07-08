from bopcat.catdata import CATData
from bopcat.catparam import CATParam
from bopcat.catcalc import CATCalc
from bopcat.output import bopcat_logo, print_format
from bopcat.plotting import plot_energy_volume
from bopcat.calc_bopfox import calc_phonon_bopfox
import sys
import os

print_format(bopcat_logo(),level=1)

# execute input file and initialize 
execfile(sys.argv[-1])
cat_controls.initialize()

# generate reference data
cat_data = CATData(controls=cat_controls)

# get atoms with energies, if structures is None will get everything
ref_atoms = cat_data.get_ref_atoms(structures=['Fe/229/0/1/*','fcc']
                                  ,quantities=['energy'])

# generate initial parameters
cat_param = CATParam(controls=cat_controls,data=cat_data)

# set up calculator with model
cat_calc = CATCalc(controls=cat_controls,model=cat_param.models[-1])

# calculate energy of ref_atoms with the model
cat_calc.set_atoms(ref_atoms)
model_data = cat_calc.get_property()
# data is now save in atoms
model_atoms = cat_calc.get_atoms()
# plot energy-volume
plot_energy_volume(ref_atoms+model_atoms,filename='EV.png',show_plot=False)

# get ground state structure
gs = cat_data.get_ground_state()
# calculate phonon, solver, problematic
#bs, dos = calc_phonon_bopfox(gs,cat_param.models[-1],show_plot=False
#                            ,solver='phon',cell_size=(4,4,4)
#                            ,k_points=['Gamma','H','P','Gamma','N'])

os.system('rm -rf *png *dat')
print 'ok.'
