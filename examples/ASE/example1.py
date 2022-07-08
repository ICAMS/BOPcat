from ase.calculators import bopfox
from ase.build import bulk

# create calculator
calc = bopfox.BOPfox(modelsbx='models.bx',task='energy',model='Madsen-2011')

# create structure
atom = bulk('Fe')
atom.set_initial_magnetic_moments([3]*len(atom))

# attach calculator to ASE atoms
atom.set_calculator(calc)

# calculate energy
ene = atom.get_potential_energy()
print 'energy: ',ene

tol = 0.0001
if abs(-8.07027289558-ene) < tol:
    print 'ok.'
else:
    print 'failed.'
