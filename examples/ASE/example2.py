from ase.calculators import bopfox
from ase.build import bulk
import bopcat.bopmodel as bm

# modelsbx is essentially infox_parameters, atomsbx and bondsbx concatenated
abx = bm.atomsbx(version='canonicaltb'
                ,atom='Fe'
                ,mass=55.845
                ,valenceelectrons=[6.9]
                ,valenceorbitals=[5]
                ,jii = [2]
                ,onsitelevels=[0]
                ,stonerintegral=[0.75,0.75,0.75]
                )

bbx = bm.bondsbx(version='bochumtb',
                 bond=['Fe','Fe'],
                 valence=['d','d'],
                 scaling=[1.0],
                 ddsigma=[-29.6584,1.5583],
                 ddpi   =[62.2039,2.0097],
                 dddelta=[-51.7261,2.6153],
                 rep1=[1221.56485*2.,3.2,0.00000],
                 rep2=[-4.39322,0.36],
                 eamcoupling  =[0.00000],
                 functions={'ddsigma':'exponential'
                           ,'ddpi':'exponential'
                           ,'dddelta':'exponential'
                           ,'rep1':'pp_exponential'
                           ,'rep2':'emb_sqrt_gaussian'},
                 rcut=3.5,dcut=0.5,r2cut=5.5,d2cut=0.5)

infox = {
        'bandwidth':'findeminemax'
        ,'terminator':'constantabn'
        ,'bopkernel':'jackson'
        ,'nexpmoments':100
        ,'moments':6
        ,'version':'bop'
        }

# generate modelsbx, atomsbx and bondsbx are LISTS of atomsbx and bondsbx objects
modelsbx = bm.modelsbx(model='test',infox_parameters=infox,atomsbx=[abx],bondsbx=[bbx])

# we can then pass this to the bopfox calculator
calc = bopfox.BOPfox(modelsbx=modelsbx,task='force',model='test')

# create structure
atom = bulk('Fe')
atom.set_initial_magnetic_moments([3]*len(atom))

# attach calculator to ASE atoms
atom.set_calculator(calc)

# calculate energy
ene = atom.get_potential_energy()
print 'energy: ', ene
tol = 0.0001
if abs(-5.0271668176408726-ene) < tol:
    print 'ok.'
else:
    print 'failed.'

