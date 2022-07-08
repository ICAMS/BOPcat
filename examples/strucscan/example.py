from bopcat.strucscan_interface import import_data
from bopcat.variables import homedirectory
import os

# list of elements, if empty will return all
elements = ['Re']

# dictionary of dft parameters, if empty will return all
# accepted keys: deltak, xc_functional, encut, spin
params = {}

# string of path to data, the trunk folder should be 'data'
path = os.getcwd()

# string of output filename
filename = 'Re.fit'

# import_data will return list of ASE atoms objects 
atoms = import_data(elements,params,path,filename)

os.system('rm -rf Re.fit *pckl')
if len(atoms) == 48:
    print 'ok.'
else:
    print 'failed.'

