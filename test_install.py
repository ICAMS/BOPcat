#!/usr/bin/env python

import os
import sys

def print_format(message,level=0,stream=False):
    space = ' '*((level+1)*4)
    out = space + message
    print(out)

def warn(msg,fatal=False):
    print_format(msg,level=2)
    if fatal:
        print_format("failed.Exiting.",level=2)
        raise RuntimeError('Failure in test.')

def is_wide_source():
    files = os.listdir('bopcat')
    files = [f for f in files if f[-3:]=='.py']
    maxcol = 250
    has_err = False
    for f in files:
        if f in ['get_strucscan_data.py']:
            continue
        lines = open('bopcat/%s'%f).readlines()
        for i in range(len(lines)):
            if len(lines[i]) <= maxcol:
                continue
            print("LONG LINE:",lines[i])
            has_err = True
            err_msg = '''Too long lines in file %s.
                         Line number %d.
                      '''%(f,i)
            print_format(err_msg,level=3) 
    return has_err

print_format('Checking source files...')
if is_wide_source():
    warn('failed.Edit source code.',True)
else:
    warn('ok.')

print_format('Installing BOPcat...')
try:
    os.system('python setup.py install > log')
    warn('ok.')
    os.system('rm -rf log build dist *egg*')
except:
    warn('failed.See log for details.',True)

print_format('Copying bopfox-ase interface..')
try:
    import ase
    p2ase = os.path.dirname(ase.__file__)
    os.system('cp bopcat/bopcal.py %s/calculators/bopfox.py'%p2ase)
    os.system('cp bopcat/bopio.py %s/io/bopfox.py'%p2ase)
    warn('ok.')
except:
    raise
    warn('failed.copying unsuccessful.',True) 

print_format("Testing bopfox-ase interface..")
try:
    from ase.calculators import bopfox as bopio
    from ase.calculators import bopfox as bopcal
    warn('ok.')
except:
    warn('failed.bopfox-ase interface not set.',True) 

print_format('Testing modules...')
try:
    from bopcat import *
    warn('ok.')
except:
    warn('failed.not all modules can be exported.',True)

print_format('Testing external modules...')
try:
    import spglib
    import scipy
    import matplotlib
    import numpy
    warn('ok.')
except:
    warn('failed.not all modules can be exported.',True)


print_format("Identifying path to bopfox..")
#path = os.environ['PATH']
#if not 'bopfox' in path.lower():
path = os.popen('which bopfox').read()
if 'bopfox' not in path.lower():
    warn('failed.path to bopfox not found.')
else:
    path = path.split(':')
    for p in path:
        if 'bopfox' in p.lower():
            warn('found bopfox in %s'%p)
            break

if sys.argv[-1].lower() == 'path':
    sys.exit()
# run test
print_format('Running tests')
test_dirs = ['ASE'
            ,'strucscan'
            ,'optimize_Fe_Madsen-2011'
            ,'optimize_Re_Cak-2014'
            ,'construct_Fe'
            #,'construct_FeNb'
            ,'test_Fe_Madsen-2011'
            ,'structmap_examples'
            ,'notebook'
            ,'penalty']
test_scripts = [['example1.py','example2.py']
               ,['example.py']
               ,['main.py input.py']
               ,['main.py input.py']
               ,['main.py input.py']
               #,['main.py input.py']
               ,['main.py input.py']
               ,['run1.py']
               ,['process1.py']
               ,['run.py']]
test_fails = []
cwd = os.getcwd()
for i in range(len(test_dirs)):
    os.chdir('examples/'+test_dirs[i])
    temp = []
    for j in range(len(test_scripts[i])): 
        print_format('%s-%s'%(test_dirs[i],test_scripts[i][j]),level=2)
        if test_scripts[i][j].split('.')[-1] == 'py':
            os.system('python %s > log'%test_scripts[i][j])
        else:
            os.system('./%s > log'%test_scripts[i][j])
        fail = True
        lines = open('log').readlines()
        for line in lines:
            if 'ok.' in line.lower():
                fail = False
                break
        if fail:
            print_format('failed.',level=3)
        else:
            print_format('ok.',level=3)
            os.system('rm -rf log')
        temp.append(fail)
    test_fails.append(temp)
    os.chdir(cwd)


