#!/usr/bin/env python

# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

from mpi4py import MPI
from .catcalc import CATCalc
import time
from .calc_bopfox import calc_efs_bopfox


def run(atoms, calcs, kwargs):
    out = []
    s = time.time()
    for i in range(len(atoms)):
        if atoms[i] is None:
            continue
        atoms[i].set_calculator(calcs[i])
        out.append(CATCalc.calculate(atoms[i], kwargs))
    return out


def main():
    comm = MPI.Comm.Get_parent()
    status = MPI.Status()
    # initialize hand shake with communicator
    comm.send(None, dest=0, tag=0)
    s = time.time()
    # receive atoms to be calculated
    input = comm.recv(source=0, tag=1)
    atoms, calcs, kwargs = input
    s = time.time()
    # run calculations
    results = run(atoms, calcs, kwargs)
    dt = time.time() - s
    # send results to parent
    comm.send((results, dt), dest=0, tag=2)
    comm.Disconnect()
    # comm.Free()


main()
