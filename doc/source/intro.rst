Introduction
==============

About
--------------
The **BOPcat** package is a software written in Python to generate
new tight-binding or bond-order potentials or optimize existing models using
the **BOPfox** code. The parameters of the models are determined by reproducing
various target properties including energies, forces, stresses, defect formation energies,
elastic constants, etc. The structures and their properties are taken from a DFT database
but can also include experiments or other data sources.
It employs the optimization libraries of **Scipy** but external optimization
modules can also be used.
The structures and their properties are handled using the **ASE** Atoms object.
These are read from a text file which are generated from an external
database.

Installation
---------------
Prior to using the code, BOPfox should be compiled and the
path to the executable should be included in ``$PATH``. As some BOPfox source files
are read to generate some variables, it is recommended to keep the executable in the
``bopfox/src`` folder. BOPcat utilizes a number of python modules
such as numpy (http://www.numpy.org/), ASE (https://wiki.fysik.dtu.dk/ase/), Scipy (http://www.scipy.org/),
matplotlib (http://www.matplotlib.org/) and pyspglib (https://atztogo.github.io/spglib/). BOPcat depends
on the the BOPfox-ASE interface to calculate the properties of the structure which is not
yet included in the original ASE module. The relevant files are included in the
BOPcat source files. The files ``bopio.py`` and ``bopcal.py`` should be
copied to the ``ase/io`` and ``ase/calculators`` folders respectively. Both files should be renamed
``bopfox.py``. In order to know the path to your ase libraries, simpy execute the following in
python::

    import ase
    ase.__file__

In case you do not have permission to make changes to the ase folder, it is necessary to
install a local version of ASE (https://wiki.fysik.dtu.dk/ase/). A more straightforward
way is to install **Anaconda** (https://www.continuum.io/downloads). The latter is recommended
as this will also update and consolidate all your python modules in a local directory.

.. note:: It may be necessary to restart your computer. 

To make sure that you have set the correct path to ``ase``, execute the following in python::

     from ase.calculators import bopfox as bopcal
     from ase.io import bopfox as bopio

To install ``BOPcat``, run the installation script

.. code-block:: console

   python setup.py install

Alternatively, one simply specify the path to the BOPcat source files, i.e. you should
append the following to your ``.bashrc``:

    ``export PYTHONPATH=<path to bopcat>:/bopcat:$PYTHONPATH``
    ``export PYTHONPATH=<path to bopcat>:$PYTHONPATH``

These make it possible to execute::
 
    from bopcat import variables
    import variables

To test if the required paths are set correctly, execute

.. code-block:: console

    python test_install.py path

Examples
-------------
The examples to test the basic functionalities of the code are found in ``examples/``.
To get started, run the examples in the following order:

1. ``ASE``
    The usage of the BOPfox ASE interface is illustrated. 
    See `example1.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/ASE/example1.py>`_ and  
    `example2.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/ASE/example2.py>`_  

2. ``strucscan``
    BOPcat reference data are constructed from strucscan.
    See `example.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/strucscan/example.py>`_

3. ``optimize_Fe-Madsen-2011``
    An existing Fe model is optimized.
    See `input2.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/optimize_Fe-Madsen-2011/input.py>`_ and
    `main2.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/optimize_Fe-Madsen-2011/main.py>`_

4. ``optimize_Re-Cak-2014``
    An existing W model is optimized for Re.
    See `input3.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/optimize_Re-Cak-2014/input.py>`_ and
    `main3.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/optimize_Re-Cak-2014/main.py>`_

5. ``construct_Fe``
    A new Fe model is constructed.
    See `input4.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/construct_Fe/input.py>`_ and
    `main4.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/construct_Fe/main.py>`_

6. ``construct_FeNb``
    A new FeNb model is constructed.
    See `input5.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/construct_FeNb/input.py>`_ and
    `main5.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/construct_FeNb/main.py>`_

7. ``test_Fe_Madsen-2011``
    BOPcat utilities are illustrated.
    See `input6.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/test_Fe_Madsen-2011/input.py>`_ and
    `main6.py <https://dev.icams.rub.de/projects/projects/smap3d/repository/changes/examples/test_Fe_Madsen-2011/main.py>`_

BOPcat is a collection of tools for the optimizing models. It is necessary for
the user to write a script to specify the procedure. In addition, the input controls
should also be provided which are handled by the ``cat_controls`` object.
Assuming that the main script is ``main.py`` and the input file is ``input.py``, the scripts should
be executed as

.. code-block:: console

    python main.py input.py

The basic form of the script is as follows:

1. Execute and initialize the input controls. These are then attributes of the ``cat_controls`` object::

    execfile(sys.argv[-1])
    cat_contols.initialize()

2. Generate reference data. The ``cat_data`` object essentially reads the text file of structures and their properties (see :ref:`refdata`)::

    cat_data = CATData(controls=cat_controls)

3. Generate or read initial model. The ``cat_param`` object can also be used to store the resulting models at each level of optimization::

    cat_param = CATParam(controls=cat_controls,data=cat_data)

4. Set up the calculator. To set up the calculator, one provides the input controls and a model, in the following, we use the last model (``models[-1]``) saved in ``cat_param`` ::
    
    cat_calc = CATCalc(controls=cat_controls,model=cat_param.models[-1])

5. To proceed with the optimization, one needs to determine which structures are included in the target set. These are then assigned to the calculator::

    ref_atoms = cat_data.get_ref_atoms(structures=['Fe/229/0/1/*'],quantities=['energy'])
    cat_calc.set_atoms(ref_atoms)
    ref_data = cat_data.get_ref_data()

6. The optimization kernel is also necessary which takes in the calculator, the array of reference data, the constraints on variables, the weights and the input controls::

    var  = [{"bond":["Fe","Fe"],'atom':'Fe','onsitelevels':[True]}]
    optfunc = CATkernel(calc=cat_calc,ref_data=ref_data,variables=var,log='log.cat'
                       ,controls=cat_controls)

7. The optimization is run and the resulting model is saved in ``cat_param``::

    optfunc.optimize()
    new_model = optfunc.get_optimized_model()
    cat_param.models.append(new_model)

This can be done iteratively for different sets of target structures, constraints, starting parameters or even a completely different functional form.
