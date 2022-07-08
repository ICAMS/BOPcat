import os, collections, gzip, sys
from ase.io import read as aseread
from .read_outcar import read_outcar
from .read_procar import read


def get_calc_params(path):
    os.chdir(path)
    calc_params = None
    for f in os.listdir(path):
        if f[0:6].lower() == "outcar":
            filename = f.strip('.gz')
            try:
                calc_params = get_calc_params_main(path, filename=filename)
                break
            except:
                pass
    return calc_params


def get_calc_params_main(path, filename=None):
    # Outputs lattice type, cut-off energy, spin, xc functional from
    # OUTCAR file.
    calc_params = {}
    kw = path.split('/')
    os.chdir(path)
    xc_list = []
    xc_functional = ''
    mark = path.find('dk')
    delta_k = float(path[mark + 3:-1])
    if filename == None:
        dirs = os.listdir(path)
        dirs = [d for d in dirs if not d[0] == "."]
        for f in dirs:
            if f[0:6].lower() == 'outcar':
                filename = f.strip('.gz')
                break
    try:
        f = gzip.open(filename + '.gz')
        l = f.readlines()
        f.close()
    except IOError:
        # In strucscan, the OUTCAR is zipped but in cases where it is not,
        try:
            f = open(filename, 'r')
            l = f.readlines()
            f.close()
        except:
            print('No OUTCAR file found.')
            l = []
            lattice_type = None
            cut_off_energy = None
            spin = None
            xc_functional = None
    for j in l:
        j = j.decode()
        if j.find('LATTYP') > 0:
            lattice_type = ''
            for k in j.split()[3:-1]:
                lattice_type += (k + ' ')
        if j.find('ENCUT') > 0:
            cut_off_energy = float(j.split()[2])
        if j.find('ISPIN') > 0:
            spin = int(j.split()[2])
        if j.find('POTCAR') > 0:  # and j.split()[1] not in xc_functional:
            if len(xc_list) < len(kw[-6].split('-')):
                xc_list.append(j.split()[1])
    for j in range(len(xc_list)):
        # if j < len(xc_list)-1:
        #   xc_functional += (xc_list[j]+'+')
        # else:
        #   xc_functional += xc_list[j]
        xc_functional = xc_list[0]  # just get first they are same after all
    calc_params['lattyp'] = lattice_type
    calc_params['encut'] = cut_off_energy
    calc_params['spin'] = spin
    calc_params['xcf'] = xc_functional
    calc_params['dk'] = delta_k
    return calc_params


def get_defect(path, extended):
    atoms_defect = {}
    os.chdir(path)
    kw = path.split('/')
    defect_jobs = ['relax_all',
                   'relax_ion',
                   'relax_volume',
                   'unrelax']
    if kw[-5] == 'defect' and kw[-3] in defect_jobs:
        for job in defect_jobs:
            atoms_defect[job] = {}
            path2atom = ''
            for word in kw:
                if word in defect_jobs:
                    word = job
                path2atom += (word + '/')
            path2atom = path2atom[0:-1]
            if os.path.isdir(path2atom) == True and len(os.listdir(path2atom)) > 0:
                os.chdir(path2atom)
                dirs = os.listdir(path2atom)
                dirs = [d for d in dirs if d[0] not in _symbols()]
                if len(dirs) > 0:
                    try:
                        atoms_defect[job]['init'] = aseread('POSCAR.initial')
                    except IOError:
                        print(('Warning no input POSCAR was found in %s' % path2atom))
                        atoms_defect[job]['init'] = None
                    except:
                        print(('Error reading POSCAR.initial in %s' % path2atom))
                        atoms_defect[job]['init'] = None
                        # raise
                    try:
                        f = 'OUTCAR.gz'
                        atom = aseread(f)
                        if extended:
                            try:
                                calc_info = read_outcar('OUTCAR.gz')
                                try:
                                    procar_info = read('PROCAR')
                                    calc_info.update({'orbital_character': procar_info})
                                except:
                                    print(('PROCAR not found in %s ' % path))
                                # atom.set_calculator_info(calc_info)
                                atom.info = dict(calc_info)
                            except:
                                print('Cannot add details to atoms from OUTCAR.gz')
                        # f.close()
                    except IOError:
                        # In strucscan, the OUTCAR is zipped but in cases where it is not,
                        try:
                            f = 'OUTCAR'
                            atom = aseread(f)
                            if extended:
                                try:
                                    calc_info = read_outcar('OUTCAR')
                                    try:
                                        procar_info = read('PROCAR')
                                        calc_info.update({'orbital_character': procar_info})
                                    except:
                                        print(('PROCAR not found in %s ' % path))
                                    # atom.set_calculator_info(calc_info)
                                    atom.info = dict(calc_info)
                                except:
                                    print('Cannot add details to atoms from OUTCAR')
                            # f.close()
                        except:
                            print(('Warning no vasp output found in %s' % path))
                            atom = None
                    except:
                        print(('Warning no vasp output found in %s' % path))
                        atom = None
                    atoms_defect[job][job] = atom
            else:
                pass
    else:
        atoms_defect = None
    return atoms_defect


def get_cluster(path, extended):
    # Extracts the structure file for every step in constructing volume-energy
    # relationship. Stores these as Atoms object
    atoms_cluster = {}
    # calc_details = {}
    os.chdir(path)
    kw = path.split('/')
    cluster_jobs = ['unrelaxed',
                    'relaxed'
                    ]
    if kw[-5] == 'cluster' and kw[-3] in cluster_jobs:
        for job in cluster_jobs:
            atoms_cluster[job] = {}
            path2atom = ''
            for word in kw:
                if word in cluster_jobs:
                    word = job
                path2atom += (word + '/')
            path2atom = path2atom[0:-1]
            if os.path.isdir(path2atom) == True and len(os.listdir(path2atom)) > 0:
                os.chdir(path2atom)
                dirs = os.listdir(path2atom)
                dirs = [d for d in dirs if d[0] not in _symbols()]
                for d in dirs:
                    if d[0:6].lower() == 'outcar':
                        try:
                            sep = d[6]
                            key = d[7:]
                            key = key.strip('.gz')
                        except:
                            continue
                        try:
                            # f = gzip.open(d)
                            atom = aseread(d)
                            if extended:
                                calc_info = read_outcar(d)
                                try:
                                    procar_info = read('PROCAR%s%s' % (sep, key))
                                    calc_info.update({'orbital_character': procar_info})
                                except:
                                    print(('PROCAR%s%s not found in %s ' % (sep, key, path)))
                                atom.info = dict(calc_info)
                                # atom.set_calculator_info(calc_info)
                            # f.close()
                        except IOError:
                            try:
                                # f = open(d, 'r')
                                atom = aseread(d)
                                if extended:
                                    try:
                                        calc_info = read_outcar(d)
                                        try:
                                            procar_info = read('PROCAR%s%s' % (sep, key))
                                            calc_info.update({'orbital_character': procar_info})
                                        except:
                                            print(('PROCAR%s%s not found in %s ' % (sep, key, path)))
                                        # atom.set_calculator_info(calc_info)
                                        atom.info = dict(calc_info)
                                    except:
                                        raise
                                        print(('Cannot add details to atoms from %s' % d))
                                # f.close()
                            except:
                                raise
                                print(('Warning no vasp output found in %s %s' % (path2atom, d)))
                                atom = None
                        except:
                            print(('Warning no vasp output found in %s %s' % (path2atom, d)))
                            atom = None
                        if len(key) > 2:
                            atoms_cluster[job][key] = atom
                        else:
                            # skip unfinished run in OUTCAR
                            pass
            else:
                pass
    else:
        atoms_cluster = None
    return atoms_cluster


def get_bulk(path, extended):
    # Extracts the structure file for every step in constructing volume-energy
    # relationship. Stores these as Atoms object
    atoms_bulk = {}
    # calc_details = {}
    os.chdir(path)
    kw = path.split('/')
    bulk_jobs = ['elastic_relaxed',
                 'elastic_unrelaxed',
                 'phonons',
                 'relax',
                 'volume_allrelaxed',
                 'volume_relaxed',
                 'volume_unrelaxed',
                 'band_structure']
    if kw[-5] == 'bulk' and kw[-3] in bulk_jobs:
        for job in bulk_jobs:
            atoms_bulk[job] = {}
            path2atom = ''
            for word in kw:
                if word in bulk_jobs:
                    word = job
                path2atom += (word + '/')
            path2atom = path2atom[0:-1]
            if os.path.isdir(path2atom) == True and len(os.listdir(path2atom)) > 0:
                if job == 'relax':
                    os.chdir(path2atom)
                    try:
                        atoms_bulk['relax']['init'] = aseread('POSCAR.initial')
                    except IOError:
                        print(('Warning no input POSCAR was found in %s' % path))
                        atoms_bulk['relax']['init'] = None
                    except:
                        print(('Error reading POSCAR.initial in %s' % path))
                        atoms_bulk['relax']['init'] = None
                        # raise
                    try:
                        f = 'OUTCAR.gz'
                        atoms_bulk['relax']['relax'] = aseread(f)
                        if extended:
                            try:
                                calc_info = read_outcar('OUTCAR.gz')
                                try:
                                    procar_info = read('PROCAR')
                                    calc_info.update({'orbital_character': procar_info})
                                except:
                                    print(('PROCAR not found in %s ' % path))
                                # atoms_bulk['relax']['relax'].set_calculator_info(calc_info)
                                atoms_bulk['relax']['relax'].info = dict(calc_info)
                            except:
                                print('Cannot add details to atoms from OUTCAR.gz')
                        # f.close()
                    except:  # IOError:
                        # In strucscan, the OUTCAR is zipped but in cases where it is not,
                        try:
                            f = 'OUTCAR'
                            atoms_bulk['relax']['relax'] = aseread(f)
                            if extended:
                                try:
                                    calc_info = read_outcar('OUTCAR')
                                    try:
                                        procar_info = read('PROCAR')
                                        calc_info.update({'orbital_character': procar_info})
                                    except:
                                        print(('PROCAR not found in %s ' % path))
                                    # atoms_bulk['relax']['relax'].set_calculator_info(calc_info)
                                    atoms_bulk['relax']['relax'].info = dict(calc_info)
                                except:
                                    print('Cannot add details to atoms from OUTCAR')
                            # f.close()
                        except:
                            print(('Warning no vasp output found in %s' % path))
                            atoms_bulk['relax']['relax'] = None
                    # except:
                    #    print 'Warning no vasp output found in %s'%(path)
                    #    atoms_bulk['relax']['relax'] = None
                else:
                    os.chdir(path2atom)
                    dirs = os.listdir(path2atom)
                    dirs = [d for d in dirs if d[0] not in _symbols()]
                    for d in dirs:
                        if d[0:6].lower() == 'outcar':
                            try:
                                sep = d[6]  # the separator for OUTCAR* files
                                key = d[7:]
                                key = key.strip('.gz')
                            except:
                                continue
                            try:
                                # f = gzip.open(d)
                                atom = aseread(d)
                                if extended:
                                    calc_info = read_outcar(d)
                                    try:
                                        procar_info = read('PROCAR%s%s' % (sep, key))
                                        calc_info.update({'orbital_character': procar_info})
                                    except:
                                        print(('PROCAR%s%s not found in %s ' % (sep, key, path2atom)))
                                    # atom.set_calculator_info(calc_info)
                                    atom.info = dict(calc_info)
                                # f.close()
                            except IOError:
                                try:
                                    # f = open(d, 'r')
                                    atom = aseread(d)
                                    if extended:
                                        try:
                                            calc_info = read_outcar(d)
                                            try:
                                                procar_info = read('PROCAR%s%s' % (sep, key))
                                                calc_info.update({'orbital_character': procar_info})
                                            except:
                                                print(('PROCAR%s%s not found in %s ' % (sep, key, path2atom)))
                                            # atom.set_calculator_info(calc_info)
                                            atom.info = dict(calc_info)
                                        except:
                                            print(('Cannot add details to atoms from %s' % d))
                                    # f.close()
                                except:
                                    print(('Warning no vasp output found in %s %s' % (path2atom, d)))
                                    atom = None
                            except:
                                print(('Warning no vasp output found in %s %s' % (path2atom, d)))
                                atom = None
                            if len(key) >= 3 and atom is not None:
                                atoms_bulk[job][key] = atom
                            else:
                                # skip unfinished run in OUTCAR
                                pass
            else:
                pass
    else:
        atoms_bulk = None
    return atoms_bulk


def _symbols():
    return "!#$%&'()*+,-./:;<=>?@[]^_`{|}~"


def CHECK(s, ls):
    # Used to check path to requested data
    out = False
    for i in ls:
        if str(i) in s:
            out = True
            break
    return out


def CHECKdk(s, ls):
    out = False
    dk = s.split('=')[-1].split('/')[0]
    try:
        dk = float(dk)
    except:
        pass
    for i in ls:
        if i == dk:
            out = True
            break
    return out


def CHECKencut(s, ls):
    out = False
    encut = s.split('=')[2].split(".")[0]
    try:
        dk = float(encut)
    except:
        pass
    for i in ls:
        if i == dk:
            out = True
            break
    return out


def CHECKspin(spin, param):
    out = False
    if spin == []:
        spin = [1, 2]
    if param['spin'] in spin:
        out = True
    return out


def get_paths(path, cmpd_type, elements, structures, calc_type, xcfunc, deltak, encut, exclude):
    # Get paths to all files under provided directory.
    paths = []
    for root, dirs, files in os.walk(path):
        files = [f for f in files if not f[0] == "."]
        # dirs[:] = [d for d in dirs if not d[0] == "."]
        for name in files:
            # p = os.path.join(root,name).replace(name,'')
            p = root + '/'
            if p not in paths:
                paths.append(p)
    # Screen paths to DFT data subject to requested criteria
    count = 0
    scpaths = []
    for i in range(len(paths)):
        kw = paths[i].split('/')

        #         for j in cmpd_type:
        #            check = 0
        #            for k in elements:
        #              try:
        #               if k in kw[-6]:
        #                  check += True
        #              except:
        #               check = 0
        #            if j == 2:
        #               print paths[i], check, j, len(kw[-6].split('-'))
        #            if check != j or len(kw[-6].split('-')) != j:
        #               paths[i] = 'None'
        #               continue
        #            if paths[i] != 'None':
        #               break
        if len(kw) < 6:
            paths[i] = None
            continue
        if len(kw[-6].split('-')) not in cmpd_type:
            paths[i] = 'None'
            continue
        cmpd = kw[-6].split('-')
        cmpd = [e.split('_')[0] for e in cmpd]
        elements = list(elements)
        check = False
        for j in range(len(cmpd)):
            if cmpd[j] not in elements:
                paths[i] = 'None'
                continue
        if structures != [] and CHECK(paths[i], structures) == False:
            paths[i] = 'None'
            continue
        # if structures != [] and not(CHECK(paths[i],exclude)) == False:
        #   paths[i] = 'None'
        #   continue
        if calc_type != [] and CHECK(paths[i], calc_type) == False:
            paths[i] = 'None'
            continue
        if xcfunc != [] and CHECK(paths[i], xcfunc) == False:
            paths[i] = 'None'
            continue
        if deltak != [] and CHECKdk(paths[i], deltak) == False:
            paths[i] = 'None'
            continue
        if encut != [] and CHECKencut(paths[i], encut) == False:
            paths[i] = 'None'
            continue
        if paths[i] != 'None':
            if len(os.listdir(paths[i])) > 0:
                scpaths.append(paths[i])
    return scpaths


def extract(
        # Extracts DFT data from standard strucscan output.
        # Compound type 1=unary, 2=binary,...default is all
        cmpd_type=None,
        # List of elements default is all
        elements=None,
        # Type of calculation, bulk, defect,cluster default is all
        calc_type=None,
        # Structure, bcc,fcc,...default is all
        structures=None,
        # XC-functional, PBE,LDA,... default is all
        xcfunc=None,
        # Cut-off energy default is all
        encut=None,
        # k-points density default is all
        deltak=None,
        # spin-polarization
        spin=None,
        # where the DFT data are located ...default is CWD
        path=None,
        # filename of output pickle file
        filename='dft_data.pckl',
        # if detailed output is required
        verbose=True,
        # extended calculator details required
        extended=[],
        # structures to be excluded
        exclude=[]
):
    ################################################################
    if cmpd_type is None:
        cmpd_type = []
    if elements is None:
        elements = []
    if calc_type is None:
        calc_type = []
    if structures is None:
        structures = []
    if xcfunc is None:
        xcfunc = []
    if encut is None:
        encut = []
    if deltak is None:
        deltak = []
    if spin is None:
        spin = []
    if exclude is None:
        exclude = []
    cwd = os.getcwd()
    # Initialize path and check if provided path is valid
    if path is None:
        path = os.getcwd()
    else:
        path = os.path.expanduser(path)
        if not os.path.isdir(path):
            raise IOError('Provided path: %s does not exist. ' % path)
    # Initialize cmpd_type list to [1:len(elements)] if empty.
    if cmpd_type == []:
        cmpd_type = [i for i in range(1, len(elements) + 1)]
    # Initialize dictionary for output DFT data
    dft_data = {}

    # Get paths to vasp output files.
    if verbose:
        print(("Searching for paths under ", path))
    paths = get_paths(path, cmpd_type, elements, structures, calc_type, xcfunc, deltak, encut, exclude)
    # Extract data and store them in dictionary.
    if verbose:
        print(
            '--------------------------------------------------------------------------------------------------------------')
        print('DFT data were extracted from strucscan database with the following specifications:')
        print(('{0:15} : {1:20} : {2:15} : {3:20} : {4:20} : {5:5}'.format('Stoichiometry',
                                                                           'Structure',
                                                                           'XC Functional',
                                                                           'Energy cut-off(eV)',
                                                                           'dk(/Angstroem)',
                                                                           'spin')))
        print(
            '--------------------------------------------------------------------------------------------------------------')
    done_systems = []
    for i in range(len(paths)):
        kw = paths[i].split('/')
        struc = kw[-4]
        isystem = '%s_%s' % (kw[-6], kw[-4])
        p = get_calc_params(paths[i])
        # filter spin subject to criteria
        if p != None and CHECKspin(spin, p) == True:
            params = p['xcf'] + '$' + str(int(p['encut'])) + '$' + str(p['dk']) + '$' + str(p['spin'])
            if isystem not in done_systems:
                done_systems.append(isystem)
                if struc in extended or 'all' in extended:
                    extend = True
                else:
                    extend = False
                atoms_bulk = get_bulk(paths[i], extend)
                atoms_defect = get_defect(paths[i], extend)
                atoms_cluster = get_cluster(paths[i], extend)
                if atoms_bulk != None:
                    try:
                        stoic = atoms_bulk['relax']['relax'].get_chemical_formula()
                    except AttributeError:
                        print(('Warning: Stoichiometry for relaxed atom in %s cannot be determined.' % paths[i]))
                        print('Obtaining it from initial structure but correctness not guaranteed.')
                        stoic = atoms_bulk['relax']['init'].get_chemical_formula()
                    except:
                        raise ValueError('Cannot extract stoichiometry from atom in %s' % paths[i])
                elif atoms_defect != None:
                    for job in list(atoms_defect.keys()):
                        try:
                            stoic = atoms_defect[job]['init'].get_chemical_formula()
                            break
                        except:
                            pass
                elif atoms_cluster != None:
                    for job in list(atoms_cluster.keys()):
                        for bl in list(atoms_cluster[job].keys()):
                            try:
                                stoic = atoms_cluster[job][bl].get_chemical_formula()
                                break
                            except:
                                pass

                if stoic not in dft_data:
                    dft_data[stoic] = {}
                if struc not in dft_data[stoic]:
                    dft_data[stoic][struc] = {}
                if params not in dft_data[stoic][struc]:
                    dft_data[stoic][struc][params] = {}
                if 'bulk' not in dft_data[stoic][struc][params]:
                    # dft_data[stoic][struc][params]['bulk'] = {}
                    dft_data[stoic][struc][params]['bulk'] = atoms_bulk
                if 'defect' not in dft_data[stoic][struc][params]:
                    # dft_data[stoic][struc][params]['defect'] = {}
                    dft_data[stoic][struc][params]['defect'] = atoms_defect
                if 'cluster' not in dft_data[stoic][struc][params]:
                    # dft_data[stoic][struc][params]['defect'] = {}
                    dft_data[stoic][struc][params]['cluster'] = atoms_cluster
                if verbose:
                    print(('{0:15} : {1:20} : {2:15} : {3:20} : {4:20} : {5:5}'.format(stoic,
                                                                                       struc,
                                                                                       p['xcf'],
                                                                                       p['encut'],
                                                                                       p['dk'],
                                                                                       p['spin']
                                                                                       )))
        os.chdir(cwd)
    import pickle
    a = open(filename, 'wb')
    pickle.dump(dft_data, a)
    return dft_data


if __name__ == "__main__":
    # elements = ['Nb']
    structures = []
    elements = ['Fe']
    cmpd_type = [1]
    calc_type = []
    xcfunc = ['PBE']
    deltak = [0.02]
    encut = [450]
    spin = []
    path = 'data'

    data = extract(
        path=path,
        elements=elements,
        cmpd_type=cmpd_type,
        structures=structures,
        calc_type=calc_type,
        xcfunc=xcfunc,
        deltak=deltak,
        encut=encut,
        spin=spin,
        filename='dft_data.pckl'
    )
