"""
Extracts information from OUTCAR 
Uses ase.io.vasp
"""

from ase.io import vasp
import numpy as np
from ase.units import GPa
import gzip
from ase.dft.kpoints import kpoint_convert, ibz_points
from .utils import get_lattice_type


def read_atom(filename='OUTCAR', energy_without_entropy='False'):
    if isinstance(filename, str):
        # any better way to determine if gzip file??
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    atom = vasp.read_vasp_out(f)
    return atom


def _unique(atom):
    sym = atom.get_chemical_symbols()
    unique = []
    for s in sym:
        if s not in unique:
            unique.append(s)
    return unique


def read_version(filename='OUTCAR'):
    """ filename = string or file-like object
    
    Returns the VASP version used which created the OUTCAR
    """
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    for line in l:
        if line.find(' vasp.') != -1:  # find the first occurence
            version = line[len(' vasp.'):].split()[0]
            break
    f.close()
    return version


def read_fermi_level(filename='OUTCAR'):
    """ filename = string or file-like object
    """
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    for line in l:
        if line.rfind('E-fermi') > -1:
            E_f = float(line.split()[2])
    f.close()
    return E_f


def read_xcf_version(filename='OUTCAR'):
    """ filename = string or file-like object
    """
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    atom = read_atom(filename)
    l = f.readlines()
    unique = _unique(atom)
    pot_ver = {}
    for i in range(len(unique)):
        for j in range(len(l)):
            if 'VRHFIN' in l[j] and unique[i] in l[j]:
                pot_ver[unique[i]] = l[j].split()[-1]
                break
    del l, unique
    f.close()
    return pot_ver


def read_number_of_iterations(filename='OUTCAR'):
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    initer = None
    for line in l:
        if line.find('- Iteration') != -1:  # find the last iteration number
            # niter = int(line.split(')')[0].split('(')[-1].strip())
            niter = int(line.split('(')[0].split()[-1].strip())
    del l
    f.close()
    return niter


def read_nbands_nkpts(filename='OUTCAR'):
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    for line in l:
        if line.rfind('NBANDS=') > -1:
            nbands = int(line.split()[-1])
            nkpts = int(line.split()[-6])
            break
    del l
    f.close()
    return [nbands, nkpts]


def read_spin(filename='OUTCAR'):
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    for line in l:
        if line.rfind('ISPIN') > -1:
            spin = int(line.split()[2])
            break
    f.close()
    return spin


def _is_special(p, special_p, tolerance=0.00000001):
    # points should be in cartesian coordinates!
    out = False
    which_point = None
    for name, point in list(special_p.items()):
        dv = point - p
        d = np.sqrt(np.dot(dv, dv))
        if d < tolerance:
            out = True
            which_point = name
            break
    return out, which_point


def read_eigenvalues(filename='OUTCAR', cartesian=True, shift_Fermi=False
                     , write=False):
    """Reads in the eigenvalues from an OUTCAR file
       if cartesian is False k_points will be in reciprocal coordinates
       write flag relevant only for band structure calculations
    """
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename  # not
        f.seek(0)
    l = f.readlines()
    atom = read_atom(filename)
    spin = read_spin(filename)
    nbands, nkpts = read_nbands_nkpts(filename)
    niter = read_number_of_iterations(filename)
    eigs = np.zeros([niter, spin, nkpts, nbands])
    occs = np.zeros([niter, spin, nkpts, nbands])
    kpts = np.zeros([niter, spin, nkpts, 3])
    i = -1
    band_struc = False
    for j in range(len(l)):
        if 'k-points for band structure' in l[j]:
            band_struc = True
        if 'band No.' in l[j]:
            try:
                ki = int(l[j - 1].split()[1]) - 1
            except:
                print(('''!!!Warning: Invalid k-point number read in 
                             line %d of %s''' % (j, filename)))
                ki = ki + 1
                print(('   Setting it to %d. This may be incorrect!' % ki))
            if ki == 0:
                if spin == 2:
                    si = int(l[j - 3].split()[-1]) - 1
                else:
                    si = 0
                if si == 0:
                    i += 1
            for k in range(nbands):
                temp = l[j + k + 1].split()
                eigs[i][si][ki][k] = temp[-2]
                occs[i][si][ki][k] = temp[-1]
            kpts[i][si][ki] = l[j - 1].split()[-3:]
    if shift_Fermi:
        eigs -= read_fermi_level(filename)
    # k-points are in reciprocal coordinates default should be 
    # cartesian(in units of 2pi)
    if cartesian:
        kpts = kpoint_convert(atom.get_cell(), skpts_kc=kpts) / (2. * np.pi)
    f.close()
    # write out k_points 
    if write:
        if not (band_struc):
            print('''Warning: This is not a bandstructure calculation but write
                     is set to True. Output distances are meaningless.''')
        lattyp, sgnum = get_lattice_type(atom)
        if sgnum == 229:
            lattyp = 'bcc'
        elif sgnum == 225:
            lattyp = 'fcc'
        else:
            pass
        i_points = dict(ibz_points[lattyp])
        temp = list(i_points.values())
        temp = kpoint_convert(atom.get_cell(), skpts_kc=temp)
        i = 0
        for key in list(i_points.keys()):
            i_points[key] = temp[i]
            i += 1
        for i in range(len(eigs)):
            for j in range(len(eigs[i])):
                if not (cartesian):
                    temp = kpoint_convert(atom.get_cell(), skpts_kc=kpts[i][j])
                else:
                    temp = np.array(kpts[i][j]) * 2 * np.pi
                f = open('%s_eigs_iter%d_spin%d.dat' % (filename, i, j), 'w')
                dist = [0.]
                s, p = _is_special(temp[0], i_points)
                print(('Special point %s is at distance 0.' % p))
                for k in range(1, len(eigs[i][j])):
                    dv = (temp[k] - temp[k - 1])
                    delta = np.sqrt(np.dot(dv, dv))
                    s, p = _is_special(temp[k], i_points)
                    if s:
                        print(('Special point %s is at distance %3.5f' \
                               % (p, delta + dist[k - 1])))
                    dist.append(delta + dist[k - 1])
                for k in range(len(eigs[i][j])):
                    f.write('%5.3f  ' % dist[k])
                    for l in range(len(eigs[i][j][k])):
                        f.write('%3.5f  ' % eigs[i][j][k][l])
                    f.write('\n')
                f.close()
    # get only last iteration
    return kpts[0], eigs[0], occs[0]


def read_ibz_k_points(filename='OUTCAR'):
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    for j in range(len(l)):
        if 'k-points for band structure' in l[j]:
            print('''Detected a band structure run. Use read_eigenvalues 
                     instead to parse k_points''')
            return [], []
    for i in range(len(l)):
        if 'KPOINTS:' in l[i]:
            if 'Gamma-point only' in l[i]:
                Nk = 1
                ibz_k_points_weight_rec = np.zeros([1, 4])
                ibz_k_points_weight_car = np.zeros([1, 4])
        if 'irreducible k-points' in l[i]:
            Nk = int(l[i].split()[1])
        if 'Following reciprocal coordinates' in l[i]:
            ibz_k_points_weight_rec = np.zeros([Nk, 4])
            for j in range(Nk):
                temp = l[i + j + 2].split()
                ibz_k_points_weight_rec[j] = temp[0:4]
        if 'Following cartesian coordinates' in l[i]:
            ibz_k_points_weight_car = np.zeros([Nk, 4])
            for j in range(Nk):
                temp = l[i + j + 2].split()
                ibz_k_points_weight_car[j] = temp[0:4]
            break
    f.close()
    return ibz_k_points_weight_rec, ibz_k_points_weight_car


def read_electronic_temperature(filename='OUTCAR'):
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    sigma = None
    for i in range(len(l)):
        if 'Fermi-smearing in eV        SIGMA' in l[i]:
            sigma = float(l[i].split('=')[1].strip())
    f.close()
    return sigma


def read_number_of_electrons(filename='OUTCAR'):
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    for i in range(len(l)):
        if 'total number of electrons' in l[i]:
            nelect = float(l[i].split('=')[1].split()[0].strip())
    f.close()
    return nelect


def read_stress(filename='OUTCAR'):
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    for i in range(len(l)):
        if 'in kB' in l[i]:
            stress = -np.array([float(a) for a in l[i].split()[2:]]) \
                [[0, 1, 2, 4, 5, 3]] \
                     * 1e-1 * GPa

    f.close()
    return stress


def read_magnetic_moments(filename='OUTCAR'):
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    atom = read_atom(filename)
    N = atom.get_number_of_atoms()
    magnetic_moments = np.zeros(N)
    for i in range(len(l)):
        if 'magnetization (x)' in l[i]:
            for m in range(N):
                magnetic_moments[m] = float(l[i + m + 4].split()[4])
            break
    f.close()
    return magnetic_moments


def read_magnetic_moment(filename='OUTCAR'):
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    atom = read_atom(filename)
    N = atom.get_number_of_atoms()
    for i in range(len(l)):
        if 'number of electron ' in l[i]:
            mag_mom = float(l[i].split()[-1])
    return mag_mom


def is_converged(filename='OUTCAR'):
    """
    Checks whether the calculation which is associated with the OUTCAR file is 
    actually converged.
    """
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    converged = None
    # First check electronic convergence
    for line in l:
        if line.rfind('IBRION') > -1:
            ibrion = int(line.split()[2])
        if line.rfind('reached required accuracy') > -1:
            relaxed = True
        if line.rfind('EDIFF  ') > -1:
            ediff = float(line.split()[2])
        if line.rfind('total energy-change') > -1:
            split = line.split(':')
            a = float(split[1].split('(')[0])
            b = float(split[1].split('(')[1][0:-2])
            if [abs(a), abs(b)] < [ediff, ediff]:
                converged = True
            else:
                converged = False
                continue
    # Then if ibrion > 0, check whether ionic relaxation condition 
    # been fulfilled

    if ibrion > 0:
        if not relaxed:
            converged = False
        else:
            converged = True
    f.close()
    return converged


def read_mass_valence(filename='OUTCAR'):
    if isinstance(filename, str):
        if filename.split('.')[-1] == 'gz':
            f = gzip.open(filename)
        else:
            f = open(filename)
    else:
        f = filename
        f.seek(0)
    l = f.readlines()
    mass = {}
    valence = {}
    for i in range(len(l)):
        if 'TITEL' in l[i]:
            element = l[i].split()[3].split('_')[0]
            for j in range(i, len(l)):
                if 'POMASS' in l[j] and 'ZVAL' in l[j]:
                    s = l[j].split()
                    m = float(s[2].strip(';'))
                    v = float(s[5])
                    mass[element] = m
                    valence[element] = v
                    break
    return mass, valence


def read_outcar(filename='OUTCAR'):
    """Reads relevant data from an OUTCAR file. These include
           eigenvalues
               eigenvalues[iteration][spin][Nk][band]
           k_points
               kpoints[Nk]
           occupation_numbers
           ibz_k_points_weights_rec
           ibz_k_points_weights_car
           number_of_iterations
           electronic_temperature
           number_of_electrons
           stress
           number_of_bands
           spin
           fermi_level
           xcf_version
           magnetic_moments
           is_converged
        Most of the functions are taken from ase.calculators.vasp 
    """
    out = {}
    for key in ['eigenvalues',
                'k_points',
                'occupation_numbers',
                'ibz_k_points_weights_rec',
                'ibz_k_points_weights_car',
                'number_of_iterations',
                'electronic_temperature',
                'number_of_electrons',
                'stress',
                'number_of_bands',
                'number_of_k_points',
                'spin',
                'fermi_level',
                'xcf_version',
                'magnetic_moments',
                'is_converged']:
        out[key] = None
    temp = read_eigenvalues(filename)
    out['mass'], out['number_of_valence_electrons'] = \
        read_mass_valence(filename)
    out['k_points'] = temp[0]
    out['eigenvalues'] = temp[1]
    out['occupation_numbers'] = temp[2]
    temp = read_ibz_k_points(filename)
    out['ibz_k_points_weights_rec'] = temp[0]
    out['ibz_k_points_weights_car'] = temp[1]
    out['number_of_iterations'] = read_number_of_iterations(filename)
    out['electronic_temperature'] = read_electronic_temperature(filename)
    out['number_of_electrons'] = read_number_of_electrons(filename)
    temp = read_nbands_nkpts(filename)
    out['number_of_bands'] = temp[0]
    out['number_of_k_points'] = temp[1]
    out['spin'] = read_spin(filename)
    out['xcf_version'] = read_xcf_version(filename)
    out['fermi_level'] = read_fermi_level(filename)
    out['stress'] = read_stress(filename)
    out['magnetic_moments'] = read_magnetic_moments(filename)
    out['coord_k'] = 'cartesian'
    return out
