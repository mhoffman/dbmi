#!/usr/bin/env python

__version__ = "0.0.1"
VERSION = __version__

import os.path

import pprint
import collections
import unittest
import pickle

import numpy as np
import scipy.signal
import espresso.io
import ase.units

import dbmi.util


import joblib
import tempfile

memory = joblib.Memory(cachedir=tempfile.mkdtemp(), verbose=0)

def get_dipole_energy(dipole1, dipole2, distance):
    """Calculate potential energy between two dipole, assuming
    they are alinged in parallel. Dipole units are electron*Angstrom
    and energy is eV.

    Potential energy between two dipoles:
    $$
    U({\bf p_1}, {\bf p_2}, {\bf \hat r}) = - \frac{{\bf p_1}\cdot {\bf p_2} - 3 ({\bf p_1} \cdot {\bf \hat r} ) (\bf p_2 \cdot \hat{r})}{ 4 \pi \epsilon_{0} r^{3}}
    $$

    which is for the simple case of two parallel dipoles here:

    $$
    U = \frac{p_{1}p_{2}}{4 \pi \epsilon_0 r^3}
    $$



    """
    return (dipole1 * ase.units.Angstrom) * (dipole2 * ase.units.Angstrom) \
        / 4 / np.pi / (distance * ase.units.Angstrom) ** 3 \
        / ase.units._eps0 / ase.units.C * ase.units.m


def get_dipole(charge, geom, z_only=True, verbose=False):
    """
        Extract the dipole in the z-direction (3rd axis).


    """
    ion_center = np.zeros((3, ))
    e_center = np.zeros((3, ))
    ion_charge = 0
    e_charge = charge.sum()

    cell = geom.cell

    if verbose:
        print("get_dipole cell = {cell}\n\n".format(**locals()))

    X, Y, Z = map(np.complex, charge.shape)
    dx, dy, dz =  map(lambda x: 1./x, charge.shape)

    if not z_only:
        XYZ = np.column_stack(map(lambda x: x.flatten(), np.mgrid[0.:1.:dx, 0.:1.:dy, 0.:1.:dz]))
        e_center = - np.dot(cell.T, np.sum(XYZ * charge.flatten()[:, None], axis=0) / 1.)

        if verbose:
            print('    E-center   {e_center} [A, A, A]'.format(**locals()))
            print('    Total electron charge {charge_sum} e'.format(charge_sum=charge.sum()))

        for atom in geom:
            ion_contrib = atom.position * dbmi.util.pseudo_charges[atom.symbol]
            ion_center += ion_contrib
            ion_charge += dbmi.util.pseudo_charges[atom.symbol]

        if verbose:
            print('    Ion-center {ion_center} [A, A, A]'.format(**locals()))
            print('    Total ion charge {ion_charge} e'.format(**locals()))

        return (ion_center + e_center)[-1]
    else:
        ion_center = 0.
        e_center = 0.

        total_electron_charge = charge.sum()

        shape = charge.shape
        if verbose:
            print('CHARGE GRID {charge.shape}'.format(**locals()))
            print('CELL {cell}'.format(**locals()))

        for atom_i, atom in enumerate(geom):
            ion_contrib = geom.get_positions()[atom_i][-1] * dbmi.util.pseudo_charges[atom.symbol]
            ion_center += ion_contrib
            ion_charge += dbmi.util.pseudo_charges[atom.symbol]

        z_charge = charge.sum(axis=0).sum(axis=0) * (1) * ion_charge / charge.sum()
        Z = (np.mgrid[0.:1.:dz] ) * cell[-1, -1]

        e_center =  np.sum(Z * z_charge)
        e_center /=  z_charge.sum()
        ion_center /= ion_charge

        ion_dipole = ion_center * ion_charge
        e_dipole = e_center * charge.sum()

        dipole = (ion_dipole - e_dipole)

        if verbose:
            print('    Z evaluation only {z_only}'.format(**locals()))
            print('    E-center   {e_center} A'.format(**locals()))
            print('    Total electron charge {charge_sum} e'.format(charge_sum=charge.sum()))
            print('    Rescaled electron charge {z_charge_sum} e'.format(z_charge_sum=z_charge.sum()))

            print('    Ion-center {ion_center} A'.format(**locals()))
            print('    Total ion charge {ion_charge} e\n'.format(**locals()))
            print('    Electron dipole {e_dipole} | ion_dipole {ion_dipole}\n\n\n'.format(**locals()))
            print('    Dipole {dipole} eA'.format(**locals()))

        return dipole



def extract_charge(pickle_filename, verbose=True, clip=True):
    """

    Extract charge from a pickle file and crop and normalize to that all voxels sum to the total charge

    WARNING: This function assumes that the charge was extract using the ase-espressso calculator's
    calc.extract_total_potential() function, which is why this methods divides by Rydberg.
    If the charge density was extract using a different (i.e. the correct method) you
    may have to multiply with Rydberg again. In any case sum up all charge density and make sure
    this agrees within less of a electron of what you expect for your system.

    :param filename: path of pickle file to open
    :type pickle_filename: str or file
    :param verbose: Switch wether function should print status messages to stdout
    :type verbose:  bool
    :param clip: Switch wether the last slice along each axis should be dropped. By default QE puts a periodic copy at the opposite end of the supercell.
    :type verbose: bool

    """
    if type(pickle_filename) is str:
        with open(pickle_filename) as infile:
            origin, cell, charge = pickle.load(infile)
    elif type(pickle_filename) is file:
        origin, cell, charge = pickle.load(infile
    else:
        raise UserWarning('Filename argument {pickle_filename} is neither a file nor a path'.format(**locals()))

    # convert unit cell to atomic units (bohr radii)
    bohr = ase.units.Bohr * (1. )
    #bohr = 0.5291772108
    cb = cell_bohr = cell / bohr

    # calculate cell volume
    volume = np.dot(np.cross(cb[0], cb[1]), cb[2])

    # clip copies from charge density
    if clip:
        charge = charge[:-1, :-1, :-1]

    voxels = np.array(charge.shape).prod()

    voxel_volume = volume / voxels
    charge *= voxel_volume / ase.units.Rydberg

    return origin, cell, charge


def merge(dict1, dict2):
    """Recursively merge two dictionaries
    Kudos: http://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge
    """
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(merge(dict1[k], dict2[k])))
            else:
                # If one of the values is not a dict, you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                yield (k, dict2[k])
                # Alternatively, replace this with exception raiser to alert
                # you of value conflicts
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])


def calculate_interaction_energy(interactions, adsorbates, DR=4, ER1=5, ER2=5, pbc=None, verbose=False, comment='', dipole_contribution=False):
    """Calculate the adsorbate-adsorbate interaction based in d-band perturbations and electrostatic dipoles
    of isolated adsorbates.

    :param interaction: Data extracted from isolated adsorbates specifying the d-band perturbuation and electrostatic dipole
    :type interaction: dict
    :param adsorbates: List of tuple encoding the relative positions of adsorbates on high-symmetry surface sites. If tuple is of the form (surface, adsorbate, site, rel_x, rel_y).
    :type adsorbates: list
    :param DR: cut-off radius for calculating the d-band mediated interaction (default: 3).
    :type DR: int
    :param ER: cut-off radius for electrostatic dipole interaction contributions (default: 4).
    :type ER: int
    :param verbose: Flag to control whether status messages are printed during the calculation.
    :type verbose: bool
    :param comment: Comment string that appear be in the resulting (verbose) output.
    :type comment: str

    """

    dipole_factor = 2.
    # Kohn, W., and K. -H. Lau.
    # Adatom Dipole Moments on Metals and Their Interactions.
    # Solid State Communications 18, no. 5 (1976): 553-555.
    # doi:10.1016/0038-1098(76)91479-4.

    if verbose:
        if pbc:
            print('\n\nUnit cell Size ({}x{})'.format(pbc[0], pbc[1]))
        else:
            print('Non-periodic calculation')

    # setup interaction lattice
    xs = [x for _, _, _, x, _ in adsorbates]
    ys = [y for _, _, _, _, y in adsorbates]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    # get-consistent dipole-dipole interaction parameter
    if pbc:
        ER1 = int(ER2 * np.sqrt(pbc[0]*pbc[1]))
    else:
        ER1 = ER2

    lattice = np.array
    # the d-bands shifts the i-th atom see, due to all other atoms
    lattice_dbands = [np.zeros(
        (max_x - min_x + 2 * DR, max_y - min_x + 2 * DR)) for _ in range(len(adsorbates))]

    # the d-bands shifts to due the i-th atom
    # so that we can subtract later from the maximum
    self_int = [np.zeros((max_x - min_x + 2 * DR, max_y - min_x + 2 * DR))
                for _ in range(len(adsorbates))]

    # calculate d-band shifts based on other adsorbates
    for i, adsorbate in enumerate(adsorbates):
        surface, molecule, site, rel_x, rel_y = adsorbate
        if molecule is None :
            continue
        if verbose:
            print("Metal: {surface},  molecule: {molecule}, site: {site}, (X, Y) = ({rel_x}, {rel_y})".format(
                **locals()))

        try:
            d_shift_sum = sum(interactions[surface][molecule][site]['V'].get(x, {}).get(y, 0)
                              for x in range(-DR, DR) for y in range(-DR, DR))
        except KeyError:
            raise UserWarning('Interaction database lacks necessary parameters for {surface} {molecule} {site}'.format(**locals()))

        X, Y = lattice_dbands[i].shape
        for x in range(X):
            for y in range(Y):
                for j in range(len(adsorbates)):
                    # if True:
                    if j != i:

                        # this factor rescale how strong
                        # adsorbates interaction with the d-band center
                        adsorbate_j = adsorbates[j][1]
                        site_j = adsorbates[j][2]
                        interaction_scaling = ((interactions[surface][adsorbate_j][site_j]['delta_D'] / interactions[surface][adsorbate_j][site_j]['delta_E'])/
                                              (interactions[surface][molecule][site]['delta_D'] / interactions[surface][molecule][site]['delta_E']))

                        if pbc is not None:
                            lattice_dbands[j][x % pbc[0], y % pbc[1]] += interactions[surface][molecule][site]['V'].get(x - (rel_x + DR), {}).get(y - (rel_y + DR), 0.) * \
                                interactions[surface][molecule][site]['delta_D'] / d_shift_sum * interaction_scaling
                        else:
                            lattice_dbands[j][x, y] += interactions[surface][molecule][site]['V'].get(x - (rel_x + DR), {}).get(y - (rel_y + DR), 0.) * \
                                interactions[surface][molecule][site]['delta_D'] / d_shift_sum * interaction_scaling
                    else:
                        if pbc is not None:
                            self_int[j][x % pbc[0], y % pbc[1]] += interactions[surface][molecule][site]['V'].get(x - (rel_x + DR), {}).get(y - (rel_y + DR), 0.) * \
                                interactions[surface][molecule][
                                    site]['delta_D'] / d_shift_sum
                        else:
                            self_int[j][x, y] += interactions[surface][molecule][site]['V'].get(x - (rel_x + DR), {}).get(y - (rel_y + DR), 0.) * \
                                interactions[surface][molecule][
                                    site]['delta_D'] / d_shift_sum

    # calculate interaction energy
    interaction_energy = 0.
    for i, adsorbate in enumerate(adsorbates):
        surface, molecule, site, rel_x, rel_y = adsorbate
        if molecule is None :
            continue
        cell = np.array(interactions[surface]['_cell'])
        interaction_contrib = 0.
        d_shift_sum = sum(interactions[surface][molecule][site]['V'].get(x, {}).get(y, 0)
                          for x in range(-DR, DR) for y in range(-DR, DR))
        for x in range(-DR, DR):
            for y in range(-DR, DR):

                #print('rel_x {rel_x}, DR {DR}, x {x}, rel_y {rel_y}, DR {DR}, y {y}'.format(**locals()))

                w = 0

                if pbc is not None:
                    w = self_int[i][(rel_x + x + DR) %
                                    pbc[0], (rel_y + y + DR) % pbc[1]]
                    interaction_contrib += interactions[surface][molecule][site]['V'].get(x, {}).get(y, 0.) * \
                        lattice_dbands[i][(rel_x + x + DR) % pbc[0], (rel_y + y + DR) % pbc[1]] / \
                        (interactions[surface][molecule]
                         [site]['delta_D'] - w) / d_shift_sum
                else:
                    w = self_int[i][x, y]
                    interaction_contrib += interactions[surface][molecule][site]['V'].get(x, {}).get(y, 0.) * \
                        lattice_dbands[i][rel_x + x + DR, rel_y + y + DR] / \
                        (interactions[surface][molecule]
                         [site]['delta_D'] - w) / d_shift_sum

        #print('interaction contrib = {interaction_contrib}'.format(**locals()))
        if dipole_contribution:
            hicov_dipole = interactions[surface][molecule][site]['hicov_dipole']
            locov_dipole = interactions[surface][molecule][site]['locov_dipole']

            hicov_cell = interactions[surface]['_hicov_cell']
            locov_cell = interactions[surface]['_locov_cell']

            hicov_ES = dipole_factor * hicov_dipole**2 * calculate_periodic_dipole_interaction(1., hicov_cell, ER1) / 2.
            locov_ES = dipole_factor * locov_dipole**2 * calculate_periodic_dipole_interaction(1., locov_cell, ER1) / 2.
            dipole_self_interaction =  (hicov_ES - locov_ES)

            dband_delta_E = interactions[surface][molecule][site]['delta_E'] - dipole_self_interaction
            if verbose:
                print('Dipole self-interaction {molecule}@{site} {hicov_ES: .3f} - {locov_ES: .3f} = {dipole_self_interaction: .3f} eV (Dipole {hicov_dipole: .3f}/{locov_dipole: .3f} eA). d-Band energy {dband_delta_E: .3f}.'.format(**locals()))
        else:
            dband_delta_E = interactions[surface][molecule][site]['delta_E']

        interaction_energy += dband_delta_E  * interaction_contrib

        if verbose :
            print('   - d-band filling factor {interaction_contrib: .3f}'.format(**locals()))

    # calculate dipole interaction energy
    ES_ENERGY = 0.
    for i, adsorbate1 in enumerate(adsorbates):
        adsorbate1_ES_energy = 0.
        surface1, molecule1, site1, rel_x1, rel_y1 = adsorbate1
        locov_cell = np.array(interactions[surface1]['_locov_cell'])

        # dipole self-interaction correction
        locov_dipole1 = interactions[surface1][molecule1][site1]['locov_dipole']
        locov_ES1 = dipole_factor * locov_dipole1**2 * calculate_periodic_dipole_interaction(1., locov_cell, ER1) / 2.
        #adsorbate1_ES_energy -= locov_ES1
        if verbose :
            print("ES locov correction {locov_ES1: .3f} eV".format(**locals()))

        for j, adsorbate2 in enumerate(adsorbates):
            surface2, molecule2, site2, rel_x2, rel_y2 = adsorbate2

            if molecule1 is None or molecule2 is None:
                continue

            for x in range(-ER2, ER2):
                for y in range(-ER2, ER2):
                    if adsorbate1 != adsorbate2 or x != 0 or y != 0:
                        d = np.array([pbc[0] * x, pbc[1] * y, 0])
                        sp1 = np.array(interactions[surface1][molecule1][site1][
                                       'site_pos']) + np.array([rel_x1, rel_y1, 0.])
                        sp2 = np.array(interactions[surface2][molecule2][site2][
                                       'site_pos']) + np.array([rel_x2, rel_y2, 0.])
                        r = np.linalg.norm(np.dot(d + sp1 - sp2, cell))

                        dp1 = np.array(
                            interactions[surface1][molecule1][site1]['dipole'])
                        dp2 = np.array(
                            interactions[surface2][molecule2][site2]['dipole'])

                        dp_energy = dipole_factor * get_dipole_energy(dp1, dp2, r) / 2.  #  Correct for double-counting
                        adsorbate1_ES_energy += dp_energy
                        #print('Distance {r} {dp_energy} eV, ({d} {sp1} {sp2})'.format(**locals()))
        if verbose:
            print('    - coverage ES energy {molecule}@{site} ({rel_x1}x{rel_x2}) {adsorbate1_ES_energy: .3f} eV'.format(**locals()))
        ES_ENERGY += adsorbate1_ES_energy - locov_ES1

    #print('CELL {cell}'.format(**locals()))
    if verbose:
        print('Total electrostatic contribution {ES_ENERGY: .3f}'.format(**locals()))
        print('Total d-Band contribution {interaction_energy: .3f}'.format(**locals()))

    if dipole_contribution:
        interaction_energy += ES_ENERGY


    if comment:
        print(
            '{comment} ---> interaction energy {interaction_energy:.3f} eV.\n'.format(**locals()))
    return interaction_energy, interaction_energy - ES_ENERGY, ES_ENERGY


@memory.cache
def calculate_periodic_dipole_interaction(dp, cell, ER):
    dipole_self_interaction = 0.
    for x in range(-ER, ER):
        for y in range(-ER, ER):
            if x or y:
                d = float(np.linalg.norm(np.dot(np.array([x,  y, 0]), cell)))
                dipole_energy = get_dipole_energy(dp, dp, d)
                dipole_self_interaction += dipole_energy # avoid double counting
    return dipole_self_interaction


def get_DOS_hilbert(pickle_filename, channels):
    with open(pickle_filename) as f:
        energies, dos, pdos = pickle.load(f)

    out_dos = sum([
        pdos[atom][band][channel] for (atom, band, channel) in channels

    ])
    hilbert_signal = np.imag(scipy.signal.hilbert(out_dos))
    return energies, out_dos, hilbert_signal


def get_e_w(energies, rhos):

    ed = np.trapz(energies * rhos, energies) / np.trapz(rhos, energies)
    wd2 = np.trapz((energies-ed)**2*rhos, energies) / np.trapz(rhos, energies)
    return ed, wd2


def collect_interaction_data(surface_name, adsorbate_name, site_name,
                             clean_surface_logfile, clean_surface_dosfile,
                             locov_logfile, locov_dosfile,
                             hicov_logfile, hicov_dosfile,
                             site_pos=[0., 0., 0.],
                             locov_densityfile=None,
                             hicov_densityfile=None,
                             verbose=False,
                             spinpol=False,
                             adsorbate_species=[
                                 'C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P'],
                             surface_tol=.5,
                             ):
    """
        Take the resulting log files from a QuantumEspresso geometry optimization and the pickled DOS file
        created using the ase-espresso interface to derive lateral interaction parameters
        suitable for a d-band mediated interaction theory.

        This function assumes that all slab geometries are aligned such that the z-axis is parallel to the surface normal.

        :param surface_name: A descriptive handle for the substrate surface.
        :type surface_name: str
        :param adsorbate_name: A descriptive handle for the interacting adsorbate.
        :type adsorbate_name: str
        :param site_name: A name for the surface adsorption site.
        :type site_name: str
        :param clean_surface_logfile: The path of a QuantumEspresso logfile of the clean (primitive) surface slab.
        :type clean_surface_logfile: str
        :param clean_surface_dosfile: The path of a pickle file storing the (projected) density of states of the clean surface.
        :type clean_surface_dosfile: str
        :param locov_logfile: The path of a QuantumEspresso logfile of the surface with the isolated adsorbate.
        :type locov_logfile: str
        :param locov_dosfile: The path of a pickle file storing the (projected) density of states of the isolated adsorbate.
        :type locov_dosfile: str
        :param hicov_logfile: The path of a QuantumEspresso logfile of the surface at the high-coverage limit.
        :type hicov_logfile: str
        :param hicov_dosfile: The path of a pickle file storing the (projected) density of states of the isolated adsorbate.
        :type hicov_dosfile: str
        :param site_pos: The position of the surface site in relative coordinate of the clean surface unit cell
        :type site_pos: iterable of 3 floats in [0, 1)

    """


    if adsorbate_species is None:
        adsorbate_species = set(ase.data.chemical_symbols) - set(['Rh', 'Pt', 'Pd', 'Au', 'Ru'])

    interactions = {}
    interactions.update({surface_name: {}})
    interactions[surface_name].update({adsorbate_name: {}})
    interactions[surface_name][adsorbate_name].update({site_name: {}})
    interactions[surface_name][adsorbate_name][site_name].update({'V': {}})
    interactions[surface_name][adsorbate_name][site_name].update({'site_pos': site_pos})

    # store filepaths to
    interactions[surface_name][adsorbate_name][site_name].update({
        '_clean_surface_logfile': os.path.realpath(clean_surface_logfile),
        '_clean_surface_dosfile': os.path.realpath(clean_surface_dosfile),
        '_locov_logfile': os.path.realpath(locov_logfile),
        '_locov_dosfile': os.path.realpath(locov_dosfile),
        '_hicov_logfile': os.path.realpath(hicov_logfile),
        '_hicov_dosfile': os.path.realpath(hicov_dosfile),
        '_locov_densityfile': os.path.realpath(locov_densityfile) if locov_densityfile else None ,
        '_hicov_densityfile': os.path.realpath(hicov_densityfile) if hicov_densityfile else None ,
    })

    # collect clean surface info
    # - energy
    if verbose:
        print('Now reading {clean_surface_logfile} ... ({surface_name} {adsorbate_name}@{site_name})'.format(**locals()))

    clean_traj = espresso.io.read_log(clean_surface_logfile)


    clean = clean_traj[-1]
    clean_energy = clean.get_calculator().get_potential_energy()
    if verbose:
        print("Clean Surface Energy {clean_energy}".format(**locals()))
        print(interactions[surface_name][adsorbate_name][site_name])

    clean_cell = clean.cell
    surface_atom = np.argmax(clean.positions[:, 2])
    clean_channels = [[surface_atom, 'd', 0]]
    if spinpol:
        clean_channels.append([surface_atom, 'd', 1])

    # - reference d-band center
    clean_energies, clean_DOS, _ = get_DOS_hilbert(
        clean_surface_dosfile, clean_channels)
    clean_E_d, _ = get_e_w(clean_energies, clean_DOS)
    if verbose:
        print('clean d-band position {clean_E_d}'.format(**locals()))
        print('clean surface total energy {clean_energy} eV.'.format(**locals()))

    # collect high coverage surface info
    # - energy
    if verbose:
        print('Now reading {hicov_logfile} ... ({surface_name} {adsorbate_name}@{site_name})'.format(**locals()))
    hicov_traj = espresso.io.read_log(hicov_logfile)

    hicov = hicov_traj[-1]
    hicov_energy = hicov.get_calculator().get_potential_energy()
    hicov_cell = hicov.cell
    hicov_surface_area = np.linalg.norm(np.outer(hicov.cell[0], hicov.cell[1]))

    hicov_x = np.linalg.norm(hicov.cell[0]) / np.linalg.norm(clean.cell[0])
    hicov_y = np.linalg.norm(hicov.cell[1]) / np.linalg.norm(clean.cell[1])
    hicov_binding_energy = hicov_energy - clean_energy * hicov_x * hicov_y

    # - d-band shift(s)
    # ok, complicated, do later ...
    adsorbate_atoms = [i for i, species in enumerate(
        hicov.get_chemical_symbols()) if species in adsorbate_species]
    slab_atoms = [i for i, species in enumerate(
        hicov.get_chemical_symbols()) if species not in adsorbate_species]
    surface_atom = slab_atoms[np.argmax(hicov.positions[slab_atoms, 2])]



    if verbose:
        print('Surface Atom {surface_atom}'.format(**locals()))

    hicov_channels = [[surface_atom, 'd', 0]]
    if spinpol:
        hicov_channels.append([surface_atom, 'd', 1])
    hicov_energies, hicov_DOS, _ = get_DOS_hilbert(
        hicov_dosfile, hicov_channels)
    hicov_E_d, _ = get_e_w(hicov_energies, hicov_DOS)

    if verbose:
        print('hicov d-band position {hicov_E_d}'.format(**locals()))

    interactions[surface_name][adsorbate_name][site_name].update(
        {'delta_D': (clean_E_d - hicov_E_d)})

    interactions[surface_name][adsorbate_name][site_name].update(
        {'_D_clean': (clean_E_d)})

    if verbose:
        print("High Cov Energy {hicov_energy}, ({hicov_x:.2f}x{hicov_y:.2f})".format(
            **locals()))

    # collect low coverage info
    # - energy

    if verbose:
        print('Now reading {locov_logfile} ... ({surface_name} {adsorbate_name}@{site_name})'.format(**locals()))
    locov_traj = espresso.io.read_log(locov_logfile)
    locov = locov_traj[-1]
    locov_energy = locov.get_calculator().get_potential_energy()
    locov_surface_area = np.linalg.norm(np.outer(locov.cell[0], locov.cell[1]))

    locov_x = np.linalg.norm(locov.cell[0]) / np.linalg.norm(clean.cell[0])
    locov_y = np.linalg.norm(locov.cell[1]) / np.linalg.norm(clean.cell[1])

    locov_binding_energy = locov_energy - clean_energy * locov_x * locov_y

    if verbose:
        print("Low Cov Energy {locov_energy}, ({locov_x:.2f}x{locov_y:.2f})".format(
            **locals()))

    delta_E = hicov_binding_energy - locov_binding_energy

    if delta_E < 0.:
        print(
            "Warning: binding energy shift ({delta_E} eV) is negative. Indicates attractive interaction.".format(**locals()))
    elif delta_E > 2.:
        print("Warning: binding energy shift ({delta_E} eV) seems very large.".format(
            **locals()))

    interactions[surface_name][adsorbate_name][
        site_name].update({'delta_E': delta_E})


    # DEBUGGING
    interactions[surface_name][adsorbate_name][
        site_name].update({'_hicov_energy': hicov_energy})
    interactions[surface_name][adsorbate_name][
        site_name].update({'_locov_energy': locov_energy})
    interactions[surface_name][adsorbate_name][
        site_name].update({'_clean_energy': clean_energy})

    if locov_densityfile :
        origin, cell, charge = extract_charge(locov_densityfile, clip=True)
        dipole = get_dipole(charge, locov)

        interactions[surface_name][adsorbate_name][
            site_name].update({'dipole': dipole})
        interactions[surface_name][adsorbate_name][
            site_name].update({'locov_dipole': dipole})
        if verbose:
            print('==> DEBUG DENSITY-FILE: {locov_densityfile}\n\tLOG-FILE: {locov_logfile}\n\tGEOM: {locov}\n\tCELL:{cell}\n\tCHARGE-SHAPE: {charge.shape}\n\t'.format(**locals()))
            print('dipole = {dipole} eA'.format(**locals()))

    if hicov_densityfile :
        origin, cell, charge = extract_charge(hicov_densityfile, clip=True)
        dipole = get_dipole(charge, hicov)

        interactions[surface_name][adsorbate_name][
            site_name].update({'dipole': dipole})
        interactions[surface_name][adsorbate_name][
            site_name].update({'hicov_dipole': dipole})
        if verbose:
            print('==> DEBUG DENSITY-FILE: {hicov_densityfile}\n\tLOG-FILE: {hicov_logfile}\n\tGEOM: {hicov}\n\tCELL:{cell}\n\tCHARGE-SHAPE: {charge.shape}\n\t'.format(**locals()))
            print('dipole = {dipole} eA'.format(**locals()))

    adsorbate_atoms = [i for i, species in enumerate(
        hicov.get_chemical_symbols()) if species in adsorbate_species]
    lowest_adsorbate_atom_hicov = adsorbate_atoms[
        np.argmin(hicov.positions[adsorbate_atoms, 2])]

    # - d-band shift(s)
    adsorbate_atoms = [i for i, species in enumerate(
        locov.get_chemical_symbols()) if species in adsorbate_species]
    slab_atoms = [i for i, species in enumerate(
        locov.get_chemical_symbols()) if species not in adsorbate_species]
    max_z = locov.positions[surface_atom, 2]
    surface_atoms = [i for i, z in enumerate(
        locov.positions[:, 2]) if abs(max_z - z) < surface_tol and i not in adsorbate_atoms]

    lowest_adsorbate_atom = adsorbate_atoms[
        np.argmin(locov.positions[adsorbate_atoms, 2])]


    # DEBUGGING
    interactions[surface_name][adsorbate_name][site_name]['dz'] = hicov.positions[lowest_adsorbate_atom_hicov, 2] - locov.positions[lowest_adsorbate_atom, 2]


    if verbose:
        print(
            'Lowest Adsorbate Atom {lowest_adsorbate_atom}'.format(**locals()))
        print('Surface Atoms {surface_atoms}'.format(**locals()))

    for surface_atom in surface_atoms:
        # get the offset belonging to the Minimum Image Convention
        mic_positions = [(locov.positions[surface_atom] + locov.cell[0] * dx +
                          locov.cell[1] * dy, dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
        key = lambda x: np.linalg.norm(
            x[0] - locov.positions[lowest_adsorbate_atom])
        mic_positions = [(x[0], x[1], x[2], key(x)) for x in mic_positions]
        mic_positions.sort(key=key)
        mic_x, mic_y = (mic_positions[0])[1:3]

        locov_channels = [[surface_atom, 'd', 0]]
        if spinpol:
            locov_channels.append([surface_atom, 'd', 1])

        if verbose:
            print("  Getting DOS from {locov_dosfile} with the channels {locov_channels}.".format(
                **locals()))

        locov_energies, locov_DOS, _ = get_DOS_hilbert(
            locov_dosfile, locov_channels)
        locov_E_d, _ = get_e_w(locov_energies, locov_DOS)

        crystal_coord = np.linalg.solve(
            clean.cell.T, locov.positions[
                surface_atom] + mic_x * locov.cell[0] + mic_y * locov.cell[1]
            - locov.positions[lowest_adsorbate_atom])
        crystal_x = int(round(crystal_coord[0]))
        crystal_y = int(round(crystal_coord[1]))

        d_shift = (locov_E_d - clean_E_d)
        if verbose:
            print(
                '  Surface Atom {surface_atom}, crystal coord {crystal_coord}'.format(**locals()))
            print(
                "    At ({crystal_x}, {crystal_y}): {d_shift} eV".format(**locals()))
        interactions[surface_name][adsorbate_name][site_name][
            'V'].setdefault(crystal_x, {})[crystal_y] = d_shift

    if verbose:
        print('locov d-band position {locov_E_d}'.format(**locals()))

    # save cell geometry
    interactions[surface_name]['_cell']  = clean.cell.tolist()
    interactions[surface_name]['_locov_cell']  = locov.cell.tolist()
    interactions[surface_name]['_hicov_cell']  = hicov.cell.tolist()

    return interactions
