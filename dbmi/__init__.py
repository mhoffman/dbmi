#!/usr/bin/env python

__version__ = "0.0.1"
VERSION = __version__


import pprint
import collections
import unittest
import pickle

import numpy as np
import scipy.signal
import espresso.io

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
                # Alternatively, replace this with exception raiser to alert you of value conflicts
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])


def calculate_interaction_energy(interactions, adsorbates, IR=4, pbc=None, verbose=False):
    if verbose:
        if pbc:
            print('Unit cell Size ({}x{})'.format(pbc[0], pbc[1]))
        else:
            print('Non-periodic calculation')

    # setup interaction lattice
    xs = [x for _, _, _, x, _ in adsorbates]
    ys = [y for _, _, _, _, y in adsorbates]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    lattice = np.array
    # the d-bands shifts the i-th atom see, due to all other atoms
    lattice_dbands = [np.zeros(
        (max_x - min_x + 2 * IR, max_y - min_x + 2 * IR)) for _ in range(len(adsorbates))]

    # the d-bands shifts to due the i-th atom
    # so that we can subtract later from the maximum
    self_int = [np.zeros((max_x - min_x + 2 * IR, max_y - min_x + 2 * IR))
                for _ in range(len(adsorbates))]

    # calculate d-band shifts based on other adsorbates
    for i, adsorbate in enumerate(adsorbates):
        print(adsorbate)
        surface, molecule, site, rel_x, rel_y = adsorbate
        if verbose:
            print("Metal: {surface},  molecule: {molecule}, site: {site}, (X, Y) = ({rel_x}, {rel_y})".format(
                **locals()))

        d_shift_sum = sum(interactions[surface][molecule][site]['V'].get(x, {}).get(y, 0)
                          for x in range(-IR, IR) for y in range(-IR, IR))

        X, Y = lattice_dbands[i].shape
        for x in range(X):
            for y in range(Y):
                for j in range(len(adsorbates)):
                    # if True:
                    if j != i:
                        if pbc is not None:
                            lattice_dbands[j][x % pbc[0], y % pbc[1]] += interactions[surface][molecule][site]['V'].get(x - (rel_x + IR), {}).get(y - (rel_y + IR), 0.) * \
                                interactions[surface][molecule][
                                    site]['delta_D'] / d_shift_sum
                        else:
                            lattice_dbands[j][x, y] += interactions[surface][molecule][site]['V'].get(x - (rel_x + IR), {}).get(y - (rel_y + IR), 0.) * \
                                interactions[surface][molecule][
                                    site]['delta_D'] / d_shift_sum
                    else:
                        if pbc is not None:
                            self_int[j][x % pbc[0], y % pbc[1]] += interactions[surface][molecule][site]['V'].get(x - (rel_x + IR), {}).get(y - (rel_y + IR), 0.) * \
                                interactions[surface][molecule][
                                    site]['delta_D'] / d_shift_sum
                        else:
                            self_int[j][x, y] += interactions[surface][molecule][site]['V'].get(x - (rel_x + IR), {}).get(y - (rel_y + IR), 0.) * \
                                interactions[surface][molecule][
                                    site]['delta_D'] / d_shift_sum

    # calculate interaction energy
    interaction_energy = 0.
    for i, adsorbate in enumerate(adsorbates):
        surface, molecule, site, rel_x, rel_y = adsorbate
        interaction_contrib = 0.
        d_shift_sum = sum(interactions[surface][molecule][site]['V'].get(x, {}).get(y, 0)
                          for x in range(-IR, IR) for y in range(-IR, IR))
        for x in range(-IR, IR):
            for y in range(-IR, IR):

                #print('rel_x {rel_x}, IR {IR}, x {x}, rel_y {rel_y}, IR {IR}, y {y}'.format(**locals()))

                w = 0

                if pbc is not None:
                    w = self_int[i][(rel_x + x + IR) %
                                    pbc[0], (rel_y + y + IR) % pbc[1]]
                    interaction_contrib += interactions[surface][molecule][site]['V'].get(x, {}).get(y, 0.) * \
                        lattice_dbands[i][(rel_x + x + IR) % pbc[0], (rel_y + y + IR) % pbc[1]] / \
                        (interactions[surface][molecule]
                         [site]['delta_D'] - w) / d_shift_sum
                else:
                    w = self_int[i][x, y]
                    interaction_contrib += interactions[surface][molecule][site]['V'].get(x, {}).get(y, 0.) * \
                        lattice_dbands[i][rel_x + x + IR, rel_y + y + IR] / \
                        (interactions[surface][molecule]
                         [site]['delta_D'] - w) / d_shift_sum

        #print('interaction contrib = {interaction_contrib}'.format(**locals()))
        interaction_energy += (interactions[surface][molecule][site]['delta_E']) * \
            interaction_contrib

    if verbose:
        print('---> interaction energy {:.3f} eV.\n'.format(interaction_energy))
    return interaction_energy


def get_DOS_hilbert(pickle_filename, channels):
    with open(pickle_filename) as f:
        energies, dos, pdos = pickle.load(f)

    out_dos = sum([
        pdos[atom][band][channel] for (atom, band, channel) in channels

    ])
    hilbert_signal = np.imag(scipy.signal.hilbert(out_dos))
    return energies, out_dos, hilbert_signal


def get_e_w(energies, rhos):
    rho_sum = sum(rhos)
    d_band_center = sum(
        [energy * rho for (energy, rho) in zip(energies, rhos)]) / rho_sum
    d_band_width = 4 * \
        np.sqrt(sum([(energy - d_band_center) ** 2 *
                     rho for (energy, rho) in zip(energies, rhos)]) / rho_sum)
    E_d = d_band_center + d_band_width / 2.

    return d_band_center, d_band_width


def collect_interaction_data(surface_name, adsorbate_name, site_name,
                             clean_surface_logfile, clean_surface_dosfile,
                             locov_logfile, locov_dosfile,
                             hicov_logfile, hicov_dosfile,
                             locov_densityfile=None,
                             verbose=False,
                             spinpol=False,
                             adsorbate_species=['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P'],
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
        :type surface_name: str
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

    """

    interactions = {}
    interactions.update({surface_name: {}})
    interactions[surface_name].update({adsorbate_name: {}})
    interactions[surface_name][adsorbate_name].update({site_name: {}})

    interactions[surface_name][adsorbate_name][site_name].update({'V': {}})

    # collect clean surface info
    # - energy
    clean_traj = espresso.io.read_log(clean_surface_logfile)

    clean = clean_traj[-1]
    clean_energy = clean.get_calculator().get_potential_energy()
    if verbose:
        print("Clean Surface Energy {clean_energy}".format(**locals()))
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

    # collect high coverage surface info
    # - energy
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

    if verbose:
        print("High Cov Energy {hicov_energy}, ({hicov_x:.2f}x{hicov_y:.2f})".format(
            **locals()))

    # collect low coverage info
    # - energy
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

    delta_E = (hicov_binding_energy - locov_binding_energy)

    if delta_E < 0.:
        print(
            "Warning: binding energy shift ({delta_E} is negative.".format(**locals()))
    elif delta_E > 2.:
        print("Warning: binding energy shift ({delta_E}) seems very large.".format(
            **locals()))

    interactions[surface_name][adsorbate_name][site_name].update({'delta_E': delta_E})

    # - d-band shift(s)
    adsorbate_atoms = [i for i, species in enumerate(
        locov.get_chemical_symbols()) if species in adsorbate_species]
    slab_atoms = [i for i, species in enumerate(
        locov.get_chemical_symbols()) if species not in adsorbate_species]
    max_z = locov.positions[surface_atom, 2]
    surface_atoms = [i for i, z in enumerate(
        locov.positions[:, 2]) if abs(max_z - z) < surface_tol and i not in adsorbate_atoms]

    lowest_adsorbate_atom = adsorbate_atoms[np.argmin(locov.positions[adsorbate_atoms, 2])]


    if verbose:
        print('Lowest Adsorbate Atom {lowest_adsorbate_atom}'.format(**locals()))
        print('Surface Atoms {surface_atoms}'.format(**locals()))

    for surface_atom in surface_atoms:
        # get the offset belonging to the Minimum Image Convention
        mic_positions = [(locov.positions[surface_atom] + locov.cell[0] * dx + locov.cell[1] * dy, dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
        key = lambda x: np.linalg.norm(x[0] - locov.positions[lowest_adsorbate_atom])
        mic_positions = [(x[0], x[1], x[2], key(x)) for x in mic_positions]
        mic_positions.sort(key=key)
        mic_x, mic_y = (mic_positions[0])[1:3]

        locov_channels = [[surface_atom, 'd', 0]]
        if spinpol:
            locov_channels.append([surface_atom, 'd', 1])

        if verbose:
            print("  Getting DOS from {locov_dosfile} with the channels {locov_channels}.".format(**locals()))

        locov_energies, locov_DOS, _ = get_DOS_hilbert(
            locov_dosfile, locov_channels)
        locov_E_d, _ = get_e_w(locov_energies, locov_DOS)


        crystal_coord = np.linalg.solve(
            clean.cell.T, locov.positions[surface_atom] + mic_x * locov.cell[0] + mic_y * locov.cell[1]
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

    return interactions
