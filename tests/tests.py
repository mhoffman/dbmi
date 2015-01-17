#!/usr/bin/env python

import pprint
import collections
import unittest
import pickle

import numpy as np

import dbmi



# all units of energy are in eV

np.set_printoptions(precision=3)

sites = {1: 'hcp', 2: 'fcc', 3: 'ontop', 4: 'bridge'}
site_nr = dict((v, k) for k, v in sites.iteritems())


species = {1: 'O', 2: 'CO'}
species_nr = dict((v, k) for k, v in species.iteritems())

# the cutoff distance of lattice interaction parameters
IR = INTERACTION_RANGE = 4
# enter some interaction information from DFT input
data = {}
data.update({'Rh': {}})
data['Rh'].update({'111': {}})
data['Rh']['111'].update({'O': {}})
data['Rh']['111']['O'].update({'fcc': {}})
data['Rh']['111']['O']['fcc'].update({'delta_E': (-1.55161294550382 - -2.35798844300829),
                                      'delta_D': -0.95209686,
                                      'V': {},
                                      })

data['Rh']['111']['O']['fcc']['V'].update({
    -2: {-2: -0.031, -1: -0.031, 0: -0.031, 1: -0.031},
    -1: {-2:  0.004, -1: -0.188, 0:  0.004, 1: -0.031},
    0: {-2: -0.031, -1: -0.188, 0: -0.188, 1: -0.031},
    1: {-2:  0.005, -1: -0.031, 0:  0.004, 1: -0.031}
})

data['Rh']['111']['O'].update({'hcp': {}})
data['Rh']['111']['O']['hcp'].update({'delta_E': (-1.40421636906558 - -2.28312863754628),
                                      'delta_D': -1.10167184,
                                      'V': {},
                                      })

data['Rh']['111']['O']['hcp']['V'].update({
    -2: {-2: -0.028, -1: -0.007, 0: -0.028, 1: -0.004},
    -1: {-2: -0.028, -1: -0.235, 0: -0.235, 1: -0.028},
    0: {-2: -0.028, -1: -0.007, 0: -0.235, 1: -0.007},
    1: {-2: -0.028, -1: -0.028, 0: -0.028, 1: -0.028}
})

interactions = data

with open('interactions.dat', 'w') as out:
    out.write(pprint.pformat(data))

# after feeding all the necessary data, let's to some practical
# calculations
verbose = False

class Test_OFCC_OFCC_nn_interaction(unittest.TestCase):

    def test(self):
        # 1st, fcc-fcc nearest neighbors
        adsorbates = [['Rh', '111', 'O', 'fcc', 0, 1],
                      ['Rh', '111', 'O', 'fcc', 0, 0],
                      ]
        lattice_size = (2, 2)
        self.assertAlmostEqual(dbmi.calculate_interaction_energy(interactions,
                                                                 adsorbates, IR, pbc=lattice_size, verbose=verbose), 0.537583665003, msg=str(adsorbates) + " Result deviates.")

class Test_OHCP_OHCP_nn_interaction(unittest.TestCase):

    def test(self):
        # 2nd, hcp-hcp nearest neighbors
        adsorbates = [['Rh', '111', 'O', 'hcp', 0, 1],
                      ['Rh', '111', 'O', 'hcp', 0, 0],
                      ]
        lattice_size = (2, 2)
        self.assertAlmostEqual(dbmi.calculate_interaction_energy(interactions, adsorbates, IR, pbc=lattice_size, verbose=verbose), 0.58594151232,
                               msg=str(adsorbates) + " Result deviates.")

class Test_3x3_interaction(unittest.TestCase):

    def test(self):
        adsorbates = [
            ['Rh', '111', 'O', 'fcc', 0, 0],
            ['Rh', '111', 'O', 'fcc', 0, 3],
            ['Rh', '111', 'O', 'fcc', 3, 0],
            ['Rh', '111', 'O', 'fcc', 3, 3],
        ]
        lattice_size = (2, 2)

        self.assertAlmostEqual(0.0174399756714, dbmi.calculate_interaction_energy(
            interactions, adsorbates, IR, verbose=verbose))
