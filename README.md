# dbmi
This python module implements an approximate theory for calculating adsorbate-adsorbate interactions based on relative shifts of d-band center. Currently only works with files from QuantumEspresso calculations using the ase-espresso interface of ASE.

# Usage

There are currently only two functions that do useful work:

    def collect_interaction_data(surface_name, adsorbate_name, site_name,
                             clean_surface_logfile, clean_surface_dosfile,
                             locov_logfile, locov_dosfile,
                             hicov_logfile, hicov_dosfile,
                             verbose=False,
                             spinpol=False,
                             adsorbate_species=['C', 'H', 'O', 'N'],
                             surface_tol=1.,
                             )

    def calculate_interaction_energy(interactions, adsorbates, IR=4, pbc=None, verbose=False)

The function `collect_interaction_data` will collect all needed information from finished QuantumEspresso calculations. It also requires that projected density of states are found in a file as produced by the [ase-espresso](https://github.com/vossjo/ase-espresso) interface. The returned dictionary can be used for many different lateral interaction situations. `dbmi.merge(dict1, dict2)` can be used to merge the lateral interaction coefficients from two or more adsorbates. It may be useful to store it somewhere.

The function `calculate_interaction_energy` return the full (a.ka. integrated) interaction energy. `interaction` is the dictionary containing all interaction coefficients, the `adsorbates` argument is a list of element of the form `[surface, molecule, site, cell_x, cell_y]`. The `pbc` parameter can be used to evaluate the adsorbates on a finite lattice wrapping around periodic boundaries in all directions. If no `pbc` is supplied an inifinite lattice is assumed.
