# dbmi

(*d-band mediated interactions*)

This python module implements an approximate theory for calculating spatially resolved adsorbate-adsorbate interactions based on relative shifts of different d-band centers.

As input it requires one DFT calculation with accompanying projected density of states for each: the empty (clean) substrate surface (primitive surface unit cell), an optimized adsorbate geometry in the low coverage limit (i.e. 3x3 - 6x6) surface slab and an optimized geometry of the high coverage limit (i.e. 1x1 or 2x2 overlayer). So far the surface are limited to those where the primitive surface unit cell consists of only one surface atom ((100), (110), or (111)). Currently only works with files from QuantumEspresso calculations using the [ase-espresso](https://github.com/vossjo/ase-espresso) interface of [ASE](https://wiki.fysik.dtu.dk/ase/).


Very experimental, work in progress. Also needs a patched version of [ase-espresso](https://github.com/vossjo/ase-espresso), drop me a line or check back later.

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

The function `collect_interaction_data` will collect all needed information from finished QuantumEspresso calculations. It requires that the projected density of states is found in a file as produced by the [ase-espresso](https://github.com/vossjo/ase-espresso) interface. The returned dictionary can be used for many different lateral interaction situations. `dbmi.merge(dict1, dict2)` can be used to merge the lateral interaction coefficients from two or more adsorbates. It may be useful to store it somewhere.

The function `calculate_interaction_energy` returns the full (a.ka. integrated) interaction energy. `interaction` is the dictionary containing all interaction coefficients, the `adsorbates` argument is a list of element of the form `[surface, molecule, site, cell_x, cell_y]`. The `pbc` parameter can be used to evaluate the adsorbates on a finite lattice wrapping around periodic boundaries in all directions. If no `pbc` is supplied an inifinite lattice is assumed.
