# NIST Triplet Finder

This repository contains a Python script for automated search and filtering of **triplet spectral transitions** from the [NIST Atomic Spectra Database](https://physics.nist.gov/PhysRefData/ASD/lines_form.html).  
The code identifies transitions that can exhibit **atomic alignment effects**, based on a set of physical selection criteria ([Rumenskikh et al. 2025](https://doi.org/10.1093/mnras/staf1038)).

---

## Overview

The script loads atomic line data for a given atom or ion and applies a series of filters to find triplets that satisfy specific quantum-mechanical conditions.  
These include requirements on level configurations, total angular momenta, metastability of lower levels, and spectral separations.
Based on the work of [Rumenskikh et al. 2025](https://doi.org/10.1093/mnras/staf1038), several key criteria for the presence of the atomic alignment effect can be identified:
- A triplet transition is required, i.e., we are looking for levels from which there are exactly three transitions to states with identical electron configurations and terms, in addition to the total angular momentum J.
- The lower level from which transitions are observed must be metastable, i.e., it must be sufficiently long-lived and not decay too quickly to lower states. As a criterion, we chose the decay to the lower metastable state to be at least 100 times faster than from the lower metastable state to even lower states (if several are added together).
- The total angular momentum J of the lower metastable level must be J > 1/2.
- In the triplet transition observed, the wavelength difference between at least two transitions must exceed 0.8 $\AA$ for them to be resolved by a spectrograph.
- To enable observation using telescopes, we chose a search range from 500 nm to 1500 nm.

The **lines_found/** folder contains the results of the script for elements up to **iron** and ionization level **III**.

---

## Features

- Queries NIST line data via [`astroquery.nist`](https://astroquery.readthedocs.io/en/latest/nist/nist.html)
- Parses and cleans the results into a structured `pandas.DataFrame`
- Finds triplet transitions that share the same electron configuration and term
- Filters lines by:
  - wavelength range (e.g., 5000–15000 $\AA$, variable)
  - total angular momentum \( J > 1/2 \)
  - spectral separations \( > 0.8 \) $\AA$ (variable)
  - metastable lower levels (based on Einstein A-coefficients)
- Exports full datasets or filtered triplets to tab-separated files

---

## Dependencies

- Python ≥ 3.9  
- [pandas](https://pandas.pydata.org/)  
- [numpy](https://numpy.org/)  
- [astropy](https://www.astropy.org/)  
- [astroquery](https://astroquery.readthedocs.io/)  
- re (standard library)

You can install them via:
```bash
pip install pandas numpy astropy astroquery
```

---

## Usage

You can search for lines directly in this script's `main()` function or copy the class to your own project.
To retrieve and fully process data, use only the `NIST_data` class constructor. It requires the name of an element similar to the [NIST](https://physics.nist.gov/PhysRefData/ASD/lines_form.html) database and the wavelength range for searching transitions in angstroms (the wider the range, the better). The found lines are saved in plain text format.

```python
data = NIST_data('He I', 300, 35e6)
data.save_triplets_formated()
```
For output see `triplets_data_He I.dat` file.

**Important**
The script returns all **Ritz** wavelengths in vacuum in angstroms. Please keep this in mind when comparing with NIST.

---

## Troubleshooting

Several errors may occur during operation:
- The NIST database may report that the required data was not found (if you specified the element correctly):
In this case, it is recommended to slightly change the search wavelength range, and then everything will work (a feature of the `astroquery` library).
- The script reports that lower levels were not found for a certain lower level, which prevents metastability assessment. In this case, the transition is not taken into account. To correct this, try specifying a shorter wavelength.
- Errors in parsing table data. Not all available elements in [NIST](https://physics.nist.gov/PhysRefData/ASD/lines_form.html) have been checked, and there may be a data format we haven't accounted for. If you encounter this issue, please let us know.

---

## Developed by

**Stas Sharipov**  
Laboratory of High-Power Laser Energy, Institute of Laser Physics SB RAS 
GitHub: [@SinjiBaka](https://github.com/SinjiBaka)
