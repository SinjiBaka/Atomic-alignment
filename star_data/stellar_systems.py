from dataclasses import dataclass
import numpy as np

from . import sun_spectrum

@dataclass
class StellarSystem:

    name: str

    # star
    Teff: float
    Rstar_Rsun: float

    # planet
    orbital_distance_AU: float

    # spectrum (flux at planetary orbit in [erg/cm^2/s])
    # wavelength in Angstrom
    spectrum_wl: np.ndarray | None = None
    spectrum_flux: np.ndarray | None = None


# general list of systems to use for photoexcitation
stellar_systems = {
    "Sun" : StellarSystem(name = "Sun",
                          Teff = 5780,
                          Rstar_Rsun = 1.0,
                          orbital_distance_AU = 1.0,
                          spectrum_wl = sun_spectrum.sun_wl,
                          spectrum_flux =sun_spectrum.sun_flux)
}