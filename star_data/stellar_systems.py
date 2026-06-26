from dataclasses import dataclass
import numpy as np

from . import sun_spectrum
from . import hatp32_spectrum
from . import kelt9_spectrum

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
                          spectrum_flux =sun_spectrum.sun_flux),

    "HAT-P-32" : StellarSystem(name = "HAT-P-32",
                          Teff = 6269,
                          Rstar_Rsun = 1.219,
                          orbital_distance_AU = 0.0343,
                          spectrum_wl = hatp32_spectrum.hatp32_wl,
                          spectrum_flux =hatp32_spectrum.hatp32_flux),

    "KELT-9" : StellarSystem(name = "KELT-9",
                          Teff = 10170,
                          Rstar_Rsun = 2.362,
                          orbital_distance_AU = 0.03368,
                          spectrum_wl = kelt9_spectrum.kelt9_wl,
                          spectrum_flux =kelt9_spectrum.kelt9_flux)
                          
}