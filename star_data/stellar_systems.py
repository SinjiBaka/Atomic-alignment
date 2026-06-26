from dataclasses import dataclass
import numpy as np

from . import sun_spectrum
from . import hatp32_spectrum
from . import kelt9_spectrum
from . import wasp33_spectrum
from . import hatp11_spectrum
from . import hd73583_spectrum
from . import wasp80_spectrum
from . import hd209458_spectrum
from . import wasp69_spectrum
from . import wasp107_spectrum
from . import gj3470_spectrum
from . import hd189733_spectrum
from . import gj436_spectrum

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
                          spectrum_flux =kelt9_spectrum.kelt9_flux),
    
    "WASP-33" : StellarSystem(name = "WASP-33",
                          Teff = 7400,
                          Rstar_Rsun = 1.444,
                          orbital_distance_AU = 0.02558,
                          spectrum_wl = wasp33_spectrum.wasp33_wl,
                          spectrum_flux =wasp33_spectrum.wasp33_flux),
                
    "HAT-P-11" : StellarSystem(name = "HAT-P-11",
                          Teff = 4780,
                          Rstar_Rsun = 0.683,
                          orbital_distance_AU = 0.053,
                          spectrum_wl = hatp11_spectrum.hatp11_wl,
                          spectrum_flux =hatp11_spectrum.hatp11_flux),
    
    "HD 73583" : StellarSystem(name = "HD 73583",
                          Teff = 4511,
                          Rstar_Rsun = 0.71,
                          orbital_distance_AU = 0.0604,
                          spectrum_wl = hd73583_spectrum.hd73583_wl,
                          spectrum_flux =hd73583_spectrum.hd73583_flux),
                    
    "WASP-80" : StellarSystem(name = "WASP-80",
                          Teff = 4145,
                          Rstar_Rsun = 0.63,
                          orbital_distance_AU = 0.0346,
                          spectrum_wl = wasp80_spectrum.wasp80_wl,
                          spectrum_flux =wasp80_spectrum.wasp80_flux),

    "HD 209458" : StellarSystem(name = "HD 209458",
                          Teff = 6092,
                          Rstar_Rsun = 1.203,
                          orbital_distance_AU = 0.04747,
                          spectrum_wl = hd209458_spectrum.hd209458_wl,
                          spectrum_flux =hd209458_spectrum.hd209458_flux),

    "WASP-69" : StellarSystem(name = "WASP-69",
                          Teff = 4715,
                          Rstar_Rsun = 0.813,
                          orbital_distance_AU = 0.04525,
                          spectrum_wl = wasp69_spectrum.wasp69_wl,
                          spectrum_flux =wasp69_spectrum.wasp69_flux),
    
    "WASP-107" : StellarSystem(name = "WASP-107",
                          Teff = 4430,
                          Rstar_Rsun = 0.66,
                          orbital_distance_AU = 0.055,
                          spectrum_wl = wasp107_spectrum.wasp107_wl,
                          spectrum_flux =wasp107_spectrum.wasp107_flux),
                        
    "GJ 3470" : StellarSystem(name = "GJ 3470",
                          Teff = 3652,
                          Rstar_Rsun = 0.48,
                          orbital_distance_AU = 0.03557,
                          spectrum_wl = gj3470_spectrum.gj3470_wl,
                          spectrum_flux =gj3470_spectrum.gj3470_flux),

    "HD 189733" : StellarSystem(name = "HD 189733",
                          Teff = 4875,
                          Rstar_Rsun = 0.805,
                          orbital_distance_AU = 0.031,
                          spectrum_wl = hd189733_spectrum.hd189733_wl,
                          spectrum_flux =hd189733_spectrum.hd189733_flux),

    "GJ 436" : StellarSystem(name = "GJ 436",
                          Teff = 3684,
                          Rstar_Rsun = 0.464,
                          orbital_distance_AU = 0.02887,
                          spectrum_wl = gj436_spectrum.gj436_wl,
                          spectrum_flux =gj436_spectrum.gj436_flux)
                                               
}