from find_lines import NIST_data


def consider_many_elements():
    """
    Searches for triplets subject to alignment effect for all relevant chemical elements
    """

    elements = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn"]

    ion_states = ["I", "II", "III", "IV"] 

    for el in elements:
        for ion in ion_states:

            linename = f"{el} {ion}"

            try:
                print(f"\nProcessing {linename}")

                data = NIST_data(
                    linename=linename,
                    lambda1=1e-6,
                    lambda2=35e7,
                    in_vacuum=True,
                    sort_lambda1=500,
                    sort_lambda2=15000,
                    system_name="GJ 436",
                    use_planck = False)

                if data.to_save:
                    data.save_triplets_formated()

            except Exception as e:
                print(f"Failed for {linename}: {e}")


def main():

    """
    Example usage of the NIST_data class for helium atom and iron ion.
    """

    # Fe II
    data = NIST_data(linename ='Fe II',
                     lambda1 = 1e-6,
                     lambda2 = 35e7,
                     in_vacuum = False,
                     sort_lambda1 = 1000,
                     sort_lambda2 = 15000,
                     system_name="Sun",
                     use_planck = True)
    
    data.save_triplets_formated()

    data.inspect_lower_level_decay("3d6.(3F2).4s",
                                   "b 4F",
                                   1.5)

    #He I
    data = NIST_data(linename ='He I',
                     lambda1 = 1e-6,
                     lambda2 = 35e7,
                     in_vacuum = False,
                     sort_lambda1 = 5000,
                     sort_lambda2 = 15000,
                     system_name="Sun",
                     use_planck = True)
    
    data.save_triplets_formated()

    data.inspect_lower_level_decay("1s.2s",
                                   "3S",
                                   1.0)

if __name__ == '__main__':
    #consider_many_elements()
    main()
