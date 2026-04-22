import pandas as pd
import re
import numpy as np
from astroquery.nist import Nist
import astropy.units as u



class NIST_data:

    """
    This class loads data from the NIST database for an atom/ion in
    a selected wavelength range. It searches for triplet transitions
    subject to atomic alignment.

    Conditions for the alignment effect:
        1. The lower level must be metastable, i.e., transitions from it
           must be only forbidden, or the relaxation time of the level must
           differ by several orders of magnitude compared to the relaxation
           time of the triplet lines.
        2. The upper sublevels must have the same electron configuration
            and the same term, but differ in their J moments.
        3. The search range for lines is 500 - 1500 nm (depends on telescope choosen).
        4. The difference between the wavelengths of the lines
           (at least two of them) must exceed 0.8 Angstrom.
        5. The J moment of the lower level is > 1/2.

    Attributes:
        full_data (DataFrame): A data frame containing the full NIST data,
        excluding transitions with missing values

        trip_data (DataFrame): A data frame containing all triplet transitions
        found subject to atomic alignment

    Usage:
        data = NIST_data('He I', 300, 35e6)
        data.save_triplets_formated()
    """

    def __init__(self, linename : str, lambda1 : float, lambda2 : float, in_vacuum : bool = True,
                 sort_lambda1 : float = 5000.0, sort_lambda2 : float = 15000.0, meta_factor : float = 100.0,
                  wavelength_sep : float = 0.8 ):

        """
        Performs all the basic steps of parsing and filtering
        the loaded data to find triplets that satisfy the alignment conditions

        Note: 
            1) For a reliable line search, the smallest possible initial wavelength 
               must be specified to determine the metastability of the lower transition level.
            2) The code does not take into account transitions where lifetime,
               oscillator strength, or term data are missing.
            3) Here used Ritz wavelengths.

        Args:
            linename (str)  : name of atom or ion (e.g. H I, HeII, Fe VI etc.)
            lambda1 (float) : specifies the start wavelength to load from NIST database in Angstroms
            lambda2 (float) : specifies the finish wavelength to load from NIST database in Angstroms
            in_vacuum (bool) : store wavelenght in vacuum (True) or in air (False) 
            lambda1 (float) : wavelength to star filtering with
            lambda2 (float) : wavelength to end filtering with 
            meta_factor (float) :  sets the threshold in A for checking metastability
            wavelength_sep (float) : wavelength separation between any triplet lines in Angstroms
        """

        self.line = linename

        if in_vacuum:
            wavelength_type = 'vacuum'
        else:
            wavelength_type = 'vac+air'

        nist_table = Nist.query( lambda1 * u.AA, lambda2 * u.AA,
                                 linename=linename, energy_level_unit='eV',
                                 wavelength_type = wavelength_type)
        
        self.raw_data = nist_table.to_pandas()
        print(len(self.raw_data))
        
        self.to_save = True
        if self.raw_data['Ritz'].isna().all():
            print(f"\nNo data found for the {self.line} in NIST!")
            self.to_save = False
            return

        self.raw_data = self.raw_data[['Ritz','Aki', 'fik', 'Ei           Ek', 'Lower level', 'Upper level', 'gi   gk', 'Type']]

        print(f"\nLoaded\t{len(self.raw_data)} rows for {self.line}")
        
        self.__define_transition_type()
        self.raw_data = self.raw_data.dropna()
        self.__clear_titles()
        self.__Ritz_to_float()
        self.__parse_energies()
        self.__parse_stat_weight()
        self.__parse_levels()
        self.__make_dataframe()
        self.__drop_unresolved_lines()
        self.__J_to_float()
        self.full_data = self.full_data.dropna()

        print(f"{self.line} data parsed")

        # just for sorting
        self.triplet_keys = [
            'lower_conf', 'lower_term', 'lower_J',
            'upper_conf', 'upper_term']

        self.__find_all_triplets()
        self.__filter_by_wavelength(sort_lambda1, sort_lambda2)
        self.__filter_by_J()
        self.__filter_by_spectral_resolution(wavelength_sep)
        self.__filter_metastable_states(factor = meta_factor)

        print(f"Data fitered, found {int(len(self.trip_data)/3)} lines")

    def __parse_energies(self):

        """"
        Makes two individual columns for level energies
        """

        self.Ed = []
        self.Eu = []
        Ed_Eu = self.raw_data[['Ei           Ek']]

        for i, l in enumerate(Ed_Eu.values):
            dig = re.findall("\\d+", l[0])
            self.Ed.append(float(f"{dig[0]}.{dig[1]}"))
            self.Eu.append(float(f"{dig[2]}.{dig[3]}"))
           
    def __define_transition_type(self):

        """
        In NIST databse allowed lines have type "NaN", this function
        interprets "NaN" type as "E1" - electric dipole transition
        """

        self.raw_data['Type'] = self.raw_data['Type'].astype(object).fillna('E1')
    
    def __parse_stat_weight(self):

        """
        Makes two individual columns for statstical weight
        """

        self.gd = []
        self.gu = []
        gd_gu = self.raw_data[['gi   gk']]
        for i, l in enumerate(gd_gu.values):
            dig = re.findall("\\d+", l[0])
            self.gd.append(int(dig[0]))
            self.gu.append(int(dig[1]))

    def __parse_levels(self):

        """
        Parses combined colunms from NIST database to separate
        electron configuration, term, full momentum J for both lower and upper levels
        """

        self.low_conf = []
        self.low_term = []
        self.low_J    = []
        levs = self.raw_data[['Lower level']]
        for i, l in enumerate(levs.values):
            Conf, Term, J = l[0].split(sep = '|')
            self.low_conf.append(Conf.strip())
            self.low_term.append(Term.strip())
            self.low_J.append(J.strip())

        self.upp_conf = []
        self.upp_term = []
        self.upp_J    = []
        levs = self.raw_data[['Upper level']]
        for i, l in enumerate(levs.values):
            Conf, Term, J = l[0].split(sep = '|')
            self.upp_conf.append(Conf.strip())
            self.upp_term.append(Term.strip())
            self.upp_J.append(J.strip())

    def __drop_unresolved_lines(self):

        """
        There are some lines with multiple J momentum in NIST database.
        Such cases correspond to mathematically unresolved lines
        (they have too similar wavelengths) and contain ',' symbol.
        This function drops such lines due its practical useless
        """

        bad_idx = []
        for i in range(len(self.low_J)):
            if ',' in self.low_J[i]: bad_idx.append(i)
            if ',' in self.upp_J[i]: bad_idx.append(i)

        bad_idx = list(set(bad_idx))

        self.full_data = self.full_data.drop(bad_idx)
        self.full_data = self.full_data.dropna()
    
    def __make_dataframe(self):

        """
        Makes DataFrame using parsed NIST table
        """

        self.full_data = pd.DataFrame({'Ritz' : self.raw_data[['Ritz']].values[:, 0],
                   'Aud' : self.raw_data[['Aki']].values[:, 0],
                   'fdu' : self.raw_data[['fik']].values[:, 0],
                   'Ed' : self.Ed,
                   'Eu' : self.Eu,       
                   'lower_conf' : self.low_conf,
                   'lower_term' : self.low_term,
                   'lower_J' : self.low_J,
                   'upper_conf' : self.upp_conf,
                   'upper_term' : self.upp_term,
                   'upper_J' : self.upp_J,
                   'gd' : self.gd,
                   'gu' : self.gu,
                   'type' : self.raw_data[['Type']].values[:, 0]})

    def __J_to_float(self):

        """
        Converts J moments to float values
        """

        self.full_data['lower_J'] = self.full_data['lower_J'].apply(self.__str_to_float)
        self.full_data['upper_J'] = self.full_data['upper_J'].apply(self.__str_to_float)

    def __Ritz_to_float(self):

        """
        Converts Ritz wavelength to float values
        """
        self.raw_data['Ritz'] = self.raw_data['Ritz'].astype(str)
        self.raw_data['Ritz'] = self.raw_data['Ritz'].apply(self.__str_to_float)

    def __group_triplets(self):
        return self.trip_data.groupby('triplet_id', group_keys=False)

    def __find_all_triplets(self):
        """
        Finds groups (lower_conf, lower_term, lower_J, upper_conf, upper_term)
        that have exactly 3 transitions from one state to states that differ 
        only in the J term. Sorts by level and wavelength.
        """
        dupls = self.full_data[self.full_data.duplicated(
        subset=self.triplet_keys,
        keep=False
        )]

        triplets = (
            dupls.groupby(self.triplet_keys, group_keys=False)
            .filter(lambda g: (len(g) == 3) and (g['upper_J'].nunique() == 3))
            .copy()
        )

        triplets['triplet_id'] = (
            triplets.groupby(self.triplet_keys).ngroup()
        )

        self.trip_data = triplets.sort_values(
            by=self.triplet_keys + ['Ritz']
        ).reset_index(drop=True)

    def __filter_by_wavelength(self, lambda1: float, lambda2: float):

        """
        Keeps only those triplets for which ALL 3 lines lie in [lambda1, lambda2].

        Args: 
            lambda1 (float) : wavelength to star with
            lambda2 (float) : wavelength to end with 
        
        Returns:
            DataFrame of lines
        """
        def in_range(g):
            cond = (g['Ritz'] >= lambda1) & (g['Ritz'] <= lambda2)
            return cond.all() and len(g) == 3

        self.trip_data = (
            self.__group_triplets()
            .filter(in_range)
            .sort_values(by=self.triplet_keys + ['Ritz'])
            .reset_index(drop=True)
        )

    def __filter_by_J(self):

        """
        Choose transitions with J > 0.5 only
        """
        self.trip_data = (
        self.__group_triplets()
        .filter(lambda g: (g['lower_J'] > 0.5).all())
        .reset_index(drop=True)
    )

    def __filter_by_spectral_resolution(self, min_sep: float):

        """
        Removes triplets for which the maximum pair of differences is <= min_sep
        That is, it leaves only those triplets where there is a pair of
        lines with a difference > min_sep.

        Args:
            min_sep (float) : wavelength separation in angstroms
        """
        def good(g):
            wl = g['Ritz'].values
            diffs = [abs(wl[i] - wl[j]) for i in range(3) for j in range(i+1, 3)]
            return max(diffs) > min_sep

        self.trip_data = (
            self.__group_triplets()
            .filter(good)
            .reset_index(drop=True)
        )

    def __filter_metastable_states(self, factor: float):

        """
        Keeps triplets only if the lower level can be considered metastable.
        Logic: if among the transitions to this level (the entire full_data database)
        there is a transition with A (max_ground_A) significantly LARGER than the triplet's
        characteristic A (sum), then the level is not metastable -> remove the triplet.

        If no transitions to a lower state could be found for a given level, then such a
        triplet is discarded, since it is impossible to say whether it is metastable or not.
        
        Args:
            factor (float) : Sets the threshold in A
        """
        def is_metastable(g):

            row = g.iloc[0]

            mask = (
                (self.full_data['upper_conf'] == row['lower_conf']) &
                (self.full_data['upper_term'] == row['lower_term']) &
                (self.full_data['upper_J']    == row['lower_J'])
            )

            decay_channels = self.full_data[mask]

            if decay_channels.empty:
                return False

            total_A_down = decay_channels['Aud'].fillna(0).astype(float).sum()

            trip_A_sum = g['Aud'].fillna(0).astype(float).sum()

            return total_A_down < factor * trip_A_sum

        self.trip_data = (
            self.__group_triplets()
            .filter(is_metastable)
            .reset_index(drop=True)
        )   

    def __str_to_float(self, s):

        """
        Converts str to float value
        If str contains '/' symbol, then this is interpreted as division.
        If str contains '+' symbol, it means the calculated Ritz wavelength, not the addition

        Args:
            s (str) : string of digits? that may be separated by '/'

        Returns:
            Derived float value
        """

        if not s:
            return np.nan

        if '/' in s:
            numerator, denominator = s.split('/')
            return float(numerator) / float(denominator)
        elif '+' in s:
            wl = s.split('+')
            return float(wl[0])
        else:
            return float(s)

    def __clear_titles(self):

        """
        In NIST there may be data such that the column names
        are repeated not only at the beginning, but also in
        some other row of the table
        (for example FeII with wavelength_type='air+vac')
        """

        self.raw_data = self.raw_data[self.raw_data['Ritz'] != 'Ritz']

    def save_full(self):

        """
        Saves full NIST data for line choosen exlude unresolved lines
        or transitions without crutial information
        """

        if not self.to_save:
            print("Noting to save!")
            return

        self.full_data.to_csv(f"full_data_{self.line}.dat", sep = "\t", index = False, na_rep = 'NaN')
        print("Full data saved")

    def save_triplets(self):

        """
        Saves triplets found that are subjected to alaignment effect
        """
        if not self.to_save:
            print("Nothing to save!")
            return
        
        if self.trip_data.empty:
            print("Nothing to save!")
            return

        self.trip_data.sort_values(
        by=self.triplet_keys + ['Ritz']
        ).to_csv(
            f"triplets_data_{self.line}.dat",
            sep="\t",
            index=False,
            na_rep='NaN'
        )

    def save_triplets_formated(self):

        """
        Saves triplets found that are subjected to alaignment effect
        in more readable form
        """

        if not self.to_save:
            print("Nothing to save!")
            return

        if self.trip_data.empty:
            print("Nothing to save!")
            return
        
        with open(f"triplets_data_{self.line}.dat", "w") as file:

            df = self.trip_data.copy()

            df["E_d(eV)-E_u(eV)"] = df["Ed"].astype(str) + "-" + df["Eu"].astype(str)
            df["g_d-g_u"] = df["gd"].astype(str) + "-" + df["gu"].astype(str)

            columns = [
                "Ritz", "Aud", "fdu", "E_d(eV)-E_u(eV)",
                "lower_conf", "lower_term", "lower_J",
                "upper_conf", "upper_term", "upper_J",
                "g_d-g_u", "type"
            ]

            widths = [
                max(df[col].astype(str).map(len).max(), len(col)) + 2
                for col in columns
            ]

            
            header_line = "".join(col.ljust(width) for col, width in zip(columns, widths))
            file.write(header_line + "\n")
            file.write("-" * len(header_line) + "\n")

            
            df = df.sort_values(by=self.triplet_keys + ['Ritz'])

            
            for _, group in df.groupby('triplet_id'):

                group = group.sort_values('Ritz')

                for _, row in group.iterrows():
                    values = [str(row[col]) for col in columns]
                    line = "".join(v.ljust(w) for v, w in zip(values, widths))
                    file.write(line + "\n")

                file.write("\n")
    
    def inspect_lower_level_decay(self, lower_conf: str, lower_term: str, lower_J: float, top_n: int = 20):

        """
        Prints all radiative decay channels for a given lower level
        belonging to a selected triplet.

        This method is intended for validation of metastability
        assumptions used in triplet selection.

        Parameters
        ----------
        lower_conf : str
            Electron configuration of the lower level (e.g. '3d6.(3F2).4s')
        lower_term : str
            Term symbol of the lower level (e.g. 'b 4F')
        lower_J : float
            Total angular momentum J of the lower level
        top_n : int, optional
            Number of strongest decay channels to display (sorted by A coefficient)

        Returns
        -------
        pandas.DataFrame
            Table of all found decay channels sorted by Einstein A coefficient

        Output
        ------
        Prints a formatted list of all transitions where the selected level
        acts as an upper level (i.e. all radiative decay channels).

        Notes
        -----
        - This includes all transitions present in the loaded NIST dataset
          within the queried wavelength range.
        - Missing transitions outside the dataset are NOT accounted for.
        - Useful for assessing metastability assumptions used in triplet filtering.
        """

        mask = (
            (self.full_data['upper_conf'] == lower_conf) &
            (self.full_data['upper_term'] == lower_term) &
            (self.full_data['upper_J']    == lower_J)
        )

        decay = self.full_data[mask].copy()

        if decay.empty:
            print("\nNo decay channels found for this level in current dataset.")
            return decay

        decay = decay.sort_values(by='Aud', ascending=False)

        print("\n" + "=" * 90)
        print(f"Decay channels for level: {lower_conf} | {lower_term} | J={lower_J}")
        print("=" * 90)

        cols = ['Ritz', 'Aud', 'fdu', 'lower_conf', 'lower_term', 'lower_J',
                'upper_conf', 'upper_term', 'upper_J', 'type']

        for i, row in decay.head(top_n).iterrows():

            A = pd.to_numeric(row['Aud'], errors='coerce')
            f = pd.to_numeric(row['fdu'], errors='coerce')

            A_str = f"{A:.3e}" if pd.notna(A) else "NaN"
            f_str = f"{f:.3e}" if pd.notna(f) else "NaN"

            print(
                f"λ={row['Ritz']:.5f} A | "
                f"A={A_str} | "
                f"f={f_str} | "
                f"{row['upper_conf']} {row['upper_term']} J={row['upper_J']} → "
                f"{row['lower_conf']} {row['lower_term']} J={row['lower_J']} | "
                f"{row['type']}"
            )

        print("=" * 90)
        print(f"Total channels found: {len(decay)}")
        print(f"Total A (approx lifetime inverse): {decay['Aud'].fillna(0).astype(float).sum():.3e}")
        print("=" * 90)

        return decay


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
                    sort_lambda1=1000,
                    sort_lambda2=15000,
                    meta_factor=100.0
                )

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
                     sort_lambda1 = 4000,
                     sort_lambda2 = 15000,
                     meta_factor = 100.0)
    
    data.save_triplets_formated()

    data.inspect_lower_level_decay("3d6.(3F2).4s",
                                   "b 4F",
                                   1.5)

    # He I
    data = NIST_data(linename ='He I',
                     lambda1 = 1e-6,
                     lambda2 = 35e7,
                     in_vacuum = False,
                     sort_lambda1 = 5000,
                     sort_lambda2 = 15000,
                     meta_factor = 100.0)
    
    data.save_triplets_formated()

    data.inspect_lower_level_decay("1s.2s",
                                   "3S",
                                   1.0)

if __name__ == '__main__':
    consider_many_elements()
    #main()
