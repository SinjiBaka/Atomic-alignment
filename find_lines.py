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
        3. The search range for lines is 500 - 1500 nm.
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

    def __init__(self, linename : str, lambda1 : float, lambda2 : float):

        """
        Performs all the basic steps of parsing and filtering
        the loaded data to find triplets that satisfy the alignment conditions

        Note: 
            1) For a reliable line search, the smallest possible initial wavelength 
               must be specified to determine the metastability of the lower transition level.
            2) The code does not take into account transitions where lifetime,
               oscillator strength, or term data are missing.
            3) Wavelengths are given in vacuum units of angstroms.

        Args:
            linename (str)  : name of atom or ion (e.g. H I, HeII, Fe VI etc.)
            lambda1 (float) : specifies the start wavelength to load from NIST database in Angstroms
            lambda2 (float) : specifies the finish wavelength to load from NIST database in Angstroms
        """

        self.line = linename
        nist_table = Nist.query( lambda1 * u.AA, lambda2 * u.AA,
                                 linename= linename, energy_level_unit='eV')
        
        self.raw_data = nist_table.to_pandas()
        self.raw_data = self.raw_data[['Ritz','Aki', 'fik', 'Ei           Ek', 'Lower level', 'Upper level', 'gi   gk', 'Type']]

        
        print(f"\nLoaded\t{len(self.raw_data)} rows for {self.line}")
        
        self.__define_transition_type()
        self.raw_data = self.raw_data.dropna()
        self.__Ritz_to_float()
        self.__parse_energies()
        self.__parse_stat_weight()
        self.__parse_levels()
        self.__make_dataframe()
        self.__drop_unresolved_lines()
        self.__J_to_float()
        self.full_data = self.full_data.dropna()

        print(f"{self.line} data parsed")

        self.__find_all_triplets()
        self.__filter_by_wavelength(5000, 15000)
        self.__filter_by_J()
        self.__filter_by_spectral_resolution()
        self.__filter_metastable_states(factor = 100)

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

    def __find_all_triplets(self):
        """
        Finds groups (lower_conf, lower_term, lower_J, upper_conf, upper_term)
        that have exactly 3 transitions from one state to states that differ 
        only in the J term. Sorts by level and wavelength.
        """
        dupls = self.full_data[self.full_data.duplicated(
            subset=['lower_conf', 'lower_term', 'lower_J', 'upper_conf', 'upper_term'],
            keep=False
        )]

        triplets = (
            dupls.groupby(
                ['lower_conf', 'lower_term', 'lower_J', 'upper_conf', 'upper_term'],
                group_keys=False
            )
            .filter(lambda g: (len(g) == 3) and (g['upper_J'].nunique() == 3))
        )

        triplets = triplets.sort_values(
            by=['lower_conf', 'lower_term', 'lower_J', 'upper_conf', 'upper_term', 'Ritz']
        ).reset_index(drop=True)

        self.trip_data = triplets

    def __filter_by_wavelength(self, lambda1: float, lambda2: float):

        """
        Keeps only those triplets for which ALL 3 lines lie in [lambda1, lambda2].

        Args: 
            lambda1 (float) : wavelength to star with
            lambda2 (float) : wavelength to end with 
        
        Returns:
            DataFrame of lines
        """
        def triplet_in_range(group):
            in_range = (group['Ritz'] >= lambda1) & (group['Ritz'] <= lambda2)
            return in_range.all() and (len(group) == 3)

        grouped = self.trip_data.groupby(
            ['lower_conf', 'lower_term', 'lower_J', 'upper_conf', 'upper_term'],
            group_keys=False
        )

        filtered = grouped.filter(triplet_in_range)

        self.trip_data = filtered.sort_values(
            by=['lower_conf', 'lower_term', 'upper_conf', 'upper_term', 'Ritz']
        ).reset_index(drop=True)

        if len(self.trip_data) % 3 != 0:
            print("Warning: missing rows - range does not fully cover triplets!")

        return self.trip_data

    def __filter_by_J(self):

        """
        Choose transitions with J > 0.5 only
        """
        self.trip_data = self.trip_data[self.trip_data['lower_J'] > 0.5]

    def __filter_by_spectral_resolution(self, min_sep: float = 0.8):

        """
        Removes triplets for which the maximum pair of differences is <= min_sep
        That is, it leaves only those triplets where there is a pair of
        lines with a difference > min_sep.

        Args:
            min_sep (float) : wavelength separation in angstroms
        """
        bad_idx = []
        for i in range(0, len(self.trip_data) - 2, 3):
            l1 = float(self.trip_data.iloc[i]['Ritz'])
            l2 = float(self.trip_data.iloc[i+1]['Ritz'])
            l3 = float(self.trip_data.iloc[i+2]['Ritz'])
            diffs = [abs(l1 - l2), abs(l1 - l3), abs(l2 - l3)]
            
            if max(diffs) <= min_sep:
                bad_idx.extend([i, i+1, i+2])

        if bad_idx:
            self.trip_data = self.trip_data.drop(self.trip_data.index[bad_idx]).reset_index(drop=True)

    def __filter_metastable_states(self, factor: float = 100.0):

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
        bad_idx = []
        for d in range(0, len(self.trip_data) - 2, 3):
            lower_conf = self.trip_data.iloc[d]['lower_conf']
            lower_term = self.trip_data.iloc[d]['lower_term']
            lower_J = self.trip_data.iloc[d]['lower_J']

            mask = (
                (self.full_data['upper_conf'] == lower_conf) &
                (self.full_data['upper_term'] == lower_term) &
                (self.full_data['upper_J']    == lower_J)
            )

            ground = self.full_data[mask]

            if ground.empty:
                print(
                    f"No ground line found for state\t{lower_conf} {lower_term} {lower_J}!"
                    " Triplet will be omitted."
                )
                bad_idx.extend([d, d+1, d+2])
                continue

            max_ground_A = ground['Aud'].dropna().astype(float).max() if not ground['Aud'].dropna().empty else 0.0

            trip_A_sum = (
                float(self.trip_data.iloc[d]['Aud'] if not pd.isna(self.trip_data.iloc[d]['Aud']) else 0.0) +
                float(self.trip_data.iloc[d+1]['Aud'] if not pd.isna(self.trip_data.iloc[d+1]['Aud']) else 0.0) +
                float(self.trip_data.iloc[d+2]['Aud'] if not pd.isna(self.trip_data.iloc[d+2]['Aud']) else 0.0)
            )

            if max_ground_A > factor * max(trip_A_sum, 1e-20):
                bad_idx.extend([d, d+1, d+2])

        if bad_idx:
            self.trip_data = self.trip_data.drop(self.trip_data.index[bad_idx]).reset_index(drop=True)   

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

    def save_full(self):

        """
        Saves full NIST data for line choosen exlude unresolved lines
        or transitions without crutial information
        """
        self.full_data.to_csv(f"full_data_{self.line}.dat", sep = "\t", index = False, na_rep = 'NaN')
        print("Full data saved")

    def save_triplets(self):

        """
        Saves triplets found that are subjected to alaignment effect
        """

        self.trip_data.to_csv(f"triplets_data_{self.line}.dat", sep = "\t", index = False, na_rep = 'NaN')
        print("Triplets saved")

    def save_triplets_formated(self):

        """
        Saves triplets found that are subjected to alaignment effect
        in more readable form
        """

        if self.trip_data.empty:
            print("Noting to save!")
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

            for i in range(0, len(df) - 2, 3):
                for j in [i, i + 1, i + 2]:
                    l = df.iloc[j]
                    values = [str(l[col]) for col in columns]
                    line = "".join(v.ljust(w) for v, w in zip(values, widths))
                    file.write(line + "\n")
                file.write("\n")
                    


def main():

    """
    Example usage of the NIST_data class for helium atom and carbor ion.
    """

    data = NIST_data('He I', 300, 35e6)
    data.save_triplets_formated()

    data = NIST_data('C II', 300, 35e6)
    data.save_triplets_formated()


if __name__ == '__main__':
    main()