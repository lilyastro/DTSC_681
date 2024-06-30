from multiprocessing.dummy import Array
import pandas as pd
import numpy as np
from sklearn.utils import resample



class CleanData:
    """A Class that returns the cleaned and transformed data set for White and Giant Dwarf classification.
    This is a prepartory """
    
    def __init__(self, file):
        self.df = pd.read_csv(file)

    
    def filter_df(self, columns: Array):
        """
        Function to filter the df to only data we need to perform the classification.
        Args: DataFrame
        Returns: DataFrame 
        """ 
        self.df = self.df[columns]
        print(f'Filtering df... {self.df}')
        return self.df[columns]

    def convert_datatypes(self):
        """
        Convert the columns from a dataframe.
        Returns: DataFrame 
        """ 
        for col in self.df.columns:
            if col != 'SpType':
                self.df[col] = pd.to_numeric(self.df[col], downcast="float", errors='coerce')
        print(f'converting data types...')
        return self.df

    def drop_nulls(self):
        """
        Drop rows with null values from the dataframe.
        Returns:
            pd.DataFrame: Dataframe with rows containing null values removed.
        """
        self.df = self.df.dropna() 
        return self.df.dropna() 



    def drop_dupes(self):
        """
        Function to convert the columns from a dataframe.
        Args: DataFrame
        Returns: None 
        """ 
        self.df = self.df.drop_duplicates()
        return self.df


    def drop_outlier_plx(self):
        """
        Function to drop outlier e_Plx and where Plx = 0 from a dataframe.
        The definition is an outlier is 2 standard deviations away from the median.
        Args: DataFrame
        Returns: DataFrame 
        """

        med = self.df['e_Plx'].med()
        std = self.df['e_Plx'].std()
        
        threshold = med + (2 * std)

        cleaned_df = self.df[self.df['e_Plx'] < threshold]
        cleaned_df = cleaned_df[cleaned_df['Plx'] != 0]

        return cleaned_df

    def drop_outlier_plx(self):
        """
        Function to drop outlier e_Plx from a dataframe.
        The definition is an outlier is 2 standard deviations away from the median.
        Args: DataFrame
        Returns: DataFrame 
        """

        mean = self.df['e_Plx'].mean()
        std = self.df['e_Plx'].std()
        
        threshold = mean + (2 * std)

        self.df = self.df[self.df['e_Plx'] < threshold]
        
        return self.df

    def add_abs_mag(self):
        """
        Function to add the Absolute Magnitiude column to the dataframe.
        Args: DataFrame
        Returns: DataFrame 
        """ 
        # drop where parallax = 0, cannot divide by 0.
        self.df = self.df[self.df['Plx'] != 0]
        self.df["Abs_Mag"] = self.df["Vmag"] + 5 * (np.log10(abs(self.df["Plx"]))+1)
        return self.df


