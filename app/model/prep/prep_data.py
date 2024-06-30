from multiprocessing.dummy import Array
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.utils import shuffle


class PrepData:
    """A Class that returns the prepped/transformed data set for White and Giant Dwarf classification."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the PrepData class with a DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing star data.
        """
        self.df = df

    def classify(self) -> pd.DataFrame:
        """
        Classify Dwarfs and Giants based on Spectral Type.
        
        Returns:
            pd.DataFrame: DataFrame with an additional 'Target' column for classification.
        """
        def categorize_star_type(sp_type: str) -> int:
            if "V" in sp_type:
                if "VII" in sp_type:
                    return 0  # Dwarfs are VII
                else:
                    return 1  # Giants are V, IV, and VI
            elif "I" in sp_type:
                return 0  # Dwarfs are I, II, III
            else:
                return 3  # Other Class

        # Apply the function to the 'SpType' column to create the 'Target' column
        self.df['Target'] = self.df['SpType'].apply(categorize_star_type)
        return self.df

    def balance(self) -> pd.DataFrame:
        """
        Balance the classes in the DataFrame.
        
        Returns:
            pd.DataFrame: Balanced DataFrame.
        """
        giants = self.df[self.df['Target'] == 1]
        others = self.df[self.df['Target'] == 3]
        dwarfs = self.df[self.df['Target']== 0]

        #significantly more other stars, then more Giants, not as many dwarfs. We need to rebalance dataset.
        resamp_giants = resample(giants, 
                                        replace=False,    
                                        n_samples=dwarfs.shape[0],     
                                        random_state=1)
        resamp_others = resample(others, 
                                        replace=False,    
                                        n_samples=dwarfs.shape[0],     
                                        random_state=1)
        
        self.df = pd.concat([resamp_others, resamp_giants, dwarfs])
        self.df.reset_index(inplace=True, drop=True)
        self.df = shuffle(self.df)
        return self.df

    def split(self, df: pd.DataFrame, features: Array, labels: Array) -> tuple:
        """
        Separate the features and labels of the dataset.
        
        Args:
            df (pd.DataFrame): DataFrame of the data.
            features (Array): Column names of the feature set.
            labels (Array): Column names of the label set.
        
        Returns:
            tuple: Tuple containing feature array (X) and label array (Y).
        """
        X = df[features].values
        Y = df[labels].values.flatten().astype('int')
        return X, Y

    def encode(self, X: np.ndarray, position) -> np.ndarray:
        """
        Encode the data for classification.
        
        Args:
            X (np.ndarray): Feature array to be encoded.
        
        Returns:
            np.ndarray: Encoded feature array.
        """
        encoder = LabelEncoder()
        X[:, position] = encoder.fit_transform(X[:, position])
        return X

    def train(self, features: np.ndarray, labels: np.ndarray, scale: bool = False) -> tuple:
        """
        Split the data into training and testing sets, and optionally scale the features.
        
        Args:
            features (np.ndarray): Feature array.
            labels (np.ndarray): Label array.
            scale (bool): Whether to scale the features.
        
        Returns:
            tuple: Tuple containing training and testing feature and label arrays.
        """
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)
        if scale:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
