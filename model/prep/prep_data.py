from multiprocessing.dummy import Array
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.utils import shuffle





class PrepData:
    """A Class that returns the prepped/transformed data set for White and Giant Dwarf classification. """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def classify(self):
        """
        Description: Function to classify Dwarfs and Giants based on Spectral Type
        Args: None
        Returns: DataFrame 
        """ 
        self.df['Target'] = self.df['SpType']
        for i in range(len(self.df['Target'])):
            if "V" in self.df.loc[i,'Target']: 
                if "VII" in self.df.loc[i,'Target']: 
                    self.df.loc[i,'Target'] = 0 # VII is Dwarf
                else:
                    self.df.loc[i,'Target'] = 1 # IV, V, VI are Giants
            elif "I" in self.df.loc[i,'Target']: 
                self.df.loc[i,'Target'] = 0 # I, II, III are Dwarfs
            else: 
                self.df.loc[i,'Target'] = 3 # None
        return

    def balance(self):
        """
        Function to balance the classes
        """

        df_giants = self.df[self.df.Target == 1]
        df_dwarfs = self.df[self.df.Target == 0]
        df_others = self.df[self.df.Target == 3]

        reample_df_giants = resample(df_giants, 
                                 replace=False,    
                                 n_samples=df_dwarfs.shape[0],     
                                 random_state=1)
        reample_df_others = resample(df_others, 
                                 replace=False,    
                                 n_samples=df_dwarfs.shape[0],     
                                 random_state=1)
 
        # Combine minority class with downsampled majority class
        self.df = pd.concat([reample_df_giants, df_dwarfs,reample_df_others])
        self.df.reset_index(inplace=True, drop=True)
        self.df = shuffle(self.df)
        return self.df

    def split(self, df: pd.DataFrame, feature: Array, labels: Array):
        """
        Description: Function to seperate the features and labels of the dataset.
        Args: 
            df: (Dataframe) Dataframe of the data
            feature: (Array) the column values of what the feature set should be
            labels: (Array) the column names of what the label set should be
        Returns: Tuple[Array, Array] 
        """ 
        X = df[feature]
        X = X.values
        Y = df[labels]
        Y = Y.values
        Y = Y.flatten().astype('int')
        return X, Y

    
    def encode(self, X):
        """
        Function to encode the data we need to perform the classification.
        Args: X: 
        Returns: X:  
        """ 
        encoded_x = LabelEncoder()
        X[:,4]=encoded_x.fit_transform(X[:,4])
        return X

    def train(self, feature, label, scale: False):
        """
        Function to encode the data we need to perform the classification.
        Args: X: 
        Returns: X:  
        """ 
        X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size = 0.25, random_state = 0)
        if scale:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

