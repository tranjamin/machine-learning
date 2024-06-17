# Data Visualisation, Preprocessing and Statistics

from __future__ import annotations

import pandas as pd
import numpy as np
import numpy.typing as nptype
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold

from abc import ABC, abstractmethod
import random
from sklearn.decomposition import PCA

pd.options.mode.chained_assignment = None

class Column(ABC):
    '''
    An Abstract Class representing a column of data in a dataset.
    '''

    @abstractmethod
    def visualise(self, **kwargs) -> None:
        '''
        Visualises the dataset by plotting it
        '''
        pass

    @abstractmethod
    def display_stats(self) -> None:
        '''
        Displays some basics statistics about the data
        '''
        pass

    def get_data(self) -> pd.Series:
        '''
        Retrieves that data as a pandas series
        '''
        return self.data

    def get_unique(self) -> nptype.NDArray:
        '''
        Gets all the unique values of this column
        '''
        return self.get_data().unique()

    def make_categorical(self):
        pass

    def make_numerical(self):
        pass
    
    def __len__(self):
        return self.total_count

class CategoricalColumn(Column):
    def __init__(self, data):
        self.data = data
        self.name = self.data.name
               
        self.categories = self.data.unique()
        self.num_categories = len(self.categories)
        self.categories_count = dict(self.data.value_counts())
        self.total_count = len(self.data)
        self.categories_proba = {k: v/self.total_count for k,v in self.categories_count.items()}

        self.encodings = None
        self.unencode = None
        self.encoder = None
    
    def onehot_encode(self) -> pd.DataFrame:
        '''
        Returns the onehot encoding of the column. Does not modify the object
        '''
        encoded = pd.get_dummies(self.data, prefix=f'{self.name}')
        return encoded.astype('int64')
    
    def entropy(self) -> float:
        running_total = 0
        for p in self.categories_proba.values():
            running_total += -(p * math.log(p, 2))
        return running_total
    
    def display_stats(self):
        print(f"Category Name: {self.name}")
        print(f"Number of categories: {len(self.categories)} | Categories: {self.categories}")
        print(f"Total datapoints: {self.total_count} | Distribution: {self.categories_proba}")
    
    def visualise(self):
        plt.bar(list(self.categories), list(self.categories_count.values()))
        plt.title(f"Distribution of '{self.name}'")
        plt.show()

    def make_numerical(self):
        '''
        Returns the label encoding of the column. Does not modify the object
        The returned object has additional properties .ecoder, .unencode, .encodings
        '''
        encoder = LabelEncoder()
        encoder.fit(self.data)
        new_data = pd.Series(encoder.transform(self.data), name=self.name)
        new_col = NumericalColumn(new_data)
        new_col.encoder = encoder
        new_col.unencode = encoder.inverse_transform
        new_col.encodings = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
        return new_col

    def make_categorical(self):
        return self

class NumericalColumn(Column):
    def __init__(self, data):
        self.data = data
        self.name = self.data.name

        self.mean = self.data.mean()
        self.std = self.data.std()
        if self.std == 0:
            self.std = 1
        self.mode = self.data.mode()[0]
        self.median = self.data.median()
        self.min = self.data.min()
        self.max = self.data.max()
        self.total_count = len(self.data)

        self.encodings = None
        self.unencode = None
        self.encoder = None
    
    def entropy(self, width) -> float:
        running_total = 0
        bins = np.arange(self.min - width, self.max + width, width)
        binned_data = pd.cut(self.data, bins)
        bin_populations = list(binned_data.value_counts())
        print(bin_populations)
        for num in bin_populations:
            proba = num/self.total_count
            if proba:
                running_total += -(proba * math.log(proba, 2))
        
        return running_total

    def display_stats(self):
        print(f"Category Name: {self.name}")
        print(f"Min: {self.min} | Max: {self.max} | Count: {self.total_count}")
        print(f"Mean: {self.mean} | Std: {self.std}")
        print(f"Mode: {self.mode} | Median: {self.median}")
    
    def visualise(self, width, min=None, max=None):
        plt.hist(self.data, bins=np.arange(self.min if min is None else min, (self.max if max is None else max) + width, width))
        plt.title(f"Binned Distribution of '{self.name}'")
        plt.show()
    
    def normalise(self, inplace=True):
        '''
        Returns the normalised data. Does not modify the object unless inplace=True
        '''
        debiased = self.data - self.mean
        normalised = debiased / self.std
        if inplace:
            self.data = normalised
        return normalised

    def fill_null_values(self):
        pass

    def make_categorical(self):
        '''
        Returns a new object. Has the additional parameters .encoder, .unencode, .encodings
        Does not modify the current object
        '''
        encoder = LabelEncoder()
        encoder.fit(self.data)
        # new_data = pd.Series(encoder.transform(self.data), name=self.name)
        new_data = self.data
        new_col = CategoricalColumn(new_data)
        new_col.encoder = encoder
        new_col.unencode = encoder.inverse_transform
        new_col.encodings = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
        return new_col
    
    def make_numerical(self):
        return self


class Data():
    '''
    A class which holds and manipulates datasets. The data itself is accessible by self.get_data(), however each column
    is also encoded as a Column object, which allows for more granular manipulation. Be careful because the
    data of the columns is only a copy (by value) of the data in the dataset.

    Error Handling: this is how the dataset handles invalid values
        err_strategy = 'ignore': does not look for invalid values. Default.
        err_strategy = 'remove': removes all rows which have invalid values.
    
    invalid values are determined by the 'errors' parameter

    Typical Usage:
    df = pd.read_csv("filename") # read data in as a dataframe \n\r
    dataset = Data(df, err_strategy='remove') # store data in the object, always set to 'remove' when first importing data \n\r
    dataset = dataset.select(['x1', 'x2', 'class_input', 'class_output']) # select the relevant columns \n\r
    dataset.normalise_numeric() # normalise all numeric data \n\r
    dataset.make_numeric('class_output') # label encode the outputs \n\r
    dataset = dataset.encode_categoricals() # onehot encode all other categories \n\r

    train_dataset, test_dataset = dataset.train_test_split(split_ratio=0.9, shuffle=True) \n\r
    trainX, trainY = train_dataset.input_output_split(outputs='class_output') \n\r
    testX, testY = test_dataset.input_output_split(outputs='class_output') \n\r

    '''
    
    def __init__(self, dataset: pd.DataFrame, err_strategy: str='ignore', errors: list[str]=['NaN', '.', 'nan']):
        # read data in
        self.df = dataset

        # remove corrupted rows
        if (err_strategy == 'remove'):
            bad_rows = self.df.apply(lambda row: row.astype(str).isin(errors)).any(axis=1)
            new_df = self.df[~bad_rows]
            print(f'Removing {len(self.df) - len(new_df)} rows')
            self.df = new_df

            # reset indices
            self.df = self.df.reset_index(drop=True)
        elif (err_strategy == "ignore"):
            pass


        # try to reconvert each row to numeric
        for col_name in self.df.columns:
            try:
                self.df[col_name] = pd.to_numeric(self.df[col_name])
            except:
                pass
        
        # store column data
        self.num_columns = self.df.columns
        self.columns = self.df.columns

        # store columns as either categorical or numerical
        self.column_objects = {}
        for col_name in self.columns:
            data = self.df.get(col_name)
            if self.df.get(col_name).dtype.name == 'object':
                self.column_objects[col_name] = CategoricalColumn(data)
            else:
                self.column_objects[col_name] = NumericalColumn(data)
        
    def copy(self) -> Data:
        '''
        Copies a dataset into a new object. This is a deepcopy so even Column objects are copied.
        '''
        new_data = Data(self.df)
        new_data.df = self.df
        new_data.num_columns = self.num_columns
        new_data.columns = self.df.columns
        new_data.column_objects = self.column_objects
        return new_data

    def preview(self) -> None:
        '''
        Displays a preview of the dataset.
        '''
        print(self.df.head())

    def get_col(self, col_name: str) -> Column:
        '''
        Retrieves a single column of data by name.\n\r
        Data is returned in an object form to allow manipulations, either a NumericalColumn or CategoricalColumn.\n\r
        Be careful manipulating these column objects, because although the columns may be updated you need to manually update the dataset. Refer to .update() for more information.
        '''
        return self.column_objects.get(col_name)
    
    def col_is_categorical(self, col_name: str) -> bool:
        '''
        Determines whether a column is a categorical or not.
        '''
        return isinstance(self.get_col(col_name), CategoricalColumn)
    
    def get_cols(self) -> dict[str, Column]:
        '''
        Get all the columns of the dataset.\n\r
        Data is returned as a dictionary, with keys being the column names and dictionaries being the columns
        stored in object form, either a NumericalColumn or CategoricalColumn\n\r
        Be careful manipulating these column objects, because although the columns may be updated you need to manually update the dataset. Refer to .update() for more information.
        '''
        return self.column_objects
    
    # def col_to_num(self, col_name):
    #     self.column_objects[col_name] = self.get_col(col_name).make_numerical()
    
    # def col_to_cat(self, col_name):
    #     self.column_objects[col_name] = self.get_col(col_name).make_categorical()

    def plot_against(self, x: str, y: str, categories: str | None =None) -> None:
        '''
        Plot a two numerical columns, x & y, against each other.
        Optionally also plot a categorical (or numerical) column using different colours.\n\r

        If you want to plot numerical vs categorical, the easiest way to do this is plot_against(x, x, categories=category)
        '''
        colormappings = None
        if categories is not None:
            if self.col_is_categorical(categories):
                colormappings = self.get_col(categories).make_numerical().get_data() # map the categories to integers
            else:
                colormappings = self.get_col(categories).get_data()
                
        plt.scatter(self.get_col(x).get_data(), self.get_col(y).get_data(), c=colormappings)
        plt.xlabel(self.get_col(x).name)
        plt.ylabel(self.get_col(y).name)
        plt.title(f'Plot with colourmappings "{self.get_col(categories).name}"')
        plt.legend()
        plt.colorbar()
        plt.ion()
        plt.show()

    def encode_categoricals(self) -> Data:
        '''
        Encodes all categorical columns using one-hot encoding. Each class in a categorical column is given its own column, where a 1 represents if a row was of that class and a 0 otherwise.
        
        Returns a new corresponding Data object, does not modify the original object.
        '''
        df = pd.DataFrame()

        for col_name in self.columns:
            obj = self.get_col(col_name)
            if (self.col_is_categorical(col_name)):
                encoded = obj.onehot_encode()
                for col in encoded:
                    df.insert(len(df.columns), col, encoded[col], allow_duplicates=False)
            else:
                df.insert(len(df.columns), col_name, obj.data, allow_duplicates=False)
        
        return Data(df)

    def make_numerical(self, col_name: str) -> None:
        '''
        Converts a categorical column to a numerical format by label encoding. Each class in the category is encoded to a distinct integer starting from 0.\n\r
        Accessing self.get_col() will now return a CategoricalColumn. The data is also updated in self.get_data().
        '''
        self.column_objects[col_name] = self.get_col(col_name).make_numerical()
        self.update_data(col_name=col_name)
    
    def make_categorical(self, col_name: str) -> None:
        self.column_objects[col_name] = self.get_col(col_name).make_categorical()
        self.update_data(col_name=col_name)

    def train_test_split(self, split_ratio: float=0.7, shuffle: bool=True, stratify:nptype.ArrayLike=None) -> tuple[Data, Data]:
        '''
        Splits a dataset into two datasets. This is usually for training and testing, but can also be used for manual hold out datasets.\n\r

        Parameters:
        split_ratio: the fraction of the dataset to allocate to training
        shuffle: true if the data should be shuffled before splittig
        stratify: None if the data should not be stratified. Otherwise, stratifies the data so there is the same proportion of labels in each set. Read more at https://scikit-learn.org/stable/modules/cross_validation.html#stratification

        Returns:
        (train_dataset, test_dataset)

        Example Usage:
        train_dataset, test_dataset = self.train_test_split(split_ratio=0.8, shuffle=True)
        '''
        train, test = train_test_split(self.df, train_size=float(split_ratio), shuffle=shuffle, stratify=stratify)

        return Data(train), Data(test)

    def filter(self, col_name: str, col_value: str, exclude: bool=False) -> Data:
        '''
        Filters out a dataset to only include rows where the value at col_name is col_value.
        Returns a new dataset which is filtered. Does not modify the original dataset.\n\r
        If exclude is set to True, filters for everything but col_value
        '''
        new_df = None
        if not exclude:
            new_df = self.df.loc[self.df[col_name] == col_value].reindex()
        else:
            new_df = self.df.loc[self.df[col_name] != col_value].reindex()
        
        return Data(new_df)
        
    def select(self, columns: str | list[str], exclude: bool=False) -> Data:
        '''
        Selects a subset of the dataset columns and returns a new Data object containing only those columns.\n\r
        columns can either a string to select a single column or an array to select multiple columns

        If exclude is set to true removes all of these columns instead.
        '''
        if not exclude:
            return Data(self.df[columns])
        else:
            return Data(self.df.drop(columns, axis=1))
    
    def replace(self, col_name: str, replace_dict: dict) -> None:
        '''
        Replaces data in a certain column according to a dictionary. Replacements are done in-place

        A dictionary {'a':'b', 'c':'d'} means that all instances of 'a' are replaced by 'b' and all instances of 'c' are replaced with 'd'
        '''
        col = self.get_col(col_name)
        col.get_data().replace(to_replace=replace_dict, inplace=True)
        self.update_data(col_name)
    
    def get_data(self) -> pd.DataFrame:
        '''
        Gets the data in this dataset, in the format of a pandas DataFrame.\n\r
        For certain external functions in pytorch, tensorflow or scikit-learn, you may need to do this to input data.
        '''
        return self.df

    def get_unique(self, col_name: str) -> nptype.NDArray:
        '''
        Gets all the unique values of a certain column.
        '''
        col = self.get_col(col_name)
        return col.get_data().unique()
    
    def normalise_numeric(self) -> None:
        '''
        Normalise all numeric columns.\n\r
        Data is updated in the self.get_data() and in each individual columnz
        '''
        for col_name in self.columns:
            if self.col_is_categorical(col_name=col_name):
                continue

            self.get_col(col_name).normalise(inplace=True)
            self.update_data(col_name)
    
    def update_data(self, col_name: str) -> None:
        '''
        Updates the data of a column so it is reflected in the dataset (i.e. wwhen calling self.get_data()).\n\r
        This only works for manipulations that only change column values, and does not add additional columns (i.e. don't use this for onehot encoding updates)
        '''
        obj = self.get_col(col_name)
        self.df[col_name] = obj.get_data()
    
    def bootstrap(self, seed=None) -> Data:
        '''
        Bootstrap samples the data with replacement.
        '''
        new_df = self.df.sample(frac=1, replace=True, axis=0, random_state=seed)
        return Data(new_df)

    def input_output_split(self, outputs: str | list[str]) -> tuple[Data, Data]:
        '''
        Splits the data into inputs and outputs. Returns two Data objects

        Parameters:
            outputs: a string or list of strings that represent the outputs in the dataset.
        
        Returns:
            (inputs, outputs): the datasets of inputs and outputs
        '''
        output_data = pd.DataFrame(self.df[outputs])
        input_data = self.df.drop(outputs, axis=1, inplace=False)
        return Data(input_data), Data(output_data)
    
    def k_fold_validation_split(self, k):
        kf = KFold(k, shuffle=True)
        splits = kf.split(self.get_data())

    def covariance_matrix(self):
        return np.cov(self.get_data(), rowvar=False)
    
    def PCA_decompose(self, n_components=None):
        pca = PCA(n_components=n_components)
        pca.fit(self.get_data())
        keys = self.get_data().columns
