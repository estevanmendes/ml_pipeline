from ml_pipeline.dataset import Dataset
import pandas as pd


class Loader:
    """
    A class that loads and processes a dataset.

    Parameters:
    path (str): The path to the CSV file containing the dataset.
    cat_columns (list): A list of column names that are categorical variables.
    date_columns (list): A list of column names that are date variables.
    cont_columns (list): A list of column names that are continuous variables.
    disc_columns (list): A list of column names that are discrete variables.
    label_columns (list, optional): A list of column names that are labels. Defaults to None.

    Attributes:
    _dataset (Dataset): An instance of the Dataset class that represents the loaded dataset.
    """

    def __init__(self, path, cat_columns, date_columns, cont_columns, disc_columns, label_columns=None):
        """
        Initializes a Loader object.

        Loads the dataset from the specified CSV file and creates a Dataset object.

        Parameters:
        path (str): The path to the CSV file containing the dataset.
        cat_columns (list): A list of column names that are categorical variables.
        date_columns (list): A list of column names that are date variables.
        cont_columns (list): A list of column names that are continuous variables.
        disc_columns (list): A list of column names that are discrete variables.
        label_columns (list, optional): A list of column names that are labels. Defaults to None.
        """
        df = pd.read_csv(path, dtype=str)
        self._dataset = Dataset(df, cat_columns, date_columns, cont_columns, disc_columns, label_columns)

    def rename_columns(self, columns_map):
        """
        Renames the columns of the dataset.

        Parameters:
        columns_map (dict): A dictionary mapping old column names to new column names.
        """
        self._dataset.df = self._dataset.df.rename(columns=columns_map)
        
    @property
    def dataset(self):
        """
        Returns the dataset.

        Returns:
        Dataset: An instance of the Dataset class representing the loaded dataset.
        """
        return self._dataset
    
