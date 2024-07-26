import pandas as pd
from ml_pipeline.dataset import Dataset

class Preprocessor:
    """
    A class that provides preprocessing methods for a given dataset.

    Args:
        dataset (Dataset): The dataset object containing the data to be preprocessed.

    Attributes:
        dataset (Dataset): The dataset object containing the data to be preprocessed.
        df (pandas.DataFrame): The DataFrame representation of the dataset.
        cat_columns (list): The list of categorical column names in the dataset.
        date_columns (list): The list of date column names in the dataset.
        cont_columns (list): The list of continuous column names in the dataset.
        disc_columns (list): The list of discrete column names in the dataset.

    Methods:
        _convert_cat(): Converts categorical columns to the 'category' data type.
        _convert_date(): Converts date columns to the 'datetime' data type.
        _convert_cont(): Converts continuous columns to the 'float' data type.
        _convert_discrete(): Converts discrete columns to the 'float' and then 'int' data type.
        fill_na(fill_value=""): Fills missing values in the dataset with the specified fill value.
        na_analysis(): Performs missing value analysis and returns the count of missing values for each column.
        drop_na(): Drops rows with missing values from the dataset.
        replacements(columns, replacements): Replaces substrings in the specified columns with the specified replacements.
        create_dummies(columns): Creates dummy variables for the specified categorical columns.
        create_date(day_column, month_column, year_column, date_column=None): Creates a new date column from the specified day, month, and year columns.
        process_date(): Processes date columns and extracts additional features such as weekday, day, month, and year.
        process_labels(): Converts the label column to categorical codes.
        cat_to_codes(columns=None): Converts categorical columns to categorical codes.
        set_types(): Sets the data types of the columns in the dataset.
        update_dataset(): Updates the dataset object with the preprocessed data.

    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.df = dataset.df
        self.cat_columns = dataset.cat_columns
        self.date_columns = dataset.date_columns
        self.cont_columns = dataset.cont_columns
        self.disc_columns = dataset.disc_columns
        self.dataset.df

    def _convert_cat(self):
        self.df.loc[:, self.cat_columns] = self.df[self.cat_columns].astype('category')

    def _convert_date(self):
        for col in self.date_columns:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

    def _convert_cont(self):
        self.df[self.cont_columns] = self.df[self.cont_columns].astype(float)

    def _convert_discrete(self):
        self.df[self.disc_columns] = self.df[self.disc_columns].astype(float).astype(int)

    def fill_na(self, fill_value=""):
        self.df = self.df.fillna(fill_value)

    def na_analysis(self):
        return self.df.isna().sum()

    def drop_na(self):
        self.df = self.df.dropna()

    def replacements(self, columns, replacements):
        if isinstance(replacements, str):
            replacements = [replacements] * len(columns)
        if isinstance(columns, str):
            columns = [columns]
        if len(columns) != len(replacements):
            raise ValueError("Columns and replacements must have the same length")

        for column, replacement in zip(columns, replacements):
            self.df[column] = self.df[column].str.replace(replacement, "")

    def create_dummies(self, columns):
        new_columns = []
        for column in columns:
            new_columns += list(map(lambda x: column + "_" + x, self.df[column].unique().tolist()))
        self.df = pd.get_dummies(self.df, columns=columns)
        for column in columns:
            if column in self.cat_columns:
                self.cat_columns.remove(column)
        self.cat_columns += new_columns

    def create_date(self, day_column, month_column, year_column, date_column=None):
        if date_column is None:
            date_column = "data_" + day_column.split("_")[-1]
        self.df[date_column] = pd.to_datetime(self.df[day_column].astype(float).astype(int).astype(str) \
                                              + "-" + self.df[month_column].astype(float).astype(int).astype(str) \
                                              + "-" + self.df[year_column].astype(float).astype(int).astype(str) \
                                              , errors='coerce')
        self.date_columns.append(date_column)

    def process_date(self):
        for name, func in zip(["weekday", "day", "month", "year"],
                              [lambda x: x.dt.weekday, lambda x: x.dt.day, lambda x: x.dt.month, lambda x: x.dt.year]):
            columns = list(map(lambda x: x + "_" + name, self.date_columns))
            self.df[columns] = self.df[self.date_columns].apply(func)
            self.disc_columns += columns

    def process_labels(self):
        self.df[self.dataset.label_column] = self.df[self.dataset.label_column].apply(
            lambda x: x.astype('category').cat.codes)

    def cat_to_codes(self, columns=None):
        if columns is None:
            columns = self.cat_columns
        print(columns)
        self.df[columns] = self.df[columns].apply(lambda x: x.astype('category').cat.codes)

    def set_types(self):
        # self._convert_cat()
        self._convert_date()
        self._convert_cont()
        self._convert_discrete()

    def update_dataset(self):
        self._convert_discrete()
        self.dataset.df = self.df
        self.dataset.cat_columns = self.cat_columns
        self.dataset.date_columns = self.date_columns
        self.dataset.cont_columns = self.cont_columns
        self.dataset.disc_columns = self.disc_columns

