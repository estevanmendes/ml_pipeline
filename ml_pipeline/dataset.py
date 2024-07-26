class Dataset:
    """
    A class representing a dataset.

    Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - cat_columns (list): A list of categorical column names.
    - date_columns (list): A list of date column names.
    - cont_columns (list): A list of continuous column names.
    - disc_column (list): A list of discrete column names.
    - label_column (str or list, optional): The label column name(s). Defaults to None.

    Attributes:
    - _df (pandas.DataFrame): The input dataframe.
    - cat_columns (list): A list of categorical column names.
    - date_columns (list): A list of date column names.
    - cont_columns (list): A list of continuous column names.
    - disc_columns (list): A list of discrete column names.
    - labeled (bool): Indicates if the dataset is labeled.
    - label_column (str or list): The label column name(s).
    
    Methods:
    - check_columns(all_columns, *columns_groups): Checks if the given columns match the dataset columns.
    - not_label_columns(date=False): Returns a list of column names excluding the label column(s).
    - set_sample(sample, type, name, subtype=None): Sets a sample for a specific type and name.
    - get_sample(type, name, subtype=None): Retrieves a sample for a specific type and name.

    """

    def __init__(self, df, cat_columns, date_columns, cont_columns, disc_column, label_column=None):
        self._df = df
        self.cat_columns = cat_columns
        self.date_columns = date_columns
        self.cont_columns = cont_columns
        self.disc_columns = disc_column
        
        if isinstance(label_column, str):
            label_column = [label_column]
            
        if label_column is not None:
            self.labeled = True
            self.label_column = label_column
        else:
            self.labeled = False
            self.label_column = []
        
    def check_columns(self, all_columns, *columns_groups):
        """
        Checks if the given columns match the dataset columns.

        Parameters:
        - all_columns (list): A list of all column names.
        - *columns_groups (list): Variable number of lists containing column names.

        Raises:
        - ValueError: If the columns do not match.

        """
        columns = []
        for group in columns_groups:
            if group:
                columns += group
        if len(all_columns) == len(columns):
            raise ValueError(f"Columns do not match: {set(all_columns).difference(columns)}")        
        
    @property
    def not_label_columns(self, date=False):
        """
        Returns a list of column names excluding the label column(s).

        Parameters:
        - date (bool, optional): Indicates if date columns should be included. Defaults to False.

        Returns:
        - list: A list of column names.

        """
        if not date:
            return self.cat_columns + self.cont_columns + self.disc_columns
        else:
            return self.cat_columns + self.date_columns + self.cont_columns + self.disc_columns

    @property
    def df(self):
        """
        Returns the input dataframe.

        Returns:
        - pandas.DataFrame: The input dataframe.

        """
        return self._df
    
    @df.setter
    def df(self, df):
        """
        Sets the input dataframe.

        Parameters:
        - df (pandas.DataFrame): The input dataframe.

        """
        self._df = df

    def set_sample(self, sample, type, name, subtype=None):
        """
        Sets a sample for a specific type and name.

        Parameters:
        - sample: The sample to be set.
        - type (str): The type of the sample (train, validation, or test).
        - name (str): The name of the sample (X or Y).
        - subtype (str, optional): The subtype of the sample (scaled or None). Defaults to None.

        Raises:
        - ValueError: If the type, name, or subtype is invalid.

        """
        if type not in ["train", "validation", "test"]:
            raise ValueError("Type must be train, validation, or test")
        if name not in ["X", "Y"]:
            raise ValueError("Name must be X or Y")
        if subtype not in ["scaled", None]:
            raise ValueError("Subtype must be scaled or none")
        
        att_name = name + "_" + type if subtype is None else name + "_" + type + "_" + subtype

        setattr(self, att_name, sample)
    
    def get_sample(self, type, name, subtype=None):
        """
        Retrieves a sample for a specific type and name.

        Parameters:
        - type (str): The type of the sample (train, validation, or test).
        - name (str): The name of the sample (X or Y).
        - subtype (str, optional): The subtype of the sample (scaled or None). Defaults to None.

        Returns:
        - The requested sample.

        Raises:
        - ValueError: If the type, name, or subtype is invalid.

        """
        if type not in ["train", "validation", "test"]:
            raise ValueError("Type must be train, validation, or test")
        if name not in ["X", "Y"]:
            raise ValueError("Name must be X or Y")
        if subtype not in ["scaled", None]:
            raise ValueError("Subtype must be scaled or none")

        att_name = name + "_" + type if subtype is None else name + "_" + type + "_" + subtype

        return getattr(self, att_name)
    
