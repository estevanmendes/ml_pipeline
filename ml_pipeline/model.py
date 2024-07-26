from sklearn.base import BaseEstimator
from typing import Union
from ml_pipeline.dataset import Dataset



class Model(BaseEstimator):
    """
    A class representing a machine learning model.

    Parameters:
    - model: The machine learning model to be used.
    - params: The parameters for the model.
    - supervised: A boolean indicating whether the model is supervised or not.
    - run_scaled: A boolean indicating whether to run the model on scaled data.
    - run_on_categorical: A boolean indicating whether to run the model on categorical columns.
    - run_on_continues: A boolean indicating whether to run the model on continuous columns.
    - created: A boolean indicating whether the model has been created or not.

    Methods:
    - camel_case_split: Splits a camel case string into separate words.
    - __init__: Initializes the Model object.
    - get_X: Retrieves the X data for a given type and dataset.
    - set_params: Sets the parameters for the model.
    - grid_search_params: Sets the parameters for grid search.
    - random_grid_search_params: Sets the parameters for random grid search.
    - model: Returns the model object.
    - set_metrics: Sets the metrics for the model.
    - show_metrics: Returns the metrics for the model.
    - fit: Fits the model to the data.
    - __str__: Returns a string representation of the model.
    - __call__: Makes predictions using the model.

    Attributes:
    - _model: The machine learning model.
    - params: The parameters for the model.
    - supervised: A boolean indicating whether the model is supervised or not.
    - created: A boolean indicating whether the model has been created or not.
    - run_scaled: A boolean indicating whether to run the model on scaled data.
    - run_on_categorical: A boolean indicating whether to run the model on categorical columns.
    - run_on_continues: A boolean indicating whether to run the model on continuous columns.
    - grid_search_params: The parameters for grid search.
    - metrics: The metrics for the model.
    """

    @staticmethod
    def camel_case_split(str):     
        """
        Splits a camel case string into separate words.

        Parameters:
        - str: The camel case string to be split.

        Returns:
        - A list of words.
        """
        start_idx = [i for i, e in enumerate(str)
                    if e.isupper()] + [len(str)]
    
        start_idx = [0] + start_idx
        return [str[x: y] for x, y in zip(start_idx, start_idx[1:])]         
    
    
    def __init__(self,model,params,supervised:bool,run_scaled:bool=False,run_on_categorical=True,run_on_continues=True,created=False) -> None:
        """
        Initializes the Model object.

        Parameters:
        - model: The machine learning model to be used.
        - params: The parameters for the model.
        - supervised: A boolean indicating whether the model is supervised or not.
        - run_scaled: A boolean indicating whether to run the model on scaled data.
        - run_on_categorical: A boolean indicating whether to run the model on categorical columns.
        - run_on_continues: A boolean indicating whether to run the model on continuous columns.
        - created: A boolean indicating whether the model has been created or not.
        """
        self._model=model
        self.params=params
        self.supervised=supervised
        if not created:
            self.set_params()
        else:
            self.created=created
        self.run_scaled=run_scaled
        self.run_on_categorical=run_on_categorical
        self.run_on_continues=run_on_continues
        if not (run_on_categorical or run_on_continues):
            raise ValueError("Model must run on categorical and/or continuous columns")
        
    def get_X(self,type:Union["train","test","validation"],dataset:Dataset):       
        """
        Retrieves the X data for a given type and dataset.

        Parameters:
        - type: The type of data to retrieve (train, test, or validation).
        - dataset: The dataset object.

        Returns:
        - The X data.
        """
        subtype="scaled" if self.run_scaled else None
        columns=dataset.not_label_columns
        if not self.run_on_categorical:
            columns=list(set(columns).difference(dataset.cat_columns))
        if not self.run_on_continues:
            columns=list(set(columns).difference(dataset.cont_columns))
            
        return dataset.get_sample(type,"X",subtype=subtype)[columns]
    
    def set_params(self):
        """
        Sets the parameters for the model.
        """
        self.created=True
        self._model=self._model(**self.params)
    
    def grid_search_params(self,**params):
        """
        Sets the parameters for grid search.

        Parameters:
        - params: The parameters for grid search.
        """
        self.grid_search_params=params

    def random_grid_search_params(self,**params):
        """
        Sets the parameters for random grid search.

        Parameters:
        - params: The parameters for random grid search.
        """
        self.grid_search_params=params

    @property
    def model(self):
        """
        Returns the model object.

        Raises:
        - ValueError: If the model has not been set.
        """
        if not hasattr(self,"created"):
            raise ValueError("Model was not set")
        
        return self._model        

    def set_metrics(self,metric,value):
        """
        Sets the metrics for the model.

        Parameters:
        - metric: The metric name.
        - value: The metric value.
        """
        if not hasattr(self,"metrics"):
            self.metrics={}
        self.metrics[metric]=value

    def show_metrics(self):
        """
        Returns the metrics for the model.
        """
        return self.metrics
    
    def fit(self,Y,X=None,type=None,dataset:Dataset=None,**kwargs):
        """
        Fits the model to the data.

        Parameters:
        - Y: The target variable.
        - X: The input data.
        - type: The type of data (train, test, or validation).
        - dataset: The dataset object.
        - kwargs: Additional keyword arguments for the fit method.

        Returns:
        - The fitted model.
        """
        if X is None and type is not None and dataset is not None:
            X=self.get_X(type,dataset)
            
        return self._model.fit(X,Y,**kwargs)

    def __str__(self) -> str:
        """
        Returns a string representation of the model.
        """
        name=(self.model.__str__()).split("(")[0]
        if self.run_scaled:
            name+="_scaled"
        if self.run_on_categorical:
            name+="_cat"
        if self.run_on_continues:
            name+="_cont"
        return name
            
    def __call__(self,X=None,type=None,dataset:Dataset=None):
        """
        Makes predictions using the model.

        Parameters:
        - X: The input data.
        - type: The type of data (train, test, or validation).
        - dataset: The dataset object.

        Returns:
        - The predictions.
        """
        if type is not None and Dataset is not None:
            X=self.get_X(type,dataset)
        if X is None:
            raise ValueError("X or type and Dataset must be passed")
        
        return self._model.predict(X)

