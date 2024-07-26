from ml_pipeline.model import Model
from ml_pipeline.dataset import Dataset
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.base import clone
import copy
import numpy as np

class ModelTunning:
    """
    The ModelTunning class is used for hyperparameter tuning of machine learning models.
    It provides methods for performing random search and grid search to find the best set of hyperparameters.

    Attributes:
        model (Model): The machine learning model to be tuned.
        dataset (Dataset): The dataset used for training the model.
        cv (int): The number of cross-validation folds.
        random_state (int): The random seed for reproducibility.
        scoring_fn (str): The scoring function used for evaluating the models.

    Methods:
        ideal_cutoff(size, cutoff, max_iter): Recursive function to calculate the ideal cutoff value.
        RandomSearch(n_iter): Performs random search to find the best set of hyperparameters.
        GridSearch(max_iter, amplitude, cutoff, params): Performs grid search to find the best set of hyperparameters.
        best_model: Returns the best model found during the search.

    Usage:
        # Create an instance of ModelTunning
        tuner = ModelTunning(model, dataset, cv=10, random_state=123, scoring_fn="accuracy")

        # Perform random search
        tuner.RandomSearch(n_iter=10)

        # Perform grid search
        tuner.GridSearch(max_iter=30, amplitude=0.5, cutoff=5, params=None)

        # Get the best model
        best_model = tuner.best_model
    """

    def __init__(self, model: Model, dataset: Dataset, cv=10, random_state=123, scoring_fn="accuracy") -> None:
        """
        Initializes a new instance of the ModelTunning class.

        Args:
            model (Model): The machine learning model to be tuned.
            dataset (Dataset): The dataset used for training the model.
            cv (int, optional): The number of cross-validation folds. Defaults to 10.
            random_state (int, optional): The random seed for reproducibility. Defaults to 123.
            scoring_fn (str, optional): The scoring function used for evaluating the models. Defaults to "accuracy".
        """
        self.model = model
        self.dataset = dataset
        self.random_state = random_state
        self.cv = cv
        self.scoring_fn = scoring_fn

    def ideal_cutoff(size, cutoff, max_iter):
        """
        Recursive function to calculate the ideal cutoff value.

        Args:
            size (int): The number of hyperparameters.
            cutoff (int): The current cutoff value.
            max_iter (int): The maximum number of iterations.

        Returns:
            int: The ideal cutoff value.
        """
        if size * cutoff > max_iter:
            cutoff -= 1
            return ModelTunning.ideal_cutoff(size, cutoff, max_iter)
        else:
            return cutoff

    def RandomSearch(self, n_iter=10):
        """
        Performs random search to find the best set of hyperparameters.

        Args:
            n_iter (int, optional): The number of iterations. Defaults to 10.

        Returns:
            dict: The results of the random search.
        """
        search = RandomizedSearchCV(
            self.model.model,
            self.model.grid_search_params,
            n_iter=n_iter,
            random_state=self.random_state,
            scoring=self.scoring_fn,
            cv=self.cv
        )

        search.fit(self.model.get_X(type="train", dataset=self.dataset), self.dataset.Y_train)
        self.random_search = search
        self.random_search_best_model = copy.deepcopy(self.model)
        self.random_search_best_model._model = clone(self.random_search.best_estimator_)
        return search.cv_results_

    def GridSearch(self, max_iter=30, amplitude=0.5, cutoff=5, params=None):
        """
        Performs grid search to find the best set of hyperparameters.

        Args:
            max_iter (int, optional): The maximum number of iterations. Defaults to 30.
            amplitude (float, optional): The amplitude for generating parameter values. Defaults to 0.5.
            cutoff (int, optional): The cutoff value for generating parameter values. Defaults to 5.
            params (dict, optional): The hyperparameters to be tuned. If None, the best parameters from random search will be used. Defaults to None.

        Returns:
            dict: The results of the grid search.
        """
        if params is None:
            if not hasattr(self, "random_search"):
                raise ValueError("Params must be passed or RandomSearch must be run")

            params = self.random_search.best_params_
            if max_iter < 3 * len(params):
                print("Max_iter is less than the number of parameters. It is being set to 3 times the number of parameters")
                max_iter = 3 * len(params)

            cutoff = self.ideal_cutoff(len(params), cutoff, max_iter)

            for param, value in params.items():
                if isinstance(value, int):
                    params[param] = np.arange(value - min(cutoff, int(value * amplitude)), value + min(cutoff, int(value * amplitude)))
                elif isinstance(value, float):
                    params[param] = np.arange(value - min(cutoff, value * amplitude), value + min(cutoff, value * amplitude))

        fine_search = GridSearchCV(
            self.model.model,
            params,
            scoring=self.scoring_fn,
            cv=self.cv
        )

        fine_search.fit(self.model.get_X(type="train", dataset=self.dataset), self.dataset.Y_train)
        self.fine_search = fine_search
        self.grid_search_best_model = copy.deepcopy(self.model)
        self.grid_search_best_model._model = clone(self.random_search.best_estimator_)
        return fine_search.cv_results_

    @property
    def best_model(self):
        """
        Returns the best model found during the search.

        Raises:
            ValueError: If no search was run.

        Returns:
            Model: The best model.
        """
        if hasattr(self, "grid_search_best_model"):
            return self.grid_search_best_model
        elif hasattr(self, "random_search_best_model"):
            return self.random_search_best_model
        else:
            raise ValueError("No search was run")


class ModelTunning:
    
    @staticmethod
    def ideal_cutoff(size,cutoff,max_iter):
        
        if size*cutoff>max_iter:
            cutoff-=1
            return ModelTunning.ideal_cutoff(size,cutoff,max_iter)
            #return cutoff
        else:
            return cutoff
    def __init__(self,model:Model,dataset:Dataset,cv=10,random_state=123,scoring_fn="accuracy") -> None:
        self.model=model
        self.dataset=dataset
        self.random_state=random_state
        self.cv=cv
        self.scoring_fn=scoring_fn

    def RandomSearch(self,n_iter=10):
        search=RandomizedSearchCV(self.model.model,self.model.grid_search_params,n_iter=n_iter,random_state=self.random_state,
                                  scoring=self.scoring_fn,cv=self.cv)
        
        search.fit(self.model.get_X(type="train",dataset=self.dataset),self.dataset.Y_train)
        self.random_search=search
        self.random_search_best_model=copy.deepcopy(self.model)
        self.random_search_best_model._model=clone(self.random_search.best_estimator_)
        return search.cv_results_
    

    
    def GridSearch(self,max_iter=30,amplitude=0.5,cutoff=5,params=None,verbose=1):
        if params is None:            
            if not hasattr(self,"random_search"):
                raise ValueError("Params must be passed or RandomSearch must be run")
            
            params=self.random_search.best_params_
            if max_iter<3*len(params):
                print("Max_iter is less than the number of parameters. It is being set to 3 times the number of parameters")
                max_iter=3*len(params)

            
            cutoff=self.ideal_cutoff(len(params),cutoff,max_iter)
        
            for param,value in params.items():                
                if isinstance(value,int):
                    params[param]=np.arange(value-min(cutoff,int(value*amplitude)),value+min(cutoff,int(value*amplitude)))
                elif isinstance(value,float):
                    params[param]=np.arange(value-min(cutoff,value*amplitude),value+min(cutoff,value*amplitude))
                elif isinstance(value,str):
                    params[param]=[value]
        if verbose:
            print("-"*20+"Grid Search Parameters"+"-"*20)
            print(params)
            print("-"*50)
                    
        fine_search=GridSearchCV(self.model.model,params,scoring=self.scoring_fn,cv=self.cv)       
        fine_search.fit(self.model.get_X(type="train",dataset=self.dataset),self.dataset.Y_train)
        self.fine_search=fine_search
        self.grid_search_best_model=copy.deepcopy(self.model)
        self.grid_search_best_model._model=clone(self.random_search.best_estimator_)
        return fine_search.cv_results_
    
    @property
    def best_model(self):
        if hasattr(self,"grid_search_best_model"):
            return self.grid_search_best_model
        elif hasattr(self,"random_search_best_model"):
            return self.random_search_best_model
        else:
            raise ValueError("No search was run")

