from typing import List
from sklearn.base import BaseEstimator
from ml_pipeline.model import Model
from ml_pipeline.dataset import Dataset
from ml_pipeline.evaluator import Evaluator
import time


class Trainer:
    """
    The Trainer class is responsible for training machine learning models.

    Args:
        models (List[BaseEstimator]): A list of machine learning models to be trained.
        dataset (Dataset): The dataset used for training and evaluation.

    Attributes:
        models (List[BaseEstimator]): A list of machine learning models.
        dataset (Dataset): The dataset used for training and evaluation.

    Methods:
        _train(model: Model, **kwargs): Trains a single machine learning model.
        train(**kwargs): Trains all models or a specific model.
        run_grid_search(random_state=123, n_iter=10): Runs grid search for models with grid search parameters.
        run_evaluation(): Evaluates all models.
        add(model: Model, train=True): Adds a new model to the list of models.

    """

    def __init__(self, models: List[BaseEstimator], dataset: Dataset) -> None:
        if not isinstance(models, list):
            models = [models]
            
        self.models = models
        self.dataset = dataset

    def _train(self, model: Model, **kwargs):
        """
        Trains a single machine learning model.

        Args:
            model (Model): The machine learning model to be trained.
            **kwargs: Additional keyword arguments to be passed to the model's fit method.

        """
        t1 = time.time()
        if model.supervised:
            model.fit(type="train", dataset=self.dataset, Y=self.dataset.Y_train, **kwargs)
        else:
            model.fit(tpe="train", dataset=self.dataset, **kwargs)
             
        t2 = time.time()
        model.__setattr__("training_time", t2 - t1)

    def train(self, **kwargs):
        """
        Trains all models or a specific model.

        Args:
            **kwargs: Additional keyword arguments to be passed to the _train method.
                If "model" is provided, only that specific model will be trained.

        """
        if "model" not in kwargs:            
            for model in self.models:                    
                self._train(model, **kwargs)
        else:
            model = kwargs.pop("model")
            self._train(model, **kwargs)


    def run_evaluation(self):
        """
        Evaluates all models.

        """
        for model in self.models:
            model.evaluate()

    def add(self, model: Model, train=True):
        """
        Adds a new model to the list of models.

        Args:
            model (Model): The machine learning model to be added.
            train (bool): Whether to train the model after adding it. Default is True.

        """
        self.models.append(model)
        if train:
            self.train(model=model)

            
class Trainer():
    def __init__(self,models:List[BaseEstimator],dataset:Dataset) -> None:
        if not isinstance(models,list):
            models=[models]
            
        self.models=models
        self.dataset=dataset




    def _train(self,model:Model,**kwargs):
            t1=time.time()
            if model.supervised:
                    model.fit(type="train",dataset=self.dataset,Y=self.dataset.Y_train,**kwargs)
            
            else:
                    model.fit(tpe="train",dataset=self.dataset,**kwargs)
             
            t2=time.time()
            model.__setattr__("training_time",t2-t1)
        

    def train(self,**kwargs):
        if "model" not in kwargs:            
            for model in self.models:                    
                self._train(model,**kwargs)
        else:
            model=kwargs.pop("model")
            self._train(model,**kwargs)
        


    def run_evaluation(self):
        eval=Evaluator(models=self.models,dataset=self.dataset)
        eval.run(sample="train",suffix="train_")

    def add(self,model:Model,train=True):
        self.models.append(model)
        if train:
            self.train(model=model)
