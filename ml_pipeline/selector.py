from ml_pipeline.model import Model
from typing import List


class Selector():
    """
    A class that selects the best model based on a specified metric.
    
    Args:
        models (List[Model]): A list of models to select from.
        aimed_metric (str): The metric to optimize for. Defaults to "accuracy".
    
    Attributes:
        models (List[Model]): A list of models to select from.
        aimed_metric (str): The metric to optimize for.
    
    Methods:
        check_metric(): Checks if each model has the specified metric.
        select(): Selects the model with the highest value for the specified metric.
    """
    
    def __init__(self, models: List[Model], aimed_metric: str = "accuracy") -> None:
        """
        Initializes a Selector object.
        
        Args:
            models (List[Model]): A list of models to select from.
            aimed_metric (str): The metric to optimize for. Defaults to "accuracy".
        """
        if not isinstance(models, list):
            models = [models]
        
        self.models = models
        self.aimed_metric = aimed_metric
    
    def check_metric(self):
        """
        Checks if each model has the specified metric.
        
        Raises:
            ValueError: If a model does not have the specified metric.
        """
        for model in self.models:
            if not hasattr(model, "metrics"):
                raise ValueError(f"{str(model)} has no metrics")
            elif model.metrics.get(self.aimed_metric) is None:
                raise ValueError(f"{str(model)} has no metrics")
            
    def select(self):
        """
        Selects the model with the highest value for the specified metric.
        
        Returns:
            Model: The selected model.
        """
        scores = []
        for model in self.models:
            scores.append(model.metrics.get(self.aimed_metric))
        
        argmax = scores.index(max(scores))

        return self.models[argmax]


