from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from ml_pipeline.model import Model
from ml_pipeline.dataset import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Evaluator():
    """
    Class to perform evaluation of models on a dataset.

    Args:
        models (List[Model]): A list of models to evaluate.
        dataset (Dataset): The dataset to evaluate the models on.

    Attributes:
        dataset (Dataset): The dataset used for evaluation.
        models (List[Model]): The models to evaluate.

    Methods:
        run(sample="validation"): Runs the evaluation on the specified sample.
        metrics(): Returns the evaluation metrics for each model.
        plot_metrics(orient="model"): Plots the evaluation metrics for each model.

    """

    def __init__(self, models: List[Model], dataset: Dataset) -> None:
        if not isinstance(models, list):
            models = [models]
            
        self.dataset = dataset
        self.models = models




    def run(self, sample=["validation"],prefix=""):
        """
        Runs the evaluation on the specified sample.

        Args:
            sample (str, optional): The sample to evaluate the models on. Defaults to "validation".

        """        
        for model in self.models:
            accuracy = accuracy_score(self.dataset.get_sample(sample, "Y"), model(type=sample, dataset=self.dataset))  
            precision = precision_score(self.dataset.get_sample(sample, "Y"), model(type=sample, dataset=self.dataset))
            recall = recall_score(self.dataset.get_sample(sample, "Y"), model(type=sample, dataset=self.dataset))
            f1 = f1_score(self.dataset.get_sample(sample, "Y"), model(type=sample, dataset=self.dataset))
            roc_auc = roc_auc_score(self.dataset.get_sample(sample, "Y"), model(type=sample, dataset=self.dataset))
            cm = confusion_matrix(self.dataset.get_sample(sample, "Y"), model(type=sample, dataset=self.dataset), labels=model.model.classes_)
            model.set_metrics(f"{prefix}accuracy", accuracy)
            model.set_metrics(f"{prefix}precision", precision)
            model.set_metrics(f"{prefix}recall", recall)
            model.set_metrics(f"{prefix}f1", f1)
            model.set_metrics(f"{prefix}roc_auc", roc_auc)
            model.__setattr__(f"{prefix}confusion_matrix", cm)
           


    def metrics(self):
        """
        Returns the evaluation metrics for each model.

        Returns:
            dict: A dictionary containing the evaluation metrics for each model.

        """
        metrics = {}
        for model in self.models:
            metrics[str(model)] = model.show_metrics()
        return metrics
    
    def plot_metrics(self, orient="model"):
        """
        Plots the evaluation metrics for each model.

        Args:
            orient (str, optional): The orientation of the plot. Defaults to "model".

        Returns:
            matplotlib.figure.Figure: The plotted figure.

        """
        if orient == "model":
            fig, axs = plt.subplot_mosaic([["accuracy", "precision", "recall"], ["f1", "roc_auc", "vazio"]], sharey=True, figsize=(10, 4))
            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                values = []
                for model in self.models:
                    values.append(model.__getattribute__("metrics")[metric])
                axs[metric].set_title(metric)
                sns.barplot(x=values, y=[str(model) for model in self.models], ax=axs[metric])
                axs[metric].set_xlim(0.5, 1)
            plt.tight_layout()
            return fig
        else:
            fig, axs = plt.subplots(len(self.models)//5+1, 5, sharey=True)
            if len(self.models)//5+1 == 1:
                axs = axs.reshape(1, -1)

            for index, model in enumerate(self.models):
                values = []
                for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                    values.append(model.__getattribute__("metrics")[metric])
                axs[index//5, index%5].set_title(model.model, fontsize=8)
                sns.barplot(x=values, y=["accuracy", "precision", "recall", "f1", "roc_auc"], ax=axs[index//5, index%5])
            plt.tight_layout()
            return fig
        

    def ṕlot_confusion_matrix(self):
        fig, axs = plt.subplots(len(self.models)//5+1, 5, sharey=True,figsize=(10, 3*(len(self.models)//5+1)))
        if len(self.models)//5+1 == 1:
            axs = axs.reshape(1, -1)
        for index,model in enumerate(self.models):
            disp = ConfusionMatrixDisplay(confusion_matrix=model.__getattribute__("confusion_matrix"),
                                        display_labels=model.model.classes_)
            disp.plot(ax=axs[index//5, index%5],colorbar=False)
            axs[index//5, index%5].set_title(model.model, fontsize=8)
            
        plt.tight_layout()
        


