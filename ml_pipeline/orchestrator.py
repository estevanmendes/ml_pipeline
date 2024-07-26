import pickle
from ml_pipeline.preprocessor import Preprocessor
from ml_pipeline.ml_preprocessor import MLPreprocessing
from ml_pipeline.trainer import Trainer
from ml_pipeline.evaluator import Evaluator
from ml_pipeline.model_tunning import ModelTunning
from ml_pipeline.selector import Selector



class Orchestrator():
    """
    The orchestrator class is responsible for coordinating the different steps of a machine learning pipeline.
    It takes in various components such as models, data loader, preprocessor, trainer, evaluator, model tunning, and model selector.
    The main purpose of this class is to provide a high-level interface to run the entire pipeline and obtain the best model.

    Args:
        models (list): A list of machine learning models to be trained and evaluated.
        loader (object): An object that loads the dataset.
        preprocessor_class (class): The class for preprocessing the dataset.
        ml_reprocessing_class (class): The class for preprocessing the dataset for machine learning.
        trainer_class (class): The class for training the models.
        evaluation_class (class): The class for evaluating the models.
        model_tunning_class (class): The class for tuning the hyperparameters of the models.
        model_selector_class (class): The class for selecting the best model based on a decision metric.

    Attributes:
        models (list): A list of machine learning models.
        loader (object): An object that loads the dataset.
        preprocessing_class (class): The class for preprocessing the dataset.
        ml_preprocessing_class (class): The class for preprocessing the dataset for machine learning.
        trainer_class (class): The class for training the models.
        evaluation_class (class): The class for evaluating the models.
        tunning_class (class): The class for tuning the hyperparameters of the models.
        model_selector_class (class): The class for selecting the best model based on a decision metric.
        preprocessor (object): An object that performs preprocessing on the dataset.
        MLpreprocessor (object): An object that performs preprocessing for machine learning on the dataset.
        trainer (object): An object that trains the models.
        evaluator (object): An object that evaluates the models.
        best_model (object): The best model selected based on a decision metric.
        tunning (object): An object that tunes the hyperparameters of the best model.
        bestmodel_MLpreprocessor (object): An object that performs preprocessing for machine learning on the dataset for the best model.
        bestmodel_trainer (object): An object that trains the best model.
        bestmodel_evaluator (object): An object that evaluates the best model.

    Methods:
        run_preprocessing: Performs preprocessing on the dataset.
        run_ml_preprocessing: Performs preprocessing for machine learning on the dataset.
        run_training: Trains the models.
        run_model_evaluation: Evaluates the models.
        run_model_selection: Selects the best model based on a decision metric.
        run_tuning: Tunes the hyperparameters of a model.
        run_pipeline: Runs the entire machine learning pipeline.
        run_retrain_best_model: Retrains the best model on the entire dataset.
        pickle_model: Saves the best model to a file.
        get_metrics: Returns the metrics of the best model.
        plot_metrics: Plots the evaluation metrics of the best model.
    """

    def __init__(self, models, loader, preprocessor_class: Preprocessor, ml_reprocessing_class: MLPreprocessing,
                 trainer_class: Trainer, evaluation_class: Evaluator, model_tunning_class: ModelTunning,
                 model_selector_class: Selector) -> None:
        """
        Initializes the orchestrator class with the provided components.

        Args:
            models (list): A list of machine learning models to be trained and evaluated.
            loader (object): An object that loads the dataset.
            preprocessor_class (class): The class for preprocessing the dataset.
            ml_reprocessing_class (class): The class for preprocessing the dataset for machine learning.
            trainer_class (class): The class for training the models.
            evaluation_class (class): The class for evaluating the models.
            model_tunning_class (class): The class for tuning the hyperparameters of the models.
            model_selector_class (class): The class for selecting the best model based on a decision metric.
        """
        self.models = models
        self.loader = loader
        self.preprocessing_class = preprocessor_class
        self.ml_preprocessing_class = ml_reprocessing_class
        self.trainer_class = trainer_class
        self.evaluation_class = evaluation_class
        self.tunning_class = model_tunning_class
        self.model_selector_class = model_selector_class

    def run_preprocessing(self, replacement_columns=["tamanho_motor", "milhas_carro"], replacements=["L", "mile"],
                          adv_day_column="dia_aviso", adv_month_column="mes_aviso", adv_year_column="ano_aviso",
                          dumies_columns=["cor", "tipo_cambio"]):
        """
        Performs preprocessing on the dataset.

        Args:
            replacement_columns (list): A list of column names to be replaced.
            replacements (list): A list of replacement values corresponding to the replacement_columns.
            adv_day_column (str): The column name for the day of the advertisement.
            adv_month_column (str): The column name for the month of the advertisement.
            adv_year_column (str): The column name for the year of the advertisement.
            dumies_columns (list): A list of column names to be converted to dummy variables.

        Returns:
            object: An object that performs preprocessing on the dataset.
        """
        preprocessor = self.preprocessing_class(self.loader.dataset)
        preprocessor.replacements(columns=replacement_columns, replacements=replacements)
        preprocessor.set_types()

        if (adv_day_column in self.loader.dataset.df.columns and "mes_aviso" in self.loader.dataset.df.columns and
                "ano_aviso" in self.loader.dataset.df.columns):
            preprocessor.create_date(day_column=adv_day_column, month_column=adv_month_column,
                                     year_column=adv_year_column)

        if all([column in self.loader.dataset.df.columns for column in dumies_columns]):
            preprocessor.create_dummies(dumies_columns)

        preprocessor.cat_to_codes()
        preprocessor.process_date()
        preprocessor.process_labels()
        preprocessor.fill_na()
        preprocessor.drop_na()
        preprocessor.update_dataset()
        return preprocessor

    def run_ml_preprocessing(self, dataset, validation=True, samples=["train", "validation", "test"]):
        """
        Performs preprocessing for machine learning on the dataset.

        Args:
            dataset (object): The dataset object to be preprocessed.
            validation (bool): Whether to include a validation set in the preprocessing.
            samples (list): A list of sample names to be preprocessed.

        Returns:
            object: An object that performs preprocessing for machine learning on the dataset.
        """
        MLpreprocessor = self.ml_preprocessing_class(dataset)
        MLpreprocessor.undersample()
        MLpreprocessor.split(validation=validation)
        MLpreprocessor.set_scaler()
        MLpreprocessor.scale(samples)
        return MLpreprocessor

    def run_training(self, models, dataset):
        """
        Trains the models.

        Args:
            models (list): A list of machine learning models to be trained.
            dataset (object): The dataset object to be used for training.

        Returns:
            object: An object that trains the models.
        """
        trainer = self.trainer_class(models, dataset)
        trainer.train()
        return trainer

    def run_model_evaluation(self, models, dataset, sample):
        """
        Evaluates the models.

        Args:
            models (list): A list of machine learning models to be evaluated.
            dataset (object): The dataset object to be used for evaluation.
            sample (str): The name of the sample to be evaluated.

        Returns:
            object: An object that evaluates the models.
        """
        evaluator = self.evaluation_class(models, dataset)
        evaluator.run(sample=sample)
        return evaluator

    def run_model_selection(self, models, dataset, decision_metric="f1", sample="validation"):
        """
        Selects the best model based on a decision metric.

        Args:
            models (list): A list of machine learning models to be evaluated and selected.
            dataset (object): The dataset object to be used for evaluation.
            decision_metric (str): The decision metric to be used for model selection.
            sample (str): The name of the sample to be used for model selection.

        Returns:
            tuple: A tuple containing the trainer object, evaluator object, and the best model object.
        """
        trainer = self.run_training(models, dataset)
        evaluator = self.run_model_evaluation(trainer.models, dataset, sample=sample)
        best_model = self.model_selector_class(evaluator.models, decision_metric).select()
        return trainer, evaluator, best_model

    def run_tuning(self, model, dataset, desicion_metric="roc_auc",cv=3, random_n_iter=10, grid_max_iter=10):
        """
        Tunes the hyperparameters of a model.

        Args:
            model (object): The model object to be tuned.
            dataset (object): The dataset object to be used for tuning.
            cv (int): The number of cross-validation folds.
            random_n_iter (int): The number of iterations for random search.
            grid_max_iter (int): The maximum number of iterations for grid search.

        Returns:
            object: An object that tunes the hyperparameters of the model.
        """
        tunning = self.tunning_class(model, dataset, cv=cv, random_state=123, scoring_fn=desicion_metric)
        tunning.RandomSearch(n_iter=random_n_iter)
        tunning.GridSearch(max_iter=grid_max_iter, amplitude=0.5, cutoff=3)
        return tunning

    def run_pipeline(self, cv, random_n_iter, grid_max_iter,decision_metric,verbose=0):
        """
        Runs the entire machine learning pipeline.

        Args:
            cv (int): The number of cross-validation folds.
            random_n_iter (int): The number of iterations for random search.
            grid_max_iter (int): The maximum number of iterations for grid search.

        Returns:
            object: The best model object.
        """
        self.preprocessor = self.run_preprocessing()
        self.MLpreprocessor = self.run_ml_preprocessing(self.preprocessor.dataset, validation=True)
        self.trainer, self.evaluator, best_model = self.run_model_selection(self.models, self.preprocessor.dataset,decision_metric)
        if verbose>0:
            print("Best model selected: ", best_model.model)
            print(self.evaluator.metrics())
        self.tunning = self.run_tuning(best_model, self.MLpreprocessor.dataset, cv=cv, random_n_iter=random_n_iter,
                                       grid_max_iter=grid_max_iter,desicion_metric=decision_metric)
        self.best_model = self.run_retrain_best_model()
        return self.best_model.model

    def run_retrain_best_model(self):
        """
        Retrains the best model on the entire dataset.

        Returns:
            object: The best model object.
        """
        best_model = self.tunning.best_model
        self.bestmodel_MLpreprocessor = self.run_ml_preprocessing(self.preprocessor.dataset, validation=False,
                                                                  samples=["train", "test"])
        self.bestmodel_trainer = self.run_training(best_model, self.bestmodel_MLpreprocessor.dataset)
        self.bestmodel_evaluator = self.run_model_evaluation(best_model,
                                                             self.bestmodel_MLpreprocessor.dataset, sample="test")
        self.pickle_model(best_model.model)
        return best_model

    def pickle_model(self, model, filename="best_model.pkl"):
        """
        Saves the best model to a file.

        Args:
            model (object): The model object to be saved.
            filename (str): The filename to save the model.

        Returns:
            None
        """
        pickle.dump(model, open(filename, "wb"))

    def get_metrics(self):
        """
        Returns the metrics of the best model.

        Returns:
            dict: A dictionary containing the metrics of the best model.
        """
        return self.best_model.show_metrics()

    def plot_metrics(self):
        """
        Plots the evaluation metrics of the best model.

        Returns:
            None
        """
        return self.evaluator.plot_metrics()

