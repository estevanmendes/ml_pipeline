from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.model_selection import train_test_split
from ml_pipeline.dataset import Dataset
import pandas as pd


class MLPreprocessing():
    """
    Class for performing preprocessing tasks on machine learning datasets.

    Attributes:
        dataset (Dataset): The dataset object containing the data.
        scaler (object): The scaler object used for feature scaling.

    Methods:
        undersample: Undersamples the dataset to balance the classes.
        set_scaler: Sets the scaler object for feature scaling.
        scale: Scales the features in the dataset.
        split: Splits the dataset into train, validation, and test sets.
        cross_validation: Performs cross-validation on the dataset.
    """

    def _split(X, Y, **kwargs):
        """
        Helper function to split the dataset into train and test sets.

        Args:
            X (DataFrame): The input features.
            Y (Series): The target variable.
            **kwargs: Additional arguments to pass to the train_test_split function.

        Returns:
            X_train (DataFrame): The training set input features.
            X_test (DataFrame): The test set input features.
            Y_train (Series): The training set target variable.
            Y_test (Series): The test set target variable.
        """
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, **kwargs)
        return X_train, X_test, Y_train, Y_test

    def __init__(self, dataset):
        """
        Initializes the MLPreprocessing object.

        Args:
            dataset (Dataset): The dataset object containing the data.
        """
        self.dataset = dataset
        self.dataset.__setattr__("scalable_columns", self.dataset.cont_columns + self.dataset.disc_columns)
        self.dataset.__setattr__("not_scalable_columns", self.dataset.cat_columns)

    def undersample(self):
        """
        Undersamples the dataset to balance the classes.
        """
        positive_sample_size = (self.dataset.df[self.dataset.label_column[0]] == 1).sum()
        self.dataset.__setattr__("full_df", self.dataset.df)
        self.dataset.df = pd.concat([
            self.dataset.df[self.dataset.df[self.dataset.label_column[0]] == 0].sample(positive_sample_size),
            self.dataset.df[self.dataset.df[self.dataset.label_column[0]] == 1]
        ], axis=0)
        print(positive_sample_size)

    def set_scaler(self, type="minmax"):
        """
        Sets the scaler object for feature scaling.

        Args:
            type (str): The type of scaler to use. Default is "minmax".
        """
        scalers = {"minmax": MinMaxScaler, "standard": StandardScaler, "robust": RobustScaler,
                   "normalizer": RobustScaler}
        self.scaler = scalers[type]()

    def scale(self, sample=["train", "validation", "test"]):
        """
        Scales the features in the dataset.

        Args:
            sample (list): The samples to scale. Default is ["train", "validation", "test"].

        Raises:
            ValueError: If the scaler was not set or the train data was not set.
        """
        if not hasattr(self, "scaler"):
            raise ValueError("Scaler was not set")
        if not hasattr(self.dataset, "X_train"):
            raise ValueError("Train data was not set")

        X_train_scaled = self.scaler.fit_transform(self.dataset.X_train[self.dataset.scalable_columns].astype(float))
        X_train_scaled = pd.concat([
            self.dataset.X_train[self.dataset.not_scalable_columns].reset_index(drop=True),
            pd.DataFrame(X_train_scaled, columns=self.dataset.scalable_columns)
        ], axis=1)
        self.dataset.set_sample(X_train_scaled, "train", "X", subtype="scaled")

        if "test" in sample and hasattr(self.dataset, "X_test"):
            X_test_scaled = self.scaler.transform(self.dataset.X_test[self.dataset.scalable_columns])
            X_test_scaled = pd.concat([
                self.dataset.X_test[self.dataset.not_scalable_columns].reset_index(drop=True),
                pd.DataFrame(X_test_scaled, columns=self.dataset.scalable_columns)
            ], axis=1)
            self.dataset.set_sample(X_test_scaled, "test", "X", subtype="scaled")

        if "validation" in sample and hasattr(self.dataset, "X_validation"):
            X_val_scaled = self.scaler.transform(self.dataset.X_validation[self.dataset.scalable_columns])
            X_val_scaled = pd.concat([
                self.dataset.X_validation[self.dataset.not_scalable_columns].reset_index(drop=True),
                pd.DataFrame(X_val_scaled, columns=self.dataset.scalable_columns)
            ], axis=1)
            self.dataset.set_sample(X_val_scaled, "validation", "X", subtype="scaled")

    def split(self, validation=True, stratified=True, columns_stratify=None):
        """
        Splits the dataset into train, validation, and test sets.

        Args:
            validation (bool): Whether to include a validation set. Default is True.
            stratified (bool): Whether to perform stratified sampling. Default is True.
            columns_stratify (list): The columns to use for stratified sampling. Default is None.

        Raises:
            ValueError: If the dataset is not labeled.
            NotImplemented: If stratified sampling is requested but columns_stratify is not provided.
        """
        train_size = 0.7 if validation else 0.80
        validation_size = 0.15
        test_size = 0.15 if validation else 0.20
        if not self.dataset.labeled:
            raise ValueError("Dataset is not labeled")

        if stratified and columns_stratify is not None:
            columns_stratify += self.dataset.label_column

        X_train, X_test, Y_train, Y_test = MLPreprocessing._split(
            self.dataset.df[self.dataset.not_label_columns],
            self.dataset.df[self.dataset.label_column],
            random_state=42,
            test_size=test_size + validation_size,
            stratify=columns_stratify
        )
        if validation:
            X_test, X_val, Y_test, Y_val = MLPreprocessing._split(
                X_test, Y_test, random_state=42, test_size=validation_size / (test_size + validation_size),
                stratify=columns_stratify
            )
        else:
            X_val, Y_val = None, None

        self.dataset.set_sample(X_train, "train", "X")
        self.dataset.set_sample(Y_train, "train", "Y")
        self.dataset.set_sample(X_val, "validation", "X")
        self.dataset.set_sample(Y_val, "validation", "Y")
        self.dataset.set_sample(X_test, "test", "X")
        self.dataset.set_sample(Y_test, "test", "Y")

    def cross_validation(self):
        """
        Placeholder method for performing cross-validation on the dataset.
        """
        pass

class MLPreprocessing():


    def _split(X,Y,**kwargs):

        X_train, X_test, Y_train, Y_test=train_test_split(X,Y,**kwargs)

        return X_train, X_test, Y_train, Y_test
   

    def __init__(self,dataset:Dataset) -> None:
        self.dataset=dataset
        self.dataset.__setattr__("scalable_columns",self.dataset.cont_columns+self.dataset.disc_columns)
        self.dataset.__setattr__("not_scalable_columns",self.dataset.cat_columns)

    def undersample(self):
        positive_sample_size=(self.dataset.df[self.dataset.label_column[0]]==1).sum()
        self.dataset.__setattr__("full_df",self.dataset.df)
        self.dataset.df=pd.concat([self.dataset.df[self.dataset.df[self.dataset.label_column[0]]==0].sample(positive_sample_size),
                                   self.dataset.df[self.dataset.df[self.dataset.label_column[0]]==1]],axis=0)
        print(positive_sample_size)


    def set_scaler(self,type="minmax"):
        scalers={"minmax":MinMaxScaler,"standard":StandardScaler,"robust":RobustScaler,"normalizer":RobustScaler}
        self.scaler=scalers[type]()

    def scale(self,sample=["train","validation","test"]):

        if not hasattr(self,"scaler"):
            raise ValueError("Scaler was not set")
        if not hasattr(self.dataset,"X_train"):
            raise ValueError("Train data was not set")        

        X_train_scaled=self.scaler.fit_transform(self.dataset.X_train[self.dataset.scalable_columns].astype(float))
        X_train_scaled=pd.concat([self.dataset.X_train[self.dataset.not_scalable_columns].reset_index(drop=True),
                                  pd.DataFrame(X_train_scaled,columns=self.dataset.scalable_columns)],axis=1)
        self.dataset.set_sample(X_train_scaled,"train","X",subtype="scaled")

        if "test" in sample and hasattr(self.dataset,"X_test"):
            X_test_scaled=self.scaler.transform(self.dataset.X_test[self.dataset.scalable_columns])
            X_test_scaled=pd.concat([self.dataset.X_test[self.dataset.not_scalable_columns].reset_index(drop=True),
                                     pd.DataFrame(X_test_scaled,columns=self.dataset.scalable_columns)],axis=1)
            self.dataset.set_sample(X_test_scaled,"test","X",subtype="scaled")

        if "validation" in sample and hasattr(self.dataset,"X_validation"):
            X_val_scaled=self.scaler.transform(self.dataset.X_validation[self.dataset.scalable_columns])
            X_val_scaled=pd.concat([self.dataset.X_validation[self.dataset.not_scalable_columns].reset_index(drop=True),
                                    pd.DataFrame(X_val_scaled,columns=self.dataset.scalable_columns)],axis=1)
            self.dataset.set_sample(X_val_scaled,"validation","X",subtype="scaled")
    

    def split(self,validation=True,stratified=True,columns_stratify=None):

         
        train_size=0.7 if validation else 0.80
        validation_size=0.15
        test_size=0.15 if validation else 0.20
        if not self.dataset.labeled:
            raise NotImplemented()
        
        if stratified and columns_stratify is not None:            
            columns_stratify+=self.dataset.label_column

              
        X_train, X_test, Y_train, Y_test=MLPreprocessing._split(self.dataset.df[self.dataset.not_label_columns],self.dataset.df[self.dataset.label_column],random_state=42,test_size=test_size+validation_size,stratify=columns_stratify)        
        if validation:
                X_test, X_val, Y_test, Y_val=MLPreprocessing._split(X_test,Y_test,random_state=42,test_size=validation_size/(test_size+validation_size),stratify=columns_stratify)
        else:
            X_val,Y_val=None,None
        
        self.dataset.set_sample(X_train,"train","X")
        self.dataset.set_sample(Y_train,"train","Y")
        self.dataset.set_sample(X_val,"validation","X")
        self.dataset.set_sample(Y_val,"validation","Y")
        self.dataset.set_sample(X_test,"test","X")
        self.dataset.set_sample(Y_test,"test","Y")



    def cross_validation():
        raise NotImplemented()
