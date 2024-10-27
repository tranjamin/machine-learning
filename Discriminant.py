from typing import Any
import sklearn
import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.linear_model
from GenericModel import GenericModel

class LDA(GenericModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(**kwargs)
        self.err_function = None

    def get_model(self) -> Any:
        return self.model
    
    def set_err_function(self, err_function):
        self.err_function = err_function
    
    def copy(self) -> GenericModel:
        # create a new version of the same class
        new = type(self)(**self.kwargs)

        # clone underlying model and copy across fields
        new.model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        new.err_function = self.err_function
        new.x_train = self.x_train
        new.x_test = self.x_test
        new.y_train = self.y_train
        new.y_test = self.y_test

        return new
    
    def fit(self) -> None:
        self.model.fit(self.x_train, self.y_train)

    def predict(self, x) -> list[float]:
        return self.model.predict(x)
    
    def augment_hyperparam(self, param: str, value: Any):
        return super().augment_hyperparam(param, value)

    def evaluate_training_performance(self, target: str = None, probabilistic=False) -> float | dict[str, float]:
        preds = self.predict(self.x_train)
        return self.err_function(self.y_train, preds)
    
    def evaluate_testing_performance(self, target: str = None, probabilistic=False) -> float | dict[str, float]:
        preds = self.predict(self.x_test)
        return self.err_function(self.y_test, preds)