import os
import sys
from src.logger import logging
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        if not os.path.exists(file_path):
            raise CustomException(f"Model file not saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            try:
                logging.info(f"Training model: {model_name}")
                if param.get(model_name):
                    gs = GridSearchCV(model, param[model_name], cv=3)
                    gs.fit(X_train, y_train)
                    model.set_params(**gs.best_params_)

                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                logging.info(
                    f"{model_name}: Train Score: {train_score}, Test Score: {test_score}"
                )
                report[model_name] = test_score
            except Exception as e:
                logging.error(f"Error training model {model_name}: {e}")

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
