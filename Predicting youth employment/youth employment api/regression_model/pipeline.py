from feature_engine.encoding import OneHotEncoder, OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler

from regression_model.config.core import config
from regression_model.processing import features as pp

YOUTH_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_variables),
        ),
        # Impute categorical variables
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars,
            ),
        ),
        # Encoding categorical Variables
        (
            "categorical_encoder",
            OneHotEncoder(
                top_categories=2,
                variables=config.model_config.categorical_vars,
                drop_last=True,
            ),
        ),
        (
            "mean_imputation",
            MeanMedianImputer(
                imputation_method="mean",
                variables=config.model_config.numerical_variables,
            ),
        ),
        # Outlier Handling
        (
            "capper",
            Winsorizer(
                capping_method="gaussian",
                tail="both",
                fold=0.1,
                variables=config.model_config.numerical_variables,
            ),
        ),
        ("scaler", StandardScaler()),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=config.model_config.n_estimators,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
