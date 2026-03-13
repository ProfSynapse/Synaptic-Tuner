# Trainers/ml/features/builder.py
# Converts FeaturesConfig into an sklearn ColumnTransformer.
# Phase 1 scope: numeric (imputer + scaler) and categorical (imputer + encoder).
# Used by pipeline_builder.py to assemble the full sklearn Pipeline.

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from Trainers.ml.config import FeaturesConfig


_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def build_preprocessor(config: FeaturesConfig) -> ColumnTransformer:
    """Build a ColumnTransformer from feature configuration.

    Args:
        config: Validated FeaturesConfig from YAML.

    Returns:
        Unfitted ColumnTransformer ready for pipeline.fit().

    Phase 1 scope: numeric + categorical.
    Phase 2 adds: text (TfidfVectorizer).
    """
    transformers: list[tuple[str, Pipeline | str, list[str]]] = []

    # --- Numeric pipeline ---
    if config.numeric is not None:
        steps = []
        nc = config.numeric
        if nc.imputer != "none":
            steps.append(("imputer", SimpleImputer(strategy=nc.imputer)))
        if nc.scaler != "none":
            steps.append(("scaler", _SCALERS[nc.scaler]()))
        if steps:
            transformers.append(("numeric", Pipeline(steps), nc.columns))
        else:
            transformers.append(("numeric", "passthrough", nc.columns))

    # --- Categorical pipeline ---
    if config.categorical is not None:
        cc = config.categorical
        imputer = SimpleImputer(strategy="constant", fill_value="__missing__")
        if cc.encoder == "onehot":
            encoder = OneHotEncoder(
                handle_unknown=cc.handle_unknown,
                sparse_output=False,
            )
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        transformers.append((
            "categorical",
            Pipeline([("imputer", imputer), ("encoder", encoder)]),
            cc.columns,
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop columns not in any transformer
    )
