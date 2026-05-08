"""Compatibility wrapper for legacy model pickle imports.

Older serialized artifacts reference ``decision_making.models``.  The current
implementation lives under ``decision_making.ml_model.models``, so we keep this
module as a thin re-export layer to preserve unpickling compatibility without
retraining artifacts.
"""

from decision_making.ml_model.models import BaseModel, RandomForestReturnModel, SklearnModel

__all__ = ["BaseModel", "SklearnModel", "RandomForestReturnModel"]
