import pandas as pd

from fdgmip.registry import HARMONIZERS


@HARMONIZERS.register("combat")
class ComBat:
    """TODO: wrap an existing implementation (e.g. neuroCombat / neuroHarmonize); do not re-implement.
    Must support fit on train batches only and transform of unseen samples from seen batches."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, batch: pd.Series):
        raise NotImplementedError("ComBat harmonisation not implemented")

    def transform(self, X: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
        raise NotImplementedError("ComBat harmonisation not implemented")
