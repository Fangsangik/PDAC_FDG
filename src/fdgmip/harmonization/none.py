import pandas as pd

from fdgmip.registry import HARMONIZERS


@HARMONIZERS.register("none")
class NoHarmonization:
    def fit(self, X: pd.DataFrame, batch: pd.Series):
        return self

    def transform(self, X: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
        return X
