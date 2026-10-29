import pandas as pd

from fdgmip.registry import HARMONIZERS


@HARMONIZERS.register("scn")
class StyleConditionedNormalization:
    """TODO: style-conditioned normalisation acts inside the network, so stage must be 'model'
    and the backbone must expose SCN layers; stub only."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, batch: pd.Series):
        raise NotImplementedError("SCN harmonisation not implemented")

    def transform(self, X: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
        raise NotImplementedError("SCN harmonisation not implemented")
