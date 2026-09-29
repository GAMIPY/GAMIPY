from typing import Any

import pandas as pd

from GAMIPY.data.categorical import CategoricalColumn
from GAMIPY.data.categorical_codec import CategoricalCodec
from GAMIPY.data.data import Data


class CategoricalEncoder:

    """
    Encodes/decodes every categorical column of a Data object to/from
    integer codes, so downstream numeric operations (covariance, distance
    metrics, ...) can treat every column uniformly. Split out of Fitness,
    which used to own this as one of three unrelated responsibilities
    (encoding, imputed-data construction, and ~11 distance metrics).

    Methods
    -------
    - **encode(data)**: Returns a Data whose categorical columns are replaced by their integer codes.
    - **decode(data, df_encoded)**: Returns a Data whose categorical columns in `df_encoded` are
      mapped back to `data`'s original category labels.

    Codes are only comparable between two encoded datasets if both use the
    same category order. Pass ``classes`` (column -> category list, e.g. from
    the full dataset) to encode every dataset with the same codes; without
    it, each dataset's own order of appearance is used.
    """

    def __init__(self, classes: dict[Any, list[Any]] | None = None) -> None:
        self.classes = classes if classes is not None else {}
        # label -> code per column with a fixed category list (same codes
        # CategoricalCodec(classes).encode produces), built once.
        self._codes = {col: {label: code for code, label in enumerate(labels)} for col, labels in self.classes.items()}

    @classmethod
    def for_data(cls, data: Data) -> "CategoricalEncoder":
        """An encoder with a fixed category list per categorical column of `data`."""
        return cls({col: CategoricalColumn(data, col).classes for col in data.get_categorical_columns})

    def encode(self, data: Data) -> Data:

        """
        Encodes all categorical columns in the data as integers.

        Args
        ----
        - **data (Data)**: The data containing categorical columns to be encoded.

        Returns
        -------
        - **Data**: A Data object with all categorical columns encoded as numerical values.
        """

        df = data.get_data.copy()
        for col in data.get_categorical_columns:
            if col in self._codes:
                df[col] = df[col].map(self._codes[col])
            else:
                cat_col = CategoricalColumn(data, col)
                df[col] = CategoricalCodec(cat_col.classes).encode(cat_col.data)
        return Data(df)

    def decode(self, data: Data, df_encoded: pd.DataFrame) -> Data:

        """
        Decodes the encoded categorical columns back to their original values.

        Args
        ----
        - **data (Data)**: The original data, whose categorical columns' class lists
          (`CategoricalColumn(data, col).classes`) are used to map codes back to labels.
        - **df_encoded (pd.DataFrame)**: The DataFrame with encoded categorical columns to be decoded.

        Returns
        -------
        - **Data**: A Data object with all categorical columns decoded back to their original values.
        """

        df = df_encoded.copy()
        for col in data.get_categorical_columns:
            codec = CategoricalCodec(CategoricalColumn(data, col).classes)
            df[col] = codec.decode(df[col])
        return Data(df)
