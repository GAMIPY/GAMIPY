from __future__ import annotations

from typing import Any

import pandas as pd


class CategoricalCodec:

    """
    Maps a categorical column's labels to/from integer codes. Split out of
    CategoricalColumn, which used to serve both summary-statistics clients
    (classes, n_classes, mode, summary) and encoding clients through one
    interface - so a client that only wanted the summary also inherited
    encode/decode.

    A codec depends only on the list of known classes, not on a Data object
    or a column name; build one from `CategoricalColumn(data, col).classes`.

    Attributes
    ----------
    - **classes (list)**: The known category labels; a label's code is its index in this list.

    Methods
    -------
    - **encode(series)**: Returns `series` with each label replaced by its integer code.
    - **decode(encoded_series)**: Returns `encoded_series` with each integer code replaced by its label.
    """

    def __init__(self, classes: list[Any]) -> None:

        """
        Initializes the codec with the category labels to encode against.

        Args
        ----
        - **classes (list)**: The known category labels, in code order.
        """

        self.classes = list(classes)

    def encode(self, series: pd.Series[Any]) -> pd.Series[Any]:

        """
        Encodes categorical labels into integer codes.

        Values not in `classes` (including missing values) become NaN rather
        than raising.

        Args
        ----
        - **series (pd.Series)**: A pandas Series of category labels.

        Returns
        -------
        - **pd.Series**: A pandas Series with the labels encoded as integers.
        """

        return series.map({cls: idx for idx, cls in enumerate(self.classes)})

    def decode(self, encoded_series: pd.Series[Any]) -> pd.Series[Any]:

        """
        Decodes integer codes back into category labels.

        Args
        ----
        - **encoded_series (pd.Series)**: A pandas Series with integer codes to be decoded.

        Returns
        -------
        - **pd.Series**: A pandas Series with the codes decoded back into category labels.
        """

        return encoded_series.map({idx: cls for idx, cls in enumerate(self.classes)})
