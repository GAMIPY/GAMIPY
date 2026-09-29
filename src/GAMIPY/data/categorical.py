from typing import Any

from GAMIPY.data.column import Column
from GAMIPY.data.data import Data


class CategoricalColumn(Column):

    """
    Represents a categorical column in a dataset.
    This class extends the Column class to provide additional functionality specific to categorical data.
    It summarizes the column's properties; mapping labels to/from integer codes is
    CategoricalCodec's job (data/categorical_codec.py), kept separate so summary-only
    clients don't depend on encode/decode.

    Attributes
    ----------
    - **classes (list)**: A list of unique categories in the column.
    - **n_classes (int)**: The number of unique categories in the column.
    - **mode (any)**: The most frequent category in the column.
    """

    def __init__(self, data: Data, column_name: str = "") -> None:

        """
        Initializes the CategoricalColumn object with a dataset and the name of the categorical column.

        Args
        ----
        - **data (Data)**: The dataset containing the categorical column.
        - **column_name (str)**: The name of the categorical column to be accessed.

        Raises
        ------
        - **ValueError**: If the specified column is not categorical or does not exist in the dataset.
        """

        super().__init__(data, column_name)
        if column_name not in data.get_categorical_columns:
            raise ValueError(f"Column '{column_name}' is not categorical.")

        # .unique() alone would include NaN itself as a "class" whenever the
        # column has missing values, so random.choice(cat_col.classes)
        # (RandomIndividual.individual(), Mutation.mutate()) could pick NaN
        # as if it were a real category - i.e. impute a missing cell with
        # another missing value.
        self.classes = data.get_column(column_name).dropna().unique().tolist()
        self.n_classes = len(self.classes)

        # Computed over the non-missing values only: the old
        # .astype(str).mode() turned every missing cell into the string
        # 'None'/'nan' and counted it as a category, so in a column that is
        # mostly missing (e.g. GAMIPY's missing-rows-only m_data) the "mode"
        # could be 'None' - which ConstantIndividual then imputed as if it
        # were a real label. Also keeps each label's original type, so the
        # mode is always one of `classes`.
        modes = data.get_column(column_name).dropna().mode()
        self.mode = modes.iloc[0] if not modes.empty else None

    @property
    def summary(self) -> dict[str, Any]:

        """
        Returns a summary of the categorical column, including its unique classes, number of classes, and mode.

        Returns
        -------
            - **dict**: A dictionary containing the column name, number of rows, number of missing
              values, unique classes, number of classes, and mode.
        """

        base_summary = super().summary
        return {
            **base_summary,
            "classes": self.classes,
            "n_classes": self.n_classes,
            "mode": self.mode
        }
