from typing import Any

from GAMIPY.data.data import Data

# One missing cell: (row label in the original data, row position in the
# missing-rows subset m_data, column).
MissingCell = tuple[Any, int, Any]


class MissingIndexLocator:

    """
    Locates every missing cell of a dataset in two coordinate systems at
    once: its row label in the original data, and its row position in the
    missing-rows-only subset (`Data.get_missing_data`, index reset). Split
    out of Data, which used to embed this positional mapping in a
    `get_missing_index` property on top of its plain DataFrame-access
    responsibility.

    Attributes
    ----------
    - **data (Data)**: The original dataset whose missing cells are located.

    Methods
    -------
    - **locate()**: Returns one `(row, row_m, column)` tuple per missing cell.
    """

    def __init__(self, data: Data) -> None:

        """
        Initializes the locator with the original dataset.

        Args
        ----
        - **data (Data)**: The original dataset (not an already row-filtered subset).
        """

        self.data = data

    def locate(self) -> list[MissingCell]:

        """
        Returns the locations of all missing values, in row-major order.

        Each tuple is `(row, row_m, column)`: `row` is the row label in the
        original data, `row_m` is that row's position among the rows that
        contain any missing value (i.e. its index in `get_missing_data`),
        and `column` is the column name.

        Returns
        -------
        - **list**: A list of `(row, row_m, column)` tuples.
        """

        # Walk rows by position, not by label, so a DataFrame with duplicate
        # index labels still gets one distinct row_m per missing row.
        missing = self.data.data.isnull()
        locations: list[MissingCell] = []
        row_m = 0
        for pos, row in enumerate(missing.index):
            row_mask = missing.iloc[pos]
            if not row_mask.any():
                continue
            for col in row_mask.index[row_mask]:
                locations.append((row, row_m, col))
            row_m += 1
        return locations
