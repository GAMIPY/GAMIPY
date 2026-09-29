import pandas as pd

from GAMIPY.data.data import Data
from GAMIPY.population.individual import Individual


class ImputedDataBuilder:

    """
    Writes an individual's genes back onto a copy of the original full
    dataset, producing one complete pd.DataFrame (originally-complete rows
    untouched, originally-missing cells filled in). Split out of GAMIPY.

    Attributes
    ----------
    - **data (Data)**: The original dataset, missing values included.
    """

    def __init__(self, data: Data) -> None:
        self.data = data

    def build(self, individual: Individual) -> pd.DataFrame:

        """
        A gene's `_m_location` row component is a position within the
        missing-only, reset-index m_data (not a label into the original
        data), so it has to be remapped through the original DataFrame's
        own missing-row order rather than used as a label directly.

        Args
        ----
        - **individual (Individual)**: The individual whose genes fill the missing cells.

        Returns
        -------
        - **pd.DataFrame**: A copy of the original data with every missing cell filled.
        """

        full_imputed = self.data.data.copy()
        original_missing_labels = self.data.data.index[self.data.data.isnull().any(axis=1)]
        for gene in individual.genes:
            orig_label = original_missing_labels[gene._m_location[0]]
            full_imputed.loc[orig_label, gene._m_location[1]] = gene.data
        return full_imputed
