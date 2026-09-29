from typing import Any


class Gene:

    """
    Represents a gene with its original location, modified location, name, data, and data type.

    Attributes
    ----------
    - **location (tuple)**: The original location of the gene in the genome.
    - **_m_location (tuple)**: The modified location of the gene, if applicable
    - **name (str)**: The name of the gene.
    - **data (object)**: The data associated with the gene, which can be of any type (e.g.,
      sequence, expression levels).
    - **dtype (str)**: The data type of the gene, such as 'num' for numerical data or 'cat' for categorical data.
    - **repr**: A string representation of the gene object, including its name, data, original
      location, modified location, and data type.

    Methods
    -------
    - **__init__**: Initializes the Gene object with the specified attributes.
    - **__repr__**: Returns a string representation of the Gene object.
    """


    # A GA run creates millions of genes: slots make them smaller and
    # faster to build than instances with a __dict__.
    __slots__ = ('_m_location', 'data', 'dtype', 'location', 'name')

    def __init__(
        self, *, orig_location: tuple[Any, Any], m_location: tuple[Any, Any], name: str, data: Any, dtype: str
    ) -> None:
        """
        Initializes the Gene object with the specified attributes.

        All arguments are required and keyword-only: a gene without a real
        location, column or dtype is never meaningful, and positional calls
        were exactly how genes used to get corrupted.

        Args
        ----
        - **orig_location (tuple)**: ``(row label, column)`` of the cell in the original data.
        - **m_location (tuple)**: ``(row position, column)`` of the cell in the missing-rows subset (m_data).
        - **name (str)**: The column name.
        - **data (Any)**: The imputed value (a number or a category label).
        - **dtype (str)**: ``'num'`` for numerical data or ``'cat'`` for categorical data.

        Raises
        ------
        - **TypeError**: If orig_location or m_location is not a tuple.
        - **ValueError**: If dtype is not one of {'num', 'cat'}.
        """
        if not isinstance(orig_location, tuple):
            raise TypeError(f"orig_location must be a tuple, got {type(orig_location).__name__}.")
        if not isinstance(m_location, tuple):
            raise TypeError(f"m_location must be a tuple, got {type(m_location).__name__}.")
        if dtype not in ("num", "cat"):
            raise ValueError(f"dtype must be 'num' or 'cat', got {dtype!r}.")

        self.location = orig_location
        self._m_location = m_location
        self.name = name
        self.data = data
        self.dtype = dtype

    def __repr__(self) -> str:
        """
        Returns a string representation of the Gene object, including its name, data,
        original location, modified location, and data type.

        Returns
        -------
        - **str**: A string representation of the Gene object.
        """

        return (
            f"Gene(name={self.name}, data={self.data}, location={self.location}, "
            f"m_location={self._m_location}, dtype={self.dtype})"
        )
