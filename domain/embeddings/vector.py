# C:\Users\erees\Documents\development\nireon_staging\nireon\domain\embeddings\vector.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np

DEFAULT_DTYPE = np.float64


@dataclass(frozen=True, slots=True)
class Vector:
    """
    Domain-level representation of an embedding vector.

    This class provides an immutable, type-safe, and mathematically-augmented
    representation of a 1D numerical vector, typically used for embeddings.
    It ensures data consistency and offers common vector operations.

    - Immutable: Once created, a Vector object cannot be changed.
    - Type-Safe: Primarily uses NumPy with a consistent float data type.
    - NumPy Backend: Leverages NumPy for efficient numerical computations.
    - Domain-Pure: No external dependencies beyond NumPy and standard Python libraries.
                   No model-loading or I/O side-effects.
    """

    data: np.ndarray  # 1-D array of DEFAULT_DTYPE

    def __post_init__(self):
        """Validates the vector data upon initialization."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(
                f"Vector data must be a NumPy array, got {type(self.data)}."
            )
        if self.data.ndim != 1:
            raise ValueError(
                f"Vector data must be 1-dimensional, got {self.data.ndim} dimensions."
            )
        if self.data.dtype != DEFAULT_DTYPE:
            # This ensures consistency, especially if Vector is instantiated directly.
            # Factory methods like from_raw already handle conversion to DEFAULT_DTYPE.
            raise TypeError(
                f"Vector data must have dtype {DEFAULT_DTYPE}, got {self.data.dtype}."
            )

    # --- Properties ---
    @property
    def dims(self) -> int:
        """Returns the dimensionality (number of elements) of the vector."""
        return int(self.data.shape[0])

    @property
    def norm(self) -> float:
        """Returns the L2 norm (Euclidean magnitude or length) of the vector."""
        return float(np.linalg.norm(self.data))

    # --- Operator Overloads ---
    def __add__(self, other: Vector) -> Vector:
        """Vector addition: self + other."""
        if not isinstance(other, Vector):
            return NotImplemented
        if self.dims != other.dims:
            raise ValueError(
                "Vectors must have the same dimensionality for addition."
            )
        return Vector(self.data + other.data)

    def __sub__(self, other: Vector) -> Vector:
        """Vector subtraction: self - other."""
        if not isinstance(other, Vector):
            return NotImplemented
        if self.dims != other.dims:
            raise ValueError(
                "Vectors must have the same dimensionality for subtraction."
            )
        return Vector(self.data - other.data)

    def __mul__(self, scalar: Union[int, float]) -> Vector:
        """Scalar multiplication: self * scalar."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector(self.data * float(scalar))

    def __rmul__(self, scalar: Union[int, float]) -> Vector:
        """Scalar multiplication (reflected): scalar * self."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Union[int, float]) -> Vector:
        """Scalar division: self / scalar."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero.")
        return Vector(self.data / float(scalar))

    def __neg__(self) -> Vector:
        """Negation of the vector: -self."""
        return Vector(-self.data)

    def __eq__(self, other: object) -> bool:
        """Checks for element-wise equality with another Vector within tolerance."""
        if not isinstance(other, Vector):
            return NotImplemented
        if self.dims != other.dims:
            return False
        return np.allclose(self.data, other.data)

    def __repr__(self) -> str:
        """A string representation of the vector, useful for debugging."""
        preview_len = 3
        if self.dims <= preview_len * 2:
            data_str = ", ".join(f"{x:.4f}" for x in self.data)
        else:
            head = ", ".join(f"{x:.4f}" for x in self.data[:preview_len])
            tail = ", ".join(f"{x:.4f}" for x in self.data[-preview_len:])
            data_str = f"{head}, ..., {tail}"
        return f"Vector(dims={self.dims}, data=[{data_str}])"

    def __str__(self) -> str:
        """A user-friendly string representation of the vector."""
        return f"Vector (dims={self.dims})"


    # --- Core Vector Operations & Metrics ---
    def dot(self, other: Vector) -> float:
        """Computes the dot product with another vector."""
        if not isinstance(other, Vector):
            raise TypeError("Dot product can only be computed with another Vector.")
        if self.dims != other.dims:
            raise ValueError(
                "Vectors must have the same dimensionality for dot product."
            )
        return float(np.dot(self.data, other.data))

    def similarity(self, other: Vector) -> float:
        """
        Computes the cosine similarity with another vector.
        Returns a value between -1 and 1 (or 0 if one/both vectors are zero).
        """
        if not isinstance(other, Vector):
            raise TypeError("Similarity can only be computed with another Vector.")
        if self.dims != other.dims:
            raise ValueError("Vectors must have the same dimensionality for similarity.")

        norm1 = self.norm
        norm2 = other.norm

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0  # Similarity with a zero vector is 0

        dot_product = self.dot(other)
        return float(dot_product / (norm1 * norm2))

    def distance_euclidean(self, other: Vector) -> float:
        """Computes the Euclidean (L2) distance to another vector."""
        if not isinstance(other, Vector):
            raise TypeError("Distance can only be computed with another Vector.")
        if self.dims != other.dims:
            raise ValueError(
                "Vectors must have the same dimensionality for Euclidean distance."
            )
        return float(np.linalg.norm(self.data - other.data))

    def distance_manhattan(self, other: Vector) -> float:
        """Computes the Manhattan (L1) distance to another vector."""
        if not isinstance(other, Vector):
            raise TypeError("Distance can only be computed with another Vector.")
        if self.dims != other.dims:
            raise ValueError(
                "Vectors must have the same dimensionality for Manhattan distance."
            )
        return float(np.sum(np.abs(self.data - other.data)))

    def elementwise_multiply(self, other: Vector) -> Vector:
        """Performs element-wise (Hadamard) multiplication with another vector."""
        if not isinstance(other, Vector):
            raise TypeError(
                "Element-wise multiplication can only be performed with another Vector."
            )
        if self.dims != other.dims:
            raise ValueError(
                "Vectors must have the same dimensionality for element-wise multiplication."
            )
        return Vector(self.data * other.data)

    # --- Utility Methods ---
    def is_zero(self, tolerance: float = 1e-9) -> bool:
        """Checks if the vector is a zero vector (all elements close to zero)."""
        return np.allclose(self.data, 0.0, atol=tolerance)

    def is_unit(self, tolerance: float = 1e-9) -> bool:
        """Checks if the vector is a unit vector (norm close to 1)."""
        return math.isclose(self.norm, 1.0, abs_tol=tolerance)

    def to_list(self) -> List[float]:
        """Converts the vector data to a Python list of floats."""
        return self.data.tolist()

    def get_normalised(self) -> Vector:
        """Returns a new Vector that is the normalised (unit) version of this vector."""
        current_norm = self.norm
        if current_norm == 0:
            return self # Already a zero vector, normalising would be 0/0
        return Vector(self.data / current_norm)

    # --- Factory Methods ---
    @classmethod
    def from_list(cls, data_list: Sequence[Union[int, float]]) -> Vector:
        """Creates a Vector from a Python list or sequence of numbers."""
        if not isinstance(data_list, Sequence) or not all(
            isinstance(x, (int, float)) for x in data_list
        ):
            raise TypeError("Input for from_list must be a sequence of numbers.")
        arr = np.array(data_list, dtype=DEFAULT_DTYPE)
        if arr.ndim != 1:
             raise ValueError("Input list must result in a 1-dimensional vector.")
        return cls(arr)

    @classmethod
    def from_raw(cls, raw: Any) -> Vector:
        """
        Creates a Vector from a raw, array-like input.
        Ensures the data is a 1D NumPy array of the default float type.
        """
        try:
            arr = np.asarray(raw, dtype=DEFAULT_DTYPE)
        except Exception as e:
            raise ValueError(f"Could not convert raw input to NumPy array: {e}") from e

        if arr.ndim != 1:
            raise ValueError(
                f"Input for Vector.from_raw must be 1-dimensional, got {arr.ndim} dimensions."
            )
        return cls(arr)

    @classmethod
    def normalised(cls, raw: Any) -> Vector:
        """
        Creates a Vector from raw input and then normalises it to unit length.
        If the input results in a zero vector, the zero vector is returned.
        """
        vec = cls.from_raw(raw)
        return vec.get_normalised()

    @classmethod
    def zeros(cls, dims: int) -> Vector:
        """Creates a zero vector of the specified dimensionality."""
        if not isinstance(dims, int) or dims <= 0:
            raise ValueError("Dimensions must be a positive integer.")
        return cls(np.zeros(dims, dtype=DEFAULT_DTYPE))

    @classmethod
    def ones(cls, dims: int) -> Vector:
        """Creates a vector of ones of the specified dimensionality."""
        if not isinstance(dims, int) or dims <= 0:
            raise ValueError("Dimensions must be a positive integer.")
        return cls(np.ones(dims, dtype=DEFAULT_DTYPE))

    @classmethod
    def random_normalised(cls, dims: int, seed: Optional[int] = None) -> Vector:
        """Creates a random normalised (unit) vector of the specified dimensionality."""
        if not isinstance(dims, int) or dims <= 0:
            raise ValueError("Dimensions must be a positive integer.")
        rng = np.random.default_rng(seed)
        random_data = rng.standard_normal(dims, dtype=DEFAULT_DTYPE)
        norm = np.linalg.norm(random_data)
        if norm == 0: # Extremely unlikely for standard_normal, but handle defensively
            return cls.zeros(dims)
        return cls(random_data / norm)