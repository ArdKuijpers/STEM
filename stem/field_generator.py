from abc import ABC, abstractmethod, abstractproperty
from typing import List, Dict, Optional, Any, Sequence, Union

import numpy as np
from random_fields.generate_field import RandomFields, ModelName

from stem.globals import VERTICAL_AXIS

available_random_field_model_names = ["Gaussian", "Exponential", "Matern", "Linear"]


class FieldGenerator(ABC):
    """
    Abstract class to generate fields as function of points coordinates (x, y and z).
    The function should implement a `generate` method to initialise the field and the `values`
    property to retrieve the generated field.

    """
    def __init__(self):
        self.generated_field: Optional[List[float]] = None

    @abstractmethod
    def generate(self, coordinates:Sequence[Sequence[float]]):
        """
        Method to generate the fields for the required coordinates.
        It has to set the generated_field attribute.

        Args:
            - coordinates (Sequence[Sequence[float]]): Sequence of points where the random field needs to be generated.

        """
        return None

    @property
    @abstractmethod
    def values(self) -> Optional[List[Any]]:
        """Returns the value of the generated field.

        Returns:
            - Optional[list[Any]]: the list of generated values for the field.

        """
        pass


class RandomFieldGenerator(FieldGenerator):
    """
    Class to generate random fields for a material property in the model as funtion of the coordinates
    of the centroid of the elements (x, y and z).

    """

    def __init__(self, model_name: str,
                 n_dim: int,
                 cov: float,
                 v_scale_fluctuation: float,
                 anisotropy: Union[float, List[float]],
                 angle: Union[float, List[float]],
                 mean_value: Optional[float] = None,
                 seed: int = 14):
        """
        Initialise a random generator field. The mean value is optional because it can be set in another moment.
        In that case it should be set before running the generate method.

        Anisotropy and angle can be given as scalar, 1-D and 2-D lists. In case the model is 3D but a 1-D or scalar
        is provided, it is assumed the same angle and anisotropy along both horizontal direction.

        Args:
            - model_name (str): Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
            - n_dim (int): number of dimensions of the model (2 or 3).
            - cov (float): The coefficient of variation of the random field.
            - v_scale_fluctuation (float): The vertical scale of fluctuation of the random field.
            - anisotropy (list): The anisotropy of the random field in the other directions (per dimension).
            - angle (list): The angle of the random field (per dimension).
            - mean_value (Optional[float]): mean value of the random field. Defaults to None. \
                In that case it should be set otherwise before running the generate method.
            - seed (int): The seed number for the random number generator.

        Raises:
            - ValueError: if the model dimensions is not 2 or 3.
            - ValueError: if the model_name is not an invalid, implemented model.
        """
        super().__init__()

        # validate the number of dimensions of the model
        if n_dim not in [2, 3]:
            raise ValueError(f"Number of dimension {n_dim} specified, but should be one of either 2 or 3.")

        # check that random field model is one of the implemented
        if model_name not in available_random_field_model_names:
            raise ValueError(f"Model name: `{model_name}` was provided but not understood or implemented yet. "
                             f"Available models are: {available_random_field_model_names}")

        # if anisotropy or angle are float, convert to list
        if isinstance(anisotropy, float):
            anisotropy = [anisotropy]
        if isinstance(angle, float):
            angle = [angle]

        # if angle or anisotropy are 1-D list but model is 3-D replicate them
        aux_key = [anisotropy, angle]
        if n_dim == 3:
            for key in aux_key:
                if len(key) == 1:
                    key += key

        self.model_name = model_name
        self.n_dim = n_dim
        self.cov = cov
        self.v_scale_fluctuation = v_scale_fluctuation
        self.anisotropy = anisotropy
        self.angle = angle
        self.mean_value = mean_value
        self.seed = seed

    def generate(self, coordinates: Sequence[Sequence[float]]):
        """
        Generate the random field parameters at the coordinates specified.
        The generated values are stored in `generated_field` attribute.

        Args:
            - coordinates (Sequence[Sequence[float]]): Sequence of points where the random field needs to be generated.

        """

        if self.mean_value is None:
            raise ValueError("The mean value of the random field is not set yet. Error.")

        variance = (self.cov * self.mean_value) ** 2

        rf_generator = RandomFields(
            n_dim=self.n_dim, mean=self.mean_value, variance=variance,
            model_name=ModelName[self.model_name],
            v_scale_fluctuation=self.v_scale_fluctuation,
            anisotropy=self.anisotropy,
            angle=self.angle,
            seed=self.seed,
            v_dim=VERTICAL_AXIS
        )
        coordinates_for_rf = np.array(coordinates)
        rf_generator.generate(coordinates_for_rf)
        self.generated_field = list(rf_generator.random_field)[0].tolist()

    @property
    def values(self) -> Optional[List[Any]]:
        """Returns the value of the generated field.

        Returns:
            - Optional[list[Any]]: the list of generated values for the field.

        """
        if self.generated_field is None:
            raise ValueError("Values for field parameters are not generated yet.")

        return self.generated_field






