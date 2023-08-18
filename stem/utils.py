from typing import Sequence, Dict, Any, List, Union, Optional

import numpy as np
import numpy.typing as npt


class Utils:
    """
    Class containing utility methods.

    """
    @staticmethod
    def check_ndim_nnodes_combinations(n_dim: int, n_nodes_element: Optional[int],
                                       available_combinations: Dict[int, List[Any]],
                                       class_name: str):
        """
        Check if the combination of number of dimensions and number of nodes per element is supported.

        Args:
            - n_dim (int): number of dimensions
            - n_nodes_element (int): number of nodes per element
            - available_combinations (Dict[int, List[int]]): dictionary containing the supported combinations of number\
               of dimensions and number of nodes per element
            - class_name (str): name of the class to be checked

        Raises:
            - ValueError: when the number of dimensions is not supported.
            - ValueError: when the combination of number of dimensions and number of nodes per element is not supported.

        """
        # check if the number of dimensions is supported
        if n_dim not in available_combinations.keys():
            raise ValueError(f"Number of dimensions {n_dim} is not supported for {class_name} elements. Supported "
                             f"dimensions are {list(available_combinations.keys())}.")

        # check if the number of nodes per element is supported
        if n_nodes_element not in available_combinations[n_dim]:
            raise ValueError(
                f"In {n_dim} dimensions, only {available_combinations[n_dim]} noded {class_name} elements are "
                f"supported. {n_nodes_element} nodes were provided."
            )

    @staticmethod
    def are_2d_coordinates_clockwise(coordinates: Sequence[Sequence[float]]):
        """
        Checks if the 2D coordinates are given in clockwise order. If the signed area is positive, the coordinates
        are given in clockwise order.

        Args:
            - coordinates (Sequence[Sequence[float]]): coordinates of the points of a surface

        Returns:
            - bool: True if the coordinates are given in clockwise order, False otherwise.
        """

        # calculate signed area of polygon
        signed_area = 0.0
        for i in range(len(coordinates) - 1):
            signed_area += (coordinates[i + 1][0] - coordinates[i][0]) * (coordinates[i + 1][1] + coordinates[i][1])

        signed_area += (coordinates[0][0] - coordinates[-1][0]) * (coordinates[0][1] + coordinates[-1][1])

        # if signed area is positive, the coordinates are given in clockwise order
        return signed_area > 0.0

    @staticmethod
    def check_dimensions(points:Sequence[Sequence[float]]):
        """

        Check if points have the same dimensions (2D or 3D).

        Args:
            - points: (Sequence[Sequence[float]]): points to be tested

        Raises:
            - ValueError: when the points have different dimensions.
            - ValueError: when the dimension is not either 2 or 3D.
        """

        lengths = [len(point) for point in points]
        if len(np.unique(lengths)) != 1:
            raise ValueError("Mismatch in dimension of given points!")

        if any([ll not in [2, 3] for ll in lengths]):
            raise ValueError("Dimension of the points should be 2D or 3D.")

    @staticmethod
    def is_collinear(point: Sequence[float], start_point: Sequence[float], end_point: Sequence[float],
                     a_tol: float = 1e-06):
        """
        Check if point is aligned with the other two on a line. Points must have the same dimension (2D or 3D)

        Args:
            - point (Sequence[float]): point coordinates to be tested
            - start_point (Sequence[float]): coordinates of first point of a line
            - end_point (Sequence[float]): coordinates of second point of a line
            - a_tol (float): absolute tolerance to check collinearity (default 1e-6)

        Raises:
            - ValueError: when there is a dimension mismatch in the point dimensions.

        Returns:
            - bool: whether the point is aligned or not
        """

        # check dimensions of points for validation
        Utils.check_dimensions([point, start_point, end_point])

        vec_1 = np.asarray(point) - np.asarray(start_point)
        vec_2 = np.asarray(end_point) - np.asarray(start_point)

        # cross product of the two vector
        cross_product = np.cross(vec_1, vec_2)
        # It should be smaller than tolerance for points to be aligned
        return np.sum(np.abs(cross_product)) < a_tol

    @staticmethod
    def is_point_between_points(point:Sequence[float], start_point:Sequence[float], end_point:Sequence[float]):
        """
        Check if point is between the other two. Points must have the same dimension (2D or 3D).

        Args:
            - point (Sequence[float]): point coordinates to be tested
            - start_point (Sequence[float]): first extreme coordinates of the line
            - end_point (Sequence[float]): second extreme coordinates of the line

        Raises:
            - ValueError: when there is a dimension mismatch in the point dimensions.

        Returns:
            - bool: whether the point is between the other two or not
        """

        # check dimensions of points for validation
        Utils.check_dimensions([point, start_point, end_point])

        # Calculate vectors between the points
        vec_1 = np.asarray(point) - np.asarray(start_point)
        vec_2 = np.asarray(end_point) - np.asarray(start_point)

        # Calculate the scalar projections of vector1 onto vector2
        scalar_projection = sum(v1 * v2 for v1, v2 in zip(vec_1, vec_2)) / sum(v ** 2 for v in vec_2)

        # Check if the scalar projection is between 0 and 1 (inclusive)
        return 0 <= scalar_projection <= 1

    @staticmethod
    def is_non_str_sequence(seq:object):
        """
        check whether object is a sequence but also not a string

        Returns:
            - bool: whether the sequence but also not a string
        """
        return isinstance(seq, Sequence) and not isinstance(seq, str)

    @staticmethod
    def chain_sequence(sequences: Sequence[Sequence[Any]]):
        """
        merges dictionary b into dictionary a. if existing keywords conflict it assumes
        they are concatenated in a list

        Args:
           - sequences (Sequence[Sequence[Any]]): sequences to chain

        Returns:
            - Iterator[Any]: chained sequences

        """
        for seq in sequences:
            yield from seq

    @staticmethod
    def merge(a: Dict[Any, Any], b: Dict[Any, Any], path: Union[List[str], Any] = None):
        """
        merges dictionary b into dictionary a. if existing keywords conflict it assumes
        they are concatenated in a list

        Args:
            - a (Dict[str,Any]): first dictionary
            - b (Dict[str,Any]): second dictionary
            - path (List[str]): object to help navigate the deeper layers of the dictionary. \
                Always place it as None

        Returns:
            - a (Dict[str,Any]): updated dictionary with the additional dictionary `b`
        """
        if path is None:
            path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    Utils.merge(a[key], b[key], path + [str(key)])
                elif a[key] == b[key]:
                    pass  # same leaf value
                elif any([not Utils.is_non_str_sequence(val) for val in (a[key], b[key])]):
                    # if none of them is a sequence and are found at the same key, then something went wrong.
                    # this should not be merge silently.
                    raise ValueError(f"Conflict of merging keys at {'->'.join(path + [str(key)])}. Two non sequence "
                                     f"values have been found.")
                else:
                    a[key] = list(Utils.chain_sequence([a[key], b[key]]))
            else:
                a[key] = b[key]
        return a

    @staticmethod
    def get_unique_objects(input_sequence: Sequence[object]):
        """
        Get the unique objects, i.e., the objects that share the same memory location.

        Args:
            - input_sequence (Sequence[object]): full list of possible duplicate objects

        Returns:
            - List[object]: list of unique objects
        """
        return list({id(obj): obj for obj in input_sequence}.values())

    @staticmethod
    def calculate_centre_of_mass(coordinates: npt.NDArray) -> npt.NDArray:
        """
        Calculate the centre of mass of a closed polygon.

        Args:
            - coordinates (npt.NDArray): coordinates of the points of a polygon

        Returns:
            - npt.NDArray: coordinates of the centre of mass

        """
        # add first point to the end of the array to close the polygon
        connected_coordinates = np.vstack((coordinates[-1], coordinates))

        # calculate length of attached lines to each point
        diff = np.diff(connected_coordinates, axis=0)

        # calculate middle coordinates of each line
        middle_coordinates = (connected_coordinates[1:] + connected_coordinates[:-1]) / 2

        # calculate weights of each line, which is the length of the line
        weights = np.sqrt(np.sum(diff ** 2, axis=1))

        # normalise weights
        normalised_weights = weights / np.sum(weights)

        # centre of mass is the weighted average of the middle coordinates
        return middle_coordinates.T.dot(normalised_weights[:, None])[:, 0]