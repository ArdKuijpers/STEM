import re

import numpy as np
import numpy.testing as npt
import pytest

from stem.utils import Utils
from tests.utils import TestUtils


class TestUtilsStem:

    def test_check_ndim_nnodes_combinations(self):
        """
        Test the check which checks if the combination of n_dim and n_nodes is valid for a certain element type.
        It checks for valid input, invalid n_dim and invalid n_nodes.

        """

        # successful checks
        Utils.check_ndim_nnodes_combinations(2,2, {2: [2]}, "test")

        # non valid n_dim
        with pytest.raises(ValueError, match=re.escape(f"Number of dimensions 3 is not supported for failed_test "
                                                       f"elements. Supported dimensions are [2].")):
            Utils.check_ndim_nnodes_combinations(3,2, {2: [2]},
                                                 "failed_test")

        # non valid n_nodes for dim
        with pytest.raises(ValueError, match=re.escape(f"In 2 dimensions, only [3, 4, 5] noded failed_test_2 elements "
                                                       f"are supported. 2 nodes were provided.")):
            Utils.check_ndim_nnodes_combinations(2,2, {2: [3,4,5]},
                                                 "failed_test_2")


    def test_is_clockwise(self):
        """
        Test the check which checks if coordinates are given in clockwise order
        """

        coordinates = [[0, 0], [2, 0], [2, 2], [0, 2]]

        assert not Utils.are_2d_coordinates_clockwise(coordinates=coordinates)
        assert Utils.are_2d_coordinates_clockwise(coordinates=coordinates[::-1])

    def test_is_clockwise_non_convex(self):
        """
        Test the check which checks if coordinates are given in clockwise order for a non-convex polygon

        """

        coordinates = [[0, 0], [2, 0], [2, 2], [1, -1], [0, 2]]

        assert not Utils.are_2d_coordinates_clockwise(coordinates=coordinates)
        assert Utils.are_2d_coordinates_clockwise(coordinates=coordinates[::-1])

    def test_collinearity_2d(self):
        """
        Check collinearity between 3 points in 2D
        """
        p1 = np.array([0, 0])
        p2 = np.array([-2, -1])

        p_test_1 = np.array([2, 1])
        p_test_2 = np.array([-5, 1])

        assert Utils.is_collinear(point=p_test_1, start_point=p1, end_point=p2)
        assert not Utils.is_collinear(point=p_test_2, start_point=p1, end_point=p2)

    def test_collinearity_3d(self):
        """
        Check collinearity between 3 points in 3D
        """

        p1 = np.array([0, 0, 0])
        p2 = np.array([-2, -2, 2])

        p_test_1 = np.array([2, 2, -2])
        p_test_2 = np.array([2, -2, 2])

        assert Utils.is_collinear(point=p_test_1, start_point=p1, end_point=p2)
        assert not Utils.is_collinear(point=p_test_2, start_point=p1, end_point=p2)

    def test_is_in_between_2d(self):
        """
        Check if point is in between other 2 points in 2D
        """

        p1 = np.array([0, 0])
        p2 = np.array([-2, -2])

        p_test_1 = np.array([2, 2])
        p_test_2 = np.array([-1, -1])

        assert not Utils.is_point_between_points(
            point=p_test_1, start_point=p1, end_point=p2
        )
        assert Utils.is_point_between_points(
            point=p_test_2, start_point=p1, end_point=p2
        )

    def test_is_in_between_3d(self):
        """
        Check if point is in between other 2 points in 3D
        """

        p1 = np.array([0, 0, 0])
        p2 = np.array([-2, -2, 2])

        p_test_1 = np.array([2, 2, -2])
        p_test_2 = np.array([-1, -1, 1])

        assert not Utils.is_point_between_points(
            point=p_test_1, start_point=p1, end_point=p2
        )
        assert Utils.is_point_between_points(
            point=p_test_2, start_point=p1, end_point=p2
        )

    def test_check_sequence_non_string(self):
        """
        Test assertion of non-string sequence
        """

        assert Utils.is_non_str_sequence([1, 2, 3, 4])
        assert Utils.is_non_str_sequence((1, 2, 3, 4))

        assert not Utils.is_non_str_sequence({"Hello world"})
        assert not Utils.is_non_str_sequence({1: {"a": "A"}, 2: {"b": "B"}})

    def test_chain_sequence(self):
        """
        Test chain of sequences
        """

        # test 1: no conflicts in arguments, merge dictionaries with common keys
        # new keys are added, common keys are expanded
        seq1 = [1, 2, 3, 4]
        seq2 = (100, 200, 300)

        actual_sequence = list(Utils.chain_sequence([seq1, seq2]))
        expected_sequence = [1, 2, 3, 4, 100, 200, 300]

        npt.assert_equal(expected_sequence, actual_sequence)

    def test_merge(self):
        """
        Test merging of dictionaries
        """

        # test 1: no conflicts in arguments, merge dictionaries with common keys
        # new keys are added, common keys are expanded
        dict1a = {1: {"a": "A"}, 2: {"b": "B"}}
        dict1b = {2: {"c": "C"}, 3: {"d": "D"}}

        actual_dict_1 = Utils.merge(dict1a, dict1b)
        expected_dict_1 = {1: {"a": "A"}, 2: {"c": "C", "b": "B"}, 3: {"d": "D"}}

        # expect it raises an error
        TestUtils.assert_dictionary_almost_equal(expected_dict_1, actual_dict_1)

        # test 2: conflicts in arguments, merge common keys argument into a list
        # common keys are expanded into list
        dict2a = {1: {"a": [1, 2, 3]}, 2: {"b": "B"}}
        dict2b = {1: {"a": [5, 6]}, 2: {"b": "D"}, 3: {"d": "D"}}

        # expect it raises an error
        with pytest.raises(
                ValueError, match="Conflict of merging keys at 2->b. Two non sequence values have been found."
        ):
            Utils.merge(dict2a, dict2b)

        # test 3: conflicts in arguments, merge common keys argument into a list
        # list of lists is preserved into a list of list + 2 extra elements
        # tuple(int) and list(str) are merged into list(str,int)
        dict3a = {1: {"a": [[1, 2, 3]]}, 2: {"b": ["B"]}}
        dict3b = {1: {"a": [5, 6]}, 2: {"b": (1, 2, 3)}, 3: {"d": "D"}}

        actual_dict_3 = Utils.merge(dict3a, dict3b)
        expected_dict_3 = {
            1: {"a": [[1, 2, 3], 5, 6]},
            2: {"b": ["B", 1, 2, 3]},
            3: {"d": "D"},
        }

        TestUtils.assert_dictionary_almost_equal(expected_dict_3, actual_dict_3)