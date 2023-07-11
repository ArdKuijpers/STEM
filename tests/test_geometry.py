import pytest
from gmsh_utils.gmsh_IO import GmshIO

from stem.geometry import *


class TestGeometry:

    @pytest.fixture
    def expected_geo_data_0D(self):
        """
        Expected geometry data for a 0D geometry group. The group is a geometry of a point

        Returns:
            - expected_geo_data (Dict[str, Any]): dictionary containing the geometry data as generated by the gmsh_io
        """
        expected_points = {1: [0, 0, 0], 2: [0.5, 0, 0]}
        return {"points": expected_points}


    @pytest.fixture
    def expected_geo_data_1D(self):
        """
        Expected geometry data for a 1D geometry group. The group is a geometry of a line

        Returns:
            - expected_geo_data (Dict[str, Any]): dictionary containing the geometry data as generated by the gmsh_io
        """
        expected_points = {4: [0, 1.0, 0], 11: [0, 2.0, 0], 12: [0.5, 2.0, 0]}
        expected_lines = {13: [4, 11], 14: [11, 12]}

        return {"points": expected_points,
                "lines": expected_lines}

    @pytest.fixture
    def expected_geo_data_2D(self):
        """
        Expected geometry data for a 2D geometry group. The group is a geometry of a square.

        Returns:
            - expected_geo_data (Dict[str, Any]): dictionary containing the geometry data as generated by the gmsh_io
        """
        expected_points = {3: [0.5, 1, 0], 4: [0, 1, 0], 11: [0, 2, 0], 12: [0.5, 2.0, 0]}
        expected_lines = { 7: [3, 4], 13: [4, 11], 14: [11, 12], 15: [12, 3]}
        expected_surfaces = {17: [-13, -7, -15, -14]}

        return {"points": expected_points,
                "lines": expected_lines,
                "surfaces": expected_surfaces}

    @pytest.fixture
    def expected_geo_data_3D(self):
        """
        Expected geometry data for a 3D geometry group. The group is a geometry of a cubic block

        Returns:
            - expected_geo_data (Dict[str, Any]): dictionary containing the geometry data as generated by the gmsh_io
        """
        expected_points = {3: [0.5, 1., 0.], 4: [0., 1., 0.], 11: [0., 2., 0.],
                           12: [0.5, 2., 0.],  18: [0.5, 1., -0.5],
                           22: [0., 1., -0.5], 23: [0., 2., -0.5], 32: [0.5, 2., -0.5]}
        expected_lines = {7: [3, 4], 13: [4, 11], 14: [11, 12], 15: [12, 3], 21: [18, 22], 29: [3, 18],
                          33: [4, 22], 41: [23, 22], 43: [18, 32], 44: [32, 23], 46: [11, 23], 55: [12, 32]}
        expected_surfaces = {17: [-13, -7, -15, -14], 34: [7, 33, -21, -29],  48: [-13, 33, -41, -46],
                             56: [-15, 55, -43, -29], 60: [-14, 46, -44, -55], 61: [41, -21, 43, 44]}
        expected_volumes = {2: [-17, 61, -48, -34, -56, -60]}

        return {"points": expected_points,
                "lines": expected_lines,
                "surfaces": expected_surfaces,
                "volumes": expected_volumes}

    def test_create_0d_geometry_from_gmsh_group(self, expected_geo_data_0D):
        """
        Test the creation of a 0D geometry from a gmsh group.

        Args:
            - expected_geo_data_0D (Dict[int, Any]): expected geometry data for a 0D geometry group.

        """
        # Read the gmsh geo file
        gmsh_io = GmshIO()
        gmsh_io.read_gmsh_geo(r"tests/test_data/gmsh_utils_column_2D.geo")
        geo_data = gmsh_io.geo_data

        # Create the geometry from the gmsh group
        geometry = Geometry().create_geometry_from_gmsh_group(geo_data, "point_group")

        # Assert that the geometry is created correctly
        assert len(geometry.points) == len(expected_geo_data_0D["points"])
        for point in geometry.points:
            assert pytest.approx(point.coordinates) == expected_geo_data_0D["points"][point.id]

    def test_create_1d_geometry_from_gmsh_group(self, expected_geo_data_1D):
        """
        Test the creation of a 1D geometry from a gmsh group.

        Args:
            - expected_geo_data_1D (Dict[int, Any]): expected geometry data for a 1D geometry group.

        """

        # Read the gmsh geo file
        gmsh_io = GmshIO()
        gmsh_io.read_gmsh_geo(r"tests/test_data/gmsh_utils_column_2D.geo")
        geo_data = gmsh_io.geo_data

        # Create the geometry from the gmsh group
        geometry = Geometry().create_geometry_from_gmsh_group(geo_data, "line_group")

        # Assert that the geometry is created correctly
        assert len(geometry.points) == len(expected_geo_data_1D["points"])
        for point in geometry.points:
            assert pytest.approx(point.coordinates) == expected_geo_data_1D["points"][point.id]

        assert len(geometry.lines) == len(expected_geo_data_1D["lines"])
        for line in geometry.lines:
            assert line.point_ids == expected_geo_data_1D["lines"][line.id]

    def test_create_2d_geometry_from_gmsh_group(self, expected_geo_data_2D):
        """
        Test the creation of a 2D geometry from a gmsh group.

        Args:
            - expected_geo_data_2D (Dict[int, Any]): expected geometry data for a 2D geometry group.

        """

        # Read the gmsh geo file
        gmsh_io = GmshIO()
        gmsh_io.read_gmsh_geo(r"tests/test_data/gmsh_utils_column_2D.geo")
        geo_data = gmsh_io.geo_data

        # Create the geometry from the gmsh group
        geometry = Geometry().create_geometry_from_gmsh_group(geo_data, "group_2")

        # Assert that the geometry is created correctly
        assert len(geometry.points) == len(expected_geo_data_2D["points"])
        for point in geometry.points:
            assert pytest.approx(point.coordinates) == expected_geo_data_2D["points"][point.id]

        assert len(geometry.lines) == len(expected_geo_data_2D["lines"])
        for line in geometry.lines:
            assert line.point_ids == expected_geo_data_2D["lines"][line.id]

        assert len(geometry.surfaces) == len(expected_geo_data_2D["surfaces"])
        for surface in geometry.surfaces:
            assert surface.line_ids == expected_geo_data_2D["surfaces"][surface.id]


    def test_create_3d_geometry_from_gmsh_group(self, expected_geo_data_3D: Dict[str, Any]):
        """
        Test the creation of a 3D geometry from a gmsh group.

        Args:
            - expected_geo_data_3D (Dict[int, Any]): expected geometry data for a 3D geometry group.

        """

        # Read the gmsh geo file
        gmsh_io = GmshIO()
        gmsh_io.read_gmsh_geo(r"tests/test_data/gmsh_utils_column_3D_tetra4.geo")
        geo_data = gmsh_io.geo_data

        # Create the geometry from the gmsh group
        geometry = Geometry().create_geometry_from_gmsh_group(geo_data, "group_2")

        # Assert that the geometry is created correctly
        assert len(geometry.points) == len(expected_geo_data_3D["points"])
        for point in geometry.points:
            assert pytest.approx(point.coordinates) == expected_geo_data_3D["points"][point.id]

        assert len(geometry.lines) == len(expected_geo_data_3D["lines"])
        for line in geometry.lines:
            assert line.point_ids == expected_geo_data_3D["lines"][line.id]

        assert len(geometry.surfaces) == len(expected_geo_data_3D["surfaces"])
        for surface in geometry.surfaces:
            assert surface.line_ids == expected_geo_data_3D["surfaces"][surface.id]

        assert len(geometry.volumes) == len(expected_geo_data_3D["volumes"])
        for volume in geometry.volumes:
            assert volume.surface_ids == expected_geo_data_3D["volumes"][volume.id]

    def test_create_geometry_from_geo_data(self, expected_geo_data_3D):
        """
        Test the creation of a 3D geometry from a geo_data dictionary.

        Args:
            - expected_geo_data_3D (Dict[int, Any]): expected geometry data for a 3D geometry group.

        """

        geo_data = expected_geo_data_3D

        # Create the geometry from the gmsh group
        geometry = Geometry().create_geometry_from_geo_data(geo_data)

        # Assert that the geometry is created correctly
        assert len(geometry.points) == len(expected_geo_data_3D["points"])
        for point in geometry.points:
            assert pytest.approx(point.coordinates) == expected_geo_data_3D["points"][point.id]

        assert len(geometry.lines) == len(expected_geo_data_3D["lines"])
        for line in geometry.lines:
            assert line.point_ids == expected_geo_data_3D["lines"][line.id]

        assert len(geometry.surfaces) == len(expected_geo_data_3D["surfaces"])
        for surface in geometry.surfaces:
            assert surface.line_ids == expected_geo_data_3D["surfaces"][surface.id]

        assert len(geometry.volumes) == len(expected_geo_data_3D["volumes"])
        for volume in geometry.volumes:
            assert volume.surface_ids == expected_geo_data_3D["volumes"][volume.id]