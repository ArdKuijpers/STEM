from typing import Optional, Union, Dict, Any

from stem.soil_material import SoilMaterial
from stem.structural_material import StructuralMaterial

from stem.geometry import Geometry
from stem.mesh import Mesh


class ModelPart:
    """
    One part of the complete model, this can be a boundary condition, a loading or another special process
    like excavation.

    Attributes:
        - name (str): name of the model part
        - nodes (None): node id followed by node coordinates in an array
        - elements (None): element id followed by connectivities in an array
        - conditions (None): condition id followed by connectivities in an array
        - geometry (Optional[:class:`stem.geometry.Geometry`]): geometry of the model part
        - parameters (Dict[Any,Any]): dictionary containing the model part parameters
    """
    def __init__(self, name: str):
        """
        Initialize the model part

        Args:
            - name (str): name of the model part
        """
        self.name: str = name
        self.geometry: Optional[Geometry] = None
        self.mesh: Optional[Mesh] = None
        self.parameters: Dict[Any, Any] = {} # todo define type

    def get_geometry_from_geo_data(self, geo_data: Dict[str, Any], name: str):
        """
        Get the geometry from the geo_data and set the nodes and elements attributes.

        Args:
            - geo_data (Dict[str, Any]): dictionary containing the geometry data as generated by the gmsh_io

        """

        self.geometry = Geometry.create_geometry_from_gmsh_group(geo_data, name)


class BodyModelPart(ModelPart):
    """
    This class contains model parts which are part of the body, e.g. a soil layer or track components.

    Inheritance:
        - :class:`ModelPart`

    Attributes:
        - name (str): name of the model part
        - nodes (None): node id followed by node coordinates in an array
        - elements (None): element id followed by connectivities in an array
        - conditions (None): condition id followed by connectivities in an array
        - parameters (Dict[str, Any]): dictionary containing the model part parameters
        - material (Union[:class:`stem.soil_material.SoilMaterial`, \
            :class:`stem.structural_material.StructuralMaterial`]): material of the model part
    """

    def __init__(self, name: str):
        """
        Initialize the body model part

        Args:
            - name (str): name of the body model part
        """
        super().__init__(name)

        self.material: Optional[Union[SoilMaterial, StructuralMaterial]] = None
