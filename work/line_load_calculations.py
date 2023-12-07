import os
import numpy as np
from enum import Enum

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.structural_material import EulerBeam, ElasticSpringDamper, NodalConcentrated

from stem.load import LineLoad, MovingLoad, UvecLoad
from stem.boundary import DisplacementConstraint, AbsorbingBoundary
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
     NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, JsonOutputParameters
from stem.stem import Stem

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

basedir = os.path.dirname(__file__)

#%matplotlib widget

class Loadcase(Enum):
    Static = 1,
    Moving = 2
    UVEC = 3

def run_stem(loadcase: Loadcase, 
             t_step = 0.01,
             t_end = 0.1, 
             output_time_interval = 0.02,
             thickness_bottom = 1,
             thickness_top = 1,
             thickness_embankement = 0.5,
             soil_width = 20,
             use_track = True
             ):
    logger.info("Set dimensions and create model")
    model = Model(ndim=3)

    case_description = f"{loadcase.name.lower()}_load"

    logger.info("Setup soil model...")

    #common
    poisson_ratio = 0.2
    porosity = 0.3

    bottom_density = 2650
    bottom_young_modulus = 30e6
    material_bottom = SoilMaterial(
        "bottom_layer_material", 
        OnePhaseSoil(model.ndim, IS_DRAINED=True, DENSITY_SOLID=bottom_density, POROSITY=porosity),
        LinearElasticSoil(YOUNG_MODULUS=bottom_young_modulus, POISSON_RATIO=poisson_ratio),
        SaturatedBelowPhreaticLevelLaw())

    top_density = 2550
    top_young_modulus = 30e6
    material_top = SoilMaterial(
        "top_layer_material", 
        OnePhaseSoil(model.ndim, IS_DRAINED=True, DENSITY_SOLID=top_density, POROSITY=porosity), 
        LinearElasticSoil(YOUNG_MODULUS=top_young_modulus, POISSON_RATIO=poisson_ratio),
        SaturatedBelowPhreaticLevelLaw())

    embankement_density = 2650
    embankement_young_modulus = 10e6
    material_embankment = SoilMaterial(
        "embankment", 
        OnePhaseSoil(model.ndim, IS_DRAINED=True, DENSITY_SOLID=embankement_density, POROSITY=porosity),
        LinearElasticSoil(YOUNG_MODULUS=embankement_young_modulus, POISSON_RATIO=poisson_ratio),
        SaturatedBelowPhreaticLevelLaw())

    logger.info("...Adding layers...")
    # Parameterized two-layer soil with embankement
    z_start = 0
    z_end = 50
    z_mid = (z_start+z_end)/2
    x_soil = soil_width

    y_base = 0
    y_bottom = thickness_bottom 
    y_top = y_bottom + thickness_top
    y_embankement = y_top + thickness_embankement
    x_embankement = 1.5
    x_max_embankement = x_embankement + thickness_embankement
    x_load = x_embankement / 2

    bottom_layer_coordinates = [(0.0, y_base, z_start),   (x_soil, y_base, z_start),   (x_soil, y_bottom, z_start),      (0.0, y_bottom, z_start)]
    top_layer_coordinates    = [(0.0, y_bottom, z_start), (x_soil, y_bottom, z_start), (x_soil, y_top, z_start),         (0.0, y_top, z_start)]
    embankment_coordinates   = [(0.0, y_top, z_start),    (x_max_embankement, y_top, z_start),  (x_embankement, y_embankement, z_start), (0.0, y_embankement, z_start)]
    model.extrusion_length = z_end - z_start

    model.add_soil_layer_by_coordinates(bottom_layer_coordinates, material_bottom, "bottom_layer")
    model.add_soil_layer_by_coordinates(top_layer_coordinates, material_top, "top_layer")
    model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment_layer")

    # add the track
    rail_parameters = EulerBeam(ndim=model.ndim, YOUNG_MODULUS=30e9, POISSON_RATIO=0.2,
                                DENSITY=7200, CROSS_AREA=0.01, I33=1e-4, I22=1e-4, TORSIONAL_INERTIA=2e-4)
    rail_pad_parameters = ElasticSpringDamper(NODAL_DISPLACEMENT_STIFFNESS=[1, 1, 1],
                                            NODAL_ROTATIONAL_STIFFNESS=[1, 1, 1],
                                            NODAL_DAMPING_COEFFICIENT=[1, 1, 1],
                                            NODAL_ROTATIONAL_DAMPING_COEFFICIENT=[1, 1, 1])
    sleeper_parameters = NodalConcentrated(NODAL_DISPLACEMENT_STIFFNESS=[0, 0, 0],
                                        NODAL_MASS=1,
                                        NODAL_DAMPING_COEFFICIENT=[0, 0, 0])

    # create a straight track with rails, sleepers and rail pads
    sleeper_spacing = 0.5
    number_of_sleepers = float(z_end - z_start)/sleeper_spacing + 1
    if number_of_sleepers != round(number_of_sleepers):
        raise ValueError("track line needs to start and end on a sleeper")
    else:
        number_of_sleepers = int(number_of_sleepers)

    load_origin_point = [x_load, y_embankement, z_start]
    model.generate_straight_track(sleeper_spacing, 
                                number_of_sleepers, 
                                rail_parameters,
                                sleeper_parameters, 
                                rail_pad_parameters, 
                                origin_point=load_origin_point, 
                                direction_vector=[0,0,1],
                                name="rail_track")

    # model.synchronise_geometry() #
    # model.show_geometry(show_surface_ids=True)

    static_load = LineLoad(active=[False, True, False], value=[0, -1000, 0])
    moving_load = MovingLoad(load=[0.0, -10000.0, 0.0], 
                             direction=[1, 1, 1], 
                             velocity=30, origin=load_origin_point,
                             offset=0.0)
    
    bogie_spacing = 2*9.95
    uvec_parameters = {
        "n_carts":1,
        "cart_inertia": (1128.8e3)/2,
        "cart_mass": (50e3)/2,
        "cart_stiffness": 2708e3,
        "cart_damping": 64e3,
        "bogie_distances": [-bogie_spacing/2, bogie_spacing/2],
        "bogie_inertia": (0.31e3)/2,
        "bogie_mass": (6e3)/2,
        "wheel_distances": [-1.25, 1.25],
        "wheel_mass": 1.5e3,
        "wheel_stiffness": 4800e3,
        "wheel_damping": 0.25e3,
        "gravity_axis": 1,
        "contact_coefficient": 9.1e-5,
        "contact_power": 1.5,
        "initialisation_steps": 100,
        }

    wheel_spacing = 2.5
    z_wheel_start = 2
    uvec_load = UvecLoad(direction=[1, 1, 1], velocity=5, origin=[x_load, y_embankement, z_start], 
                         wheel_configuration=[z_wheel_start, 
                                              z_wheel_start+wheel_spacing, 
                                              z_wheel_start+bogie_spacing,
                                              z_wheel_start+wheel_spacing+bogie_spacing],
                         uvec_file=os.path.join(basedir, "uvec_ten_dof_vehicle_2D", "uvec.py"), uvec_function_name="uvec", uvec_parameters=uvec_parameters)

    match loadcase:
        case Loadcase.Static:
            load = static_load            
        case Loadcase.Moving:    
            load = moving_load
        case Loadcase.UVEC:
            load = uvec_load

    if use_track:
        model.add_load_on_line_model_part("rail_track", load, f"{case_description}_track")
    else:
        #load_path = [(x_load, y_embankement, z_start), (0.0, y_embankement, z_mid), (x_load, y_embankement, z_end)]
        load_path = [(x_load, y_embankement, z_start), (x_load, y_embankement, z_end)]
        model.add_load_by_coordinates(load_path, load, f"{case_description}_line")

    logger.info("...Adding boundary conditions...")

    no_displacement = DisplacementConstraint(active=[True, True, True],
        is_fixed=[True, True, True], value=[0, 0, 0])
    roller_displacement = DisplacementConstraint(active=[True, True, True],
        is_fixed=[True, False, True], value=[0, 0, 0])
    absorbing = AbsorbingBoundary(absorbing_factors=[1,1], virtual_thickness=10)

    surface_bc=2
    model.add_boundary_condition_by_geometry_ids(surface_bc, [1], 
                                                 no_displacement, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(surface_bc, [2, 5, 6, 7, 11, 12, 16, 17],
                                                absorbing, "outersides_absorbing")
    model.add_boundary_condition_by_geometry_ids(surface_bc, [4, 10, 15],
                                                roller_displacement, "innersides_roller")

    logger.info("...Setup analysis parameters...")
    model.set_mesh_size(element_size=1.0)
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(
        start_time=0.0, 
        end_time=t_end, 
        delta_time=t_step, 
        reduction_factor=1.0, # fixed time step
        increase_factor=1.0) # fixed time step

    solver_settings = SolverSettings(
        analysis_type=AnalysisType.MECHANICAL, solution_type=SolutionType.DYNAMIC,
        stress_initialisation_type=StressInitialisationType.NONE,
        time_integration=time_integration,
        is_stiffness_matrix_constant=False, are_mass_and_damping_constant=False,
        convergence_criteria=DisplacementConvergenceCriteria(
            displacement_relative_tolerance=1.0e-4,
            displacement_absolute_tolerance=1.0e-9),
        strategy_type=NewtonRaphsonStrategy(), 
        scheme=NewmarkScheme(),
        linear_solver_settings=Amgcl(),
        rayleigh_k=0.12, #  Rayleigh damping, stiffness
        rayleigh_m=0.0001, #  Rayleigh damping, mass
        )

    # Set up problem data
    model.project_parameters = Problem(
        problem_name=f"calculate_{case_description}_on_embankment_3d", 
        number_of_threads=8,
        settings=solver_settings)

    # Set postprocessing parameters
    model.add_output_settings(
        part_name="porous_computational_model_part",
        output_name="output_vtk",
        output_parameters=VtkOutputParameters(
            output_interval=np.floor(output_time_interval / t_step), # NOTE: this is not a time interval but specifies to generate output each N steps
            # output_interval=output_time_interval,
            nodal_results=[NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION],
            output_control_type="step")
    )

    surface_following_line = [
        # (0,                y_embankement, z_mid), 
        # (x_load,           y_embankement, z_mid), 
        (x_embankement,    y_embankement, z_mid)]
    x_step = 2
    x_output = x_max_embankement
    while x_output < x_soil:
        surface_following_line.append((x_output, y_top, z_mid))
    surface_following_line.append((x_soil, y_top, z_mid))

    output_coordinates_part_name = "output_coordinates_part"
    model.add_output_settings_by_coordinates(     
        coordinates=surface_following_line,
        part_name=output_coordinates_part_name,
        output_name="output_coordinates",
        output_parameters=JsonOutputParameters(
            output_interval=output_time_interval, 
            nodal_results=[NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY]
        )
    )

    model.synchronise_geometry() #
    model.show_geometry(show_surface_ids=True)
    stem = Stem(model, ensure_tmp_work_folder(case_description))
    stem.write_all_input_files()
    stem.run_calculation()
    output_nodes_dict = [mp.mesh.nodes for mp in model.process_model_parts if mp.name == output_coordinates_part_name][0]

def ensure_tmp_work_folder(folder_name: str):
    folder = os.path.join("work", "tmp", folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder

# logger.info("Run static load analysis with STEM...")
# run_stem(Loadcase.Static, t_end=0.1)   

logger.info("Run dynamic load analysis with STEM...")
run_stem(Loadcase.UVEC, t_step=0.002, t_end=1.9999)   

logger.info("Done")

