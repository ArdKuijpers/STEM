import os

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import LineLoad, MovingLoad
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
     NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, Output
from stem.stem import Stem

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#%matplotlib widget


def get_model(is_moving_load: bool, t_end = 0.1):
    logger.info("Set dimensions and create model")
    model = Model(ndim=3)

    logger.info("Setup soil model...")

    #common
    poisson_ratio = 0.2
    porosity = 0.3

    solid_density_1 = 2650
    young_modulus_1 = 30e6
    material_soil_1 = SoilMaterial(
        "soil_1", 
        OnePhaseSoil(model.ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_1, POROSITY=porosity),
        LinearElasticSoil(YOUNG_MODULUS=young_modulus_1, POISSON_RATIO=poisson_ratio),
        SaturatedBelowPhreaticLevelLaw())

    solid_density_2 = 2550
    young_modulus_2 = 30e6
    material_soil_2 = SoilMaterial(
        "soil_2", 
        OnePhaseSoil(model.ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_2, POROSITY=porosity), 
        LinearElasticSoil(YOUNG_MODULUS=young_modulus_2, POISSON_RATIO=poisson_ratio),
        SaturatedBelowPhreaticLevelLaw())

    solid_density_3 = 2650
    young_modulus_3 = 10e6
    material_embankment = SoilMaterial(
        "embankment", 
        OnePhaseSoil(model.ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_3, POROSITY=porosity),
        LinearElasticSoil(YOUNG_MODULUS=young_modulus_3, POISSON_RATIO=poisson_ratio),
        SaturatedBelowPhreaticLevelLaw())

    logger.info("...Adding layers...")
    # z-coordinate = lengterichting
    z_start = 0
    z_end = 50
    soil1_coordinates = [(0.0, 0.0, z_start), (5.0, 0.0, z_start), (5.0, 1.0, z_start), (0.0, 1.0, z_start)]
    soil2_coordinates = [(0.0, 1.0, z_start), (5.0, 1.0, z_start), (5.0, 2.0, z_start), (0.0, 2.0, z_start)]
    embankment_coordinates = [(0.0, 2.0, z_start), (3.0, 2.0, z_start), (1.5, 3.0, z_start), (0.75, 3.0, z_start), (0, 3.0, z_start)]
    model.extrusion_length = z_end - z_start

    model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil_1, "soil_layer_1")
    model.add_soil_layer_by_coordinates(soil2_coordinates, material_soil_2, "soil_layer_2")
    model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment_layer")

    load_coordinates = [(0.75, 3.0, z_start), (0.75, 3.0, z_end)]
    static_load = LineLoad(active=[False, True, False], value=[0, -1000, 0])
    ## VRAAG Is het  niet logischer om de load alleen in z richting te laten verplaatsen (langs spoorlijn)
    # moving_load = MovingLoad(load=[0.0, -10000.0, 0.0], 
    #                          direction=[1, 1, 1], 
    #                          velocity=30, origin=[0.75, 3.0, 0.0],
    #                          offset=0.0)
    moving_load = MovingLoad(load=[0.0, -10000.0, 0.0], 
                             direction=[0, 0, 1], 
                             velocity=30, origin=[0.75, 3.0, 0.0],
                             offset=0.0)
    if is_moving_load:
        model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")
    else:
        model.add_load_by_coordinates(load_coordinates, static_load, "line_load")

    model.synchronise_geometry()
    model.show_geometry(show_surface_ids=True)

    ## VRAAG: hoe krijg ik afbeelding met gelijkwaardige assen?

    logger.info("...Adding boundary conditions...")

    no_displacement = DisplacementConstraint(active=[True, True, True],
                                            is_fixed=[True, True, True], value=[0, 0, 0])
    roller_displacement = DisplacementConstraint(active=[True, True, True],
                                                is_fixed=[True, False, True], value=[0, 0, 0])

    onderzijde_id = 1

    model.add_boundary_condition_by_geometry_ids(2, [onderzijde_id], no_displacement, "base_fixed")
    model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17],
                                                roller_displacement, "sides_roller")

    logger.info("...Setup analysis parameters...")
    model.set_mesh_size(element_size=1.0)
    # Set up start and end time of calculation, time step and etc
    time_integration = TimeIntegration(
        start_time=0.0, 
        end_time=t_end, 
        delta_time=0.01, 
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
    problem_name = "calculate_moving_load_on_embankment_3d" if is_moving_load else "calculate_static_load_on_embankment_3d"

    model.project_parameters = Problem(
        problem_name=problem_name, 
        number_of_threads=1,
        settings=solver_settings)

    # Set postprocessing parameters
    output = Output(
        part_name="porous_computational_model_part",
        output_name="vtk_output",
        output_dir="output",
        output_parameters=VtkOutputParameters(
            output_interval=1,
            nodal_results=[NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION],
            gauss_point_results=[],
            output_control_type="step")
    )
    model.output_settings = [output]
    return model

def ensure_tmp_work_folder(folder_name: str):
    folder = os.path.join("work", "tmp", folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder


# logger.info("Run static load analysis with STEM...")
# model = get_model(is_moving_load=False, )   
# stem = Stem(model, ensure_tmp_work_folder("static_load"))
# stem.write_all_input_files()
# stem.run_calculation()

logger.info("Run dynamic load analysis with STEM...")
model = get_model(is_moving_load=True, t_end=2)   
stem = Stem(model, ensure_tmp_work_folder("dynamic_load"))
stem.write_all_input_files()
stem.run_calculation()

logger.info("Done")

