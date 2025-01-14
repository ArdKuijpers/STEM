input_files_dir = "uvec_load"
results_dir = "output_uvec_load"

from stem.model import Model
from stem.soil_material import OnePhaseSoil, LinearElasticSoil, SoilMaterial, SaturatedBelowPhreaticLevelLaw
from stem.load import MovingLoad, UvecLoad
from stem.boundary import DisplacementConstraint
from stem.solver import AnalysisType, SolutionType, TimeIntegration, DisplacementConvergenceCriteria,\
     NewtonRaphsonStrategy, NewmarkScheme, Amgcl, StressInitialisationType, SolverSettings, Problem
from stem.output import NodalOutput, VtkOutputParameters, Output, GaussPointOutput
from stem.stem import Stem


ndim = 3
model = Model(ndim)

solid_density_1 = 2650
porosity_1 = 0.3
young_modulus_1 = 30e6
poisson_ratio_1 = 0.2
soil_formulation_1 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_1, POROSITY=porosity_1)
constitutive_law_1 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_1, POISSON_RATIO=poisson_ratio_1)
retention_parameters_1 = SaturatedBelowPhreaticLevelLaw()
material_soil_1 = SoilMaterial("soil_1", soil_formulation_1, constitutive_law_1, retention_parameters_1)


solid_density_2 = 2550
porosity_2 = 0.3
young_modulus_2 = 30e6
poisson_ratio_2 = 0.2
soil_formulation_2 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_2, POROSITY=porosity_2)
constitutive_law_2 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_2, POISSON_RATIO=poisson_ratio_2)
retention_parameters_2 = SaturatedBelowPhreaticLevelLaw()
material_soil_2 = SoilMaterial("soil_2", soil_formulation_2, constitutive_law_2, retention_parameters_2)


solid_density_3 = 2650
porosity_3 = 0.3
young_modulus_3 = 10e6
poisson_ratio_3 = 0.2
soil_formulation_3 = OnePhaseSoil(ndim, IS_DRAINED=True, DENSITY_SOLID=solid_density_3, POROSITY=porosity_3)
constitutive_law_3 = LinearElasticSoil(YOUNG_MODULUS=young_modulus_3, POISSON_RATIO=poisson_ratio_3)
retention_parameters_3 = SaturatedBelowPhreaticLevelLaw()
material_embankment = SoilMaterial("embankment", soil_formulation_3, constitutive_law_3, retention_parameters_3)


soil1_coordinates = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
soil2_coordinates = [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
embankment_coordinates = [(0.0, 2.0, 0.0), (3.0, 2.0, 0.0), (1.5, 3.0, 0.0), (0.75, 3.0, 0.0), (0, 3.0, 0.0)]
model.extrusion_length = 50


model.add_soil_layer_by_coordinates(soil1_coordinates, material_soil_1, "soil_layer_1")
model.add_soil_layer_by_coordinates(soil2_coordinates, material_soil_2, "soil_layer_2")
model.add_soil_layer_by_coordinates(embankment_coordinates, material_embankment, "embankment_layer")


load_coordinates = [(0.75, 3.0, 0.0), (0.75, 3.0, 50.0)]
# moving_load = MovingLoad(load=[0.0, -10000.0, 0.0], direction=[1, 1, 1], velocity=30, origin=[0.75, 3.0, 0.0],
                         # offset=0.0)
# model.add_load_by_coordinates(load_coordinates, moving_load, "moving_load")

uvec_parameters = {"n_carts":1,
                   "cart_inertia": (1128.8e3)/2,
                   "cart_mass": (50e3)/2,
                   "cart_stiffness": 2708e3,
                   "cart_damping": 64e3,
                   "bogie_distances": [-9.95, 9.95],
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

uvec_load = UvecLoad(direction=[1, 1, 1], velocity=5, origin=[0.75, 3.0, 0.0], wheel_configuration=[2, 4.5, 4.5+9.95*2-2.5, 4.5+9.95*2+2.5],
                    uvec_file=r"C:\Users\zuada\STEM\uvec_ten_dof_vehicle_2D\uvec.py", uvec_function_name="uvec",uvec_parameters=uvec_parameters)


model.add_load_by_coordinates(load_coordinates, uvec_load, "uvec_load")

model.synchronise_geometry()

# model.show_geometry(show_surface_ids=True)



no_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                    is_fixed=[True, True, True], value=[0, 0, 0])
roller_displacement_parameters = DisplacementConstraint(active=[True, True, True],
                                                        is_fixed=[True, False, True], value=[0, 0, 0])

model.add_boundary_condition_by_geometry_ids(2, [1], no_displacement_parameters, "base_fixed")
model.add_boundary_condition_by_geometry_ids(2, [2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17],
                                             roller_displacement_parameters, "sides_roller")


model.set_mesh_size(element_size=1.0)



analysis_type = AnalysisType.MECHANICAL
solution_type = SolutionType.DYNAMIC
# Set up start and end time of calculation, time step and etc
time_integration = TimeIntegration(start_time=0.0, end_time=1.4999999, delta_time=0.001, reduction_factor=1.0,
                                   increase_factor=1.0)
convergence_criterion = DisplacementConvergenceCriteria(displacement_relative_tolerance=1.0e-4,
                                                        displacement_absolute_tolerance=1.0e-9)
strategy_type = NewtonRaphsonStrategy()
scheme_type = NewmarkScheme()
linear_solver_settings = Amgcl()
stress_initialisation_type = StressInitialisationType.NONE
solver_settings = SolverSettings(analysis_type=analysis_type, solution_type=solution_type,
                                 stress_initialisation_type=stress_initialisation_type,
                                 time_integration=time_integration,
                                 is_stiffness_matrix_constant=False, are_mass_and_damping_constant=False,
                                 convergence_criteria=convergence_criterion,
                                 strategy_type=strategy_type, scheme=scheme_type,
                                 linear_solver_settings=linear_solver_settings, rayleigh_k=0.12,
                                 rayleigh_m=0.0001)

# Set up problem data
problem = Problem(problem_name="calculate_moving_load_on_embankment_3d", number_of_threads=4,
                  settings=solver_settings)

model.project_parameters = problem


nodal_results = [NodalOutput.DISPLACEMENT, NodalOutput.VELOCITY, NodalOutput.ACCELERATION]
gauss_point_results = [GaussPointOutput.CAUCHY_STRESS_VECTOR]

vtk_output_process = Output(
    part_name="porous_computational_model_part",
    output_name="vtk_output",
    output_dir="output",
    output_parameters=VtkOutputParameters(
        output_interval=10,
        nodal_results=nodal_results,
        gauss_point_results=gauss_point_results,
        output_control_type="step"
   )
)
model.output_settings = [vtk_output_process]

stem = Stem(model, input_files_dir)

stem.write_all_input_files()

stem.run_calculation()