import numpy as np
import numpy.typing as npt


def cantilever_hinge(x: npt.NDArray[np.float64], L: float, E: float, I: float, k: float,
                     V: float) -> npt.NDArray[np.float64]:
    r"""
    Calculate the displacement of a cantilever beam with a hinge at the end, following the deflection curve.
    All units are SI units.

    /|                   ! V
    /|-------------------o
    /|

     |--> x

    Args:
        - x (npt.NDArray): The discretisation of the beam [m]
        - L (float): The length of the beam [m]
        - E (float): The Young's modulus [Pa]
        - I (float): The second moment of inertia [m^4]
        - k (float): The spring constant [Nm/rad]
        - V (float): The applied force [N]
    Returns:
        - np.ndarray: The displacement at each x value
    """

    # Calculate the constants
    B = -((V * k * L**2) /
          (4 * E * I * k * L + 4 * E * I**2)) - (2 * E * I * V * L) / (4 * E * I * k * L + 4 * E * I**2)
    A = V / (6 * E * I)

    # displacement
    u = A * x**3 + B * x**2
    return u


if __name__ == "__main__":
    L = 20
    E = 200
    I = 1
    k = 20000
    V = 20
    x = np.linspace(0, L, 100)
    disp = cantilever_hinge(x, L, E, I, k, V)
    import matplotlib.pyplot as plt
    plt.plot(x, disp)
    plt.show()
