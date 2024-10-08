class SISOVesselSystem:
    """
    A class used to represent a Single Input Single Output (SISO) vessel system.

    Attributes:
        m11 (float): Mass of the vessel.
        X_u (float): Linear damping coefficient.
        X_absuu (float): Quadratic damping coefficient.
        position (float): Position of the vessel.
        velocity (float): Velocity of the vessel.
    """
    def __init__(self):
        """
        Constructs all the necessary attributes for the SISOVesselSystem object. 
        Example parameters are taken from "Modelling and Trajectory Planning for a Small-Scale Surface Ship by Steen & Zetterqvist
        """
        # Vessel parameters
        self.m11 = 17.06
        self.X_u = 0.2
        self.X_absuu = -0.79

        # Vessel state variables
        self._position = 0  # Position (x)
        self._velocity = 0  # Velocity (xdot)
    
    def step(self, control_input: float, disturbance: float, dt:float) -> float:
        """
        Updates the state of the vessel system for a given time step.
        
        Args:
            control_input (float): Control input to the system.
            disturbance (float): Disturbance acting on the system.
            dt (float): Time step duration.
        Returns:
            float: New position of the vessel after the time step.
        """
        # Update acceleration based on the control input and disturbance
        acceleration = 1/self.m11 * (-self.X_u * self._velocity + self.X_absuu * abs(self._velocity) * self._velocity + control_input + disturbance)

        # Update velocity and position using Euler integration
        self._velocity += acceleration * dt
        self._position += self._velocity * dt
        
        return self._position