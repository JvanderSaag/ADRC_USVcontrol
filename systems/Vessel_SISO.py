class SISOVesselSystem:
    """
    A class to represent a Single Input Single Output (SISO) USV system.  
    Considers a typical surge motion model for a USV, removing coupling effects from other degrees of freedom.
    """
    
    def __init__(self):
        """
        Constructs all the necessary attributes for the SISOVesselSystem object. 
        Example parameters are taken from "Modelling and Trajectory Planning for a Small-Scale Surface Ship by Steen & Zetterqvist
        
        Parameters:
        -----------
        mass : float, optional
            Mass of the vessel (default is 1.0).
        damping_coefficient : float, optional
            Damping coefficient of the vessel (default is 0.1).
        """
        # Vessel parameters
        self.m11 = 17.06
        self.X_u = 0.2
        self.X_absuu = -0.79

        # Vessel state variables
        self.position = 0  # Position (x)
        self.velocity = 0  # Velocity (xdot)
    
    def step(self, control_input, disturbance, dt):
        """
        Updates the state of the vessel system for a given time step.
        
        Parameters:
        -----------
        control_input : float
            The control input applied to the vessel.
        disturbance : float
            The external disturbance acting on the vessel.
        dt : float
            The time step for the simulation.
        
        Returns:
        --------
        float
            The updated position of the vessel.
        """
        # Update acceleration based on the control input and disturbance
        acceleration = 1/self.m11 * (-self.X_u * self.velocity + self.X_absuu * abs(self.velocity) * self.velocity + control_input + disturbance)
        
        # Update velocity and position using Euler integration
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        return self.position