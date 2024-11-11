from controllers.ADRC import ADRCController
class PIDController:
    """
    Proportional-Integral-Derivative (PID) Controller class with control signal saturation.

    Attributes:
        Public:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        setpoint (float): Desired setpoint.
        max_u (float): Maximum control signal limit.
        min_u (float): Minimum control signal limit.
        
        Private:
        _prev_error (float): Previous error for derivative term.
        _integral (float): Integral term.
    """

    def __init__(self, Kp: float, Ki: float, Kd: float, setpoint: float = 0, tau: float = 0):
        """
        Initialize the PIDController with given parameters.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            setpoint (float): Desired setpoint.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.tau = tau

        self._prev_error = 0.0
        self._integral = 0.0
        self._comp_x1 = 0.0
        self._comp_x2 = 0.0

    def compute_control_signal(self, measurement: float, dt: float) -> float:
        """
        Compute the control signal based on the measurement and time step.

        Args:
            measurement (float): Current measurement of the system.
            dt (float): Time step duration.

        Returns:
            float: Control signal after applying PID control and saturation.
        """
        # Calculate error
        error = self.setpoint - measurement

        # Proportional term
        P = self.Kp * error

        # Integral term
        self._integral += error * dt
        I = self.Ki * self._integral

        # Derivative term (set to 0 if first timestep)
        if dt == 0:
            D = 0
        else:
            derivative = (error - self._prev_error) / dt
            D = self.Kd * derivative

        # Update error
        self._prev_error = error

        # Control signal before saturation
        control_signal = P + I + D
        
        # Compensator block, derivative determined with another tracking differentiator
        if not self.tau == 0:
            # compensated_control_signal = control_signal + (self.tau / dt) * (control_signal - self._last_control_signal)
            comp_x1 = self._comp_x1 + dt * self._comp_x2
            comp_x2 = self._comp_x2 + dt * ADRCController._fst(self._comp_x1 - control_signal, self._comp_x2, 1e5, dt)
            self._comp_x1, self._comp_x2 = comp_x1, comp_x2
            compensated_control_signal = comp_x1 + self.tau * comp_x2
        else:
            compensated_control_signal = control_signal

        return compensated_control_signal
