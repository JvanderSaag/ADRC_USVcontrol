import numpy as np

class PIDController:
    """
    PID Controller class with control signal saturation.

    Attributes:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        setpoint (float): Desired setpoint.
        max_control (float): Maximum control signal.
        min_control (float): Minimum control signal.
        prev_error (float): Previous error value.
        integral (float): Integral of the error.
    """

    def __init__(self, Kp, Ki, Kd, setpoint=0):
        """
        Initialize the PIDController with given parameters.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            setpoint (float): Desired setpoint.
            max_control (float): Maximum control signal.
            min_control (float): Minimum control signal.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def compute_control_signal(self, measurement, dt):
        # Calculate error
        error = self.setpoint - measurement

        # Proportional term
        P = self.Kp * error

        # Integral term
        self.integral += error * dt
        I = self.Ki * self.integral

        # Derivative term (set to 0 if first timestep)
        if dt == 0 or self.prev_error == 0:  # Skip derivative for the first step
            D = 0
        else:
            derivative = (error - self.prev_error) / dt
            D = self.Kd * derivative

        # Update error
        self.prev_error = error

        # Control signal before saturation
        control_signal = P + I + D

        return control_signal

