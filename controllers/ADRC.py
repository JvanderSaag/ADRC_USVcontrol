import numpy as np

class ADRCController:
    """
    Active Disturbance Rejection Controller (ADRC) class. Based on "The Control and Simulation for the ADRC of USV" by Chen, Xiong, and Fu (2024) 
    and "From PID to ADRC" by Han, J. (2009).

    Attributes:
        setpoint (float): Controller setpoint.
        h0 (float): Filter factor for the tracking differentiator.
        r0 (float): Tracking speed for the tracking differentiator.
        b0 (float): Control coefficient.
        beta01 (float): Parameter for the Extended State Observer (ESO).
        beta02 (float): Parameter for the Extended State Observer (ESO).
        beta03 (float): Parameter for the Extended State Observer (ESO).
        k1 (float): Parameter for the Nonlinear State Error Feedback (NLSEF).
        k2 (float): Parameter for the Nonlinear State Error Feedback (NLSEF).
        alpha1 (float): Parameter for the Nonlinear State Error Feedback (NLSEF).
        alpha2 (float): Parameter for the Nonlinear State Error Feedback (NLSEF).
    """

    def __init__(self, setpoint=0, h0=0.001, r0=1, b0=5, beta01=0, beta02=0, beta03=0, k1=0, k2=0, alpha1=0, alpha2=0):
        # Controller setpoint
        self.setpoint = setpoint
        self.h = 0  # Timestep

        # Tracking differentiator (TD)
        self._x1 = 0  # State estimate
        self._x2 = 0  # State derivative estimate
        self.h0 = h0  # Filter factor
        self.r0 = r0  # Tracking speed

        # Extended State Observer (ESO)
        self._z1 = 0  # State estimate
        self._z2 = 0  # State derivative estimate
        self._z3 = 0  # Disturbance estimate
        self.beta01 = beta01
        self.beta02 = beta02
        self.beta03 = beta03
        self.b0 = b0  # Control coefficient

        # Nonlinear State Error Feedback (NLSEF)
        self.k1 = k1
        self.k2 = k2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        # Remember last control signal to compute ESO
        self._last_control_signal = 0

    def _update_tracking_differentiator(self, x: float) -> None:
        """
        Update the tracking differentiator.

        Args:
            x (float): Input signal (setpoint).
        """
        x1_next = self._x1 + self.h * self._x2
        x2_next = self._x2 + self.h * self._fst(self._x1 - x, self._x2, self.r0, self.h0)

        self._x1 = x1_next
        self._x2 = x2_next

    def _update_extended_state_observer(self, y: float, u: float) -> None:
        """
        Update the Extended State Observer (ESO).

        Args:
            y (float): Output of the plant (current measurement).
            u (float): Control signal.
        """
        e = self._z1 - y

        z1_next = self._z1 + self.h * self._z2 - self.beta01 * e
        z2_next = self._z2 + self.h * (self._z3 + self.b0 * u) - self.beta02 * self._fal(e, 0.5, self.h)
        z3_next = self._z3 - self.beta03 * self._fal(e, 0.25, self.h)

        self._z1 = z1_next
        self._z2 = z2_next
        self._z3 = z3_next

    def _compute_nlsef(self) -> float:
        """
        Compute the Nonlinear State Error Feedback (NLSEF).

        Returns:
            float: Control signal after nonlinear state error feedback.
        """
        e1 = self._x1 - self._z1
        e2 = self._x2 - self._z2
        u0 = self.k1 * self._fal(e1, self.alpha1, self.h) + self.k2 * self._fal(e2, self.alpha2, self.h)
        u = u0 - self._z3 / self.b0

        return u

    def _fst(self, x1: float, x2: float, r0: float, h0: float) -> float:
        """
        Fastest synthesis function.

        Args:
            x1 (float): State input.
            x2 (float): State derivative input.
            r0 (float): Tracking speed.
            h0 (float): Filter factor.

        Returns:
            float: Output of the fst function.
        """
        d = r0 * h0
        d0 = d * h0

        y = x1 + h0 * x2
        alpha0 = np.sqrt(d ** 2 + 8 * r0 * abs(y))

        if abs(y) > d0:
            alpha = x2 + 0.5 * ((alpha0 - d) * np.sign(y))
        else:
            alpha = x2 + y / h0

        if abs(alpha) <= d:
            return -r0 * alpha / d
        else:
            return -r0 * np.sign(alpha)

    def _fal(self, e: float, alpha: float, delta: float) -> float:
        """
        Fastest linear function (nonlinear function in ESO/NLSEF).

        Args:
            e (float): Error signal.
            alpha (float): Nonlinear coefficient.
            delta (float): Small threshold for linearization.

        Returns:
            float: Output of the fal function.
        """
        if abs(e) > delta:
            return abs(e) ** alpha * np.sign(e)
        else:
            return e / (delta ** (1 - alpha))

    def compute_control_signal(self, measurement: float, dt: float) -> float:
        """
        Compute the control signal based on the measurement and time step.

        Args:
            measurement (float): Current measurement of the system.
            dt (float): Time step duration.

        Returns:
            float: Control signal after applying ADRC.
        """
        self.h = dt  # Update timestep
        self._update_extended_state_observer(measurement, self._last_control_signal)  # Update ESO with the current measurement and last control signal

        self._update_tracking_differentiator(self.setpoint)  # Update tracking differentiator
        control_signal = self._compute_nlsef()  # Compute the control signal using NLSEF

        self._last_control_signal = control_signal  # Save the last control signal for the next ESO update

        return control_signal

