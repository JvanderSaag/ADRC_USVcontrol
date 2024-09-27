import numpy as np

class ADRCController:
    """
    Active Disturbance Rejection Controller (ADRC) class. Based on "The Control and Simulation for the ADRC of USV" by Chen, Xiong and Fu (2024)

    Attributes:
        setpoint (float): Controller setpoint.
        h0 (float): Filter factor for the tracking differentiator.
        r0 (float): Tracking speed for the tracking differentiator.
        b (float): Control coefficient.
        beta01 (float): Parameter for the Extended State Observer (ESO).
        beta02 (float): Parameter for the Extended State Observer (ESO).
        beta03 (float): Parameter for the Extended State Observer (ESO).
        delta (float): Parameter for the Extended State Observer (ESO).
        k1 (float): Parameter for the Nonlinear State Error Feedback (NLSEF).
        k2 (float): Parameter for the Nonlinear State Error Feedback (NLSEF).
        alpha1 (float): Parameter for the Nonlinear State Error Feedback (NLSEF).
        alpha2 (float): Parameter for the Nonlinear State Error Feedback (NLSEF).
    """

    def __init__(self, setpoint=0, h0=0.001, r0=1, b=5, beta01=0, beta02=0, beta03=0, delta=0.0001, k1=0, k2=0, alpha1=0, alpha2=0):
        # Controller setpoint
        self.setpoint = setpoint

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
        self.delta = delta
        self.b = b  # Control coefficient
        
        # Nonlinear State Error Feedback (NLSEF)
        self.k1 = k1
        self.k2 = k2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
    def _update_td(self, x, h):
        """
        Update the tracking differentiator. 

        Args:
            x (float): Input x.
            xdot (float): Derivative of x.
            h (float): Parameter h.

        Returns:
            None
        """
        x1_prime = self._x1 + h * self._x2
        x2_prime = self._x2 + h * self._fst(self._x1 - x, self._x2, self.r0, self.h0) 
        
        return x1_prime, x2_prime

    def _update_eso(self, y, u):
        """
        Update the Extended State Observer (ESO).

        Args:
            y (float): Output of plant.
            u (float): Control signal.
        """
        e = self._z1 - y
        z1_prime = self._z2 - self.beta01 * e
        z2_prime = self._z3 - self.beta02 * self._fal(e, 0.5, self.delta) + self.b * u
        z3_prime = -self.beta03 * self._fal(e, 0.25, self.delta)

        return z1_prime, z2_prime, z3_prime
    
    def _update_nlsef(self):
        """
        Update the Nonlinear State Error Feedback (NLSEF).

        Args:
            x (float): Input x.
            xdot (float): Derivative of x.
            h (float): Parameter h.

        Returns:
            float: Output of the NLSEF.
        """
        e1 = self._x1 - self._z1
        e2 = self._x2 - self._z2
        u0 = self.k1 * self._fal(e1, self.alpha1, self.delta) + self.k2 * self._fal(e2, self.alpha2, self.delta)
        u = u0 - self._z3 / self.b

        return u

    def _fst(self, x1, x2, r0, h0):
        """
        Compute the fastest synthesis function.

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
        alpha0 = np.sqrt(d**2 + 8 * r0 * abs(y))

        if abs(y) > d0:
            alpha = x2 + 0.5 * ((alpha0 - d) * np.sign(y))
        elif abs(y) <= d0:
            alpha = x2 + y / h0

        if abs(alpha) <= d:
            fst = -r0 * alpha / d
        elif abs(alpha) > d:
            fst = -r0 * np.sign(alpha)
        return fst

    def _fal(self, e, alpha, delta):
        """
        Compute the fastest linear function.
        
        Args:
            e (float): Error signal.
            alpha (float): Parameter alpha.
            delta (float): Parameter delta.
            
        Returns:
            float: Output of the fal function.
        """
        if abs(e) > delta:
            fal = abs(e) ** alpha * np.sign(e)
        elif abs(e) <= delta:
            fal = e / (delta ** (1 - alpha))
        return fal
    
    def compute_control_signal(self, measurement, dt):
        """
        Compute the control signal based on the measurement and time step.

        Args:
            measurement (float): Current state measurement.
            dt (float): Time step.

        Returns:
            float: Control signal after applying ADRC control and saturation.
        """
        self._x1, self._x2 = self._update_td(self.setpoint, dt) # Update tracking differentiator to get the new state estimates
        u = self._update_nlsef() # Compute the control signal using the NLSEF. 
        self._z1, self._z2, self._z3 = self._update_eso(measurement, u) # Update the ESO with the output of the plant and control signal
       
        return u