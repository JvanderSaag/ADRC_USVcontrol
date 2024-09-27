import matplotlib.pyplot as plt
import numpy as np

from controllers.PID import PIDController
from controllers.ADRC import ADRCController
from systems.Vessel_SISO import SISOVesselSystem

def calculate_mse(reference, signal):
    """Calculate the Mean Squared Error (MSE) between the reference and signal."""
    return np.mean((reference - signal) ** 2)

def main():
    # Simulation parameters
    time_end = 180  # Simulation time in seconds
    dt = 0.01      # Time step
    num_steps = int(time_end / dt)
    time = np.linspace(0, time_end, num_steps)
    
    # Step input: initial setpoint is 0, then steps to 1 after 10 seconds
    setpoint = np.ones(num_steps)  # Create an array of ones
    setpoint[:int(10 / dt)] = 0    # Set the first 10 seconds to 0 (before the step)

    np.random.seed(42)  # Set random seed for reproducibility

    # System and controller initialization
    ship1 = SISOVesselSystem()
    ship2 = SISOVesselSystem()
    pid = PIDController(Kp=5, Ki=1, Kd=15)  # Setpoint of 1 for the controller
    adrc = ADRCController(h0=0.1, r0=1, b=1, beta01=.20, beta02=.25, beta03=.20, delta=0.05, k1=0.1, k2=0.1, alpha1=0.75, alpha2=1.4)

    # Arrays to store results
    position_pid = np.zeros(num_steps)
    position_adrc = np.zeros(num_steps)
    pid_control_signal = np.zeros(num_steps)
    adrc_control_signal = np.zeros(num_steps)
    disturbance = np.zeros(num_steps)

    # Probability of applying a disturbance
    disturbance_probability = 0.001

    # Run the simulation
    disturbance_until_time = 0  # Disturbance applied until this time
    for i in range(1, num_steps):
        # Determine if a disturbance should be applied (5% chance)
        if np.random.rand() < disturbance_probability and i > disturbance_until_time:  # Apply disturbance if not locked
            disturbance_length = np.random.randint(1/dt * 5, 1/dt * 15)  # Random disturbance length
            disturbance[i:i + disturbance_length] = np.random.normal(-0.5, 0.5)  # Apply disturbance
            disturbance_until_time = i + disturbance_length
        elif i > disturbance_until_time:
            disturbance[i] = 0  # No disturbance applied

        # Update setpoints for both controllers (step input occurs at 10 seconds)
        pid.setpoint = setpoint[i]
        adrc.setpoint = setpoint[i]

        # Get control signals from PID and ADRC
        pid_control_signal[i] = pid.compute_control_signal(position_pid[i - 1], dt)
        adrc_control_signal[i] = adrc.compute_control_signal(position_adrc[i - 1], dt)

        # Update ship system with control signals and disturbance for PID
        position_pid[i] = ship1.step(pid_control_signal[i], disturbance[i], dt)

        # Update ship system with control signals and disturbance for ADRC
        position_adrc[i] = ship2.step(adrc_control_signal[i], disturbance[i], dt)

    # Calculate MSE for both controllers
    mse_pid = calculate_mse(setpoint, position_pid)
    mse_adrc = calculate_mse(setpoint, position_adrc)

    # Plotting the results
    plt.figure(figsize=(12, 10))

    # PID Controller Position
    plt.subplot(2, 1, 1)
    plt.plot(time, position_pid, label=f"Ship Position (PID Control) \nMSE: {mse_pid:.3f}")
    plt.plot(time, position_adrc, label=f"Ship Position (ADRC Control) \nMSE: {mse_adrc:.3f}")
    plt.plot(time, setpoint, color='r', linestyle='--', label="Setpoint (Step to 1 at t=10s)")
    plt.title(f'Ship Position vs Setpoint')
    plt.ylabel('Position')
    plt.legend()

    # Control Signals
    plt.subplot(2, 1, 2)
    plt.plot(time, pid_control_signal, label="PID Control Signal")
    plt.plot(time, adrc_control_signal, label="ADRC Control Signal", linestyle='--')
    plt.plot(time, disturbance, label="Disturbance", linestyle="dotted")
    plt.title('Control Signals')
    plt.ylabel('Signal')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()