import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from controllers.PID import PIDController
from controllers.ADRC import ADRCController
from systems.Vessel_SISO import SISOVesselSystem


def calculate_rmse(reference: np.ndarray, signal: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between the reference and signal.

    Args:
        reference (np.ndarray): The reference signal (desired output).
        signal (np.ndarray): The actual signal to be compared against the reference.

    Returns:
        float: The calculated mean squared error.
    """
    return np.sqrt(np.mean((reference - signal) ** 2))


def run_simulation(Kp: float, Ki: float, Kd: float, h0: float, r0: float, b0: float, 
                   k1: float, k2: float, tau_propeller: float) -> tuple:
    """
    Run the simulation with the given PID and ADRC gains and propeller dynamics.

    Args:
        Kp (float): Proportional gain of the PID controller.
        Ki (float): Integral gain of the PID controller.
        Kd (float): Derivative gain of the PID controller.
        h0 (float): ADRC filter factor of TD.
        r0 (float): ADRC tracking speed of TD.
        b0 (float): ADRC control gain, higher values means less aggressive actuator response.
        k1 (float): ADRC proportional gain.
        k2 (float): ADRC derivative gain.
        tau_propeller (float): Propeller time constant.

    Returns:
        tuple: Contains time array, setpoint, positions for PID and ADRC, 
               control signals for both controllers, disturbances, and RMSE for both controllers.
    """
    # Simulation parameters
    time_end = 120  # Simulation time in seconds
    dt = 0.01
    num_steps = int(time_end / dt)
    time = np.linspace(0, time_end, num_steps)

    # Step input: initial setpoint is 0, then steps to 1 after 10 seconds
    setpoint = np.ones(num_steps)  # Create an array of ones
    setpoint[:int(10 / dt)] = 0    # Set the first 10 seconds to 0 (before the step)
    setpoint[int(40 / dt):] = 0.5    # Set setpoint to 0.5 after 40 seconds
    setpoint[int(80 / dt):] = 2   # Set setpoint to 2 after 80 seconds
    np.random.seed(42)  # Set random seed for reproducibility

    # System and controller initialization
    ship1 = SISOVesselSystem()
    ship2 = SISOVesselSystem()
    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=0, tau=tau_propeller)
    adrc = ADRCController(
        h0=h0, r0=r0, b0=b0,
        k1=k1, k2=k2, tau=tau_propeller
    )

    # Arrays to store results
    position_pid = np.zeros(num_steps)
    position_adrc = np.zeros(num_steps)
    pid_control_signal = np.zeros(num_steps)
    adrc_control_signal = np.zeros(num_steps)
    pid_control_signal_filtered = np.zeros(num_steps)
    adrc_control_signal_filtered = np.zeros(num_steps)
    disturbance = np.zeros(num_steps)

    # Probability of applying a disturbance
    disturbance_probability = 0.001

    # Run the simulation
    disturbance_until_time = 0  # Disturbance applied until this time
    for i in range(1, num_steps):
        # Determine if a disturbance should be applied (0.1% chance)
        if np.random.rand() < disturbance_probability and i > disturbance_until_time:  # Apply disturbance if not locked
            disturbance_length = np.random.randint(1/dt * 1, 1/dt * 3)  # Random disturbance length
            disturbance[i:i + disturbance_length] = np.random.normal(-2, 2)  # Apply disturbance
            disturbance_until_time = i + disturbance_length
        elif i > disturbance_until_time:
            disturbance[i] = 0  # No disturbance applied

        # Update setpoints for both controllers (step input occurs at 10 seconds)
        pid.setpoint = setpoint[i]
        adrc.setpoint = setpoint[i]

        # Get control signals from PID and ADRC
        pid_control_signal[i] = pid.compute_control_signal(position_pid[i - 1], dt)
        adrc_control_signal[i] = adrc.compute_control_signal(position_adrc[i - 1], dt)

        # Update ship system with control signals and disturbances
        position_pid[i] = ship1.step(pid_control_signal[i], disturbance[i], dt)
        position_adrc[i] = ship2.step(adrc_control_signal[i], disturbance[i], dt)

        # Save the filtered control signals
        pid_control_signal_filtered[i] = ship1.last_control_input
        adrc_control_signal_filtered[i] = ship2.last_control_input

    # Calculate RMSE for both controllers
    mse_pid = calculate_rmse(setpoint, position_pid)
    mse_adrc = calculate_rmse(setpoint, position_adrc)

    return (time, setpoint, position_pid, position_adrc, 
            pid_control_signal_filtered, adrc_control_signal_filtered, 
            disturbance, mse_pid, mse_adrc)


def update(val: float) -> None:
    """
    Update the simulation when a slider value changes.

    Args:
        val (float): The value of the slider that triggered the update.
    """
    # Access global variables
    global pos_pid_plot, pos_adrc_plot, setpoint_plot
    global pid_signal_plot, adrc_signal_plot, disturbance_plot
    global ax1, ax2, fig
    global pid_Kp_slider, pid_Ki_slider, pid_Kd_slider
    global adrc_h0_slider, adrc_r0_slider, adrc_b0_slider
    global k1_slider, k2_slider
    global tau_propeller_slider
    
    # Get the current values of the sliders
    Kp = pid_Kp_slider.val
    Ki = pid_Ki_slider.val
    Kd = pid_Kd_slider.val
    
    h0 = adrc_h0_slider.val
    r0 = adrc_r0_slider.val
    b0 = adrc_b0_slider.val
    k1 = k1_slider.val
    k2 = k2_slider.val
    tau_propeller = tau_propeller_slider.val

    # Re-run the simulation with the new values
    _, setpoint, position_pid, position_adrc, pid_control_signal, adrc_control_signal, disturbance, rmse_pid, rmse_adrc = run_simulation(
        Kp, Ki, Kd, h0, r0, b0, k1, k2, tau_propeller)

    # Update the plot data
    pos_pid_plot.set_ydata(position_pid)
    pos_adrc_plot.set_ydata(position_adrc)
    setpoint_plot.set_ydata(setpoint)
    
    pid_signal_plot.set_ydata(pid_control_signal)
    adrc_signal_plot.set_ydata(adrc_control_signal)
    disturbance_plot.set_ydata(disturbance)

    pos_pid_plot.set_label(f"Ship Position (PID Control) \nRMSE: {rmse_pid:.3f}")
    pos_adrc_plot.set_label(f"Ship Position (ADRC Control) \nRMSE: {rmse_adrc:.3f}")

    # Redraw the plot
    ax1.relim()
    ax1.autoscale_view()
    ax1.legend()
    ax2.relim()
    ax2.autoscale_view()
    ax2.legend()
    fig.canvas.draw_idle()


# Plotting and GUI setup
if __name__ == "__main__":
    # Initial values
    tau_propeller = 2 / 4 # Settling time of 2 seconds, divided by 4 to get time constant for first-order systems.
    Kp, Ki, Kd = 35, 2.5, 45
    h0, r0, b0 = 0.140, 0.805, 0.103
    k1, k2 = 0.7, 12.28

    # Initial simulation run with default values
    time, setpoint, position_pid, position_adrc, pid_control_signal, adrc_control_signal, disturbance, rmse_pid, rmse_adrc = run_simulation(
        Kp=Kp, Ki=Ki, Kd=Kd, h0=h0, r0=r0, b0=b0, k1=k1, k2=k2, tau_propeller=tau_propeller)

    # Create the figure and the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot ship positions
    pos_pid_plot, = ax1.plot(time, position_pid, label=f"Ship Position (PID Control) \nRMSE: {rmse_pid:.3f}")
    pos_adrc_plot, = ax1.plot(time, position_adrc, label=f"Ship Position (ADRC Control) \nRMSE: {rmse_adrc:.3f}")
    setpoint_plot, = ax1.plot(time, setpoint, color='0', linestyle=':', label="Setpoint (Step to 1 at t=10s)")
    ax1.set_title('Ship Position vs Setpoint')
    ax1.set_ylabel('Position [m]')
    ax1.legend()

    # Plot control signals and disturbance
    pid_signal_plot, = ax2.plot(time, pid_control_signal, label="PID Control Signal")
    adrc_signal_plot, = ax2.plot(time, adrc_control_signal, label="ADRC Control Signal", linestyle='--')
    disturbance_plot, = ax2.plot(time, disturbance, label="Disturbance", linestyle="dotted")
    ax2.set_title('Control Signals and Disturbances')
    ax2.set_ylabel('Force [N]')
    ax2.set_xlabel('Time (s)')
    ax2.legend()

    # Adjust layout to fit the sliders
    plt.subplots_adjust(top=0.96, bottom=0.07, left=0.05, right=0.7, hspace=0.15) 

    # Slider configuration
    ax_tau = plt.axes([0.74, 0.9, 0.22, 0.05])  
    tau_propeller_slider = Slider(ax_tau, 'Tau (linear)', 0, 2.0, valinit=tau_propeller)

    ax_Kp = plt.axes([0.74, 0.85, 0.22, 0.05]) 
    pid_Kp_slider = Slider(ax_Kp, 'PID Kp', 0.1, 40.0, valinit=Kp)

    ax_Ki = plt.axes([0.74, 0.8, 0.22, 0.05])
    pid_Ki_slider = Slider(ax_Ki, 'PID Ki', 0.01, 10.0, valinit=Ki)

    ax_Kd = plt.axes([0.74, 0.75, 0.22, 0.05])
    pid_Kd_slider = Slider(ax_Kd, 'PID Kd', 1, 100.0, valinit=Kd)

    ax_h0 = plt.axes([0.74, 0.7, 0.22, 0.05])
    adrc_h0_slider = Slider(ax_h0, 'ADRC h0', 0.001, 0.5, valinit=h0)

    ax_r0 = plt.axes([0.74, 0.65, 0.22, 0.05])
    adrc_r0_slider = Slider(ax_r0, 'ADRC r0', 0.001, 2.0, valinit=r0)

    ax_b0 = plt.axes([0.74, 0.6, 0.22, 0.05])
    adrc_b0_slider = Slider(ax_b0, 'ADRC b0', 0.001, 2.5, valinit=b0)

    ax_k1 = plt.axes([0.74, 0.55, 0.22, 0.05])
    k1_slider = Slider(ax_k1, 'ADRC k1', 0., 20.0, valinit=k1)

    ax_k2 = plt.axes([0.74, 0.5, 0.22, 0.05])
    k2_slider = Slider(ax_k2, 'ADRC k2', 0, 20.0, valinit=k2)


    # Link slider updates to the update function
    for slider in [
        pid_Kp_slider, pid_Ki_slider, pid_Kd_slider,
        adrc_h0_slider, adrc_r0_slider, adrc_b0_slider,
        k1_slider, k2_slider, tau_propeller_slider
    ]:
        slider.on_changed(update)

    plt.show()
