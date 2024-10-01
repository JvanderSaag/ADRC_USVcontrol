import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from controllers.PID import PIDController
from controllers.ADRC import ADRCController
from systems.Vessel_SISO import SISOVesselSystem


def calculate_mse(reference: np.ndarray, signal: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between the reference and signal.

    Args:
        reference (np.ndarray): The reference signal (desired output).
        signal (np.ndarray): The actual signal to be compared against the reference.

    Returns:
        float: The calculated mean squared error.
    """
    return np.mean((reference - signal) ** 2)


def run_simulation(Kp: float, Ki: float, Kd: float, h0: float, r0: float, b0: float, 
                   beta01: float, beta02: float, beta03: float, k1: float, k2: float, 
                   alpha1: float, alpha2: float, tau_propeller: float) -> tuple:
    """
    Run the simulation with the given PID and ADRC gains and propeller dynamics.

    Args:
        Kp (float): Proportional gain of the PID controller.
        Ki (float): Integral gain of the PID controller.
        Kd (float): Derivative gain of the PID controller.
        h0 (float): ADRC parameter.
        r0 (float): ADRC parameter.
        b0 (float): ADRC parameter.
        beta01 (float): ADRC tuning parameter.
        beta02 (float): ADRC tuning parameter.
        beta03 (float): ADRC tuning parameter.
        k1 (float): ADRC control gain.
        k2 (float): ADRC control gain.
        alpha1 (float): ADRC parameter.
        alpha2 (float): ADRC parameter.
        tau_propeller (float): Propeller time constant.

    Returns:
        tuple: Contains time array, setpoint, positions for PID and ADRC, 
               control signals for both controllers, disturbances, and MSE for both controllers.
    """
    # Simulation parameters
    time_end = 120  # Simulation time in seconds
    dt = 0.01       # Time step
    num_steps = int(time_end / dt)
    time = np.linspace(0, time_end, num_steps)

    # Step input: initial setpoint is 0, then steps to 1 after 10 seconds
    setpoint = np.ones(num_steps)  # Create an array of ones
    setpoint[:int(10 / dt)] = 0    # Set the first 10 seconds to 0 (before the step)

    np.random.seed(42)  # Set random seed for reproducibility

    # Input constraints
    min_input = -15  # Minimum control input
    max_input = 15   # Maximum control input

    # System and controller initialization
    ship1 = SISOVesselSystem()
    ship2 = SISOVesselSystem()
    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, min_u=min_input, max_u=max_input)
    adrc = ADRCController(
        h0=h0, r0=r0, b0=b0, beta01=beta01, beta02=beta02, beta03=beta03,
        k1=k1, k2=k2, alpha1=alpha1, alpha2=alpha2,
        min_u=min_input, max_u=max_input
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

    # Propeller spin-up dynamics initialisation
    pid_propeller_output = 0  # Initial output of the propeller (stationary)
    adrc_propeller_output = 0  # Initial output of the propeller (stationary)

    # Run the simulation
    disturbance_until_time = 0  # Disturbance applied until this time
    for i in range(1, num_steps):
        # Determine if a disturbance should be applied (0.1% chance)
        if np.random.rand() < disturbance_probability and i > disturbance_until_time:  # Apply disturbance if not locked
            disturbance_length = np.random.randint(1/dt * 5, 1/dt * 15)  # Random disturbance length
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

        # Apply propeller dynamics (first-order lag)
        if tau_propeller == 0:
            pid_propeller_output = pid_control_signal[i]
            adrc_propeller_output = adrc_control_signal[i]
        else:
            pid_propeller_output += (dt / tau_propeller) * (pid_control_signal[i] - pid_propeller_output)
            adrc_propeller_output += (dt / tau_propeller) * (adrc_control_signal[i] - adrc_propeller_output)

        pid_control_signal_filtered[i] = pid_propeller_output
        adrc_control_signal_filtered[i] = adrc_propeller_output

        # Update ship system with control signals and disturbance for PID
        position_pid[i] = ship1.step(pid_control_signal_filtered[i], disturbance[i], dt)

        # Update ship system with control signals and disturbance for ADRC
        position_adrc[i] = ship2.step(adrc_control_signal_filtered[i], disturbance[i], dt)

    # Calculate MSE for both controllers
    mse_pid = calculate_mse(setpoint, position_pid)
    mse_adrc = calculate_mse(setpoint, position_adrc)

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
    global adrc_h0_slider, adrc_r0_slider, adrc_b0_slider, adrc_beta01_slider, adrc_beta02_slider, adrc_beta03_slider
    global adrc_k1_slider, adrc_k2_slider, adrc_alpha1_slider, adrc_alpha2_slider
    global tau_propeller_slider

    # Get the current values of the sliders
    Kp = pid_Kp_slider.val
    Ki = pid_Ki_slider.val
    Kd = pid_Kd_slider.val
    
    h0 = adrc_h0_slider.val
    r0 = adrc_r0_slider.val
    b0 = adrc_b0_slider.val
    beta01 = adrc_beta01_slider.val
    beta02 = adrc_beta02_slider.val
    beta03 = adrc_beta03_slider.val
    k1 = adrc_k1_slider.val
    k2 = adrc_k2_slider.val
    alpha1 = adrc_alpha1_slider.val
    alpha2 = adrc_alpha2_slider.val
    tau_propeller = tau_propeller_slider.val

    # Re-run the simulation with the new values
    time, setpoint, position_pid, position_adrc, pid_control_signal, adrc_control_signal, disturbance, mse_pid, mse_adrc = run_simulation(
        Kp, Ki, Kd, h0, r0, b0, beta01, beta02, beta03, k1, k2, alpha1, alpha2, tau_propeller)

    # Update the plot data
    pos_pid_plot.set_ydata(position_pid)
    pos_adrc_plot.set_ydata(position_adrc)
    setpoint_plot.set_ydata(setpoint)
    
    pid_signal_plot.set_ydata(pid_control_signal)
    adrc_signal_plot.set_ydata(adrc_control_signal)
    disturbance_plot.set_ydata(disturbance)

    pos_pid_plot.set_label(f"Ship Position (PID Control) \nMSE: {mse_pid:.3f}")
    pos_adrc_plot.set_label(f"Ship Position (ADRC Control) \nMSE: {mse_adrc:.3f}")

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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Initial run with default values
    run_simulation(Kp=1.0, Ki=0.5, Kd=0.1, h0=0.1, r0=0.1, b0=0.1, beta01=1.0, beta02=1.0, beta03=1.0,
                   k1=1.0, k2=1.0, alpha1=1.0, alpha2=1.0)

    pos_pid_plot, = ax1.plot([], [], label='Ship Position (PID Control)')
    pos_adrc_plot, = ax1.plot([], [], label='Ship Position (ADRC Control)')
    setpoint_plot, = ax1.plot([], [], 'k--', label='Setpoint')
    ax1.set_title('Ship Position Control')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position')
    ax1.grid()
    ax1.legend()

    pid_signal_plot, = ax2.plot([], [], label='PID Control Signal')
    adrc_signal_plot, = ax2.plot([], [], label='ADRC Control Signal')
    disturbance_plot, = ax2.plot([], [], 'r--', label='Disturbance')
    ax2.set_title('Control Signals')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Signal')
    ax2.grid()
    ax2.legend()

    # Slider configuration for PID gains
    axcolor = 'lightgoldenrodyellow'
    ax_Kp = plt.axes([0.1, 0.9, 0.65, 0.03], facecolor=axcolor)
    ax_Ki = plt.axes([0.1, 0.85, 0.65, 0.03], facecolor=axcolor)
    ax_Kd = plt.axes([0.1, 0.8, 0.65, 0.03], facecolor=axcolor)
    ax_h0 = plt.axes([0.1, 0.75, 0.65, 0.03], facecolor=axcolor)
    ax_r0 = plt.axes([0.1, 0.7, 0.65, 0.03], facecolor=axcolor)
    ax_b0 = plt.axes([0.1, 0.65, 0.65, 0.03], facecolor=axcolor)
    ax_beta01 = plt.axes([0.1, 0.6, 0.65, 0.03], facecolor=axcolor)
    ax_beta02 = plt.axes([0.1, 0.55, 0.65, 0.03], facecolor=axcolor)
    ax_beta03 = plt.axes([0.1, 0.5, 0.65, 0.03], facecolor=axcolor)
    ax_k1 = plt.axes([0.1, 0.45, 0.65, 0.03], facecolor=axcolor)
    ax_k2 = plt.axes([0.1, 0.4, 0.65, 0.03], facecolor=axcolor)
    ax_alpha1 = plt.axes([0.1, 0.35, 0.65, 0.03], facecolor=axcolor)
    ax_alpha2 = plt.axes([0.1, 0.3, 0.65, 0.03], facecolor=axcolor)
    ax_tau_propeller = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axcolor)

    pid_Kp_slider = Slider(ax_Kp, 'Kp', 0.0, 2.0, valinit=1.0)
    pid_Ki_slider = Slider(ax_Ki, 'Ki', 0.0, 1.0, valinit=0.5)
    pid_Kd_slider = Slider(ax_Kd, 'Kd', 0.0, 1.0, valinit=0.1)
    
    adrc_h0_slider = Slider(ax_h0, 'h0', 0.0, 2.0, valinit=0.1)
    adrc_r0_slider = Slider(ax_r0, 'r0', 0.0, 2.0, valinit=0.1)
    adrc_b0_slider = Slider(ax_b0, 'b0', 0.0, 2.0, valinit=0.1)
    adrc_beta01_slider = Slider(ax_beta01, 'beta01', 0.0, 5.0, valinit=1.0)
    adrc_beta02_slider = Slider(ax_beta02, 'beta02', 0.0, 5.0, valinit=1.0)
    adrc_beta03_slider = Slider(ax_beta03, 'beta03', 0.0, 5.0, valinit=1.0)
    adrc_k1_slider = Slider(ax_k1, 'k1', 0.0, 5.0, valinit=1.0)
    adrc_k2_slider = Slider(ax_k2, 'k2', 0.0, 5.0, valinit=1.0)
    adrc_alpha1_slider = Slider(ax_alpha1, 'alpha1', 0.0, 5.0, valinit=1.0)
    adrc_alpha2_slider = Slider(ax_alpha2, 'alpha2', 0.0, 5.0, valinit=1.0)
    tau_propeller_slider = Slider(ax_tau_propeller, 'Tau Propeller', 0.0, 5.0, valinit=0)

    # Link slider updates to the update function
    for slider in [
        pid_Kp_slider, pid_Ki_slider, pid_Kd_slider,
        adrc_h0_slider, adrc_r0_slider, adrc_b0_slider,
        adrc_beta01_slider, adrc_beta02_slider, adrc_beta03_slider,
        adrc_k1_slider, adrc_k2_slider, adrc_alpha1_slider, adrc_alpha2_slider,
        tau_propeller_slider
    ]:
        slider.on_changed(update)

    plt.show()
