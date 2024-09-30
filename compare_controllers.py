import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from controllers.PID import PIDController
from controllers.ADRC import ADRCController
from systems.Vessel_SISO import SISOVesselSystem

def calculate_mse(reference, signal):
    """Calculate the Mean Squared Error (MSE) between the reference and signal."""
    return np.mean((reference - signal) ** 2)

def run_simulation(Kp, Ki, Kd, h0, r0, b0, beta01, beta02, beta03, k1, k2, alpha1, alpha2):
    """Run the simulation with the given PID and ADRC gains."""
    # Simulation parameters
    time_end = 120  # Simulation time in seconds
    dt = 0.01      # Time step
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
    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, min_u=min_input, max_u=max_input)  # Setpoint of 1 for the controller
    adrc = ADRCController(h0=h0, r0=r0, b0=b0, beta01=beta01, beta02=beta02, beta03=beta03, k1=k1, k2=k2, alpha1=alpha1, alpha2=alpha2, min_u=min_input, max_u=max_input)

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

    return time, setpoint, position_pid, position_adrc, pid_control_signal, adrc_control_signal, disturbance, mse_pid, mse_adrc

def update(val):
    # Access global variables
    global pos_pid_plot, pos_adrc_plot, setpoint_plot
    global pid_signal_plot, adrc_signal_plot, disturbance_plot
    global ax1, ax2, fig
    global pid_Kp_slider, pid_Ki_slider, pid_Kd_slider
    global adrc_h0_slider, adrc_r0_slider, adrc_b0_slider, adrc_beta01_slider, adrc_beta02_slider, adrc_beta03_slider
    global adrc_k1_slider, adrc_k2_slider, adrc_alpha1_slider, adrc_alpha2_slider

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

    # Re-run the simulation with the new values
    time, setpoint, position_pid, position_adrc, pid_control_signal, adrc_control_signal, disturbance, mse_pid, mse_adrc = run_simulation(
        Kp, Ki, Kd, h0, r0, b0, beta01, beta02, beta03, k1, k2, alpha1, alpha2)

    # Update the plot data
    pos_pid_plot.set_ydata(position_pid)
    pos_adrc_plot.set_ydata(position_adrc)
    setpoint_plot.set_ydata(setpoint)
    
    pid_signal_plot.set_ydata(pid_control_signal)
    adrc_signal_plot.set_ydata(adrc_control_signal)
    disturbance_plot.set_ydata(disturbance)

    pos_pid_plot.set_label(f"Ship Position (PID Control) \nMSE: {mse_pid:.3f}")
    pos_adrc_plot.set_label(f"Ship Position (ADRC Control) \nMSE: {mse_adrc:.3f}")
    ax1.legend()
    ax2.legend()

    # Update axes limits based on new data
    ax1.relim()
    ax1.autoscale_view()

    ax2.relim()
    ax2.autoscale_view()

    # Redraw the plots
    fig.canvas.draw_idle()

def main():
    global pos_pid_plot, pos_adrc_plot, setpoint_plot
    global pid_signal_plot, adrc_signal_plot, disturbance_plot
    global ax1, ax2, fig
    global pid_Kp_slider, pid_Ki_slider, pid_Kd_slider
    global adrc_h0_slider, adrc_r0_slider, adrc_b0_slider, adrc_beta01_slider, adrc_beta02_slider, adrc_beta03_slider
    global adrc_k1_slider, adrc_k2_slider, adrc_alpha1_slider, adrc_alpha2_slider

    # Init values
    Kp, Ki, Kd = 35, 2.5, 45

    h0, r0, b0 = 0.01, 1, 1
    beta01, beta02, beta03 = 1, 1/(2*0.01**0.5), 2/(25*0.01**1.2)
    k1, k2, alpha1, alpha2 = 0.1, 0.2, 0.3, 0.1


    # Initial simulation run with default values
    time, setpoint, position_pid, position_adrc, pid_control_signal, adrc_control_signal, disturbance, mse_pid, mse_adrc = run_simulation(
        Kp=Kp, Ki=Ki, Kd=Kd, h0=h0, r0=r0, b0=b0, beta01=beta01, beta02=beta02, beta03=beta03, k1=k1, k2=k2, alpha1=alpha1, alpha2=alpha2)

    # Create the figure and the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot ship positions
    pos_pid_plot, = ax1.plot(time, position_pid, label=f"Ship Position (PID Control) \nMSE: {mse_pid:.3f}")
    pos_adrc_plot, = ax1.plot(time, position_adrc, label=f"Ship Position (ADRC Control) \nMSE: {mse_adrc:.3f}")
    setpoint_plot, = ax1.plot(time, setpoint, color='0', linestyle=':', label="Setpoint (Step to 1 at t=10s)")
    ax1.set_title('Ship Position vs Setpoint')
    ax1.set_ylabel('Position')
    ax1.legend()

    # Plot control signals and disturbance
    pid_signal_plot, = ax2.plot(time, pid_control_signal, label="PID Control Signal")
    adrc_signal_plot, = ax2.plot(time, adrc_control_signal, label="ADRC Control Signal", linestyle='--')
    disturbance_plot, = ax2.plot(time, disturbance, label="Disturbance", linestyle="dotted")
    ax2.set_title('Control Signals')
    ax2.set_ylabel('Signal')
    ax2.set_xlabel('Time (s)')
    ax2.legend()

    # Adjust layout to fit the sliders
    plt.subplots_adjust(top=0.96, bottom=0.07, left=0.05, right=0.7, hspace=0.15)  # Adjust right side for sliders

    # Add sliders for PID controller gains on the right
    ax_Kp = plt.axes([0.74, 0.85, 0.22, 0.05])  # Changed position for vertical layout
    pid_Kp_slider = Slider(ax_Kp, 'PID Kp', 0.1, 40.0, valinit=Kp)

    ax_Ki = plt.axes([0.74, 0.8, 0.22, 0.05])
    pid_Ki_slider = Slider(ax_Ki, 'PID Ki', 0.01, 10.0, valinit=Ki)

    ax_Kd = plt.axes([0.74, 0.75, 0.22, 0.05])
    pid_Kd_slider = Slider(ax_Kd, 'PID Kd', 1, 100.0, valinit=Kd)

    # Add sliders for ADRC controller gains on the right
    ax_h0 = plt.axes([0.74, 0.7, 0.22, 0.05])
    adrc_h0_slider = Slider(ax_h0, 'ADRC h0', 0.0001, 0.01, valinit=h0)

    ax_r0 = plt.axes([0.74, 0.65, 0.22, 0.05])
    adrc_r0_slider = Slider(ax_r0, 'ADRC r0', 0.1, 10.0, valinit=r0)

    ax_b0 = plt.axes([0.74, 0.6, 0.22, 0.05])
    adrc_b0_slider = Slider(ax_b0, 'ADRC b0', 0.1, 10.0, valinit=b0)

    ax_beta01 = plt.axes([0.74, 0.55, 0.22, 0.05])
    adrc_beta01_slider = Slider(ax_beta01, 'ADRC beta01', 0.1, 5.0, valinit=beta01)

    ax_beta02 = plt.axes([0.74, 0.5, 0.22, 0.05])
    adrc_beta02_slider = Slider(ax_beta02, 'ADRC beta02', 0.1, 10, valinit=beta02)

    ax_beta03 = plt.axes([0.74, 0.45, 0.22, 0.05])
    adrc_beta03_slider = Slider(ax_beta03, 'ADRC beta03', 0.1, 30, valinit=beta03)

    ax_k1 = plt.axes([0.74, 0.4, 0.22, 0.05])
    adrc_k1_slider = Slider(ax_k1, 'ADRC k1', 0.01, 1.0, valinit=k1)

    ax_k2 = plt.axes([0.74, 0.35, 0.22, 0.05])
    adrc_k2_slider = Slider(ax_k2, 'ADRC k2', 0.01, 1.0, valinit=k2)

    ax_alpha1 = plt.axes([0.74, 0.3, 0.22, 0.05])
    adrc_alpha1_slider = Slider(ax_alpha1, 'ADRC alpha1', 0.01, 1.0, valinit=alpha1)

    ax_alpha2 = plt.axes([0.74, 0.25, 0.22, 0.05])
    adrc_alpha2_slider = Slider(ax_alpha2, 'ADRC alpha2', 0.01, 1.0, valinit=alpha2)

    # Call update function on slider value change
    pid_Kp_slider.on_changed(update)
    pid_Ki_slider.on_changed(update)
    pid_Kd_slider.on_changed(update)

    adrc_h0_slider.on_changed(update)
    adrc_r0_slider.on_changed(update)
    adrc_b0_slider.on_changed(update)
    adrc_beta01_slider.on_changed(update)
    adrc_beta02_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
