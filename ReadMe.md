# USV Control Simulation

## Overview

This project simulates the control of a Unmanned Surface Vehicle (USV) using two different controllers: PID (Proportional-Integral-Derivative) and ADRC (Active Disturbance Rejection Control). The simulation uses a Single Input Single Output (SISO) system model to represent the USV's surge motion, allowing for the comparison of control strategies in the presence of disturbances.

## Project structure
```
/ADRC_USVcontrol
│ 
├── /controllers 
│   ├── PID.py # PID Controller implementation 
│   └── ADRC.py # ADRC Controller implementation 
│ 
├── /system 
│   └── Vessel_SISO.py # SISO vessel system implementation 
│   
├── compare_controllers.py # Main simulation script 
└── README.md # This README file
```

## Requirements

To run this simulation, ensure you have the following libraries installed:

- numpy
- matplotlib

You can install the required libraries using pip:

``` 
pip install numpy matplotlib 
```


## Components

### Controllers

#### PID Controller (PID.py)

This file contains the implementation of the PID control algorithm, which computes the control signal based on the error between the desired setpoint and the current position of the vessel.

#### ADRC Controller (ADRC.py)

This file contains the implementation of the ADRC control algorithm, which is designed to reject disturbances and ensure that the system behaves as desired.

### SISO Vessel System

`Vessel_SISO.py` contains the class `SISOVesselSystem`, which represents a Single Input Single Output system for the USV. It includes methods to update the vessel's state based on control inputs and disturbances:

#### Attributes:
- `m11`: Mass of the vessel.
- `X_u`: Damping coefficient.
- `X_absuu`: Non-linear damping term.
- `position`: Current position of the vessel.
- `velocity`: Current velocity of the vessel.

#### Methods:
- `__init__`: Initializes the vessel parameters.
- `step(control_input, disturbance, dt)`: Updates the vessel state based on control input and disturbances over a time step.

### Main Simulation Script

`compare_controllers.py` is the main script that runs the simulation:

#### Functionality:
- It sets up the simulation parameters, initializes the vessels and controllers, and executes the simulation.
- The `run_simulation` function runs the control simulation over a specified time period and collects results.
- The `update` function updates the plots when the user changes the controller parameters via sliders.
- The `main` function initializes the controllers, runs the simulation, and displays the results in interactive plots.

## Running the Simulation

To run the simulation, simply execute the `compare_controllers.py` script:

```
python3 compare_controllers.py
```


## Interactive Controls

The simulation includes interactive sliders for adjusting the parameters of both the PID and ADRC controllers in real-time:

- PID gains: `Kp`, `Ki`, `Kd`
- ADRC parameters: `h0`, `r0`, `b0`, `β01`, `β02`, `β03`, `k1`, `k2`, `α1`, `α2`
- Propeller time constant: `τ`

## Results

The simulation plots the position of the vessel controlled by both the PID and ADRC controllers against the desired setpoint. The sliders can be used to tweak the above parameters and directly see the effect on the output.
