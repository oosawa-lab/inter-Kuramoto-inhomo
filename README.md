# inter-Kuramoto-inhomo
Interactive Kuramoto Model with Inhomogeneous Coupling

This Python script simulates the Kuramoto model with inhomogeneous coupling between oscillators. The simulation allows real-time visualization of the oscillators' phases on a unit circle, as well as interactive controls to start, stop, pause, and adjust the simulation.

Table of Contents
Overview
Required Libraries
Installation Instructions
Usage
Command-Line Arguments
Natural Frequencies (ω)
Interactive Features
Notes
License
Overview
This script simulates the Kuramoto model, where oscillators with phases interact with each other, and their synchronization behavior is modeled over time.

Key Features

Kuramoto Model simulation using a custom coupling matrix (K)
Real-time visualization: Plot oscillators' phases on the unit circle
Interactive Features: Drag oscillators to change their phases, pause/resume/stop the simulation
Required Libraries
This script depends on the following Python libraries:

numpy: for numerical calculations
matplotlib: for plotting and visualization
argparse: for parsing command-line arguments
sys, time: for system-related functionalities
You can install these libraries if you don't have them already by running the following command:

pip install numpy matplotlib
Installation Instructions
Clone the repository:
git clone https://github.com/yourusername/kuramoto-model.git
cd kuramoto-model
Install the required dependencies:
pip install -r requirements.txt
Once the dependencies are installed, you're ready to run the script.
Usage
Prepare the Coupling Matrix File
The script requires a file that defines the coupling strength between oscillators. This file should contain a square matrix (i.e., number of rows and columns must be equal).
Example of coupling_matrix.txt:

1.0  0.5  0.2

0.5  1.0  0.3

0.2  0.3  1.0

This matrix defines the strength of interaction between each pair of oscillators.

Run the Script
To run the simulation, execute the following command:

python kuramoto.py --dt 0.01 --coupling_matrix_file coupling_matrix.txt
--dt: The time step for the simulation. The default value is 0.01.
--coupling_matrix_file: The path to the coupling matrix file (this is required).
Command-Line Arguments
--dt: The time step for the simulation (default is 0.01).
--coupling_matrix_file: The path to the coupling matrix file (this is required).
Natural Frequencies (ω)
In the Kuramoto model, each oscillator has a natural frequency (ω) that determines its intrinsic frequency of oscillation. These frequencies can be uniform or inhomogeneous, depending on the system setup.

Uniform Natural Frequencies: In this script, all oscillators initially have the same natural frequency, which is set to ω = 1.0 for all oscillators. This means that all oscillators are assumed to have the same natural frequency in the absence of coupling interactions.
Inhomogeneous Natural Frequencies: If you'd like to simulate oscillators with different natural frequencies, you can modify the omega array. The natural frequencies can be set to random values, specific constants, or any other desired distribution.
Example:

omega = np.random.uniform(0.5, 1.5, N)  # Random natural frequencies between 0.5 and 1.5
The natural frequency of each oscillator influences how quickly it oscillates in the absence of interactions. When oscillators are coupled, their phases evolve over time depending on both their natural frequencies and the coupling strength defined by the matrix K.

Interactive Features
This script provides an interactive visualization. Here are some features you can use:

Phase Change: You can click on an oscillator in the unit circle plot to select it. Once selected, you can drag it to a new position to change its phase. The phase will update immediately on the plot.
Pause/Resume Simulation: You can pause or resume the simulation by clicking the "Pause/Resume" button located at the bottom of the plot.
Start/Stop Simulation: You can start or stop the simulation by clicking the "Start/Stop" button located at the top right of the plot.
Exit: Press Enter to exit the simulation, and the window will close.
Notes
Coupling Matrix: The coupling matrix (coupling_matrix.txt) must be a square matrix. The size of the matrix determines the number of oscillators (N).
Exiting the Simulation: After stopping the simulation, pressing Enter will close the plot and terminate the script.
License
This project is licensed under the MIT License. See the LICENSE file for more information.
