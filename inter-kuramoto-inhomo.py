# -----------------------------------------------------------------------------
# Copyright (c) 2025 Chikoo Oosawa, Kyushu Institute of Technology
#
# This project is licensed under the MIT License.
# You can find the full license at: https://opensource.org/licenses/MIT
#
# Repository: https://github.com/oosawa-lab/inter-Kuramoto-inhomo/
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
import argparse
import sys
import time

# Argument parsing
parser = argparse.ArgumentParser(description='Kuramoto Model Simulation with Inhomogeneous Coupling')
parser.add_argument('--dt', type=float, default=0.01, help='Time step for simulation')
parser.add_argument('--coupling_matrix_file', type=str, required=True, help='File containing the inhomogeneous coupling matrix')
args = parser.parse_args()

# Load coupling matrix from file
try:
    K = 1 * np.loadtxt(args.coupling_matrix_file)
    N = K.shape[0]  # Number of oscillators is determined by the coupling matrix size
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"The coupling matrix must be square (found {K.shape[0]}x{K.shape[1]} matrix).")
except Exception as e:
    print(f"Error reading coupling matrix: {e}")
    raise ValueError("The coupling matrix must be square and match the number of oscillators.")

# Kuramoto model parameters
omega = np.ones(N)  # 固有振動数ωは、全部同じ1

# Kuramoto equation
def kuramoto(phase):
    phase_diff = phase[:, None] + phase[None, :]  # 相互作用の修正
    M = 2.0  # K-meansのKをM
    return omega + np.sum(K * np.sin((M / 2.0) * phase_diff), axis=1)

# 4th Order Runge-Kutta method
def runge_kutta(phase, dt):
    k1 = dt * kuramoto(phase)
    k2 = dt * kuramoto(phase + 0.5 * k1)
    k3 = dt * kuramoto(phase + 0.5 * k2)
    k4 = dt * kuramoto(phase + k3)
    return phase + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Initial conditions
np.random.seed(int(time.time()))
initial_phase = np.random.rand(N) * 2 * np.pi  # Random initial phase
result = [initial_phase]

# Create a custom colormap for grayscale (0 is white, ±π are black)
cmap = LinearSegmentedColormap.from_list('gray_phase', ['black', 'white', 'black'], N=256)

# Set up the figure with two subplots (left for matrix, right for Kuramoto phase)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Plot 1: Heatmap of the coupling matrix (left side)
def plot_heatmap(ax, matrix):
    cax = ax.imshow(matrix, cmap='jet', interpolation='nearest')
    fig.colorbar(cax, ax=ax, label='kij')
    ax.set_title("Coupling Matrix Heatmap")
    ax.set_xticks([])
    ax.set_yticks([])

plot_heatmap(ax1, K)

# Plot 2: Phase plot (cos(phase), sin(phase)) (right side)
colors = plt.cm.viridis(np.linspace(0, 1, N))
lines = [ax2.plot([], [], 'o', markersize=10, color=color)[0] for color in colors]
texts = [ax2.text(0, 0, '', fontsize=12, ha='center') for _ in range(N)]
time_text = ax2.text(0, 1.3, '', fontsize=14, ha='center')  # Time text position adjusted
order_text = ax2.text(0, 1.2, '', fontsize=14, ha='center')  # Order parameter position adjusted

# Draw unit circle
theta = np.linspace(0, 2 * np.pi, 100)
unit_circle_x = np.cos(theta)
unit_circle_y = np.sin(theta)
ax2.plot(unit_circle_x, unit_circle_y, 'k--', label='Unit Circle')

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal', adjustable='box')
ax2.set_title(f'Oscillator Dynamics Over Time N={N}')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.tick_params(left=False, right=False, bottom=False, top=False)

# Turn on interactive mode
plt.ion()

paused = False  # To keep track of whether the simulation is paused or not
simulation_started = False  # Track whether simulation is started or not
frame = 0  # Start from the first frame
dragging = False  # To track dragging state
selected_oscillator = None  # Selected oscillator index

# Event handler for key press
def on_key(event):
    global paused
    if event.key == 'enter':  # Handle return/enter key press
        print("Exiting simulation...")
        plt.close(fig)  # Close the plot window and exit the simulation
        sys.exit()  # Ensure the program exits completely

# Event handler for mouse click (to select oscillator to drag)
def on_click(event):
    global selected_oscillator, dragging
    if event.inaxes == ax2 and paused:
        x_click, y_click = event.xdata, event.ydata
        for i in range(N):
            # Calculate distance from the click position to the oscillator position
            x_oscillator = np.cos(result[-1][i])
            y_oscillator = np.sin(result[-1][i])
            distance = np.sqrt((x_click - x_oscillator)**2 + (y_click - y_oscillator)**2)
            if distance < 0.2:  # Sensitivity threshold
                selected_oscillator = i  # Select the oscillator
                dragging = True  # Start dragging
                print(f"Oscillator {i} selected")
                break

# Event handler for mouse drag (to update oscillator phase)
def on_motion(event):
    global selected_oscillator, dragging
    if selected_oscillator is not None and event.inaxes == ax2 and paused and dragging:
        x_drag, y_drag = event.xdata, event.ydata
        # Convert the drag position back to a phase angle
        new_phase = np.arctan2(y_drag, x_drag)  # Convert (x, y) to phase
        result[-1][selected_oscillator] = new_phase  # Update phase of the selected oscillator
        
        # Update the display with the new phase position
        lines[selected_oscillator].set_data(x_drag, y_drag)  # Move the oscillator point to the new position
        texts[selected_oscillator].set_position((x_drag, y_drag))  # Update the position of text labels
        texts[selected_oscillator].set_text(f'[{selected_oscillator}] {omega[selected_oscillator]:.2f}')  # Update label
        
        print(f"Oscillator {selected_oscillator} phase updated to {new_phase:.3f}")

# Event handler for mouse release (when dragging stops)
def on_release(event):
    global dragging
    dragging = False  # Stop dragging when mouse is released
    print("Drag ended")

# Start/Stop Button callback function
def start_stop_button_callback(event):
    global simulation_started, paused
    if not simulation_started:
        simulation_started = True  # Mark the simulation as started
        button_start_stop.label.set_text('Stop')  # Change label to 'Stop'
        print("Simulation started")
    else:
        simulation_started = False  # Stop the simulation
        button_start_stop.label.set_text('Start')  # Change label to 'Start'
        print("Simulation stopped")
        plt.close(fig)  # Close the plot window when stopped
        sys.exit()  # Exit the program completely

# Pause/Resume Button callback function
def toggle_button_callback(event):
    global paused
    paused = not paused  # Toggle the paused state
    button_pause.label.set_text('Resume' if paused else 'Pause')  # Change button text
    print(f"Simulation {'paused' if paused else 'resumed'}")

# Create a Start/Stop button
ax_button_start_stop = plt.axes([0.75, 0.05, 0.15, 0.075])  # Position of the start/stop button (top-right)
button_start_stop = Button(ax_button_start_stop, 'Start')  # Initial label is 'Start'
button_start_stop.on_clicked(start_stop_button_callback)  # Link to callback

# Create a Pause/Resume button
ax_button_pause = plt.axes([0.55, 0.05, 0.15, 0.075])  # Position for left-bottom, symmetric
button_pause = Button(ax_button_pause, 'Pause')  # Initial label is 'Pause'
button_pause.on_clicked(toggle_button_callback)  # Link to callback

# Connect events to figure
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# Main loop to update the simulation (infinite loop now)
while True:
    if simulation_started and not paused:
        # Perform the simulation step
        new_phase = runge_kutta(result[-1], args.dt)
        result.append(new_phase)  # Store the new phase values after one time step
        frame += 1  # Increment frame

        # Update phase plot
        for i in range(N):
            xdata = np.cos(result[frame][i])  # Phase to Cartesian coordinates
            ydata = np.sin(result[frame][i])  # Phase to Cartesian coordinates
            lines[i].set_data(xdata, ydata)  # Set the new coordinates
            texts[i].set_position((xdata, ydata))  # Update the position of the text label
            texts[i].set_text(f'[{i}] {omega[i]:.2f}')  # Display oscillator index and frequency

        # Update time text
        time = frame * args.dt
        time_text.set_text(f'Time: {time:.2f} s')

        # Update order parameter text
        R_current = np.abs(np.sum(np.exp(1j * result[frame])) / N)
        order_text.set_text(f'Order Parameter: {R_current:.3f}')

    # Update figure
    plt.draw()

    # Pause briefly
    plt.pause(0.001)

plt.ioff()  # Turn off interactive mode
plt.show()  # Explicitly show the figure after exiting the loop





    
    
    
