'''
=============
Vibe coded!!
=============
'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy.ndimage import gaussian_filter
import tensorflow as tf

import config


class MagneticFieldPainter:
    """
    Interactive tool to draw magnetic field configurations on a 2D grid.

    Features:
    - Click and drag to draw magnetic field vectors
    - Adjustable brush size
    - Gaussian smoothing for realistic field distributions
    - Export to model-compatible format
    - Visualization of field magnitude and direction
    """

    def __init__(self, grid_size=224, smoothing_sigma=5.0):
        """
        Initialize the magnetic field painter.

        Args:
            grid_size: Resolution of the grid (default 224 to match model input)
            smoothing_sigma: Gaussian smoothing parameter (higher = smoother)
        """
        self.grid_size = grid_size
        self.smoothing_sigma = smoothing_sigma

        # Initialize magnetic field components (H_x, H_y)
        self.H_x = np.zeros((grid_size, grid_size))
        self.H_y = np.zeros((grid_size, grid_size))

        # Drawing state
        self.drawing = False
        self.brush_size = 25
        self.brush_strength = 100.0  # Field strength in A/m

        # Track previous mouse position for drag direction
        self.prev_x = None
        self.prev_y = None

        # Flag to control window closing
        self.window_open = True

        # Setup the figure and axes
        self._setup_figure()

    def _setup_figure(self):
        """Setup the matplotlib figure with controls"""
        self.fig = plt.figure(figsize=(14, 8))

        # Main field visualization
        self.ax_field = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)

        # Magnitude visualization
        self.ax_magnitude = plt.subplot2grid((3, 3), (0, 2))

        # Controls
        self.ax_brush_size = plt.subplot2grid((3, 3), (1, 2))
        self.ax_brush_strength = plt.subplot2grid((3, 3), (2, 2))

        # Buttons
        self.ax_btn_clear = plt.axes([0.7, 0.02, 0.1, 0.04])
        self.ax_btn_smooth = plt.axes([0.81, 0.02, 0.1, 0.04])
        self.ax_btn_done = plt.axes([0.92, 0.02, 0.07, 0.04])

        self.btn_clear = Button(self.ax_btn_clear, 'Clear')
        self.btn_smooth = Button(self.ax_btn_smooth, 'Smooth')
        self.btn_done = Button(self.ax_btn_done, 'Done')

        self.btn_clear.on_clicked(self.clear_field)
        self.btn_smooth.on_clicked(self.smooth_field)
        self.btn_done.on_clicked(self.close_window)

        # Sliders
        self.slider_brush_size = Slider(
            self.ax_brush_size, 'Brush Size', 1, 50, valinit=self.brush_size, valstep=1
        )
        self.slider_brush_strength = Slider(
            self.ax_brush_strength, 'Strength', 10, 500, valinit=self.brush_strength
        )

        self.slider_brush_size.on_changed(self.update_brush_size)
        self.slider_brush_strength.on_changed(self.update_brush_strength)

        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Initial visualization
        self.update_visualization()

        # Instructions
        self.ax_field.set_title(
            'Draw Magnetic Field (Click & Drag to paint field in direction of motion)',
            fontsize=10
        )

    def on_press(self, event):
        """Handle mouse button press"""
        if event.inaxes == self.ax_field:
            self.drawing = True
            self.prev_x = event.xdata
            self.prev_y = event.ydata

    def on_release(self, event):
        """Handle mouse button release"""
        self.drawing = False
        self.prev_x = None
        self.prev_y = None

    def on_motion(self, event):
        """Handle mouse motion"""
        if self.drawing and event.inaxes == self.ax_field:
            if self.prev_x is not None and self.prev_y is not None:
                # Calculate drag direction
                dx = event.xdata - self.prev_x
                dy = event.ydata - self.prev_y
                self.draw_at_position(event.xdata, event.ydata, dx, dy)

            self.prev_x = event.xdata
            self.prev_y = event.ydata

    def draw_at_position(self, x, y, dx, dy):
        """Draw magnetic field at the specified position based on drag direction"""
        if x is None or y is None:
            return

        # Convert from data coordinates to grid indices
        i = int((y + 0.5) * self.grid_size)
        j = int((x + 0.5) * self.grid_size)

        # Check bounds
        if i < 0 or i >= self.grid_size or j < 0 or j >= self.grid_size:
            return

        # Calculate field components from drag direction
        drag_magnitude = np.sqrt(dx**2 + dy**2)
        if drag_magnitude > 0:
            # Normalize and scale by brush strength
            H_x_component = (dx / drag_magnitude) * self.brush_strength
            H_y_component = (dy / drag_magnitude) * self.brush_strength
        else:
            # No movement, skip drawing
            return

        # Apply brush with circular footprint
        for di in range(-self.brush_size, self.brush_size + 1):
            for dj in range(-self.brush_size, self.brush_size + 1):
                ni, nj = i + di, j + dj

                # Check bounds
                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                    # Distance from brush center
                    dist = np.sqrt(di**2 + dj**2)

                    # Apply with falloff
                    if dist <= self.brush_size:
                        weight = 1.0 - (dist / self.brush_size)
                        self.H_x[ni, nj] += H_x_component * weight
                        self.H_y[ni, nj] += H_y_component * weight

        self.update_visualization()

    def smooth_field(self, event=None):
        """Apply Gaussian smoothing to the field"""
        self.H_x = gaussian_filter(self.H_x, sigma=self.smoothing_sigma)
        self.H_y = gaussian_filter(self.H_y, sigma=self.smoothing_sigma)
        self.update_visualization()

    def clear_field(self, event=None):
        """Clear the magnetic field"""
        self.H_x = np.zeros((self.grid_size, self.grid_size))
        self.H_y = np.zeros((self.grid_size, self.grid_size))
        self.update_visualization()

    def close_window(self, event=None):
        """Close the painter window"""
        self.window_open = False
        plt.close(self.fig)

    def update_brush_size(self, val):
        """Update brush size from slider"""
        self.brush_size = int(val)

    def update_brush_strength(self, val):
        """Update brush strength from slider"""
        self.brush_strength = val

    def update_visualization(self):
        """Update the field visualization"""
        # Clear previous plots
        self.ax_field.clear()
        self.ax_magnitude.clear()

        # Calculate magnitude
        H_magnitude = np.sqrt(self.H_x**2 + self.H_y**2)

        # Plot magnitude as heatmap on main axis
        extent = [-0.5, 0.5, -0.5, 0.5]
        im = self.ax_field.imshow(
            H_magnitude,
            extent=extent,
            origin='lower',
            cmap='viridis',
            alpha=0.8,
            vmin=0,
            vmax=500
        )

        # Plot vector field (downsampled)
        skip = max(1, self.grid_size // 20)
        x = np.linspace(-0.5, 0.5, self.grid_size)
        y = np.linspace(-0.5, 0.5, self.grid_size)
        X, Y = np.meshgrid(x, y)

        self.ax_field.quiver(
            X[::skip, ::skip],
            Y[::skip, ::skip],
            self.H_x[::skip, ::skip],
            self.H_y[::skip, ::skip],
            color='white',
            alpha=0.6,
            scale=np.max(H_magnitude) * 20 if np.max(H_magnitude) > 0 else 1
        )

        self.ax_field.set_xlim(extent[0], extent[1])
        self.ax_field.set_ylim(extent[2], extent[3])
        self.ax_field.set_xlabel('x (normalized)')
        self.ax_field.set_ylabel('y (normalized)')
        self.ax_field.set_title(
            'Draw Magnetic Field (Click & Drag to paint field in direction of motion)',
            fontsize=10
        )

        # Plot magnitude on side panel
        self.ax_magnitude.imshow(
            H_magnitude,
            origin='lower',
            cmap='viridis',
            vmin=0,
            vmax=500
        )
        self.ax_magnitude.set_title('Field Magnitude', fontsize=9)
        self.ax_magnitude.axis('off')

        # Add statistics
        max_field = np.max(H_magnitude)
        mean_field = np.mean(H_magnitude)
        self.ax_magnitude.text(
            0.5, -0.1,
            f'Max: {max_field:.1f} A/m\nMean: {mean_field:.1f} A/m',
            transform=self.ax_magnitude.transAxes,
            fontsize=8,
            ha='center'
        )

        self.fig.canvas.draw_idle()

    def get_field_array(self, normalize=True):
        """
        Get the magnetic field as a numpy array compatible with the model.

        Args:
            normalize: Whether to normalize the field (as done in data.py)

        Returns:
            np.ndarray: Shape (224, 224, 2) with H_x and H_y components
        """
        # Stack H_x and H_y
        H = np.stack([self.H_x, self.H_y], axis=-1)

        if normalize:
            # Apply same normalization as in data.py
            H_MEAN = 0.0
            H_STD = 1000.0
            H = (H - H_MEAN) / H_STD

        return H.astype(np.float32)

    def export_field(self, event=None, filename='drawn_field.npy'):
        """Export the field to a numpy file"""
        H = self.get_field_array(normalize=True)
        np.save(filename, H)
        print(f"Field exported to {filename}")
        print(f"Shape: {H.shape}")
        print(f"Range: [{H.min():.3f}, {H.max():.3f}]")


    def show(self, block=True):
        """Display the interactive painter"""
        try:
            plt.tight_layout()
        except:
            pass  # Ignore tight_layout warnings

        if block:
            # Ensure window is shown and brought to front
            self.fig.show()
            plt.show(block=True)
        else:
            plt.show(block=False)


def create_normalised_magnetic_field(grid_size=224, smoothing_sigma=3.0, normalize=True):
    """
    Interactive function to draw and return a magnetic field.

    Opens an interactive painter window. Draw your field, then close the window
    to return the field array.

    Args:
        grid_size: Resolution of the grid (default 224 to match model input)
        smoothing_sigma: Gaussian smoothing parameter
        normalize: Whether to normalize the output field

    Returns:
        np.ndarray: Magnetic field array of shape (grid_size, grid_size, 2)

    Example:
        >>> from magnetic_field_painter import create_magnetic_field
        >>> H = create_magnetic_field()
        # Draw your field, then close the window
        >>> print(H.shape)
        (224, 224, 2)
    """
    painter = MagneticFieldPainter(
        grid_size=grid_size,
        smoothing_sigma=smoothing_sigma
    )

    print("=== Magnetic Field Painter ===")
    print("Controls:")
    print("  - Click and drag to draw magnetic field in the direction of motion")
    print("  - Use sliders to adjust brush size and field strength")
    print("  - Click 'Smooth' to apply Gaussian smoothing")
    print("  - Click 'Clear' to reset the field")
    print("  - Click 'Done' when finished to return the field\n")

    painter.show(block=True)

    # Return the field array
    H = painter.get_field_array(normalize=normalize)
    print(f"\nField returned with shape: {H.shape}")
    print(f"Range: [{H.min():.3f}, {H.max():.3f}]")

    return H


def main():
    """
    Main function to run the magnetic field painter.

    Usage:
        python magnetic_field_painter.py

    Controls:
        - Click and drag: Draw magnetic field in the direction of drag
        - Brush Size slider: Adjust brush size
        - Strength slider: Adjust field strength
        - Smooth button: Apply Gaussian smoothing
        - Clear button: Reset field to zero
        - Done button: Close window and return field
    """
    painter = MagneticFieldPainter(
        grid_size=224,
        smoothing_sigma=3.0
    )

    print("=== Magnetic Field Painter ===")
    print("Controls:")
    print("  - Click and drag to draw magnetic field in the direction of motion")
    print("  - Use sliders to adjust brush size and field strength")
    print("  - Click 'Smooth' to apply Gaussian smoothing")
    print("  - Click 'Clear' to reset the field")
    print("  - Click 'Done' when finished")
    print("\nAfter drawing, you can call painter.predict_magnet_params() to infer magnet properties")
    print("or painter.export_field('myfield.npy') to save the field.\n")

    painter.show()

    return painter


if __name__ == "__main__":
    painter = main()