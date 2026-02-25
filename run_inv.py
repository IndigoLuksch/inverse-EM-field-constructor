import tensorflow as tf
import numpy as np
import magpylib as magpy
import matplotlib
matplotlib.use('MacOSX')  # Ensure proper backend for macOS
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

import config
import magnetic_field_painter
import data
Dataset = data.Dataset()
import magnet_field_tf
import model as Model

def visualise_H(filename, H1, H2, title1="H Field 1", title2="H Field 2", show_vectors=True, denormalize=True):
    """
    Visualize two H fields side by side for comparison.

    Args:
        H1: First H field, shape (224, 224, 2) - [Hx, Hy], normalized
        H2: Second H field, shape (224, 224, 2) - [Hx, Hy], normalized
        title1: Title for first field
        title2: Title for second field
        show_vectors: Whether to show vector arrows
        denormalize: Whether to denormalize fields for display
    """
    # Pre-calculate magnitudes to determine shared colorbar scale
    H_magnitudes = []
    for H in [H1, H2]:
        if denormalize:
            Hx = H[:, :, 0] * Dataset.H_STD
            Hy = H[:, :, 1] * Dataset.H_STD
        else:
            Hx = H[:, :, 0]
            Hy = H[:, :, 1]
        H_magnitudes.append(np.sqrt(Hx**2 + Hy**2))

    # Calculate shared colorbar limits
    vmin = min(np.min(H_magnitudes[0]), np.min(H_magnitudes[1]))
    vmax = max(np.max(H_magnitudes[0]), np.max(H_magnitudes[1]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, (H, title, ax) in enumerate([(H1, title1, axes[0]), (H2, title2, axes[1])]):
        # Extract Hx and Hy and denormalize if requested
        if denormalize:
            Hx = H[:, :, 0] * Dataset.H_STD
            Hy = H[:, :, 1] * Dataset.H_STD
        else:
            Hx = H[:, :, 0]
            Hy = H[:, :, 1]
        H_magnitude = H_magnitudes[idx]

        # Plot magnitude as heatmap with shared colorbar limits
        extent = [-config.AOI_CONFIG['x_dim']/2, config.AOI_CONFIG['x_dim']/2,
                  -config.AOI_CONFIG['y_dim']/2, config.AOI_CONFIG['y_dim']/2]
        im = ax.imshow(H_magnitude, extent=extent, origin='lower', cmap='viridis', alpha=0.8,
                      vmin=vmin, vmax=vmax)

        # Plot vector field (downsampled)
        if show_vectors:
            # Get actual H field shape from input
            h_shape = H.shape[0]
            skip = max(1, h_shape // 20)  # Show ~20x20 arrows
            x = np.linspace(-config.AOI_CONFIG['x_dim']/2, config.AOI_CONFIG['x_dim']/2, h_shape)
            y = np.linspace(-config.AOI_CONFIG['y_dim']/2, config.AOI_CONFIG['y_dim']/2, h_shape)
            X, Y = np.meshgrid(x, y)

            ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                     Hx[::skip, ::skip], Hy[::skip, ::skip],
                     color='white', alpha=0.6, scale=vmax*20 if vmax > 0 else 1)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='|H| (A/m)')
        cbar.ax.tick_params(labelsize=10)

        # Add stats text
        stats_text = (f'Max: {np.max(H_magnitude):.2f} A/m\n'
                     f'Mean: {np.mean(H_magnitude):.2f} A/m\n'
                     f'Min: {np.min(H_magnitude):.2f} A/m')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()
    #plt.savefig(f"scratch/run_inv/{filename}.png")
    plt.close()

#---parameters

#---load model---
print("\n\n---Loading model---")
model_name = "model4.keras"
model = tf.keras.models.load_model(f"models/{model_name}",
                                   custom_objects={'custom_loss' : Model.custom_loss_polar})
print("Model loaded")

#---iterate---
magnets = magpy.Collection()
maes = []

def test_model(trials=1, iterations=5, visualise=False, use_generated_data=True, scale=1.0, verbose=False):
    '''
    trials: number of magnets to test
    iterations: number of prediction iterations per magnet (residual loss)
    visualise: visualise H fields after each iteration?
    scale: scale factor for predicted magnetisation
    '''
    for _ in tqdm(range(trials)):
        # ---get magnetic field---
        if use_generated_data:
            print("\n---Generating test data---")
            # Generate random magnet parameters
            import random

            random.seed()  # Random seed for different data each time

            x_actual = random.uniform(-config.AOI_CONFIG['x_dim'], config.AOI_CONFIG['x_dim'])
            y_actual = random.uniform(-config.AOI_CONFIG['y_dim'], config.AOI_CONFIG['y_dim'])
            a_actual = random.uniform(config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max'])
            b_actual = random.uniform(config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max'])
            Mr_actual = random.uniform(config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max'])
            Mtheta_actual = random.uniform(-np.pi, np.pi)

            # Convert polar to Cartesian for magnet creation
            Mx_actual = Mr_actual * np.cos(Mtheta_actual)
            My_actual = Mr_actual * np.sin(Mtheta_actual)

            print(f"Ground truth magnet parameters:")
            print(f"  Position: ({x_actual:.2f}, {y_actual:.2f}) m")
            print(f"  Dimensions: ({a_actual:.2f}, {b_actual:.2f}) m")
            print(f"  Magnetization: Mr={Mr_actual:.3f} T, θ={Mtheta_actual:.3f} rad (Mx={Mx_actual:.3f}, My={My_actual:.3f}) T")

            # Generate H field using magpylib
            magnet_actual = magpy.magnet.Cuboid(
                polarization=(Mx_actual, My_actual, 0),
                dimension=(a_actual, b_actual, 1),
                position=(x_actual, y_actual, 2.5)
            )
            H_field = magpy.getH(magnet_actual, Dataset.points)[:, :2]

            # Reshape and normalize (same as training data pipeline)
            H_field = tf.reshape(H_field, [
                int(config.AOI_CONFIG['x_dim'] / config.AOI_CONFIG['resolution']) + 1,
                int(config.AOI_CONFIG['y_dim'] / config.AOI_CONFIG['resolution']) + 1,
                2
            ])
            H_actual = (H_field / Dataset.H_STD).numpy()  # Normalize and convert to numpy

            print("Test data generated")
        else:
            print("\n---Draw magnetic field---")
            H_actual = magnetic_field_painter.create_normalised_magnetic_field(grid_size=301)
            print("Magnetic field created")
        H = H_actual  # already normalised

        for i in range(iterations):
            #---predict---
            H_in = np.expand_dims(H, axis=0)
            params_normalised = model.predict(H_in, verbose=0) #input normalised H, output normalised params
            params_normalised /= 1

            #denormalise params (polar coordinates)
            x_pred = params_normalised[0][0] * (2 * config.AOI_CONFIG['x_dim']) - config.AOI_CONFIG['x_dim']
            y_pred = params_normalised[0][1] * (2 * config.AOI_CONFIG['y_dim']) - config.AOI_CONFIG['y_dim']
            a = params_normalised[0][2] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min']
            b = params_normalised[0][3] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min']
            Mr = params_normalised[0][4] * (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']) + config.MAGNET_CONFIG['M_min']
            Mtheta = params_normalised[0][5] * (2 * np.pi) - np.pi

            # Convert polar to Cartesian for field calculations
            Mx = Mr * np.cos(Mtheta)
            My = Mr * np.sin(Mtheta)

            #print(f"\nIteration {i+1} - Predicted magnet parameters:")
            #print(f"  Position: ({x_pred:.2f}, {y_pred:.2f}) m")
            #print(f"  Dimensions: ({a:.2f}, {b:.2f}) m")
            #print(f"  Magnetization: Mr={Mr:.3f} T, θ={Mtheta:.3f} rad (Mx={Mx:.3f}, My={My:.3f}) T")

            #---create magnet and compute predicted H field---
            polarisation = (scale*Mx, scale*My, 0)
            magnet = magpy.magnet.Cuboid(polarization=polarisation,
                                         dimension=(np.abs(a), np.abs(b), 1),
                                         position=(x_pred, y_pred, 2.5)
                                         )
            magnets.add(magnet)

            H_pred = magnet_field_tf.compute_H_field(
                observers=Dataset.points,
                dimension=(np.abs(a), np.abs(b), 1),
                polarization=polarisation,
                position=(x_pred, y_pred, 2.5)
            )
            #drop z
            H_pred = H_pred[:, :2]
            #reshape from ~90,000, 2 --> 301, 301, 2
            H_pred = tf.reshape(H_pred, [int(config.AOI_CONFIG['x_dim'] / config.AOI_CONFIG['resolution']) + 1,
                                         int(config.AOI_CONFIG['y_dim'] / config.AOI_CONFIG['resolution']) + 1,
                                         2])
            #normalise (keep 301x301 resolution to match model input)
            H_pred = (H_pred / Dataset.H_STD).numpy()

            #---results---
            #if 6 < i < 24:
            if visualise == True:
                visualise_H(str(i), H, H_pred, title1="Target", title2="Model")
            H_dif = H - H_pred
            mae = np.mean(np.abs(H_dif))
            maes.append(mae)
            H = H_dif

            if verbose:
                print(f"Iteration {i+1}: mae = {mae:.5f}")

    return maes



plt.plot(maes)
plt.xlabel('iteration')
plt.ylabel('mae')
plt.show()