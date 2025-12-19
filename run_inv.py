import tensorflow as tf
import numpy as np
import magpylib as magpy
import matplotlib
matplotlib.use('MacOSX')  # Ensure proper backend for macOS
import matplotlib.pyplot as plt

import config
import magnetic_field_painter
import data
Dataset = data.Dataset()

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
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, (H, title, ax) in enumerate([(H1, title1, axes[0]), (H2, title2, axes[1])]):
        # Extract Hx and Hy and denormalize if requested
        if denormalize:
            Hx = H[:, :, 0] * Dataset.H_STD
            Hy = H[:, :, 1] * Dataset.H_STD
        else:
            Hx = H[:, :, 0]
            Hy = H[:, :, 1]
        H_magnitude = np.sqrt(Hx**2 + Hy**2)

        # Plot magnitude as heatmap
        extent = [-config.AOI_CONFIG['x_dim']/2, config.AOI_CONFIG['x_dim']/2,
                  -config.AOI_CONFIG['y_dim']/2, config.AOI_CONFIG['y_dim']/2]
        im = ax.imshow(H_magnitude, extent=extent, origin='lower', cmap='viridis', alpha=0.8)

        # Plot vector field (downsampled)
        if show_vectors:
            skip = max(1, 224 // 20)  # Show ~20x20 arrows
            x = np.linspace(-config.AOI_CONFIG['x_dim']/2, config.AOI_CONFIG['x_dim']/2, 224)
            y = np.linspace(-config.AOI_CONFIG['y_dim']/2, config.AOI_CONFIG['y_dim']/2, 224)
            X, Y = np.meshgrid(x, y)

            ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                     Hx[::skip, ::skip], Hy[::skip, ::skip],
                     color='white', alpha=0.6, scale=np.max(H_magnitude)*20 if np.max(H_magnitude) > 0 else 1)

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
iterations = 200

#---load model---
print("\n\n---Loading model---")
model_name = "model1.keras"
model = tf.keras.models.load_model(f"models/{model_name}")
print("Model loaded")

#---draw magnetic field---
H_actual = magnetic_field_painter.create_normalised_magnetic_field()
print("Magnetic field created")

#---iterate---
H = H_actual #already normalised
magnets = magpy.Collection()
maes = []

for i in range(iterations):
    #predict
    H_in = np.expand_dims(H, axis=0)
    params_normalised = model.predict(H_in, verbose=0) #input normalised H, output normalised params
    params_normalised /= 1

    #denormalise params
    x = params_normalised[0][0] * (2 * config.AOI_CONFIG['x_dim']) - config.AOI_CONFIG['x_dim']
    y = params_normalised[0][1] * (2 * config.AOI_CONFIG['y_dim']) - config.AOI_CONFIG['y_dim']
    a = params_normalised[0][2] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min']
    b = params_normalised[0][3] * (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']) + config.MAGNET_CONFIG['dim_min']
    Mx = params_normalised[0][4] * (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']) + config.MAGNET_CONFIG['M_min']
    My = params_normalised[0][5] * (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']) + config.MAGNET_CONFIG['M_min']

    #print(f"\nIteration {i+1} - Predicted magnet parameters:")
    #print(f"  Position: ({x:.2f}, {y:.2f}) m")
    #print(f"  Dimensions: ({a:.2f}, {b:.2f}) m")
    #print(f"  Magnetization: ({Mx:.3f}, {My:.3f}) T")

    #create magnet
    magnet = magpy.magnet.Cuboid(polarization=(Mx, My, 0),
                                 dimension=(np.abs(a), np.abs(b), 1),
                                 position=(x, y, 2.5)
                                 )
    magnets.add(magnet)

    #calculate resulting H, metrics
    H_pred = magpy.getH(magnets, Dataset.points)

    #drop z
    H_pred = H_pred[:, :2]

    #reshape from ~90,000, 2 --> 301, 301, 2
    H_pred = tf.reshape(H_pred, [int(config.AOI_CONFIG['x_dim'] / config.AOI_CONFIG['resolution']) + 1,
                                 int(config.AOI_CONFIG['y_dim'] / config.AOI_CONFIG['resolution']) + 1,
                                 2])
    #reshape to 224, 224, 2 using downsampling
    H_pred = tf.image.resize(H_pred, [224, 224], method='bilinear')

    #normalise
    H_pred = H_pred / Dataset.H_STD

    #visualise_H(str(i), H, H_pred, title1="Target", title2="Model")
    H_dif = H - H_pred
    mae = np.mean(np.abs(H_dif))
    maes.append(mae)
    print(f"Iteration {i+1}: mae = {mae:.5f}")
    H = H_dif

plt.plot(maes)
plt.xlabel('iteration')
plt.ylabel('mae')
plt.show()