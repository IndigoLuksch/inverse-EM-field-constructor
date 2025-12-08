import numpy as np
from tqdm import tqdm
import magpylib as magpy
import random
import math
from scipy.stats import qmc
import time
import config
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

random.seed(config.RANDOM_SEED)

#magnetic field data generation
class CuboidDataGenerator:
    '''
    Generates magnetic field data for: 
    • Cuboid magnets with random positions (x, y), side lengths (a, b), and magnetisations (Mx, My)
    • Associated magnetic field maps (H_x, H_y) over the area of interest 
    using latin hypercube sampling and following the specifications in config.py
    '''
    def __init__(self):
        self.magnets = None
        self.H = None
        self.points = None

    def generate_data(self, batch_size=2000):
        #---generate magpy magnet collection---
        pbar = tqdm(total=config.TRAINING_CONFIG['dataset_size'], desc="Creating magnets")
        sampler = qmc.LatinHypercube(d=6)
        samples = sampler.random(n=config.TRAINING_CONFIG['dataset_size'])

        x_samples = qmc.scale(samples[:,0:1], -config.AOI_CONFIG['x_dim'], config.AOI_CONFIG['x_dim']).flatten() #twice AOI size
        y_samples = qmc.scale(samples[:,1:2], -config.AOI_CONFIG['y_dim'], config.AOI_CONFIG['y_dim']).flatten() #twice AOI size
        a_samples = qmc.scale(samples[:,2:3], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
        b_samples = qmc.scale(samples[:, 3:4], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
        Mx_samples = qmc.scale(samples[:,4:5], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()
        My_samples = qmc.scale(samples[:, 5:6], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()

        magnets = []
        for i in range(config.TRAINING_CONFIG['dataset_size']):
            magnet = magpy.magnet.Cuboid(polarization=(Mx_samples[i], My_samples[i], 0),
                                         dimension=(a_samples[i], b_samples[i], 1),
                                         position=(x_samples[i], y_samples[i], 0))
            magnets.append(magnet)
            pbar.update(1)
        pbar.close()

        #---generate AOI points---
        x = np.arange(-config.AOI_CONFIG['x_dim'] / 2,
                    config.AOI_CONFIG['x_dim'] / 2 + config.AOI_CONFIG['resolution'],
                    config.AOI_CONFIG['resolution'])

        y = np.arange(-config.AOI_CONFIG['y_dim'] / 2,
                      config.AOI_CONFIG['y_dim'] / 2 + config.AOI_CONFIG['resolution'],
                      config.AOI_CONFIG['resolution'])

        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        #---calculate magnetic field at each AOI point
        pbar = tqdm(total = math.ceil(config.TRAINING_CONFIG['dataset_size']/batch_size), desc='Calculating H fields', unit='batch')

        for batch_start in range(0, config.TRAINING_CONFIG['dataset_size'], batch_size):
            batch_end = min(batch_start + batch_size, config.TRAINING_CONFIG['dataset_size'])
            H_batch = np.zeros((batch_end-batch_start,  # number of magnets
                          points.shape[0],  # number of points
                          2))  # Hx, Hy
            for i in range(batch_end-batch_start):
                H_batch[i] = magpy.getH(magnets[i], points)[:, :2]  #only store Hx, Hy
            if batch_start == 0:
                np.savez('generated_data.npz', H=H_batch, magnets=magnets, points=points)
            else:
                data = np.load('generated_data.npz')
                H_cumulative = np.concatenate( [ data['H'], H_batch ] )
                np.savez('generated_data.npz', H=H_cumulative, magnets=magnets, points=points)
            pbar.update(1)
        pbar.close()

        #store as instance variables
        data = np.load('generated_data.npz')
        H = data['H']
        self.magnets = magnets
        self.H = H
        self.points = points

    def visualize_random_sample(self):
        '''
        VIBE CODED
        • note that field is only calculated in AOI
        Visualizes a random sample from the generated dataset showing:
        • Magnetic field magnitude as heatmap
        • Field direction as vector arrows
        • Magnet position and dimensions as a rectangle
        • Magnetization vector as an arrow
        '''
        if self.magnets is None or self.H is None:
            raise ValueError("No data to visualize. Run generate_magnets() first.")

        #select random sample
        idx = random.randint(0, len(self.magnets) - 1)
        magnet = self.magnets[idx]
        H_sample = self.H[idx]

        #get magnet properties
        pos = magnet.position
        dim = magnet.dimension
        pol = magnet.polarization

        #reshape H field for visualization
        grid_size = int(np.sqrt(self.points.shape[0]))
        Hx = H_sample[:, 0].reshape(grid_size, grid_size)
        Hy = H_sample[:, 1].reshape(grid_size, grid_size)
        H_magnitude = np.sqrt(Hx**2 + Hy**2)

        #create meshgrid for quiver plot
        x = np.linspace(-config.AOI_CONFIG['x_dim']/2, config.AOI_CONFIG['x_dim']/2, grid_size)
        y = np.linspace(-config.AOI_CONFIG['y_dim']/2, config.AOI_CONFIG['y_dim']/2, grid_size)
        X, Y = np.meshgrid(x, y)

        #downsample for quiver plot (every nth point)
        skip = max(1, grid_size // 20)

        #create single figure
        fig, ax = plt.subplots(figsize=(10, 9))

        #plot magnitude as heatmap
        im = ax.imshow(H_magnitude, extent=[-config.AOI_CONFIG['x_dim']/2, config.AOI_CONFIG['x_dim']/2,
                                            -config.AOI_CONFIG['y_dim']/2, config.AOI_CONFIG['y_dim']/2],
                       origin='lower', cmap='viridis', alpha=0.8)

        #plot vector field (downsampled)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  Hx[::skip, ::skip], Hy[::skip, ::skip],
                  color='white', alpha=0.6, scale=np.max(H_magnitude)*20)

        #plot magnet as rectangle
        ax.add_patch(Rectangle((pos[0] - dim[0]/2, pos[1] - dim[1]/2),
                                dim[0], dim[1],
                                fill=False, edgecolor='red', linewidth=3))

        #plot magnetization vector
        ax.arrow(pos[0], pos[1], pol[0]*3, pol[1]*3,
                 head_width=1, head_length=1, fc='red', ec='red', linewidth=3)

        ax.set_title(f'Magnetic Field Distribution (Sample {idx})', fontsize=14, fontweight='bold')
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)

        cbar = plt.colorbar(im, ax=ax, label='|H| (A/m)')
        cbar.ax.tick_params(labelsize=10)

        #add info text
        info_text = (f'Magnet Position: ({pos[0]:.2f}, {pos[1]:.2f}) m\n'
                     f'Dimensions: ({dim[0]:.2f}, {dim[1]:.2f}) m\n'
                     f'Polarization: ({pol[0]:.3f}, {pol[1]:.3f}) T')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def load_training_data(self, filename='generated_data.npz'):
        data = np.load(filename)
        self.H = data['H']
        self.magnets = data['magnets']
        self.points = data['points']
        print("Data loaded")

if __name__ == '__main__':
    generator = CuboidDataGenerator()
    generator.generate_data(1000)
    generator.visualize_random_sample()
