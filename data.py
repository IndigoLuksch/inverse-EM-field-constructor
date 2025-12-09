import numpy as np
from tqdm import tqdm
import tensorflow as tf
import magpylib as magpy
import random
from scipy.stats import qmc
import os
from google.cloud import storage
import time
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config
random.seed(config.RANDOM_SEED)

#magnetic field data generation
class Dataset:
    '''
    Generates magnetic field data for: 
    • Magnets with random positions (x, y), dimensions (a, b), and magnetisations (Mx, My)
    • Associated magnetic field maps (H_x, H_y) over the area of interest 
    using latin hypercube sampling and following the specifications in config.py
    '''
    def __init__(self):
        self.magnets = None
        self.H = None
        self.points = None
        self.local_path = '/tfrecords'
        self.gcs_path = None
        self.bucket = None

    def setup_gcloud(self):
        '''Initialise Google Cloud Storage and bucket'''
        storage_client = storage.Client()
        bucket_name = config.DATASET_CONFIG['bucket_name']
        self.bucket = storage_client.get_bucket(bucket_name)
        print(f"Bucket {bucket_name} created")

    def upload_to_gcloud(self, local_path, gcs_path):
        '''Upload a file to GCS'''
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

    @staticmethod
    def serialise_example(H, params): #params is a np array
        '''Convert H, params to tfrecord file format'''
        feature = {
            'H' : tf.train.Feature(float_list=tf.train.FloatList(value=H.flatten())),
            'params' : tf.train.Feature(float_list=tf.train.FloatList(value=params)),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def generate_cubiod_data(self, samples_per_batch): #cuboid-shaped magnets
        #---generate magpy magnet collection---
        sampler = qmc.LatinHypercube(d=6)
        samples = sampler.random(n=config.DATASET_CONFIG['dataset_size'])

        x_samples = qmc.scale(samples[:,0:1], -config.AOI_CONFIG['x_dim'], config.AOI_CONFIG['x_dim']).flatten() #twice AOI size
        y_samples = qmc.scale(samples[:,1:2], -config.AOI_CONFIG['y_dim'], config.AOI_CONFIG['y_dim']).flatten() #twice AOI size
        a_samples = qmc.scale(samples[:,2:3], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
        b_samples = qmc.scale(samples[:, 3:4], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
        Mx_samples = qmc.scale(samples[:,4:5], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()
        My_samples = qmc.scale(samples[:, 5:6], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()

        magnets = []
        for i in tqdm(range(config.DATASET_CONFIG['dataset_size'])):
            magnet = magpy.magnet.Cuboid(polarization=(Mx_samples[i], My_samples[i], 0),
                                         dimension=(a_samples[i], b_samples[i], 1),
                                         position=(x_samples[i], y_samples[i], 0))
            magnets.append(magnet)

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

        #---save metadata---
        np.savez(f'{self.local_path}/metadata.npz', magnets=magnets, points=points)
        self.upload_to_gcloud(local_path=f'{self.local_path}/metadata.npz', gcs_path=f'{self.gcs_path}/metadata.npz')

        #---calculate magnetic field at each AOI point; save as tfrecord; save in batches (e.g. to reduce gcs API calls)
        num_batches = math.ceil(config.DATASET_CONFIG['dataset_size'] / samples_per_batch)
        for batch_idx in range(num_batches):
            batch_start = batch_idx * samples_per_batch
            batch_end = min(batch_start + samples_per_batch, config.DATASET_CONFIG['dataset_size'])

            filename = f'{batch_idx:04d}-of-{num_batches}.tfrecord'
            local_fullpath = f'{self.local_path}/{filename}'
            gcs_fullpath = f'{self.gcs_path}/{filename}'

            #generate H and save tfrecord
            with tf.io.TFRecordWriter(local_fullpath) as writer:
                H_single = magpy.getH(magnets[i], points)[:, :2].astype(np.float32)
                params = np.array([
                    magnets[i].position[0], magnets[i].position[1],
                    magnets[i].dimension[0], magnets[i].dimension[1],
                    magnets[i].polarization[0], magnets[i].polarization[1]
                ], dtype=np.float32)

                writer.write(self.serialise_example(H_single, params))

            #upload to gcs
            self.upload_to_gcloud(local_fullpath, gcs_fullpath)
            os.remove(local_fullpath)

            print("Data generated and saved to gcloud")


    #need to fix:
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
        if self.magnets is None:
            metadata = np.load(f'{self.local_path}/metadata.npz', allow_pickle=True)
            self.magnets = metadata['magnets']
            self.points = metadata['points']

            # Select random sample
        idx = random.randint(0, len(self.magnets) - 1)
        magnet = self.magnets[idx]

        # Load the corresponding H field
        H_sample = np.load(f'data/H_{idx:06d}.npy')

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

