import numpy as np
from tqdm import tqdm
import tensorflow as tf
import magpylib as magpy
import random
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
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
        self.num_points = None
        self.local_path = 'tfrecords'
        self.gcs_path = 'tfrecords'
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

    @staticmethod
    def deserialise_example(serialised_example):
        '''parse single tfrecord to tensor'''

        feature = {
            'H': tf.io.FixedLenFeature([self.num_points * 2], tf.float32),
            'params': tf.io.FixedLenFeature([6], tf.float32),
        }

        parsed = tf.io.parse_single_example(serialised_example , feature)

        # Reshape H from flat array back to original shape
        H = tf.reshape(parsed['H'], [self.num_points, 2])  # [n_points, 2] for (Hx, Hy)
        params = parsed['params']

        return H, params

    def generate_cubiod_data(self, num_batches=10): #cuboid-shaped magnets
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
        self.num_points = len(points)

        #---save metadata---
        #np.savez(f'{self.local_path}/metadata.npz', magnets=magnets, points=points)
        #self.upload_to_gcloud(local_path=f'{self.local_path}/metadata.npz', gcs_path=f'{self.gcs_path}/metadata.npz')

        # ---split batch indices---
        split_idx = np.arange(num_batches)
        train_split_idx, val_split_idx = train_test_split(split_idx,
                                              test_size=(config.DATASET_CONFIG['val_split'] + config.DATASET_CONFIG[
                                                  'test_split']),
                                              random_state=config.RANDOM_SEED)
        val_split_idx, test_split_idx = train_test_split(val_split_idx,
                                             test_size=(config.DATASET_CONFIG[
                                                 'test_split']/(config.DATASET_CONFIG['val_split'] + config.DATASET_CONFIG[
                                                 'test_split'])), # test/(test+val)
                                             random_state=config.RANDOM_SEED)

        samples_per_batch = -(config.DATASET_CONFIG['dataset_size'] // -num_batches)  # ceiling division

        #---calculate magnetic field at each AOI point; save as tfrecord; save in batches (e.g. to reduce gcs API calls)

        for i, split in enumerate([train_split_idx, val_split_idx, test_split_idx]):
            for batch_idx in tqdm(split, desc=f"Generating {'train' if i==0 else 'val' if i==1 else 'test'} data"):
                name = ['train', 'val', 'test'][i]
                filename = f'{name}-{batch_idx:04d}.tfrecord'
                local_fullpath = f'{self.local_path}/{filename}'
                os.makedirs(os.path.dirname(local_fullpath), exist_ok=True)
                gcs_fullpath = f'{self.gcs_path}/{filename}'

                #indices
                batch_start = batch_idx * samples_per_batch
                batch_end = min(batch_start + samples_per_batch, config.DATASET_CONFIG['dataset_size'])

                #generate H and save tfrecord
                with tf.io.TFRecordWriter(local_fullpath) as writer:
                    for j in range(batch_start, batch_end):
                        if magnets[j] is not None: #last batch may not contain samples_per_batch samples
                            H_single = magpy.getH(magnets[j], points)[:, :2].astype(np.float32)
                            params = np.array([
                                magnets[j].position[0], magnets[j].position[1],
                                magnets[j].dimension[0], magnets[j].dimension[1],
                                magnets[j].polarization[0], magnets[j].polarization[1]
                            ], dtype=np.float32)

                            writer.write(self.serialise_example(H_single, params))

                #upload to gcs®
                self.upload_to_gcloud(local_fullpath, gcs_fullpath)
                os.remove(local_fullpath)


    def load_split_datasets(self, split='train'):
        '''tf dataset with AUTOTUNE to load from gcloud'''
        fullpath = f'{self.gcs_path}/{split}-*.tfrecord'
        files = tf.io.gfile.glob(fullpath)

        dataset = tf.data.Dataset.from_tensor_slices(files) #dataset of filenames

        #load data (not immediately)
        dataset = dataset.interleave( #interleave --> simultaneous
            lambda x : tf.data.TFRecordDataset(x), #filename --> TFRec reads file
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE #AUTOTUNE for optimised parallel loading
        )

        #parse tfrecord to tensors
        dataset = dataset.map(self.deserialise_example,
                              num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(config.DATASET_CONFIG['batch_size'])

        dataset = dataset.prefetch(tf.data.AUTOTUNE) #prefetch next batch during training


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

generator = Dataset()
generator.setup_gcloud()
generator.generate_cubiod_data()  # num_batches should <= dataset_size
