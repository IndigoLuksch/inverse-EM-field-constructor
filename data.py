import numpy as np
from tqdm import tqdm
import tensorflow as tf
import magpylib as magpy
import random
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
import os
#from google.cloud import storage
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

        #---file paths---
        self.local_path = './data/default_folder'
        self.gcs_blob_path = 'tfrecords'
        self.gcs_bucket_name = 'inverse-em-2'
        self.bucket = None

        #---AOI points---
        x = np.arange(-config.AOI_CONFIG['x_dim'] / 2,
                      config.AOI_CONFIG['x_dim'] / 2 + config.AOI_CONFIG['resolution'],
                      config.AOI_CONFIG['resolution'])

        y = np.arange(-config.AOI_CONFIG['y_dim'] / 2,
                      config.AOI_CONFIG['y_dim'] / 2 + config.AOI_CONFIG['resolution'],
                      config.AOI_CONFIG['resolution'])

        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        self.points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        self.num_points = len(self.points)

        #---data normalisation---
        self.H_STD = 1000

    def visualise_random_sample(self, use_gcloud=False, split='train', num_samples=1):
        '''
        VIBE CODED

        Visualizes random samples from the tfrecord dataset showing:
        • Magnetic field magnitude as heatmap
        • Field direction as vector arrows
        • Magnet position and dimensions as a rectangle
        • Magnetization vector as an arrow

        Args:
            use_gcloud: If True, load from GCS bucket. If False, load from local tfrecords
            split: Which split to load from ('train', 'val', or 'test')
            num_samples: Number of random samples to visualize
        '''
        # Get list of tfrecord files
        if use_gcloud:
            fullpath = f'gs://{self.gcs_bucket_name}/{self.gcs_blob_path}/{split}-*.tfrecord'
            files = tf.io.gfile.glob(fullpath)
            if not files:
                print(f"No files found at {fullpath}")
                return
        else:
            import glob
            fullpath = f'{self.local_path}/{split}-*.tfrecord'
            files = glob.glob(fullpath)
            if not files:
                print(f"No files found at {fullpath}. Make sure tfrecords exist locally.")
                return

        print(f"Found {len(files)} tfrecord files")

        # Load dataset
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(lambda x: tf.io.parse_single_example(x, {
            'H': tf.io.FixedLenFeature([self.num_points * 2], tf.float32),
            'params': tf.io.FixedLenFeature([6], tf.float32),
        }))

        # Randomly sample
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.take(num_samples)

        # Visualize each sample
        for idx, sample in enumerate(dataset):
            # Extract H field and params
            H_flat = sample['H'].numpy()
            params = sample['params'].numpy()

            # Reshape H field: flat -> (301, 301, 2)
            grid_size = int(config.AOI_CONFIG['x_dim'] / config.AOI_CONFIG['resolution']) + 1
            H_field = H_flat.reshape(grid_size, grid_size, 2)
            Hx = H_field[:, :, 0]
            Hy = H_field[:, :, 1]
            H_magnitude = np.sqrt(Hx**2 + Hy**2)

            # Extract magnet parameters (unnormalized)
            pos_x, pos_y = params[0], params[1]
            dim_a, dim_b = params[2], params[3]
            pol_Mx, pol_My = params[4], params[5]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 9))

            # Plot magnitude as heatmap
            extent = [-config.AOI_CONFIG['x_dim']/2, config.AOI_CONFIG['x_dim']/2,
                      -config.AOI_CONFIG['y_dim']/2, config.AOI_CONFIG['y_dim']/2]
            im = ax.imshow(H_magnitude, extent=extent, origin='lower', cmap='viridis', alpha=0.8)

            # Plot vector field (downsampled)
            skip = max(1, grid_size // 20)
            x = np.linspace(-config.AOI_CONFIG['x_dim']/2, config.AOI_CONFIG['x_dim']/2, grid_size)
            y = np.linspace(-config.AOI_CONFIG['y_dim']/2, config.AOI_CONFIG['y_dim']/2, grid_size)
            X, Y = np.meshgrid(x, y)

            ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                     Hx[::skip, ::skip], Hy[::skip, ::skip],
                     color='white', alpha=0.6, scale=np.max(H_magnitude)*20 if np.max(H_magnitude) > 0 else 1)

            # Plot magnet as rectangle
            ax.add_patch(Rectangle((pos_x - dim_a/2, pos_y - dim_b/2),
                                   dim_a, dim_b,
                                   fill=False, edgecolor='red', linewidth=3))

            # Plot magnetization vector (scaled for visibility)
            arrow_scale = min(dim_a, dim_b) * 0.5
            ax.arrow(pos_x, pos_y, pol_Mx*arrow_scale, pol_My*arrow_scale,
                    head_width=1, head_length=1, fc='red', ec='red', linewidth=3)

            ax.set_title(f'Magnetic Field Distribution (Sample {idx+1})', fontsize=14, fontweight='bold')
            ax.set_xlabel('x (m)', fontsize=12)
            ax.set_ylabel('y (m)', fontsize=12)

            cbar = plt.colorbar(im, ax=ax, label='|H| (A/m)')
            cbar.ax.tick_params(labelsize=10)

            # Add info text
            info_text = (f'Magnet Position: ({pos_x:.2f}, {pos_y:.2f}) m\n'
                        f'Dimensions: ({dim_a:.2f}, {dim_b:.2f}) m\n'
                        f'Polarization: ({pol_Mx:.3f}, {pol_My:.3f}) T\n'
                        f'Max |H|: {np.max(H_magnitude):.2f} A/m\n'
                        f'Mean |H|: {np.mean(H_magnitude):.2f} A/m')
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            plt.show()

        print(f"Visualized {num_samples} sample(s) from {split} split")

    def load_training_data(self, filename='generated_data.npz'):
        data = np.load(filename)
        self.H = data['H']
        self.magnets = data['magnets']
        self.points = data['points']
        print("Data loaded")

    def generate_cuboid_data_TF_dataset(self):
        """
        generate data as tf.dataset and save locally
        """
        # ---generate magpy magnet collection---
        sampler = qmc.LatinHypercube(d=6)
        samples = sampler.random(n=config.DATASET_CONFIG['dataset_size'])

        x_samples = qmc.scale(samples[:, 0:1], -config.AOI_CONFIG['x_dim'],
                              config.AOI_CONFIG['x_dim']).flatten()  # twice AOI size
        y_samples = qmc.scale(samples[:, 1:2], -config.AOI_CONFIG['y_dim'],
                              config.AOI_CONFIG['y_dim']).flatten()  # twice AOI size
        a_samples = qmc.scale(samples[:, 2:3], config.MAGNET_CONFIG['dim_min'],
                              config.MAGNET_CONFIG['dim_max']).flatten()
        b_samples = qmc.scale(samples[:, 3:4], config.MAGNET_CONFIG['dim_min'],
                              config.MAGNET_CONFIG['dim_max']).flatten()
        Mx_samples = qmc.scale(samples[:, 4:5], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()
        My_samples = qmc.scale(samples[:, 5:6], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()

        #slice arrays
        x = qmc.scale(samples[:, 0:1], -config.AOI_CONFIG['x_dim'], config.AOI_CONFIG['x_dim']).flatten()
        y = qmc.scale(samples[:, 1:2], -config.AOI_CONFIG['y_dim'], config.AOI_CONFIG['y_dim']).flatten()
        a = qmc.scale(samples[:, 2:3], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
        b = qmc.scale(samples[:, 3:4], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
        Mx = qmc.scale(samples[:, 4:5], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()
        My = qmc.scale(samples[:, 5:6], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()

        #magnetisation: cartesian --> polar
        Mr = np.sqrt(Mx ** 2 + My ** 2)
        Mtheta = np.arctan2(My, Mx)

        # Stack parameters: (N, 6)
        all_params = np.stack([x, y, a, b, Mr, Mtheta], axis=1).astype(np.float32)

        #---generator function for TF dataset---
        def generate_cuboid_data_TF_dataset(self):
            """
            generate data as tf.dataset and save locally
            """
            # ---generate magpy magnet collection---
            sampler = qmc.LatinHypercube(d=6)
            samples = sampler.random(n=config.DATASET_CONFIG['dataset_size'])

            x_samples = qmc.scale(samples[:, 0:1], -config.AOI_CONFIG['x_dim'],
                                  config.AOI_CONFIG['x_dim']).flatten()  # twice AOI size
            y_samples = qmc.scale(samples[:, 1:2], -config.AOI_CONFIG['y_dim'],
                                  config.AOI_CONFIG['y_dim']).flatten()  # twice AOI size
            a_samples = qmc.scale(samples[:, 2:3], config.MAGNET_CONFIG['dim_min'],
                                  config.MAGNET_CONFIG['dim_max']).flatten()
            b_samples = qmc.scale(samples[:, 3:4], config.MAGNET_CONFIG['dim_min'],
                                  config.MAGNET_CONFIG['dim_max']).flatten()
            Mx_samples = qmc.scale(samples[:, 4:5], config.MAGNET_CONFIG['M_min'],
                                   config.MAGNET_CONFIG['M_max']).flatten()
            My_samples = qmc.scale(samples[:, 5:6], config.MAGNET_CONFIG['M_min'],
                                   config.MAGNET_CONFIG['M_max']).flatten()

            # slice arrays
            x = qmc.scale(samples[:, 0:1], -config.AOI_CONFIG['x_dim'], config.AOI_CONFIG['x_dim']).flatten()
            y = qmc.scale(samples[:, 1:2], -config.AOI_CONFIG['y_dim'], config.AOI_CONFIG['y_dim']).flatten()
            a = qmc.scale(samples[:, 2:3], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
            b = qmc.scale(samples[:, 3:4], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
            Mx = qmc.scale(samples[:, 4:5], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()
            My = qmc.scale(samples[:, 5:6], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()

            # magnetisation: cartesian --> polar
            Mr = np.sqrt(Mx ** 2 + My ** 2)
            Mtheta = np.arctan2(My, Mx)

            # Stack parameters: (N, 6)
            all_params = np.stack([x, y, a, b, Mr, Mtheta], axis=1).astype(np.float32)

            #---generator function for TF dataset---
            def gen_dataset_split(start_idx, end_idx, batch_size=100):
                def data_generator():
                    total_items = end_idx - start_idx
                    total_batches = int(np.ceil(total_items / batch_size))

                    with tqdm(total=total_batches, desc=f"Generating {start_idx}-{end_idx}", unit="batch") as pbar:
                        for i in range(start_idx, end_idx, batch_size):
                            batch_end = min(i + batch_size, end_idx)
                            current_batch_size = batch_end - i

                            #create magnet batch of size batch_size
                            magnets = [
                                magpy.magnet.Cuboid(
                                    polarization=(Mx[j], My[j], 0),
                                    dimension=(a[j], b[j], 1),
                                    position=(x[j], y[j], 2.5)
                                ) for j in range(i, batch_end)
                            ]

                            H_batch = magpy.getH(magnets, self.points)
                            H_batch = H_batch[:, :, :2].astype(np.float32)

                            for k in range(current_batch_size):
                                yield all_params[i + k], H_batch[k]
                            pbar.update(1)

                return tf.data.Dataset.from_generator(
                    data_generator,
                    output_signature=(
                        tf.TensorSpec(shape=(6,), dtype=tf.float32),
                        tf.TensorSpec(shape=(90601, 2), dtype=tf.float32)
                    )
                )

            #---generate datasets---
            dataset_size = config.DATASET_CONFIG['dataset_size']
            train_size = int(dataset_size * config.DATASET_CONFIG['train_split'])
            val_size = int(dataset_size * config.DATASET_CONFIG['val_split'])

            train_ds = gen_dataset_split(0, train_size)
            val_ds = gen_dataset_split(train_size, train_size + val_size)
            test_ds = gen_dataset_split(train_size + val_size, dataset_size)

            tf.data.Dataset.save(train_ds, self.local_path + "/train_ds")
            tf.data.Dataset.save(val_ds, self.local_path + "/val_ds")
            tf.data.Dataset.save(test_ds, self.local_path + "/test_ds")


    #--------------------------------------
    #---BELOW FUNCTIONS NO LONGER IN USE---
    #previously used to save data as TFRecord and upload to gcloud
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

    def deserialise_normalise_example(self, serialised_example):
        '''parse single tfrecord to tensor and reshape to model input size'''

        feature = {
            'H': tf.io.FixedLenFeature([self.num_points * 2], tf.float32),
            'params': tf.io.FixedLenFeature([6], tf.float32),
        }

        parsed = tf.io.parse_single_example(serialised_example , feature)

        #reshape H to input shape of ResNet50
        H = tf.reshape(parsed['H'], [int(config.AOI_CONFIG['x_dim'] / config.AOI_CONFIG['resolution']) + 1,
                                      int(config.AOI_CONFIG['y_dim'] / config.AOI_CONFIG['resolution']) + 1,
                                      2])
        H = tf.image.resize(H, [224, 224], method='bilinear')

        #normalise H
        H_MEAN = 0.0
        H_STD = self.H_STD  #estimated typical max H (across all data)
        H = (H - H_MEAN) / H_STD

        #normalise params to [0, 1] using ranges from config.py
        params = parsed['params']
        params = tf.stack([
            (params[0] + config.AOI_CONFIG['x_dim']) / (2 * config.AOI_CONFIG['x_dim']),  # x: -30 to 30 -> 0 to 1
            (params[1] + config.AOI_CONFIG['y_dim']) / (2 * config.AOI_CONFIG['y_dim']),  # y: -30 to 30 -> 0 to 1
            (params[2] - config.MAGNET_CONFIG['dim_min']) / (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']),  # a
            (params[3] - config.MAGNET_CONFIG['dim_min']) / (config.MAGNET_CONFIG['dim_max'] - config.MAGNET_CONFIG['dim_min']),  # b
            (params[4] - config.MAGNET_CONFIG['M_min']) / (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']),  # Mx
            (params[5] - config.MAGNET_CONFIG['M_min']) / (config.MAGNET_CONFIG['M_max'] - config.MAGNET_CONFIG['M_min']),  # My
        ])

        return H, params

    def generate_cuboid_data_TFRECORD(self, use_gcs=False, num_batches=1000):
        """
        generate data as TFRecord and upload to gcloud (this is OVERKILL for the ~80GB dataset I'm using)
        """
        #---generate magpy magnet collection---
        sampler = qmc.LatinHypercube(d=6)
        samples = sampler.random(n=config.DATASET_CONFIG['dataset_size'])

        x_samples = qmc.scale(samples[:,0:1], -config.AOI_CONFIG['x_dim'], config.AOI_CONFIG['x_dim']).flatten() #twice AOI size
        y_samples = qmc.scale(samples[:,1:2], -config.AOI_CONFIG['y_dim'], config.AOI_CONFIG['y_dim']).flatten() #twice AOI size
        a_samples = qmc.scale(samples[:,2:3], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
        b_samples = qmc.scale(samples[:, 3:4], config.MAGNET_CONFIG['dim_min'], config.MAGNET_CONFIG['dim_max']).flatten()
        Mx_samples = qmc.scale(samples[:,4:5], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()
        My_samples = qmc.scale(samples[:, 5:6], config.MAGNET_CONFIG['M_min'], config.MAGNET_CONFIG['M_max']).flatten()

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

        #---calculate magnetic field at each AOI point; save as tfrecord; save in batches (e.g. to reduce gcs API calls)---
        #if use_gcs=True: save locally, upload to gcs, delete local copy
        #if use_gcs=False: save locally

        #generate, upload data
        for i, split in enumerate([train_split_idx, val_split_idx, test_split_idx]):
            for batch_idx in tqdm(split, desc=f"Generating {'train' if i==0 else 'val' if i==1 else 'test'} data"):
                name = ['train', 'val', 'test'][i]
                filename = f'{name}-{batch_idx:04d}.tfrecord'
                local_fullpath = f'{self.local_path}/{filename}'
                os.makedirs(os.path.dirname(local_fullpath), exist_ok=True)
                if use_gcs:
                    gcs_blob_fullpath = f'{self.gcs_blob_path}/{filename}'

                #indices
                batch_start = batch_idx * samples_per_batch
                batch_end = min(batch_start + samples_per_batch, config.DATASET_CONFIG['dataset_size'])

                #generate H and save tfrecord
                with tf.io.TFRecordWriter(local_fullpath) as writer:
                    for j in range(batch_start, batch_end):
                        #if x_samples[j] is not None: #last batch may not contain samples_per_batch samples
                        magnet = magpy.magnet.Cuboid(polarization=(Mx_samples[j], My_samples[j], 0),
                                                     dimension=(a_samples[j], b_samples[j], 1),
                                                     position=(x_samples[j], y_samples[j], 2.5))
                        H_single = magpy.getH(magnet, self.points)[:, :2].astype(np.float32)
                        params = np.array([x_samples[j], y_samples[j],
                                           a_samples[j], b_samples[j],
                                           Mx_samples[j], My_samples[j]
                                           ], dtype=np.float32)

                        writer.write(self.serialise_example(H_single, params))

                #upload to gcs
                if use_gcs:
                    self.upload_to_gcloud(local_fullpath, gcs_blob_fullpath)
                    os.remove(local_fullpath)

    #goes with above generate_cuboid_data_TFRECORD function
    def load_datasets_TFRECORD(self, use_gcs=False, split='train', prop_to_load=1.0):
        '''tf dataset with AUTOTUNE to load from gcloud
        prop_to_load: proportion of dataset to load (float)'''
        if use_gcs:
            fullpath = f'gs://{self.gcs_bucket_name}/{self.gcs_blob_path}/{split}-*.tfrecord'
            files = tf.io.gfile.glob(fullpath)
            if not files:
                print(f"No files found at {fullpath}")
        else:
            import glob
            fullpath = f'{self.local_path}/{split}-*.tfrecord'
            files = glob.glob(fullpath)
            if prop_to_load != 1.0:
                random.shuffle(files)
                files = files[0:(int(prop_to_load*len(files)))]

            if not files:
                print(f"No files found at {fullpath}. Make sure tfrecords exist locally.")

        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(files, dtype=tf.string))
        dataset = dataset.shuffle(buffer_size=len(files))  #shuffle for better parallelisation

        #better fcs performance
        options = tf.data.Options()
        options.experimental_deterministic = False

        #load data with optimised settings
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=1),
            cycle_length=16,  #read 16 files simultan
            block_length=8,  #read 8 records per file
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )

        dataset = dataset.with_options(options)

        #parse and batch
        dataset = dataset.map(self.deserialise_normalise_example,
                              num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(config.TRAINING_CONFIG['batch_size'],
                                drop_remainder=True)  #drop incomplete batches

        dataset = dataset.repeat()

        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    #---ABOVE FUNCTIONS NO LONGER IN USE---
    #--------------------------------------


if __name__ == '__main__':
    print(f"running data.py ...")
    time.sleep(10)
    #generator = Dataset()
    #generator.generate_cuboid_data()  #num_batches should <= dataset_size