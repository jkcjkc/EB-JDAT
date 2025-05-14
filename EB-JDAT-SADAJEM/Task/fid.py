# """Calculates the Frechet Inception Distance (FID) to evalulate GANs

# The FID metric calculates the distance between two distributions of images.
# Typically, we have summary statistics (mean & covariance matrix) of one
# of these distributions, while the 2nd distribution is given by a GAN.

# When run as a stand-alone program, it compares the distribution of
# images that are stored as PNG/JPEG at a specified location with a
# distribution given by summary statistics (in pickle format).

# The FID is calculated by assuming that X_1 and X_2 are the activations of
# the pool_3 layer of the inception net for generated samples and real world
# samples respectively.

# See --help to see further details.

# Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
# of Tensorflow

# Copyright 2018 Institute of Bioinformatics, JKU Linz

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# """

# import os
# import pathlib
# from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# import numpy as np
# import torch
# import torchvision.transforms as TF
# from PIL import Image
# from scipy import linalg
# from torch.nn.functional import adaptive_avg_pool2d
# import ssl
# import urllib.request

# # 创建一个不验证证书的上下文
# ssl._create_default_https_context = ssl._create_unverified_context

# try:
#     from tqdm import tqdm
# except ImportError:
#     # If tqdm is not available, provide a mock version of it
#     def tqdm(x):
#         return x


# from inception import InceptionV3

# parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
# parser.add_argument(
#     "--num-workers",
#     type=int,
#     help=(
#         "Number of processes to use for data loading. " "Defaults to `min(8, num_cpus)`"
#     ),
# )
# parser.add_argument(
#     "--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu"
# )
# parser.add_argument(
#     "--dims",
#     type=int,
#     default=2048,
#     choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
#     help=(
#         "Dimensionality of Inception features to use. "
#         "By default, uses pool3 features"
#     ),
# )
# parser.add_argument(
#     "--save-stats",
#     action="store_true",
#     help=(
#         "Generate an npz archive from a directory of "
#         "samples. The first path is used as input and the "
#         "second as output."
#     ),
# )
# parser.add_argument(
#     "path",
#     type=str,
#     nargs=2,
#     help=("Paths to the generated images or " "to .npz statistic files"),
# )

# IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


# class ImagePathDataset(torch.utils.data.Dataset):
#     def __init__(self, files, transforms=None):
#         self.files = files
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, i):
#         path = self.files[i]
#         img = Image.open(path).convert("RGB")
#         if self.transforms is not None:
#             img = self.transforms(img)
#         return img


# def get_activations(
#     files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
# ):
#     """Calculates the activations of the pool_3 layer for all images.

#     Params:
#     -- files       : List of image files paths
#     -- model       : Instance of inception model
#     -- batch_size  : Batch size of images for the model to process at once.
#                      Make sure that the number of samples is a multiple of
#                      the batch size, otherwise some samples are ignored. This
#                      behavior is retained to match the original FID score
#                      implementation.
#     -- dims        : Dimensionality of features returned by Inception
#     -- device      : Device to run calculations
#     -- num_workers : Number of parallel dataloader workers

#     Returns:
#     -- A numpy array of dimension (num images, dims) that contains the
#        activations of the given tensor when feeding inception with the
#        query tensor.
#     """
#     model.eval()

#     if batch_size > len(files):
#         print(
#             (
#                 "Warning: batch size is bigger than the data size. "
#                 "Setting batch size to data size"
#             )
#         )
#         batch_size = len(files)

#     dataset = ImagePathDataset(files, transforms=TF.ToTensor())
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         drop_last=False,
#         num_workers=num_workers,
#     )

#     pred_arr = np.empty((len(files), dims))

#     start_idx = 0

#     for batch in tqdm(dataloader):
#         batch = batch.to(device)

#         with torch.no_grad():
#             pred = model(batch)[0]

#         # If model output is not scalar, apply global spatial average pooling.
#         # This happens if you choose a dimensionality not equal 2048.
#         if pred.size(2) != 1 or pred.size(3) != 1:
#             pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

#         pred = pred.squeeze(3).squeeze(2).cpu().numpy()

#         pred_arr[start_idx : start_idx + pred.shape[0]] = pred

#         start_idx = start_idx + pred.shape[0]

#     return pred_arr


# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """Numpy implementation of the Frechet Distance.
#     The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
#     and X_2 ~ N(mu_2, C_2) is
#             d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

#     Stable version by Dougal J. Sutherland.

#     Params:
#     -- mu1   : Numpy array containing the activations of a layer of the
#                inception net (like returned by the function 'get_predictions')
#                for generated samples.
#     -- mu2   : The sample mean over activations, precalculated on an
#                representative data set.
#     -- sigma1: The covariance matrix over activations for generated samples.
#     -- sigma2: The covariance matrix over activations, precalculated on an
#                representative data set.

#     Returns:
#     --   : The Frechet Distance.
#     """

#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)

#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)

#     assert (
#         mu1.shape == mu2.shape
#     ), "Training and test mean vectors have different lengths"
#     assert (
#         sigma1.shape == sigma2.shape
#     ), "Training and test covariances have different dimensions"

#     diff = mu1 - mu2

#     # Product might be almost singular
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         msg = (
#             "fid calculation produces singular product; "
#             "adding %s to diagonal of cov estimates"
#         ) % eps
#         print(msg)
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

#     # Numerical error might give slight imaginary component
#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             raise ValueError("Imaginary component {}".format(m))
#         covmean = covmean.real

#     tr_covmean = np.trace(covmean)

#     return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# def calculate_activation_statistics(
#     files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
# ):
#     """Calculation of the statistics used by the FID.
#     Params:
#     -- files       : List of image files paths
#     -- model       : Instance of inception model
#     -- batch_size  : The images numpy array is split into batches with
#                      batch size batch_size. A reasonable batch size
#                      depends on the hardware.
#     -- dims        : Dimensionality of features returned by Inception
#     -- device      : Device to run calculations
#     -- num_workers : Number of parallel dataloader workers

#     Returns:
#     -- mu    : The mean over samples of the activations of the pool_3 layer of
#                the inception model.
#     -- sigma : The covariance matrix of the activations of the pool_3 layer of
#                the inception model.
#     """
#     act = get_activations(files, model, batch_size, dims, device, num_workers)
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma


# def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1):
#     if path.endswith(".npz"):
#         with np.load(path) as f:
#             m, s = f["mu"][:], f["sigma"][:]
#     else:
#         path = pathlib.Path(path)
#         files = sorted(
#             [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
#         )
#         m, s = calculate_activation_statistics(
#             files, model, batch_size, dims, device, num_workers
#         )

#     return m, s


# def calculate_fid_given_paths(images, images_gt, device, dims, num_workers=1):
#     """Calculates the FID of two paths"""
#     # for p in paths:
#     #     if not os.path.exists(p):
#     #         raise RuntimeError("Invalid path: %s" % p)

#     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

#     model = InceptionV3([block_idx]).to(device)

#     m1, s1 = calculate_activation_statistics(
#         images, model, 50, dims, device, num_workers
#     )
#     m2, s2 = calculate_activation_statistics(
#         images_gt, model, 50, dims, device, num_workers
#     )
#     fid_value = calculate_frechet_distance(m1, s1, m2, s2)

#     return fid_value


# def save_fid_stats(paths, batch_size, device, dims, num_workers=1):
#     """Saves FID statistics of one path"""
#     if not os.path.exists(paths[0]):
#         raise RuntimeError("Invalid path: %s" % paths[0])

#     if os.path.exists(paths[1]):
#         raise RuntimeError("Existing output file: %s" % paths[1])

#     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

#     model = InceptionV3([block_idx]).to(device)

#     print(f"Saving statistics for {paths[0]}")

#     m1, s1 = compute_statistics_of_path(
#         paths[0], model, batch_size, dims, device, num_workers
#     )

#     np.savez_compressed(paths[1], mu=m1, sigma=s1)


# def main():
#     args = parser.parse_args()

#     if args.device is None:
#         device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
#     else:
#         device = torch.device(args.device)

#     if args.num_workers is None:
#         try:
#             num_cpus = len(os.sched_getaffinity(0))
#         except AttributeError:
#             # os.sched_getaffinity is not available under Windows, use
#             # os.cpu_count instead (which may not return the *available* number
#             # of CPUs).
#             num_cpus = os.cpu_count()

#         num_workers = min(num_cpus, 8) if num_cpus is not None else 0
#     else:
#         num_workers = args.num_workers

#     if args.save_stats:
#         save_fid_stats(args.path, args.batch_size, device, args.dims, num_workers)
#         return

#     fid_value = calculate_fid_given_paths(
#         args.path, args.batch_size, device, args.dims, num_workers
#     )
#     print("FID: ", fid_value)


# if __name__ == "__main__":
#     main()


# from __future__ import absolute_import, division, print_function
# import numpy as np
# import os
# import gzip, pickle
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# from scipy.misc import imread
# from scipy import linalg
# import pathlib
# import urllib
# import tarfile
# import warnings

# MODEL_DIR = './imagenet'
# DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pool3 = None

# class InvalidFIDException(Exception):
#     pass

# #-------------------------------------------------------------------------------
# def get_fid_score(images, images_gt):
#     images = np.stack(images, 0)
#     images_gt = np.stack(images_gt, 0)

#     with tf.Session(config=config) as sess:
#         m1, s1 = calculate_activation_statistics(images, sess)
#         m2, s2 = calculate_activation_statistics(images_gt, sess)
#         fid_value = calculate_frechet_distance(m1, s1, m2, s2)

#     print("Obtained fid value of {}".format(fid_value))
#     return fid_value


# def create_inception_graph(pth):
#     """Creates a graph from saved GraphDef file."""
#     # Creates graph from saved graph_def.pb.
#     with tf.gfile.FastGFile( pth, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString( f.read())
#         _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')
# #-------------------------------------------------------------------------------


# # code for handling inception net derived from
# #   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# def _get_inception_layer(sess):
#     """Prepares inception net for batched usage and returns pool_3 layer. """
#     layername = 'FID_Inception_Net/pool_3:0'
#     pool3 = sess.graph.get_tensor_by_name(layername)
#     ops = pool3.graph.get_operations()
#     for op_idx, op in enumerate(ops):
#         for o in op.outputs:
#             shape = o.get_shape()
#             if shape._dims != []:
#               shape = [s.value for s in shape]
#               new_shape = []
#               for j, s in enumerate(shape):
#                 if s == 1 and j == 0:
#                   new_shape.append(None)
#                 else:
#                   new_shape.append(s)
#               o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
#     return pool3
# #-------------------------------------------------------------------------------


# def get_activations(images, sess, batch_size=50, verbose=False):
#     """Calculates the activations of the pool_3 layer for all images.

#     Params:
#     -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
#                      must lie between 0 and 256.
#     -- sess        : current session
#     -- batch_size  : the images numpy array is split into batches with batch size
#                      batch_size. A reasonable batch size depends on the disposable hardware.
#     -- verbose    : If set to True and parameter out_step is given, the number of calculated
#                      batches is reported.
#     Returns:
#     -- A numpy array of dimension (num images, 2048) that contains the
#        activations of the given tensor when feeding inception with the query tensor.
#     """
#     # inception_layer = _get_inception_layer(sess)
#     d0 = images.shape[0]
#     if batch_size > d0:
#         print("warning: batch size is bigger than the data size. setting batch size to data size")
#         batch_size = d0
#     n_batches = d0//batch_size
#     n_used_imgs = n_batches*batch_size
#     pred_arr = np.empty((n_used_imgs,2048))
#     for i in range(n_batches):
#         if verbose:
#             print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
#         start = i*batch_size
#         end = start + batch_size
#         batch = images[start:end]
#         pred = sess.run(pool3, {'ExpandDims:0': batch})
#         pred_arr[start:end] = pred.reshape(batch_size,-1)
#     if verbose:
#         print(" done")
#     return pred_arr
# #-------------------------------------------------------------------------------


# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """Numpy implementation of the Frechet Distance.
#     The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
#     and X_2 ~ N(mu_2, C_2) is
#             d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
#     Stable version by Dougal J. Sutherland.

#     Params:
#     -- mu1 : Numpy array containing the activations of the pool_3 layer of the
#              inception net ( like returned by the function 'get_predictions')
#              for generated samples.
#     -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
#                on an representive data set.
#     -- sigma1: The covariance matrix over activations of the pool_3 layer for
#                generated samples.
#     -- sigma2: The covariance matrix over activations of the pool_3 layer,
#                precalcualted on an representive data set.

#     Returns:
#     --   : The Frechet Distance.
#     """

#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)

#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)

#     assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
#     assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

#     diff = mu1 - mu2

#     # product might be almost singular
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
#         warnings.warn(msg)
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

#     # numerical error might give slight imaginary component
#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             raise ValueError("Imaginary component {}".format(m))
#         covmean = covmean.real

#     tr_covmean = np.trace(covmean)

#     return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
# #-------------------------------------------------------------------------------


# def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
#     """Calculation of the statistics used by the FID.
#     Params:
#     -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
#                      must lie between 0 and 255.
#     -- sess        : current session
#     -- batch_size  : the images numpy array is split into batches with batch size
#                      batch_size. A reasonable batch size depends on the available hardware.
#     -- verbose     : If set to True and parameter out_step is given, the number of calculated
#                      batches is reported.
#     Returns:
#     -- mu    : The mean over samples of the activations of the pool_3 layer of
#                the incption model.
#     -- sigma : The covariance matrix of the activations of the pool_3 layer of
#                the incption model.
#     """
#     act = get_activations(images, sess, batch_size, verbose)
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma
# #-------------------------------------------------------------------------------


# #-------------------------------------------------------------------------------
# # The following functions aren't needed for calculating the FID
# # they're just here to make this module work as a stand-alone script
# # for calculating FID scores
# #-------------------------------------------------------------------------------
# def check_or_download_inception(inception_path):
#     ''' Checks if the path to the inception file is valid, or downloads
#         the file if it is not present. '''
#     INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
#     if inception_path is None:
#         inception_path = '/tmp'
#     inception_path = pathlib.Path(inception_path)
#     model_file = inception_path / 'classify_image_graph_def.pb'
#     if not model_file.exists():
#         print("Downloading Inception model")
#         from urllib import request
#         import tarfile
#         fn, _ = request.urlretrieve(INCEPTION_URL)
#         with tarfile.open(fn, mode='r') as f:
#             f.extract('classify_image_graph_def.pb', str(model_file.parent))
#     return str(model_file)


# def _handle_path(path, sess):
#     if path.endswith('.npz'):
#         f = np.load(path)
#         m, s = f['mu'][:], f['sigma'][:]
#         f.close()
#     else:
#         path = pathlib.Path(path)
#         files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
#         x = np.array([imread(str(fn)).astype(np.float32) for fn in files])
#         m, s = calculate_activation_statistics(x, sess)
#     return m, s


# def calculate_fid_given_paths(paths, inception_path):
#     ''' Calculates the FID of two paths. '''
#     inception_path = check_or_download_inception(inception_path)

#     for p in paths:
#         if not os.path.exists(p):
#             raise RuntimeError("Invalid path: %s" % p)

#     create_inception_graph(str(inception_path))
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         m1, s1 = _handle_path(paths[0], sess)
#         m2, s2 = _handle_path(paths[1], sess)
#         fid_value = calculate_frechet_distance(m1, s1, m2, s2)
#         return fid_value


# def _init_inception():
#   global pool3
#   if not os.path.exists(MODEL_DIR):
#     os.makedirs(MODEL_DIR)
#   filename = DATA_URL.split('/')[-1]
#   filepath = os.path.join(MODEL_DIR, filename)
#   if not os.path.exists(filepath):
#     def _progress(count, block_size, total_size):
#       sys.stdout.write('\r>> Downloading %s %.1f%%' % (
#           filename, float(count * block_size) / float(total_size) * 100.0))
#       sys.stdout.flush()
#     filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#     print()
#     statinfo = os.stat(filepath)
#     print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
#   tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
#   with tf.gfile.FastGFile(os.path.join(
#       MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     _ = tf.import_graph_def(graph_def, name='')
#   # Works with an arbitrary minibatch size.
#   with tf.Session() as sess:
#     pool3 = sess.graph.get_tensor_by_name('pool_3:0')
#     ops = pool3.graph.get_operations()
#     for op_idx, op in enumerate(ops):
#         for o in op.outputs:
#             shape = o.get_shape()
#             if shape._dims != []:
#               shape = [s.value for s in shape]
#               new_shape = []
#               for j, s in enumerate(shape):
#                 if s == 1 and j == 0:
#                   new_shape.append(None)
#                 else:
#                   new_shape.append(s)
#               o.__dict__['_shape_val'] = tf.TensorShape(new_shape)


# if pool3 is None:
#   _init_inception()
#!/usr/bin/env python3
''' Calculates the Frechet Inception Distance (FID) to evalulate GANs (TensorFlow 2.x version).'''

# import numpy as np
# import os
# import gzip, pickle
# import tensorflow as tf
# from tensorflow import keras
# from scipy import linalg  # 注意：需使用 scipy >= 1.7.0
# import pathlib
# import urllib
# import tarfile
# import warnings
# import imageio

# MODEL_DIR = './inception'
# DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# class InvalidFIDException(Exception):
#     pass

# #-------------------------------------------------------------------------------
# def get_fid_score(images, images_gt):
#     images = np.stack(images, 0)
#     images_gt = np.stack(images_gt, 0)

#     # 使用 TensorFlow 2.x 方式运行
#     m1, s1 = calculate_activation_statistics(images)
#     m2, s2 = calculate_activation_statistics(images_gt)
#     fid_value = calculate_frechet_distance(m1, s1, m2, s2)

#     print(f"Obtained fid value of {fid_value}")
#     return fid_value

# def create_inception_graph(pth):
#     """Creates a graph from saved GraphDef file."""
#     with tf.io.gfile.GFile(pth, 'rb') as f:
#         graph_def = tf.compat.v1.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.compat.v1.import_graph_def(graph_def, name='FID_Inception_Net')

# #-------------------------------------------------------------------------------


# #-------------------------------------------------------------------------------
# def load_inception_model():
#     """加载预训练Inception模型"""
#     if not hasattr(load_inception_model, '_model'):
#         # 创建Inception计算图
#         model_path = check_or_download_inception(None)
#         create_inception_graph(model_path)
        
#         # 构建Keras模型
#         tf.compat.v1.disable_eager_execution()
#         inputs = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3])
#         with tf.compat.v1.Session() as sess:
#             pool3 = sess.graph.get_tensor_by_name('pool_3:0')
#             load_inception_model._model = tf.compat.v1.keras.Model(
#                 inputs, tf.reshape(pool3, [-1, 2048]))
            
#     return load_inception_model._model


# import tensorflow as tf
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Reshape
# def load_inception_model():
#     """加载预训练Inception模型"""
#     if not hasattr(load_inception_model, '_model'):
#         # 禁用 Eager Execution，模拟 TensorFlow 1.x 行为
#         tf.compat.v1.disable_eager_execution()

#         # 创建输入层
#         input_tensor = tf.keras.Input(shape=(None, None, 3))
#         weights_path = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

#         # 加载 InceptionV3 模型，不包含顶层，使用 imagenet 权重
#         base_model = InceptionV3(
#             include_top=False,
#             weights=weights_path,
#             input_tensor=input_tensor,
#             pooling='avg'
#         )

#         # 提取池化层输出（可以修改为您需要的层）
#         # output = Reshape(base_model.output, [-1, 2048])
#         output = Reshape((-1, 2048))(base_model.output)

#         # 构建最终的 Keras 模型
#         load_inception_model._model = Model(inputs=input_tensor, outputs=output)

#     return load_inception_model._model


# #-------------------------------------------------------------------------------
# def calculate_activation_statistics(images, batch_size=50, verbose=False):
#     """计算激活统计数据"""
#     act = get_activations(images, batch_size, verbose)
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma

# #-------------------------------------------------------------------------------
# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """Numpy implementation of the Frechet Distance."""
#     # 保持原有实现不变
#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)

#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)

#     diff = mu1 - mu2

#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

#     if np.iscomplexobj(covmean):
#         covmean = covmean.real

#     tr_covmean = np.trace(covmean)
    
#     return (diff.dot(diff) + np.trace(sigma1) + 
#             np.trace(sigma2) - 2 * tr_covmean)

# #-------------------------------------------------------------------------------
# def check_or_download_inception(inception_path):
#     """检查并下载Inception模型"""
#     if inception_path is None:
#         inception_path = pathlib.Path(MODEL_DIR)
#     else:
#         inception_path = pathlib.Path(inception_path)
    
#     model_file = inception_path / 'classify_image_graph_def.pb'
#     if not model_file.exists():
#         print("Downloading Inception model...")
#         inception_path.mkdir(parents=True, exist_ok=True)
#         filename = DATA_URL.split('/')[-1]
#         filepath = inception_path / filename
#         urllib.request.urlretrieve(DATA_URL, filepath)
        
#         with tarfile.open(filepath, 'r:gz') as tar:
#             tar.extract('classify_image_graph_def.pb', str(inception_path))
            
#     return str(model_file)

# #-------------------------------------------------------------------------------
# def _handle_path(path):
#     """处理输入路径"""
#     if path.endswith('.npz'):
#         with np.load(path) as f:
#             return f['mu'], f['sigma']
#     else:
#         path = pathlib.Path(path)
#         files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
#         x = np.array([imageio.imread(str(fn)).astype(np.float32) for fn in files])
#         return calculate_activation_statistics(x)


from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pytorch_image_generation_metrics

from pytorch_image_generation_metrics import get_fid
import torch


# compute KID
def compute_kid(
    real_images,
    fake_images,
    max_subset_size=10,
    device="cuda",
    feature="logits_unbiased",
    normalize=True,
):
    batch_size = 500

    kid = KernelInceptionDistance(
        feature=feature, subset_size=max_subset_size, normalize=normalize
    ).to(device=device)

    for i in range(0, real_images.size(0), batch_size):
        batch = real_images[i : i + batch_size].to(device)
        kid.update(batch, real=True)

    for i in range(0, fake_images.size(0), batch_size):
        batch = fake_images[i : i + batch_size].to(device)
        kid.update(batch, real=False)

    KID = kid.compute()

    return KID[0], KID[1]


def compute_is(imgs, device="cuda"):
    """
    Compute the Inception Score
    Imgs is a tensor of shape (N, C, H, W) with values in [0, 1]
    """
    batch_size = 512
    inception = InceptionScore(feature="logits_unbiased", normalize=True).to(device)

    for i in range(0, imgs.size(0), batch_size):
        batch = imgs[i : i + batch_size].to(device)
        inception.update(batch)
    IS = inception.compute()
    return IS[0], IS[1]


def compute_fid(real_images, fake_images, device="cuda", feature=2048, normalize=True):
    """
    All images will be resized to 299 x 299 which is the size of the original training data.
    The boolian flag real determines if the images should update the statistics of the real
    distribution or the fake distribution.
    real_images: tensor of shape (N, C, H, W) with values in [0, 1], IT SHOULD BE CIFAR-10 TRAIN SET or TEST SET
    fake_images: tensor of shape (M, C, H, W) with values in [0, 1]
    """
    ### compute the FID score with torchmetrics

    batch_size = 512
    fid = FrechetInceptionDistance(feature=feature, normalize=normalize).to(
        device=device
    )

    for i in range(0, real_images.size(0), batch_size):
        batch = real_images[i : i + batch_size].to(device)
        fid.update(batch, real=True)

    for i in range(0, fake_images.size(0), batch_size):
        batch = fake_images[i : i + batch_size].to(device)
        fid.update(batch, real=False)

    ### compute the FID score with python-gan-metrics
    # FID = get_fid(
    #     fake_images,
    #     real_images,
    # )

    return fid.compute()

def compute_lpips(train_loader, fake_images, device="cuda", net="alex"):
    """
    Compute the LPIPS score
    fake_images is a tensor of shape (N, C, H, W) with values in [0, 1]
    train loader is the dataloader of the training set
    """

    imgs, labels = [], []
    for i, (img, label) in enumerate(train_loader):
        imgs.append(img)
        labels.append(label)

    imgs = torch.cat(imgs, dim=0)
    labels = torch.cat(labels, dim=0)

    # sort the images for class, to compute the LPIPS
    imgs = imgs[torch.argsort(labels)]

    assert imgs.size() == fake_images.size()

    batch_size = 1000
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True).to(
        device=device
    )

    for i in range(0, imgs.size(0), batch_size):
        batch_real = imgs[i : i + batch_size].to(device)
        batch_fake = fake_images[i : i + batch_size].to(device)
        lpips.update(batch_real, batch_fake)

    return lpips.compute()



#-------------------------------------------------------------------------------
# if __name__ == '__main__':
#     # 示例用法
#     real_images = np.random.randn(100, 299, 299, 3).astype(np.float32) * 255
#     fake_images = np.random.randn(100, 299, 299, 3).astype(np.float32) * 255
    
#     fid_value = get_fid_score(fake_images, real_images)
#     print(f"FID Score: {fid_value}")