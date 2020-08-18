"""A Statistical Comparative Study on Image Reconstruction and Clustering With Novel VAE Cost Function """



# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x 
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from tensorflow.python.layers import base
#import tensorflow.contrib.slim as slim
#from tensorflow.examples.tutorials.mnist import input_data
import seaborn as sns
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import SpectralClustering 

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
print("GPU is", "available" if tf.test.is_gpu_available() else "not available")


#---------------------------------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# add empty color dimension
#x_train2 = np.expand_dims(x_train2, -1)
#x_test2 = np.expand_dims(x_test2, -1)
# Normalize data set to 0-to-1 range

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(np.max(x_train))
print(np.max(y_test))
#---------------------------------------------------------------------------------------------------


# Helper Class 
# Reference: https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
class Dataset:
  def __init__(self,data):
      self._index_in_epoch = 0
      self._epochs_completed = 0
      self._data = data
      self._num_examples = data.shape[0]
      pass

  @property
  def data(self):
      return self._data

  def next_batch(self,batch_size,shuffle = False):
      start = self._index_in_epoch
      if start == 0 and self._epochs_completed == 0:
          idx = np.arange(0, self._num_examples)  # get all possible indexes
          #np.random.shuffle(idx)  # shuffle indexe
          self._data = self.data[idx]  # get list of `num` random samples

      # go to the next batch
      if start + batch_size > self._num_examples:
          self._epochs_completed += 1
          rest_num_examples = self._num_examples - start
          data_rest_part = self.data[start:self._num_examples]
          idx0 = np.arange(0, self._num_examples)  # get all possible indexes
          #np.random.shuffle(idx0)  # shuffle indexes
          self._data = self.data[idx0]  # get list of `num` random samples

          start = 0
          self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
          end =  self._index_in_epoch  
          data_new_part =  self._data[start:end]  
          return np.concatenate((data_rest_part, data_new_part), axis=0)
      else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          return self._data[start:end]

x_train2 = Dataset(x_train)
#---------------------------------------------------------------------------------------------------

def calculate_cl_acc(ground_truth, est, nb_all_clusters, cluster_offset=0, label_correction=False):

    majority = np.zeros(nb_all_clusters)
    population = np.zeros(nb_all_clusters)

    if label_correction:
      
        est = correct_labels(ground_truth, est)

    for cluster in range(cluster_offset, nb_all_clusters + cluster_offset):
        if np.bincount(ground_truth[est==cluster]).size != 0:
            majority[cluster-cluster_offset] = np.bincount(ground_truth[est==cluster]).max()
            population[cluster-cluster_offset] = np.bincount(ground_truth[est==cluster]).sum()

    cl_acc = majority[majority>0].sum()/population[population>0].sum()

    return cl_acc, population.sum()


def correct_labels(ground_truth, est):

    corrested_est = np.zeros_like(est, dtype='int')

    for cluster in range(est.max()+1):
        if np.bincount(ground_truth[est==cluster]).size != 0:
            true_label = np.bincount(ground_truth[est==cluster]).argmax()
            corrested_est[est==cluster] = true_label

    return corrested_est

#---------------------------------------------------------------------------------------------------
#tf.reset_default_graph()
tf.compat.v1.reset_default_graph()
# Input image
X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
# Output image
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
# reshape image array into vector of lenght 784
Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
# Dropout probability hyperparameter
keep_prob0 = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
keep_prob1 = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob1')
keep_prob2 = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob2')
keep_prob=[keep_prob0,keep_prob1,keep_prob2]
dec_in_channels = 1
n_latent= 32
reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels // 2

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

#---------------------------------------------------------------------------------------------------

def encoder(X_in, keep_prob0,keep_prob,keep_prob2):
    activation = lrelu
    ff= []
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 28, 28, 1], name='reshaped')
        print('one',X.get_shape())

        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation,name='x1',
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        
        #x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)

        print('two',x.get_shape())
        #x = tf.nn.dropout(x, keep_prob0)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation,name='x2',
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        print('three',x.get_shape())
        #x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)

        #x = tf.nn.dropout(x, keep_prob1)

        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation,name='x3', 
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
       	#x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)

        print('four',x.get_shape())
        #x = tf.nn.dropout(x, keep_prob2)
        x = tf.contrib.layers.flatten(x)
        print('five',x.get_shape())
        mn = tf.layers.dense(x, units=n_latent,name="mn")
        print('six',mn.get_shape())
        sd       = 0.5 * tf.layers.dense(x, units=n_latent, name='sd')            
        print('seven',sd.get_shape())
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        print('eight',z.get_shape())
        ff.append(mn)
        return z, mn, sd
print("----------------------------------------------------------")           

#---------------------------------------------------------------------------------------------------
def decoder(sampled_z, keep_prob0,keep_prob1,keep_prob2):
    with tf.variable_scope("decoder", reuse=None):
        
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        print('one_decoder',x.get_shape())

        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)

        print('2_decoder',x.get_shape())
        x = tf.reshape(x, reshaped_dim)
        print('3_decoder',x.get_shape())
        
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu,name='x1_d',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        print('con1_decoder',x.get_shape())   

       	#x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
        #x = tf.nn.dropout(x, keep_prob0)
        
      
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.leaky_relu,name='x2_d',  
                                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        print('con2_decoder',x.get_shape())
        
       	#x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)        
        #x = tf.nn.dropout(x, keep_prob1)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.leaky_relu,name='x3_d', 
                                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        print('con3_decoder',x.get_shape()) 
       	#x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
        #x = tf.nn.dropout(x, keep_prob2)

        x = tf.contrib.layers.flatten(x)
        print('faltten_decoder',x.get_shape())
        x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
        print('x-hat',x.get_shape())
        return img      

#---------------------------------------------------------------------------------------------------
sampled, mn, sd = encoder(X_in, keep_prob0,keep_prob1,keep_prob2)
dec = decoder(sampled, keep_prob0,keep_prob1,keep_prob2)
unreshaped = tf.reshape(dec, [-1, 28*28])

img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1) 

latent_loss = -0.5 * tf.keras.backend.sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0
batch_size =64
#---------------------------------------------------------------------------------------------------
im1 = tf.image.convert_image_dtype( Y_flat, tf.float32)
im2 = tf.image.convert_image_dtype( unreshaped, tf.float32)
im1 = tf.expand_dims(im1, axis = 2)
im2 = tf.expand_dims(im2, axis = 2)
ssim2 = tf.image.ssim(im1,im2 , max_val=1.0) #ssim2 = tf.image.ssim(X_in,dec , max_val=1.0)
#---------------------------------------------------------------------------------------------------

#img_loss = img_loss * tf.math.log(ssim2) 
#loss = tf.reduce_mean(img_loss*(1+tf.nn.softplus(1-ssim2))+ latent_loss) 
#loss = tf.reduce_mean(img_loss*(tf.math.sigmoid(1/(1-ssim2)))+ latent_loss) 
loss = tf.reduce_mean( img_loss*((1-ssim2)**2 )+ latent_loss )
loss = loss + reg_constant * sum(reg_losses)
optimizer = tf.train.AdamOptimizer(0.0008).minimize(loss) #0.0008
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#---------------------------------------------------------------------------------------------------

import time
start_time = time.time()
reconstruction_loss1 = []
KL_diver = []
iter = 20000
for m in range(1):
  for i in range(iter):
      #batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
      batch = [np.reshape(b, [28, 28]) for b in x_train2.next_batch(batch_size=64)]
      sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob0: 0.8,keep_prob1: 0.8,keep_prob2: 0.8})
      if not i % 2000:
          ls, d, i_ls, d_ls, mu, sigm, ssimm = sess.run([loss, dec, img_loss, latent_loss, mn, sd, ssim2], 
                                                 feed_dict = {X_in: batch, Y: batch, keep_prob0: 1,keep_prob1: 1,keep_prob2: 1})
         # print("iteration",i, "total loss is ", ls,"reconstruction loss is ", np.mean(i_ls), "KL loss is ",np.mean(d_ls))
  reconstruction_loss1.append(np.mean(i_ls))
  KL_diver.append(np.mean(d_ls))


print("reconstruction loss is ", np.mean( reconstruction_loss1), "KL loss is ",np.mean(KL_diver))
print("--- %s seconds ---" % (time.time() - start_time))
#---------------------------------------------------------------------------------------------------
x_test2 = Dataset(x_test)
batch2 = [np.reshape(b, [28, 28]) for b in x_test2.next_batch(batch_size=10000)]
_, d2, _, _, mu2,_,_= sess.run([loss, dec, img_loss, latent_loss, mn, sd,ssim2], 
                            feed_dict = {X_in: batch2, Y: batch2,  keep_prob0: 1,keep_prob1: 1,keep_prob2: 1})
#---------------------------------------------------------------------------------------------------


#@title Default title text
plt.rcParams['figure.figsize'] = [12, 8]
fig = plt.figure()
plt.subplot(251)
plt.imshow(np.reshape(batch[1], [28, 28]), cmap='gray')
#plt.show()
plt.subplot(256)
plt.imshow(d[1], cmap='gray')
plt.subplot(252)
plt.imshow(batch[36], cmap='gray')
plt.subplot(257)
plt.imshow(d[36], cmap='gray')
plt.subplot(253)
plt.imshow(batch[24], cmap='gray')
plt.subplot(258)
plt.imshow(d[24], cmap='gray')
plt.subplot(254)
plt.imshow(batch[5], cmap='gray')
plt.subplot(259)
plt.imshow(d[5], cmap='gray')
plt.subplot(255)
plt.imshow(batch[4], cmap='gray')
fig.add_subplot(2,5,10)
plt.imshow(d[4], cmap='gray')
plt.savefig('01.pdf')
#---------------------------------------------------------------------------------------------------


nb_clusters =10
acc_output = []
estimator = KMeans(init='k-means++', n_clusters=nb_clusters, n_init=nb_clusters)
estimator.fit(mu2)
acc_output.append(calculate_cl_acc(y_test, estimator.labels_,nb_clusters,label_correction=True)[0])    
print('The Accuracy from kmeans is: ',acc_output)
#---------------------------------------------------------------------------------------------------

from sewar.full_ref import mse, uqi, ssim,  vifp
# mse, rmse, psnr, rmse_sw, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
#!pip install sewar
#from sewar.full_ref import mse, rmse, psnr, rmse_sw, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
def mm(y1):
  t1,t2,t3,t4,t5,t6=[],[],[],[],[],[]

  for i in range(32):
    t1.append(uqi(batch2[i+y1],d2[i+y1]))
    t2.append(mse(batch2[i+y1],d2[i+y1]))
    #t3.append(rmse(batch2[i],d2[i]))
    t6.append(vifp(batch2[i+y1],d2[i+y1]))

  #print("Structural Similarity Index (SSIM) is %0.3f" % np.mean(f)) # Good
  #print("Universal Quality Image Index (UQI) is %0.3f" % np.mean(t1)) # Good 
  #print("Mean Squared Error (MSE) is %0.3f" % np.mean(t2) ) # Good
  #print("Visual Information Fidelity (VIF) is %0.3f" %  np.mean(t6) ) # Good 
  #print("Root Mean Sqaured Error (RMSE) is %0.3f" % np.mean(t3)) # Good 
  #print("Spatial Correlation Coefficient (SCC) is %0.3f" % np.mean(t4) )
  #print("Spectral Angle Mapper (SAM) is %0.3f" % np.mean(t5) )
  return ( np.mean(t1) , np.mean(t2), np.mean(t6))

print('UQI'+'-----------------------------'+'MSE'+'----------------------'+'VIF')
#---------------------------------------------------------------------------------------------------
for i in range(10):
  print( mm(1000*i))

"""# SSIM"""
f =[]
y= 0
for i in range(32):
  
  im1 = tf.image.convert_image_dtype( batch2[i+y], tf.float32)
  im2 = tf.image.convert_image_dtype( d2[i+y], tf.float32)
  im1 = tf.expand_dims(im1, axis = 2)
  im2 = tf.expand_dims(im2, axis = 2)
  ssim2 = tf.image.ssim(im1, im2, max_val=1.0)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #sess.run(tf.initialize_all_variables()) #
    tf_ssim1 = sess.run(ssim2)
    f.append(tf_ssim1)
print('SSIM %0.3f' %np.mean(f))