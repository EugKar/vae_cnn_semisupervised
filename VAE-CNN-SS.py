import tensorflow as tf
import input_data
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
slim = tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli
Normal = tf.contrib.distributions.Normal
Categorical = tf.contrib.distributions.Categorical
MultivariateNormalDiag = tf.contrib.distributions.MultivariateNormalDiag

def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature, name='gumbel_softmax')

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

K = 8 # number of classes for z
N = 5 # number of generated samples
M = 10 # number of label classes

# input image x labeled (shape=(batch_size,784))
x_l = tf.placeholder(tf.float32, [None, 784])
x_l_in = tf.reshape(x_l, [-1, 28, 28, 1])

# labels
y_l = tf.placeholder(tf.float32, [None, M])

# input image x unlabeled (shape=(batch_size,784))
x_u = tf.placeholder(tf.float32, [None, 784])
x_u_in = tf.reshape(x_u, [-1, 28, 28, 1])

# variational posterior q(y|x) for unlabeled data only
net = slim.conv2d(x_u_in, 32, [5, 5], scope='conv_yx_1')
net = slim.conv2d(net, 64, [5, 5], scope='conv_yx_2')
net = slim.conv2d(net, 128, [5, 5], scope='conv_yx_3')
# unnormalized logits for N separate K-categorical distributions (shape=(batch_size,M))
logits_y_u = slim.fully_connected(slim.flatten(net), M, activation_fn=None,
                                           scope='fc_yx')
logits_y_u = tf.tile(logits_y_u, [N, 1])
q_y_prob = tf.nn.softmax(logits_y_u, name='q_y_u_prob')
#log_q_y = tf.log(q_y_prob+1e-20)
#q_y_u = Categorical(logits=logits_y_u)

# temperature
tau = tf.Variable(5.0, name="temperature")
# sample and reshape back (shape=(batch_size * N,M))
# set hard=True for ST Gumbel-Softmax
y_u = gumbel_softmax(logits_y_u, tau, hard=False)
q_y_u = tf.reduce_sum(q_y_prob * y_u, 1)

# classification network for labeled data only
net = slim.conv2d(x_l_in, 32, [5, 5], scope='conv_yx_1', reuse=True)
net = slim.conv2d(net, 64, [5, 5], scope='conv_yx_2', reuse=True)
net = slim.conv2d(net, 128, [5, 5], scope='conv_yx_3', reuse=True)
# unnormalized logits for M-categorical distributions (shape=(batch_size,M))
logits_y_l = slim.fully_connected(slim.flatten(net),M,activation_fn=None,
                                           scope='fc_yx', reuse=True)
y_l_net = tf.nn.softmax(logits_y_l, name = 'y_l_net')
q_y_l = Bernoulli(logits = logits_y_l, name = 'q_y_l')
y_l_out = q_y_l.sample(name = 'y_l_out')

# variational posterior q(z|x,y) for unlabeled data, i.e. the encoder

shape = x_u_in.get_shape().as_list()
y_tiled_u = tf.tile(y_u[:, None, None, :], [1, shape[1], shape[2], 1])
x_u_in_tiled = tf.tile(x_u_in, [N, 1, 1, 1], name='x_u_in_tiled')
x_and_y_u = tf.concat((x_u_in_tiled, y_tiled_u), axis=3, name='x_and_y_u')

net = slim.conv2d(x_and_y_u, 32, [5, 5], scope='conv_zxy_mu_1')
net = slim.conv2d(net, 64, [5, 5], scope='conv_zxy_mu_2')
net = slim.conv2d(net, 64, [5, 5], scope='conv_zxy_mu_3')
mu_u = slim.fully_connected(slim.flatten(net), K, activation_fn=None,
                                      scope='fc_zxy_mu')

net = slim.conv2d(x_and_y_u, 32, [5, 5], scope='conv_zxy_sigma_1')
net = slim.conv2d(net, 64, [5, 5], scope='conv_zxy_sigma_2')
net = slim.conv2d(net, 64, [5, 5], scope='conv_zxy_sigma_3')
sigma_u = slim.fully_connected(slim.flatten(net), K, activation_fn=None,
                                      scope='fc_zxy_sigma')

#z_u = tf.random_normal(tf.shape(mu_u)) * sigma_u + mu_u
mu_u = tf.tile(mu_u, [N, 1])
sigma_u = tf.tile(sigma_u, [N, 1])
q_z_u = MultivariateNormalDiag(mu_u, sigma_u, name='q_z_u')
#z_u = q_z_u.sample(name='z_u')
z_u = tf.random_normal(tf.shape(mu_u), name='z_u') * sigma_u + mu_u

# variational posterior q(z|x,y) for labeled data, i.e. the encoder

y_tiled_l = tf.tile(y_l[:, None, None, :], [1, shape[1], shape[2], 1])
x_and_y_l = tf.concat((x_l_in, y_tiled_l), axis=3, name='x_and_y_l')

net = slim.conv2d(x_and_y_l, 32, [5, 5], scope='conv_zxy_mu_1', reuse=True)
net = slim.conv2d(net, 64, [5, 5], scope='conv_zxy_mu_2', reuse=True)
net = slim.conv2d(net, 64, [5, 5], scope='conv_zxy_mu_3', reuse=True)
mu_l = slim.fully_connected(slim.flatten(net), K, activation_fn=None,
                                      scope='fc_zxy_mu', reuse=True)

net = slim.conv2d(x_and_y_l, 32, [5, 5], scope='conv_zxy_sigma_1', reuse=True)
net = slim.conv2d(net, 64, [5, 5], scope='conv_zxy_sigma_2', reuse=True)
net = slim.conv2d(net, 64, [5, 5], scope='conv_zxy_sigma_3', reuse=True)
sigma_l = slim.fully_connected(slim.flatten(net), K, activation_fn=None,
                                      scope='fc_zxy_sigma', reuse=True)

#z_l = tf.random_normal(tf.shape(mu_l)) * sigma_l + mu_l
#q_z_l = Normal(mu_l, sigma_l)
mu_l = tf.tile(mu_l, [N, 1])
sigma_l = tf.tile(sigma_l, [N, 1])
q_z_l = MultivariateNormalDiag(mu_l, sigma_l, name='q_z_l')
#z_l = q_z_l.sample(name='z_l')
z_l = tf.random_normal(tf.shape(mu_l), name='z_l') * sigma_l + mu_l

# generative model p(x|y,z) for unlabeled data, i.e. the decoder
z_and_y_u = tf.concat((z_u, tf.tile(y_u, [N, 1])), axis=1)

net = slim.fully_connected(z_and_y_u, 64, scope='fc_xyz_1', activation_fn=None)
net = tf.reshape(net, [-1, 8, 8, 1])
#net = slim.conv2d_transpose(net, 128, [3, 3], stride=2, scope='convt_xyz_1')
net = slim.conv2d_transpose(net, 64, [3, 3], stride=2, scope='convt_xyz_2')
net = slim.conv2d_transpose(net, 32, [3, 3], stride=2, scope='convt_xyz_3')
#net = slim.conv2d_transpose(net, 32, [3, 3], stride=2, scope='convt_xyz_4')
net = slim.flatten(net)
logits_x_u = slim.fully_connected(net, 784, scope='fc_xyz_2', activation_fn=None)
# (shape=(batch_size * N * N,784))
#logits_x_u = tf.tile(logits_x_u, [N, 1])
p_x_u = Bernoulli(logits=logits_x_u, name='p_x_u')

# generative model p(x|y,z) for labeled data, i.e. the decoder
z_and_y_l = tf.concat((z_l, tf.tile(y_l, [N, 1])), axis=1)

net = slim.fully_connected(z_and_y_l, 64, scope='fc_xyz_1', activation_fn=None, reuse=True)
net = tf.reshape(net, [-1, 8, 8, 1])
#net = slim.conv2d_transpose(net, 128, [3, 3], stride=2, scope='convt_xyz_1')
net = slim.conv2d_transpose(net, 64, [3, 3], stride=2, scope='convt_xyz_2', reuse=True)
net = slim.conv2d_transpose(net, 32, [3, 3], stride=2, scope='convt_xyz_3', reuse=True)
#net = slim.conv2d_transpose(net, 32, [3, 3], stride=2, scope='convt_xyz_4')
net = slim.flatten(net)
logits_x_l = slim.fully_connected(net, 784, scope='fc_xyz_2', activation_fn=None, reuse=True)
# (shape=(batch_size * N,784))
#logits_x_l = tf.tile(logits_x_l, [N, 1])
p_x_l = Bernoulli(logits=logits_x_l, name='p_x_l')

# loss for labeled samples
p_z_l = MultivariateNormalDiag(tf.zeros_like(mu_l), tf.ones_like(sigma_l), name='p_z_l')
kl_tmp_labeled = q_z_l.pdf(z_l) * (q_z_l.log_pdf(z_l) - tf.log(1.0/M) - p_z_l.log_pdf(z_l))
kl_tmp_labeled = tf.reshape(kl_tmp_labeled, [-1, N], name='KL_tmp_labeled')
KL_labeled = tf.reduce_sum(kl_tmp_labeled, 1, name='KL_labeled')
E_x_l = tf.reduce_sum(p_x_l.log_prob(tf.tile(x_l, [N, 1])), 1, name='E_x_l_1')
E_x_l = tf.reshape(E_x_l, [-1, N], name='E_x_l_2')
E_x_l = tf.reduce_mean(E_x_l, 1, name='E_x_l_3')
L = E_x_l - KL_labeled

# loss for unlabeled samples
p_z_u = MultivariateNormalDiag(tf.zeros_like(mu_u), tf.ones_like(sigma_u), name='p_z_u')
q_y_u = tf.tile(q_y_u, [N]) # shape [batch_size * N * N]
kl_tmp_unlabeled = q_z_u.pdf(z_u) * q_y_u * (tf.log(q_y_u) + q_z_u.log_pdf(z_u) -
 tf.log(1.0 / M) - p_z_u.log_pdf(z_u))
kl_tmp_unlabeled = tf.reshape(kl_tmp_unlabeled, [-1, N, N], name='KL_tmp_unlabeled')
KL_unlabeled = tf.reduce_sum(kl_tmp_unlabeled, [1, 2], name='KL_unlabeled')
E_x_u = tf.reduce_sum(p_x_u.log_prob(tf.tile(x_u, [N * N, 1])), 1, name='E_x_u_1')
E_x_u = tf.reshape(E_x_u, [-1, N, N], name='E_x_u_2')
E_x_u = tf.reduce_mean(E_x_u, [1, 2], name='E_x_u_3')
U = E_x_u - KL_unlabeled

# full objective
alpha = 1.0 #0.3
mean_L = tf.reduce_mean(-L, name='mean_L')
mean_U = tf.reduce_mean(-U, name='mean_U')
KL_U = tf.reduce_mean(KL_unlabeled)
#sup_loss = -tf.reduce_mean(q_y_l.log_prob(y_l), name='sup_loss')
sup_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_l, logits = logits_y_l, name = 'sup_loss'))
loss = mean_L + mean_U + alpha * sup_loss
lr = tf.constant(0.001)
train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss, var_list = slim.get_model_variables(), name = 'train_op')
init_op = tf.initialize_all_variables()

# get data
num_labeled = np.int32(100)
mnist = input_data.read_data_sets('/tmp/', n_labeled=num_labeled, one_hot=True)
data = mnist.train

BATCH_SIZE=100
NUM_ITERS=30000
tau0=1.0 # initial temperature
np_temp=tau0
np_lr=0.001
ANNEAL_RATE=0.00003
MIN_TEMP=0.5

dat=[]
sess=tf.InteractiveSession()
file_writer = tf.summary.FileWriter('/home/ekaraulov/temp/tf_logs', sess.graph)
save_path = "/home/ekaraulov/temp/vae_cnn.ckpt"
checkpoint_file = Path(save_path + '.meta')
saver = tf.train.Saver()
if checkpoint_file.is_file():
  # saver = tf.train.import_meta_graph(str(checkpoint_file))
  saver.restore(sess, tf.train.latest_checkpoint(str(checkpoint_file.parent)))
else:
  # saver = tf.train.Saver()
  sess.run(init_op)
  for i in range(1,NUM_ITERS):
    np_x,np_y=data.next_batch(BATCH_SIZE)
    np_x_l = np_x[:100,:]
    np_x_u = np_x[100:, :]
    np_y_l = np_y[:100, :]
    _,np_loss,np_L,np_U,np_sup_loss,np_KL_U=sess.run([train_op,loss,mean_L,mean_U,sup_loss, KL_U],{
        x_l:np_x_l,
        x_u:np_x_u,
        y_l:np_y_l,
        tau:np_temp,
        lr:np_lr
      })
    if i % 100 == 1:
      dat.append([i,np_temp,np_loss])
    if i % 1000 == 1:
      np_temp=np.maximum(tau0*np.exp(-ANNEAL_RATE*i),MIN_TEMP)
      np_lr*=0.9
    if i % 5000 == 1:
      print('Step %d, ELBO: %0.3f, L: %0.3f, U: %0.3f, sup_loss: %0.3f, KL_U: %0.3f' % (i,-np_loss, np_L, np_U, np_sup_loss, np_KL_U))
      save_path = saver.save(sess, save_path)
      print("Model saved in file: %s" % save_path)

data_t = mnist.test
nbatches = np.floor(data_t.labels.shape[0] / BATCH_SIZE).astype(int)
np_correct = 0
for i in range(1, nbatches):
  np_x_t, np_y_t = data_t.next_batch(BATCH_SIZE)
  np_y_l_out = sess.run([y_l_out], {x_u:np_x_t, x_l:np_x_t})
  np_correct = np_correct + np.sum(np_y_l_out * np_y_t)
print("Accuracy: %f" % (np_correct / data_t.labels.shape[0] * 100))
