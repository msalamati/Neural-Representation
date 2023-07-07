"""By running this code, one classification-based neural network is trained and saved:
for a given state vector the trained NN produces a vector which must assign higher values to the valid control inputs."""
import os
import numpy as np
import math
import time
import scipy.io
import mat73

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import env_pool

def discrete_sys_size_gen():
    """ This function computes a vector that contains number of
     discrete states for every dimension of state and input spaces."""
    discrete_sys_size = np.zeros(dim_x + dim_u)
    for ii in range(0, dim_x):
        discrete_sys_size[ii] = math.floor((X_range[ii, 1] - X_range[ii, 0] - eta_x[ii]) / eta_x[ii] + 1)
    for ii in range(dim_x, dim_x + dim_u):
        discrete_sys_size[ii] = math.floor(
            (U_range[ii - dim_x, 1] - U_range[ii - dim_x, 0] - eta_u[ii - dim_x]) / eta_u[ii - dim_x] + 1)
    return discrete_sys_size.astype(int)


"""def NN_structure():
    model = Sequential()
    model.add(Dense(20, input_dim=dim_x, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.tanh))
    model.add(Dense(160, activation=tf.nn.tanh))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(250, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(num_dis_inputs, activation='linear'))
    return model"""


def NN_structure():
    """Create the model for guessing valid control inputs."""
    model = Sequential()
    model.add(Dense(20, input_dim=dim_x, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))

    """model.add(Dense(250, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))"""
    model.add(Dense(num_dis_inputs, activation='linear'))
    return model




def My_Custom_Generator_TS(x, out_filename, num_samples, inp_dim, out_dim, batch_size):
    """On-the-fly data Generator for the training process (random-read)."""
    inputs = []
    targets = []
    batchcount = 0
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    #x = np.memmap(inp_filename, dtype='float32', mode='r', shape=(num_samples, inp_dim), offset=0)
    y = np.memmap(out_filename, dtype='float32', mode='r', shape=(num_samples, out_dim), offset=0)
    while True:
        for line in indices:
            inputs.append([x[line, :]])
            targets.append([y[line, :]])
            batchcount += 1
            if batchcount > batch_size:
                batch_x = np.array(inputs, dtype='float32')
                batch_y = np.array(targets, dtype='float32')
                yield (batch_x, batch_y)
                inputs = []
                targets = []
                batchcount = 0


# Setting the parameters
env = env_pool.tora()

X_range = env.X_range  # state-space
U_range = env.U_range  # input space
sample_time = env.sample_time  # sampling time in seconds
scale = env.scale
eta_x = env.eta_x/scale  # state-space discretization size
eta_u = env.eta_u  # input-space discretization size
# parallelization parameters
length = env.length
num_tasks_per_step = env.num_tasks_per_step
# learning related settings
epochs = env.epochs
batch_size = env.batch_size
learning_rate = env.learning_rate
# defining filenames for loading the controller
win_domain_filename_not_scaled = env.win_domain_filename_not_scaled
win_domain_filename_scaled = env.win_domain_filename_scaled
# defining filenames for the processed output data
onehot_filename = env.onehot_filename
num_valid_cont_list_filename = env.num_valid_cont_list_filename
# defining paths for saving the trained NNs
checkpoint_path_NN_controller = env.checkpoint_path_NN_controller


# Extract descriptive parameters of the system
rr_x = eta_x / 2  # radius of the partitions in the state-space
rr_u = eta_u / 2  # radius of the partitions in the input-space
dim_x = np.shape(X_range)[0]  # dimension of the state-space
dim_u = np.shape(U_range)[0]  # dimension of the input-space
discrete_sys_size = discrete_sys_size_gen()  # vector containing number of discrete pointsalong each dimension in the
# state and input spaces
num_dis_inputs = np.prod(discrete_sys_size[dim_x:dim_x + dim_u]).astype(int)  # size of the input-space
# winning domain size
win_domain_not_scaled = scipy.io.loadmat(win_domain_filename_not_scaled)["win_domain"]
#win_domain = mat73.loadmat(win_domain_filename)['win_domain']
win_domain_size_not_scaled = win_domain_not_scaled.shape[0]
win_domain_size_scaled = win_domain_size_not_scaled * (scale**dim_x)
win_domain_scaled = np.memmap(win_domain_filename_scaled, dtype='float32', mode='r', shape=(win_domain_size_scaled, dim_x))
#controller = scipy.io.loadmat(controller_filename)["controller"]
#onehot = np.memmap(onehot_filename, dtype='float32', mode='r', shape=(win_domain_size, num_dis_inputs))  #  np.zeros((win_domain_size, num_dis_inputs))
#num_valid_cont_list = np.memmap(num_valid_cont_list_filename, dtype='float32', mode='r', shape=(win_domain_size))

# Training using TF 2.x
num_batches = int(np.ceil(win_domain_size_scaled / batch_size))
# chunck_size = batch_size_TS * 100

# Compile the model
NN_controller = NN_structure()

# Training using TF 2.x
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=math.ceil(win_domain_size_scaled/num_batches),
    decay_rate=.999,
    staircase=True)
NN_controller.compile(loss='mean_squared_error',
               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate/2, beta_1=.999, beta_2=.99999, name='Adam'),
               metrics=['mean_squared_error'])
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_NN_controller,
                                                 save_weights_only=True,
                                                 verbose=1)


# Fit data to models using data written on disk+save the trained models
start = time.time()
history1 = NN_controller.fit(My_Custom_Generator_TS(win_domain_scaled, onehot_filename, win_domain_size_scaled, 1, num_dis_inputs, batch_size),
                     steps_per_epoch=num_batches, epochs=epochs, verbose=1, validation_split=0, callbacks=[cp_callback])
print("Execution time for full training", time.time() - start)

