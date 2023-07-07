"""By running this code, four Neural networks are trained and saved: two for estimating forward and backward end-points
 for each discrete state-input pairs; and two for estimating forward and backward growth bounds with respect
to discrete inputs."""
import os
import numpy as np
import math
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import env_pool
#import levenberg_marquardt as lm

def discrete_sys_size_gen(x_range):
    """ This function computes a vector that contains number of
     discrete states for every dimension of state and input spaces."""
    discrete_sys_size = np.zeros(dim_x + dim_u)
    for ii in range(0, dim_x):
        discrete_sys_size[ii] = math.floor((x_range[ii, 1] - x_range[ii, 0] - eta_x[ii]) / eta_x[ii] + 1)
    for ii in range(dim_x, dim_x + dim_u):
        discrete_sys_size[ii] = math.floor(
            (U_range[ii - dim_x, 1] - U_range[ii - dim_x, 0] - eta_u[ii - dim_x]) / eta_u[ii - dim_x] + 1)
    return discrete_sys_size.astype(int)


def NN_structure_TS():
    """Create the model for estimating the transition system."""
    model = Sequential()
    model.add(Dense(40, input_dim=dim_x+dim_u, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(800, activation=tf.nn.relu))
    model.add(Dense(nn, activation='linear'))
    return model



def My_Custom_Generator_TS(inp_filename, out_filename, num_samples, inp_dim, out_dim, batch_size):
    """On-the-fly data Generator for the training process (random-read)."""
    inputs = []
    targets = []
    batchcount = 0
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    x = np.memmap(inp_filename, dtype='float32', mode='r', shape=(num_samples, inp_dim), offset=0)
    y = np.memmap(out_filename, dtype='float32', mode='r', shape=(num_samples, out_dim), offset=0)
    while True:
        for line in indices:
            """if FW_or_BW:
                linee = transform_ind_BW_to_FW(line)
            else:"""
            linee = line
            inputs.append([x[linee, :]])
            targets.append([y[linee, :]])
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
eta_x = env.eta_x  # state-space discretization size
eta_u = env.eta_u  # input-space discretization size
shift_no = env.shift_no  # shifting the index vectors to avoid negative values
# parallelization parameters
length = env.length
num_tasks_per_step = env.num_tasks_per_step
# learning related settings
epochs_TS = env.epochs_TS
batch_size_TS = env.batch_size_TS
learning_rate = env.learning_rate
# defining filenames for saving the transition system (note that )
forw_inp_TS_filename = env.forw_inp_TS_filename
forw_out_TS_filename = env.forw_out_TS_filename
back_inp_TS_filename = env.back_inp_TS_filename
back_out_TS_filename = env.back_out_TS_filename

# defining paths for saving the trained NNs
checkpoint_path_TS_forw = env.checkpoint_path_TS_forw
checkpoint_path_TS_back = env.checkpoint_path_TS_back





# Extract descriptive parameters of the system
rr_x = eta_x / 2  # radius of the partitions in the state-space
rr_u = eta_u / 2  # radius of the partitions in the input-space
dim_x = np.shape(X_range)[0]  # dimension of the state-space
dim_u = np.shape(U_range)[0]  # dimension of the input-space
discrete_sys_size = discrete_sys_size_gen(X_range)  # vector containing number of discrete pointsalong each dimension in the
discrete_inp_size = discrete_sys_size[dim_x:dim_x+dim_u]
state_space_size = discrete_sys_size[0:dim_x]


# state and input spaces
num_dis_states = np.prod(discrete_sys_size[0:dim_x]).astype(int)  # size of the state-space
num_dis_inputs = np.prod(discrete_sys_size[dim_x:dim_x + dim_u]).astype(int)  # size of the input-space
num_state_inp_pairs = np.prod(discrete_sys_size).astype(int)  # number of state-input pairs
nn = 2 * (np.sum(discrete_sys_size[0:dim_x])+3*shift_no*dim_x)  # dimension of the vector at the output of the trained NN

# Training using TF 2.x
num_batches_TS = int(np.ceil(np.prod(discrete_sys_size).astype(int) / batch_size_TS))

# Compile the model
NN_TS_forw = NN_structure_TS()
NN_TS_back = NN_structure_TS()
# Training using TF 2.x
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=math.ceil(num_state_inp_pairs/num_batches_TS),
    decay_rate=.999,
    staircase=True)
NN_TS_forw.compile(loss='mean_squared_error',
               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate/10, beta_1=.999, beta_2=.99999, name='Adam'),
               metrics=['mean_squared_error'])
NN_TS_back.compile(loss='mean_squared_error',
               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate/10, beta_1=.999, beta_2=.99999, name='Adam'),
               metrics=['mean_squared_error'])

# using the lm method for curve fitting
"""x_train = np.memmap(forw_inp_TS_filename, dtype='float32', mode='r', shape=(num_state_inp_pairs, dim_x+dim_u), offset=0)
y_train = np.memmap(forw_out_TS_filename, dtype='float32', mode='r', shape=(num_state_inp_pairs, nn), offset=0)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(num_state_inp_pairs)
train_dataset = train_dataset.batch(batch_size_TS).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

model = NN_structure_TS()
model.summary()
model_wrapper = lm.ModelWrapper(
    tf.keras.models.clone_model(model))
model_wrapper.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
    loss=lm.MeanSquaredError())
history1 = model_wrapper.fit(train_dataset, epochs=epochs_TS)"""
# Create a callback that saves the model's weights
cp_callback_TS_forw = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_TS_forw,
                                                 save_weights_only=True,
                                                 verbose=1)
cp_callback_TS_back = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_TS_back,
                                                 save_weights_only=True,
                                                 verbose=1)


# Fit data to models using data written on disk+save the trained models
start = time.time()

"""history1 = NN_TS_forw.fit(My_Custom_Generator_TS(forw_inp_TS_filename, forw_out_TS_filename, num_state_inp_pairs, dim_x+dim_u, nn, batch_size_TS),
                     steps_per_epoch=num_batches_TS, epochs=epochs_TS, verbose=1, validation_split=0, callbacks=[cp_callback_TS_forw])
print("Execution time for FW training", time.time() - start)"""
history2 = NN_TS_back.fit(My_Custom_Generator_TS(back_inp_TS_filename, back_out_TS_filename, num_state_inp_pairs, dim_x+dim_u, nn, batch_size_TS),
                     steps_per_epoch=num_batches_TS, epochs=epochs_TS, verbose=1, validation_split=0, callbacks=[cp_callback_TS_back])
print("Execution time for FW+BW training", time.time() - start)
print("Execution time for full training", time.time() - start)