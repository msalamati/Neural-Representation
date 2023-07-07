"""Running this code generates the training data needed for controller
 compression, including the onehot (or multi-hot vector)!"""
import numpy as np
import math
import time
import ray
import copy
import env_pool
import scipy.io
import mat73

# Defining the functions


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


def generate_training_data():
    """This function is invoked when training data are needed to be written on Disk"""
    print("win_domain_size is", win_domain_size_not_scaled)
    print("controller size is", controller_not_scaled.shape)
    for step in range(win_domain_size_not_scaled // num_tasks_per_step+1):

        num_tasks_in_this_step = min(num_tasks_per_step, win_domain_size_not_scaled-step*num_tasks_per_step)
        #  start1 = time.time()
        #  ray.init(_plasma_directory="/tmp")
        output_coded0 = [single_data_generation.remote(step * num_tasks_per_step // length + i, num_tasks_in_this_step-i*length, controller_coded, win_domain_coded)[0] for i in
                       range(0, num_tasks_in_this_step // length+1)]
        output_coded1 = [single_data_generation.remote(step * num_tasks_per_step // length + i, num_tasks_in_this_step-i*length, controller_coded, win_domain_coded)[1] for i in
                       range(0, num_tasks_in_this_step // length+1)]
        output_coded2 = [single_data_generation.remote(step * num_tasks_per_step // length + i, num_tasks_in_this_step-i*length, controller_coded, win_domain_coded)[2] for i in
                       range(0, num_tasks_in_this_step // length+1)]
        #  print('the time taken by remoting is', time.time() - start)
        output_coded3 = [single_data_generation.remote(step * num_tasks_per_step // length + i, num_tasks_in_this_step-i*length, controller_coded, win_domain_coded)[3] for i in
                        range(0, num_tasks_in_this_step // length+1)]
        #  print('the time taken by remoting is', time.time() - start)
        #start = time.time()
        output_decoded0 = ray.get(output_coded0)
        output_decoded1 = ray.get(output_coded1)
        output_decoded2 = ray.get(output_coded2)
        #  print('the time taken by ray.get is', time.time() - start)
        output_decoded3 = ray.get(output_coded3)
        del output_coded0, output_coded1, output_coded2, output_coded3
        #  print('the time taken by ray.get is', time.time()-start)
        for i in range(num_tasks_in_this_step // length+1):
            length_for_this_step = min(length, num_tasks_in_this_step - i * length)
            for j in range(length_for_this_step*(scale**dim_x)):
                win_domain_scaled[(step*num_tasks_per_step+i*length)*(scale**dim_x)+j, :] = output_decoded0[i][j]
                controller_scaled[(step*num_tasks_per_step+i*length)*(scale**dim_x)+j, :] = output_decoded1[i][j]
                onehot[(step*num_tasks_per_step+i*length)*(scale**dim_x)+j, :] = output_decoded2[i][j]
                num_valid_cont_list[(step*num_tasks_per_step+i*length)*(scale**dim_x)+j] = output_decoded3[i][j]
        del output_decoded0, output_decoded1, output_decoded2, output_decoded3
        #  print('the time taken for each iteration is', time.time()-start1)
    onehot.flush()
    num_valid_cont_list.flush()


@ray.remote(num_returns=4)
def single_data_generation(ii, num_tasks, controller_codedd, win_domain_codedd):
    length_for_this_step = min(length, num_tasks)
    x = []
    c = []
    y = []
    z = []
    bin_len = dim_x  # math.ceil(math.log(scale ** dim_x, scale))
    for k in range(length_for_this_step):
        state_main = win_domain_codedd[ii*length+k, :]
        onehot_vec = np.zeros(num_dis_inputs)  # initiating the array containing an input
        controller_row = controller_codedd[ii*length+k, :]
        state_ind_set = np.where(np.all(controller_codedd[:, 0:dim_x] == state_main, axis=1))[0]
        num_valid_cont = len(state_ind_set)
        for j in range(num_valid_cont):
            inp_ind = cell_to_ind(controller_codedd[state_ind_set[j], dim_x:dim_x + dim_u]).astype(int)
            onehot_vec[inp_ind] = 1
        for sub_cell_idx in range(scale**dim_x):
            bin_sub_cell_idx = binary_converter(sub_cell_idx, bin_len)
            sub_cell_center = state_main - np.squeeze(eta_x) * scale/2 + np.squeeze(eta_x)/2+ bin_sub_cell_idx*np.squeeze(eta_x)
            x.append(copy.copy(sub_cell_center))
            c.append(copy.copy(controller_row))
            y.append(copy.copy(onehot_vec))
            z.append(copy.copy(num_valid_cont))
    return [x, c, y, z]


def cell_to_ind(vec):
    """Transforming a given vector in the continuous state space into a scalar index (for the input space)"""
    vectorized_ind = np.maximum(np.zeros(dim_u), np.minimum((vec - U_range[:, 0])//np.squeeze(eta_u), discrete_inp_size-np.ones((np.shape(discrete_inp_size)))))
    ind = vectorized_ind[0]
    for ii in range(1, dim_u):
        ind += vectorized_ind[ii]*np.prod(discrete_sys_size[0:ii-1])
    return ind


def binary_converter(x, bits):
    """"***FOR scale != 2 this function doesn't work***"""
    """function to convert a number x into a vector of bits"""
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)])


# Listing the example-specific parameters
env = env_pool.tora()

X_range = env.X_range  # state-space
U_range = env.U_range  # input space
sample_time = env.sample_time  # sampling time in seconds
scale = env.scale  # Each cell is broken into "scale^n" number of cells
eta_x = env.eta_x/scale  # state-space discretization size
eta_u = env.eta_u  # input-space discretization size
# parallelization parameters
length = env.length
num_tasks_per_step = env.num_tasks_per_step
# defining filenames for loading the controller
win_domain_filename_not_scaled = env.win_domain_filename_not_scaled
win_domain_filename_scaled = env.win_domain_filename_scaled
controller_filename_not_scaled = env.controller_filename_not_scaled
controller_filename_scaled = env.controller_filename_scaled
# defining filenames for the processed output data
onehot_filename = env.onehot_filename
num_valid_cont_list_filename = env.num_valid_cont_list_filename

# Extract descriptive parameters of the system
rr_x = eta_x / 2  # radius of the partitions in the state-space
rr_u = eta_u / 2  # radius of the partitions in the input-space
dim_x = np.shape(X_range)[0]  # dimension of the state-space
dim_u = np.shape(U_range)[0]  # dimension of the input-space
discrete_sys_size = discrete_sys_size_gen()  # vector containing number of discrete pointsalong each dimension in the
discrete_inp_size = discrete_sys_size[dim_x:dim_x+dim_u]
# state and input spaces
num_dis_inputs = np.prod(discrete_sys_size[dim_x:dim_x + dim_u]).astype(int)  # size of the input-space
# print(num_dis_inputs)

# Generate training data with writing over disk
ray.init(_plasma_directory="/tmp", log_to_driver=False)
start = time.time()
"""win_domain = scipy.io.loadmat(win_domain_filename)["win_domain"]
win_domain_coded = ray.put(win_domain)
#win_domain = mat73.loadmat(win_domain_filename)['win_domain']
win_domain_size = win_domain.shape[0]"""
win_domain_not_scaled = scipy.io.loadmat(win_domain_filename_not_scaled)["win_domain"]
# win_domain_not_scaled = mat73.loadmat(win_domain_filename_not_scaled)['win_domain']
win_domain_coded = ray.put(win_domain_not_scaled)
win_domain_size_not_scaled = win_domain_not_scaled.shape[0]
win_domain_size_scaled = win_domain_size_not_scaled * (scale**dim_x)
print("I'm here 1")
controller_not_scaled = scipy.io.loadmat(controller_filename_not_scaled)["controller"]
#print(np.min(controller_not_scaled[:, 2]))
#print(np.max(controller_not_scaled[:, 2]))
#print(np.min(controller_not_scaled[:, 3]))
#print(np.max(controller_not_scaled[:, 3]))
#print(np.shape(np.unique(controller_not_scaled[:, 2], axis=0)))
# controller_not_scaled = mat73.loadmat(controller_filename_not_scaled)['controller']
print("I'm here 2")
controller_coded = ray.put(controller_not_scaled)
win_domain_scaled = np.memmap(win_domain_filename_scaled, dtype='float32', mode='w+', shape=(win_domain_size_scaled, dim_x))
controller_scaled = np.memmap(controller_filename_scaled, dtype='float32', mode='w+', shape=(controller_not_scaled.shape[0]*(scale**dim_x), dim_x+dim_u))
onehot = np.memmap(onehot_filename, dtype='float32', mode='w+', shape=(win_domain_size_scaled, num_dis_inputs))  #  np.zeros((win_domain_size, num_dis_inputs))
num_valid_cont_list = np.memmap(num_valid_cont_list_filename, dtype='float32', mode='w+', shape=(win_domain_size_scaled))
generate_training_data()
"""controller_ind = 0
for i in range(win_domain_size):
    start1 = time.time()
    #state_ind_set = np.where(np.all(controller[:, 0:dim_x] == win_domain[i, :], axis=1))[0]
    state_ind_set = []
    x = win_domain[i, :]
    goOn = True
    while goOn:
        if (abs(controller[controller_ind, 0:dim_x] - x) < 1e-4).all():
            state_ind_set.append(controller_ind)
            controller_ind += 1
        else:
            goOn = False
            print("The number of matches is", len(state_ind_set))
    length = len(state_ind_set)
    if length == 0:
        print("The domain index is", i)
        print("controller index is", controller_ind)
    num_valid_cont_list[i] = length
    for j in range(length):
        inp_ind = cell_to_ind(controller[state_ind_set[j], dim_x:dim_x+dim_u]).astype(int)
        onehot[i, inp_ind] = 1
    print("time is", time.time()-start1)"""
ray.shutdown()
print("Finished!")
print("The writing time is:", time.time()-start)



