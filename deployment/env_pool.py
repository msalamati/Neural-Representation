"""Here, one can insert specification for each of the case-studies,
 related to the controller compression project"""
import os
import numpy as np
import math
# 2D-car (dim=2+2)
class vehicle_2d:
    def __init__(self):
        self.X_range = np.array([[-.025, 10.025], [-.025, 10.025]])  # state-space
        self.U_range = np.array([[-1.2, 1.2], [-1.2, 1.2]])  # input space
        self.sample_time = .4  # sampling time in seconds
        self.eta_x = np.array([[.05], [.05]])  # state-space discretization size
        self.eta_u = np.array([[.3], [.3]])  # input-space discretization size
        self.scale = 2  # Each cell is broken into "scale^n" number of cells
        current_path = os.getcwd()
        # parallelization parameters
        self.length = 10
        self.num_tasks_per_step = 200
        # learning related settings
        self.epochs = 20
        self.batch_size = 64
        self.learning_rate = 0.0005
        # defining filenames for for loading the controller
        self.win_domain_filename_not_scaled = os.path.join(current_path, 'examples/2D_car/win_domain.mat')
        self.win_domain_filename_scaled = os.path.join(current_path, 'examples/2D_car/win_domain_scaled.dat')
        self.controller_filename_not_scaled = os.path.join(current_path, 'examples/2D_car/controller.mat')
        self.controller_filename_scaled = os.path.join(current_path, 'examples/2D_car/controller_scaled.dat')
        # defining filenames for the processed output data
        self.onehot_filename = os.path.join(current_path, 'examples/2D_car/onehot.dat')
        self.num_valid_cont_list_filename = os.path.join(current_path,
                                                         'examples/2D_car/num_valid_cont_list_filename.dat')
        # defining paths for saving the trained NNs
        self.checkpoint_path_NN_controller = os.path.join(current_path, 'examples/2D_car/NN/')

# inverted pendulum (dim=2+1)
class inv_pend:
    def __init__(self):
        self.X_range = np.array([[-3.505, 3.505], [-2.005, 2.005]])
        self.U_range = np.array([[-1.15, 1.15]])  # input space
        self.sample_time = .01  # sampling time in seconds
        self.eta_x = np.array([[.01], [.01]])
        self.eta_u = np.array([[.3]])
        self.scale = 1  # Each cell is broken into "scale^n" number of cells
        current_path = os.getcwd()
        # parallelization parameters
        self.length = 10
        self.num_tasks_per_step = 200
        # learning related settings
        self.epochs = 20
        self.batch_size = 64
        self.learning_rate = 0.001
        # defining filenames for for loading the controller
        self.win_domain_filename_not_scaled = os.path.join(current_path, 'examples/inv_pend_2D/win_domain.mat')
        self.win_domain_filename_scaled = os.path.join(current_path, 'examples/inv_pend_2D/win_domain_scaled.dat')
        self.controller_filename_not_scaled = os.path.join(current_path, 'examples/inv_pend_2D/controller.mat')
        self.controller_filename_scaled = os.path.join(current_path, 'examples/inv_pend_2D/controller_scaled.dat')
        # defining filenames for the processed output data
        self.onehot_filename = os.path.join(current_path, 'examples/inv_pend_2D/onehot.dat')
        self.num_valid_cont_list_filename = os.path.join(current_path, 'examples/inv_pend_2D/num_valid_cont_list_filename.dat')
        # defining paths for saving the trained NNs
        self.checkpoint_path_NN_controller = os.path.join(current_path, 'examples/inv_pend_2D/NN/')

# TORA (dim=4+1)
class tora:
    def __init__(self):
        self.X_range = np.array([[-2.05, 2.05], [-2.05, 2.05], [-2.05, 2.05], [-2.05, 2.05]])
        self.U_range = np.array([[-1.15, 1.15]])  # input space
        self.sample_time = .5  # sampling time in seconds
        self.eta_x = np.array([[.1], [.1], [.1], [.1]])
        self.eta_u = np.array([[.3]])
        self.scale = 2  # Each cell is broken into "scale^n" number of cells
        current_path = os.getcwd()
        # parallelization parameters
        self.length = 10
        self.num_tasks_per_step = 32 * self.length
        # learning related settings
        self.epochs = 20
        self.batch_size = 64
        self.learning_rate = 0.001
        # defining filenames for for loading the controller
        self.win_domain_filename_not_scaled = os.path.join(current_path, 'examples/tora/win_domain.mat')
        self.win_domain_filename_scaled = os.path.join(current_path, 'examples/tora/win_domain_scaled.dat')
        self.controller_filename_not_scaled = os.path.join(current_path, 'examples/tora/controller.mat')
        self.controller_filename_scaled = os.path.join(current_path, 'examples/tora/controller_scaled.dat')
        # defining filenames for the processed output data
        self.onehot_filename = os.path.join(current_path, 'examples/tora/onehot.dat')
        self.num_valid_cont_list_filename = os.path.join(current_path, 'examples/tora/num_valid_cont_list_filename.dat')
        # defining paths for saving the trained NNs
        self.checkpoint_path_NN_controller = os.path.join(current_path, 'examples/tora/NN/')



# unicycle(dim = 3+2)
class unicycle:
    def __init__(self):
        self.X_range = np.array([[-.1, 10.1], [-.1, 10.1], [-3.6, 3.6]])  # state-space
        self.U_range = np.array([[-1.2, 1.2], [-1.2, 1.2]])  # input space
        self.sample_time = .3  # sampling time in seconds
        self.eta_x = np.array([[.2], [.2], [.2]])  # np.array([[.2], [.2], [.2]])  # state-space discretization size
        self.eta_u = np.array([[.3], [.3]])  # input-space discretization size
        self.scale = 2  # Each cell is broken into "scale^n" number of cells
        current_path = os.getcwd()
        # parallelization parameters
        self.length = 50
        self.num_tasks_per_step = 1000
        # learning related settings
        self.epochs = 20
        self.batch_size = 64
        self.learning_rate = 0.0003
        # defining filenames for for loading the controller
        self.win_domain_filename_not_scaled = os.path.join(current_path, 'examples/unicycle/win_domain.mat')
        self.win_domain_filename_scaled = os.path.join(current_path, 'examples/unicycle/win_domain_scaled.dat')
        self.controller_filename_not_scaled = os.path.join(current_path, 'examples/unicycle/controller.mat')
        self.controller_filename_scaled = os.path.join(current_path, 'examples/unicycle/controller_scaled.dat')
        # defining filenames for the processed output data
        self.onehot_filename = os.path.join(current_path, 'examples/unicycle/onehot.dat')
        self.num_valid_cont_list_filename = os.path.join(current_path, 'examples/unicycle/num_valid_cont_list_filename.dat')
        # defining paths for saving the trained NNs
        self.checkpoint_path_NN_controller = os.path.join(current_path, 'examples/unicycle/NN/')

# 4-D vehicle (dim = 4+2)
class vehicle_4d:
    def __init__(self):
            self.X_range = np.array([[-.1, 10.1], [-.1, 10.1], [-3.6, 3.6], [-1.1, 1.1]])  # state-space
            self.U_range = np.array([[-2.1, 2.1], [-2.1, 2.1]])  # input space
            self.sample_time = .4  # sampling time in seconds
            self.eta_x = np.array([[.2], [.2], [.2], [.2]])  # state-space discretization size
            self.eta_u = np.array([[.2], [.2]])  # input-space discretization size
            self.scale = 1  # Each cell is broken into "scale^n" number of cells
            current_path = os.getcwd()
            # parallelization parameters
            self.length = 200
            self.num_tasks_per_step = 5000
            # learning related settings
            self.epochs = 20
            self.batch_size = 64
            self.learning_rate = 0.001
            # defining filenames for for loading the controller
            self.win_domain_filename_not_scaled = os.path.join(current_path, 'examples/4D_car/win_domain.mat')
            self.win_domain_filename_scaled = os.path.join(current_path, 'examples/4D_car/win_domain_scaled.dat')
            self.controller_filename_not_scaled = os.path.join(current_path, 'examples/4D_car/controller.mat')
            self.controller_filename_scaled = os.path.join(current_path, 'examples/4D_car/controller_scaled.dat')
            # defining filenames for the processed output data
            self.onehot_filename = os.path.join(current_path, 'examples/4D_car/onehot.dat')
            self.num_valid_cont_list_filename = os.path.join(current_path, 'examples/4D_car/num_valid_cont_list_filename.dat')
            # defining paths for saving the trained NNs
            self.checkpoint_path_NN_controller = os.path.join(current_path, 'examples/4D_car/NN/')


# 5D car (dim = 5+2)
class vehicle_5d:
    def __init__(self):
        self.X_range = np.array([[-.1, 10.1], [-.1, 10.1], [-3.5, 3.5], [-1.1, 1.1], [-1.1, 1.1]])  # state-space
        self.U_range = np.array([[-2.1, 2.1], [-2.1, 2.1]])  # input space
        self.sample_time = .4  # sampling time in seconds
        self.eta_x = np.array([[.1], [.1], [.2], [.2], [.2]])  # state-space discretization size
        self.eta_u = np.array([[.2], [.2]])  # input-space discretization size
        current_path = os.getcwd()
        # parallelization parameters
        self.length = 200
        self.num_tasks_per_step = 50000
        # learning related settings
        self.epochs = 20
        self.batch_size = 64
        self.learning_rate = 0.001
        # defining filenames for for loading the controller
        self.win_domain_filename = os.path.join(current_path, 'examples/5D_car/win_domain.mat')
        self.controller_filename = os.path.join(current_path, 'examples/5D_car/controller.mat')
        # defining filenames for the processed output data
        self.onehot_filename = os.path.join(current_path, 'examples/5D_car/onehot.dat')
        self.num_valid_cont_list_filename = os.path.join(current_path, 'examples/5D_car/num_valid_cont_list_filename.dat')
        # defining paths for saving the trained NNs
        self.checkpoint_path_NN_controller = os.path.join(current_path, 'examples/5D_car/NN/')


# Traffic control (dim = 5+2)
class traffic_control:
    def __init__(self):
        self.X_range = np.array([[0, 10], [0, 10], [0, 10], [0, 10], [0, 10]])  # state-space
        self.U_range = np.array([[-.5, 1.5], [-.5, 1.5]])  # input space
        self.Goal_range = np.array([[3, 4], [3, 4], [3, 4], [3, 4], [3, 4]])  # target range
        self.Obs_range = np.array([[.2, .6], [.2, .6], [-3, 3]])  # obstacle range
        self.smoothing_factor = 1
        self.sample_time = .0018  # sampling time in seconds
        self.eta_x = np.array([[.25], [.25], [.25], [.25], [.25]])  # state-space discretization size
        self.eta_u = np.array([[1], [1]])  # input-space discretization size
        self.W = np.array([[.3], [.3], [.3], [.3], [.3]])  # disturbance bounds
        self.nbins = 10  # used for computing the integral
        self.time_span = np.linspace(0, self.sample_time, self.nbins)  # used for computing the integral
        self.is_GB_NN_needed = 0  # set to 1 only if the Jacobian matrix is input-dependent
        current_path = os.getcwd()
        # parallelization parameters
        self.length = 1000
        self.num_tasks_per_step = 32000
        # learning related settings
        self.epochs_TS = 12
        self.epochs_GB = 200
        self.batch_size_TS = 64
        self.batch_size_GB = 1
        self.learning_rate = 0.0001
        # defining filename for loading the nominal trajectory computed by Altro
        self.nom_traj_filename = os.path.join(current_path, '5D_traffic/nom_tr.txt')
        # defining filenames for saving the transition system
        self.forw_inp_TS_filename = os.path.join(current_path, '5D_traffic/forw_inp_TS.dat')
        self.forw_out_TS_filename = os.path.join(current_path, '5D_traffic/forw_out_TS.dat')
        self.back_inp_TS_filename = os.path.join(current_path, '5D_traffic/back_inp_TS.dat')
        self.back_out_TS_filename = os.path.join(current_path, '5D_traffic/back_out_TS.dat')
        # defining filenames for saving the input-GB sets
        self.discrete_inp_set_filename = os.path.join(current_path, '5D_traffic/discrete_inp_set.dat')
        self.forw_GB_filename = os.path.join(current_path, '5D_traffic/forw_GB.dat')
        self.back_GB_filename = os.path.join(current_path, '5D_traffic/back_GB.dat')
        # defining paths for saving the trained NNs
        self.checkpoint_path_TS_forw = os.path.join(current_path, '5D_traffic/NN_forw/TS/')
        self.checkpoint_path_TS_back = os.path.join(current_path, '5D_traffic/NN_back/TS/')
        self.checkpoint_path_GB_forw = os.path.join(current_path, '5D_traffic/NN_forw/GB/')
        self.checkpoint_path_GB_back = os.path.join(current_path, '5D_traffic/NN_back/GB/')
        # defining filenames for saving the approximation error vector
        self.e_vec_TS_forw_filename = os.path.join(current_path, '5D_traffic/e_vec_TS_forw.dat')
        self.e_vec_TS_back_filename = os.path.join(current_path, '5D_traffic/e_vec_TS_back.dat')
        self.e_vec_GB_forw_filename = os.path.join(current_path, '5D_traffic/e_vec_GB_forw.dat')
        self.e_vec_GB_back_filename = os.path.join(current_path, '5D_traffic/e_vec_GB_back.dat')
        # defining filenames for saving the safety margins
        self.safety_margin_FW_filename = os.path.join(current_path, '5D_traffic/safety_margin_forw.npy')
        self.safety_margin_BW_filename = os.path.join(current_path, '5D_traffic/safety_margin_back.npy')
        # defining filenames for saving the mismatch edges per state-input pair
        self.mismatch_list_forw_filename = os.path.join(current_path, '5D_traffic/mismatch_list_forw.dat')
        self.num_transitions_forw_filename = os.path.join(current_path, '5D_traffic/num_transitions_forw.dat')
        self.R_analytical_filename = os.path.join(current_path, '5D_traffic/R_analytical.dat')
        self.R_NN_filename = os.path.join(current_path, '5D_traffic/R_NN.dat')
        # defining filenames for saving the controller and winning domain
        self.controller_filename = os.path.join(current_path, '5D_traffic/controller.dat')
        self.winning_domain_filename = os.path.join(current_path, '5D_traffic/won_domain.pkl')
    # Defining the functions
    def dynamics_forw(self, z, t, u):
        """Defining forward dynamics for the target system"""
        return [-200*(z[0]-z[4])+6*u[0]/self.sample_time, -(200+.25/self.sample_time)*z[1]+200*z[0],
                -200*(z[2]-z[1])+8*u[1]/self.sample_time, -(200+.25/self.sample_time)*z[3]+200*z[2],
                -200*(z[4]-z[3])]  # for the 5-d traffic control example


    def dynamics_back(self, z, t, u):
        """Defining backward dynamics for the target system"""
        return [200 * (z[0] - z[4]) - 6 * u[0] / self.sample_time, (200 + .25 / self.sample_time) * z[1] - 200 * z[0],
                200 * (z[2] - z[1]) - 8 * u[1] / self.sample_time, (200 + .25 / self.sample_time) * z[3] - 200 * z[2],
                200 * (z[4] - z[3])]  # for the 5-d traffic control example

    def Jacob_forw(self, inp):
        """Defining Jacobian of forward dynamics"""
        L = np.array([[-200, 0, 0, 0, 200], [200, -(200+.25/self.sample_time), 0, 0, 0], [0, 200, -200, 0, 0],
                      [0, 0, 200, -(200+.25/self.sample_time), 0], [0, 0, 0, 200, -200]])  # for the 5-d traffic control example
        return L


    def Jacob_back(self, inp):
        """Defining Jacobian of backward dynamics"""
        L = np.array([[200, 0, 0, 0, 200], [200, (200 + .25 / self.sample_time), 0, 0, 0], [0, 200, 200, 0, 0],
                      [0, 0, 200, (200 + .25 / self.sample_time), 0], [0, 0, 0, 200, 200]])  # for the 5-d traffic control example
        return L

# Planar Quadrotor (dim = 6+2)
class vehicle_6d:
    def __init__(self):
        self.X_range = np.array([[0, 5], [0, 5], [-3, 3], [-3, 3], [-1.5, 1.5], [-1.1, 1.1]])  # state-space
        self.U_range = np.array([[-5.25, 15.75], [-.5, 1.5]])  # input space
        self.sample_time = .3  # sampling time in seconds
        self.eta_x = np.array([[.2], [.2], [.2], [.2], [.2], [.2]])  # state-space discretization size
        self.eta_u = np.array([[10.5], [1]])  # input-space discretization size
        self.W = np.array([[0], [0], [0], [0], [0], [0]])  # disturbance bounds
        self.nbins = 10  # used for computing the integral
        self.time_span = np.linspace(0, self.sample_time, self.nbins)  # used for computing the integral
        self.is_GB_NN_needed = 0  # set to 1 only if the Jacobian matrix is input-dependent
        current_path = os.getcwd()
        # parallelization parameters
        self.length = 20
        self.num_tasks_per_step = 1000
        # learning related settings
        self.epochs_TS = 20
        self.epochs_GB = 200
        self.batch_size_TS = 64
        self.batch_size_GB = 1
        self.learning_rate = 0.001
        # defining filename for loading the nominal trajectory computed by Altro
        self.nom_traj_filename = os.path.join(current_path, '6D_quad/nom_tr.txt')
        # defining filenames for saving the transition system
        self.forw_inp_TS_filename = os.path.join(current_path, '6D_quad/forw_inp_TS.dat')
        self.forw_out_TS_filename = os.path.join(current_path, '6D_quad/forw_out_TS.dat')
        self.back_inp_TS_filename = os.path.join(current_path, '6D_quad/back_inp_TS.dat')
        self.back_out_TS_filename = os.path.join(current_path, '6D_quad/back_out_TS.dat')
        # defining filenames for saving the input-GB sets
        self.discrete_inp_set_filename = os.path.join(current_path, '6D_quad/discrete_inp_set.dat')
        self.forw_GB_filename = os.path.join(current_path, '6D_quad/forw_GB.dat')
        self.back_GB_filename = os.path.join(current_path, '6D_quad/back_GB.dat')
        # defining paths for saving the trained NNs...
        # defining filenames for saving the approximation error vector
        self.e_vec_TS_forw_filename = os.path.join(current_path, '6D_quad/e_vec_TS_forw.dat')
        self.e_vec_TS_back_filename = os.path.join(current_path, '6D_quad/e_vec_TS_back.dat')
        self.e_vec_GB_forw_filename = os.path.join(current_path, '6D_quad/e_vec_GB_forw.dat')
        self.e_vec_GB_back_filename = os.path.join(current_path, '6D_quad/e_vec_GB_back.dat')
        # defining filenames for saving the safety margins
        self.safety_margin_FW_filename = os.path.join(current_path, '6D_quad/safety_margin_forw.npy')
        self.safety_margin_BW_filename = os.path.join(current_path, '6D_quad/safety_margin_back.npy')
        # defining filenames for saving the mismatch edges per state-input pair
        self.mismatch_list_forw_filename = os.path.join(current_path, '6D_quad/mismatch_list_forw.dat')
        self.num_transitions_forw_filename = os.path.join(current_path, '6D_quad/num_transitions_forw.dat')
        self.R_analytical_filename = os.path.join(current_path, '6D_quad/R_analytical.dat')
        self.R_NN_filename = os.path.join(current_path, '6D_quad/R_NN.dat')
        # defining filenames for saving the controller and winning domain
        self.controller_filename = os.path.join(current_path, '6D_quad/controller.dat')
        self.winning_domain_filename = os.path.join(current_path, '6D_quad/won_domain.pkl')
    # Defining the functions
    def dynamics_forw(self, z, t, u):
        """Defining forward dynamics for the target system"""
        return [z[2], z[3], -u[0]*math.sin(z[4]), u[0]*math.cos(z[4])-9.81, z[5], u[1]]  # for the 6-d quadrotor example


    def dynamics_back(self, z, t, u):
        """Defining backward dynamics for the target system"""
        return [-z[2], -z[3], u[0]*math.sin(z[4]), -u[0]*math.cos(z[4])+9.81, -z[5], -u[1]]  # for the 6-d quadrotor example

    def Jacob_forw(self, inp):
        """Defining Jacobian of forward dynamics"""
        L = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, abs(inp[0]), 0], [0, 0, 0, 0, abs(inp[0]), 0],
                      [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]])  # for the 6-d quadrotor example
        return L


    def Jacob_back(self, inp):
        """Defining Jacobian of backward dynamics"""
        L = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, abs(inp[0]), 0], [0, 0, 0, 0, abs(inp[0]), 0],
                      [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]])  # for the 6-d quadrotor example
        return L