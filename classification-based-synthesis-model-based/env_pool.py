"""This code lists properties for each of the case-studies."""
import os
import numpy as np
import math
# 2D-car (dim=2+2)
class vehicle_2d:
    def __init__(self):
        self.num_samples = 5  # number of samples to be taken for each state-input pair
        self.gamma = 0 * np.ones(2)  # The added bias value
        self.X_range = np.array([[0, 5], [0, 5]])  # state-space
        self.X_range_FW = np.array([[-1, 6], [-1, 6]])  # state-space for the FW computations
        self.U_range = np.array([[-1.1, 1.1], [-1.1, 1.1]])  # input space
        self.Goal_range = np.array([[3.2, 4.8], [3.2, 4.8]])  # target range
        self.Obs_range = np.array([[.2, .6], [.2, .6]])  # obstacle range
        self.smoothing_factor = 1
        self.sample_time = .4  # sampling time in seconds
        self.Lip_const = np.array([self.sample_time, self.sample_time])  # component-wise Lipschitz constant (not exatly Lipschitz constant: measuring the maximum changes in every component per sample time)
        self.eta_x = np.array([[.05], [.05]])  # state-space discretization size
        self.eta_u = np.array([[.23], [.23]])  # input-space discretization size
        self.alpha = 0.02  # factor of eta_x that will be used for shifiting the reachable set box
        self.shift_no = 10  #  shifting the index vectors to avoid negative values
        self.W = np.array([[0], [0]])  # disturbance bounds
        self.nbins = 10  # used for computing the integral
        self.time_span = np.linspace(0, self.sample_time, self.nbins)  # used for computing the integral
        current_path = os.getcwd()
        # parallelization parameters
        self.length = 200
        self.num_tasks_per_step = 32 * self.length  # num_tasks_per_step = num_cpu *  length
        # learning related settings
        self.epochs_TS = 30
        self.epochs_GB = 50
        self.batch_size_TS = 64
        self.batch_size_GB = 1
        self.learning_rate = 0.0005
        # defining filenames for saving the transition system
        self.forw_inp_TS_filename = os.path.join(current_path, '2D_car/forw_inp_TS.dat')
        self.forw_out_TS_filename = os.path.join(current_path, '2D_car/forw_out_TS.dat')  # for the classifier-based formulation
        self.forw_out_TS_c_filename = os.path.join(current_path, '2D_car/forw_out_TS_c.dat')
        self.forw_out_TS_r_filename = os.path.join(current_path, '2D_car/forw_out_TS_r.dat')
        self.forw_out_TS_r1_filename = os.path.join(current_path, '2D_car/forw_out_TS_r1.dat')
        self.forw_out_TS_r2_filename = os.path.join(current_path, '2D_car/forw_out_TS_r2.dat')
        self.forw_sub_inp_TS_filename = os.path.join(current_path, '2D_car/forw_sub_inp_TS.dat')
        self.forw_sub_out_TS_c_filename = os.path.join(current_path, '2D_car/forw_sub_out_TS_c.dat')
        self.forw_sub_out_TS_r_filename = os.path.join(current_path, '2D_car/forw_sub_out_TS_r.dat')
        self.back_inp_TS_filename = os.path.join(current_path, '2D_car/back_inp_TS.dat')
        self.back_out_TS_filename = os.path.join(current_path, '2D_car/back_out_TS.dat')  # for the classifier-based formulation
        self.back_out_TS_c_filename = os.path.join(current_path, '2D_car/back_out_TS_c.dat')
        self.back_out_TS_r_filename = os.path.join(current_path, '2D_car/back_out_TS_r.dat')
        self.back_out_TS_r1_filename = os.path.join(current_path, '2D_car/back_out_TS_r1.dat')
        self.back_out_TS_r2_filename = os.path.join(current_path, '2D_car/back_out_TS_r2.dat')
        # defining paths for saving the trained NNs
        self.checkpoint_path_TS_forw = os.path.join(current_path, '2D_car/NN_forw/TS/')
        self.checkpoint_path_TS_back = os.path.join(current_path, '2D_car/NN_back/TS/')
        self.checkpoint_path_GB1_forw = os.path.join(current_path, '2D_car/NN_forw/GB1/')
        self.checkpoint_path_GB2_forw = os.path.join(current_path, '2D_car/NN_forw/GB2/')
        self.checkpoint_path_GB1_back = os.path.join(current_path, '2D_car/NN_back/GB1/')
        self.checkpoint_path_GB2_back = os.path.join(current_path, '2D_car/NN_back/GB2/')
        # defining filenames for saving the approximation error vector
        self.e_vec_TS_forw_filename = os.path.join(current_path, '2D_car/e_vec_TS_forw.dat')
        self.e_vec_TS_forw1_filename = os.path.join(current_path, '2D_car/e_vec_TS_forw1.dat')
        self.e_vec_TS_forw2_filename = os.path.join(current_path, '2D_car/e_vec_TS_forw2.dat')
        self.e_vec_TS_back_filename = os.path.join(current_path, '2D_car/e_vec_TS_back.dat')
        self.e_vec_TS_back1_filename = os.path.join(current_path, '2D_car/e_vec_TS_back1.dat')
        self.e_vec_TS_back2_filename = os.path.join(current_path, '2D_car/e_vec_TS_back2.dat')
        # defining filenames for saving the safety margins
        self.safety_margin_FW1_filename = os.path.join(current_path, '2D_car/safety_margin_forw1.npy')
        self.safety_margin_FW2_filename = os.path.join(current_path, '2D_car/safety_margin_forw2.npy')
        self.safety_margin_BW1_filename = os.path.join(current_path, '2D_car/safety_margin_back1.npy')
        self.safety_margin_BW2_filename = os.path.join(current_path, '2D_car/safety_margin_back2.npy')
        # defining filenames for saving the mismatch edges per state-input pair
        self.mismatch_list_forw_filename = os.path.join(current_path, '2D_car/mismatch_list_forw.dat')
        self.mismatch_list_back_filename = os.path.join(current_path, '2D_car/mismatch_list_back.dat')
        self.num_transitions_forw_filename = os.path.join(current_path, '2D_car/num_transitions_forw.dat')
        self.R_analytical_filename = os.path.join(current_path, '2D_car/R_analytical.dat')
        self.R_NN_filename = os.path.join(current_path, '2D_car/R_NN.dat')
        # defining filenames for saving the controller and winning domain
        self.controller_filename = os.path.join(current_path, '2D_car/controller.dat')
        self.winning_domain_filename = os.path.join(current_path, '2D_car/won_domain.pkl')
    # Defining the functions
    def dynamics_forw(self, z, t, u):
        """Defining forward dynamics for the target system"""
        return [u[0], u[1]]  # for the 2-d vehicle example


    def dynamics_back(self, z, t, u):
        """Defining backward dynamics for the target system"""
        return [-u[0], -u[1]]  # for the 2-d vehicle example

    def Jacob_forw(self, inp):
        """Defining Jacobian of forward dynamics"""
        L = np.array([[0, 0], [0, 0]])  # for the 2-d vehicle example
        return L


    def Jacob_back(self, inp):
        """Defining Jacobian of backward dynamics"""
        L = np.array([[0, 0], [0, 0]])  # for the 2-d vehicle example
        return L

# inverted pendulum (dim = 2)
class inv_pend:
    def __init__(self):
        self.X_range = np.array([[-math.pi, math.pi], [-2, 2]])
        self.U_range = np.array([[-1.1, 1.1]])  # input space
        self.Goal_range = np.array([[-.5, .5], [-1, 1]])  # target range
        self.Obs_range = np.array([[.2, .6], [.2, .6]])  # obstacle range
        self.sample_time = .05  # sampling time in seconds
        self.eta_x = np.array([[.1], [.1]])
        self.eta_u = np.array([[.3]])
        self.alpha = 0.02  # factor of eta_x that will be used for shifiting the reachable set box
        self.shift_no = 10  # shifting the index vectors to avoid negative values
        self.W = np.array([[0], [0]])
        current_path = os.getcwd()
        self.is_GB_NN_needed = 0  # set to 1 only if the Jacobian matrix is input-dependent
        self.nbins = 3  # used for computing the integral
        self.time_span = np.linspace(0, self.sample_time, self.nbins)  # used for computing the integral
        # parallelization parameters
        self.length = 20
        self.num_tasks_per_step = 1000
        # learning related settings
        self.epochs_TS = 100
        self.epochs_GB = 200
        self.batch_size_TS = 32
        self.batch_size_GB = 1
        self.learning_rate = 0.001
        # defining filenames for saving the transition system
        self.forw_inp_TS_filename = os.path.join(current_path, 'inv_pend_2D/forw_inp_TS.dat')
        self.forw_out_TS_filename = os.path.join(current_path, 'inv_pend_2D/forw_out_TS.dat')
        self.forw_out_TS_c_filename = os.path.join(current_path, 'inv_pend_2D/forw_out_TS_c.dat')
        self.forw_out_TS_r_filename = os.path.join(current_path, 'inv_pend_2D/forw_out_TS_r.dat')
        self.forw_out_TS_r1_filename = os.path.join(current_path, 'inv_pend_2D/forw_out_TS_r1.dat')
        self.forw_out_TS_r2_filename = os.path.join(current_path, 'inv_pend_2D/forw_out_TS_r2.dat')
        self.forw_sub_inp_TS_filename = os.path.join(current_path, 'inv_pend_2D/forw_sub_inp_TS.dat')
        self.forw_sub_out_TS_c_filename = os.path.join(current_path, 'inv_pend_2D/forw_sub_out_TS_c.dat')
        self.forw_sub_out_TS_r_filename = os.path.join(current_path, 'inv_pend_2D/forw_sub_out_TS_r.dat')
        self.back_inp_TS_filename = os.path.join(current_path, 'inv_pend_2D/back_inp_TS.dat')
        self.back_out_TS_filename = os.path.join(current_path, 'inv_pend_2D/back_out_TS.dat')
        self.back_out_TS_c_filename = os.path.join(current_path, 'inv_pend_2D/back_out_TS_c.dat')
        self.back_out_TS_r_filename = os.path.join(current_path, 'inv_pend_2D/back_out_TS_r.dat')
        self.back_out_TS_r1_filename = os.path.join(current_path, 'inv_pend_2D/back_out_TS_r1.dat')
        self.back_out_TS_r2_filename = os.path.join(current_path, 'inv_pend_2D/back_out_TS_r2.dat')
        # defining paths for saving the trained NNs
        self.checkpoint_path_TS_forw = os.path.join(current_path, 'inv_pend_2D/NN_forw/TS/')
        self.checkpoint_path_TS_back = os.path.join(current_path, 'inv_pend_2D/NN_back/TS/')
        # defining filenames for saving the approximation error vector
        self.e_vec_TS_forw_filename = os.path.join(current_path, 'inv_pend_2D/e_vec_TS_forw.dat')
        self.e_vec_TS_forw1_filename = os.path.join(current_path, 'inv_pend_2D/e_vec_TS_forw1.dat')
        self.e_vec_TS_forw2_filename = os.path.join(current_path, 'inv_pend_2D/e_vec_TS_forw2.dat')
        self.e_vec_TS_back_filename = os.path.join(current_path, 'inv_pend_2D/e_vec_TS_back.dat')
        self.e_vec_TS_back1_filename = os.path.join(current_path, 'inv_pend_2D/e_vec_TS_back1.dat')
        self.e_vec_TS_back2_filename = os.path.join(current_path, 'inv_pend_2D/e_vec_TS_back2.dat')
        # defining filenames for saving the safety margins
        self.safety_margin_FW1_filename = os.path.join(current_path, 'inv_pend_2D/safety_margin_forw1.npy')
        self.safety_margin_FW2_filename = os.path.join(current_path, 'inv_pend_2D/safety_margin_forw2.npy')
        self.safety_margin_BW1_filename = os.path.join(current_path, 'inv_pend_2D/safety_margin_back1.npy')
        self.safety_margin_BW2_filename = os.path.join(current_path, 'inv_pend_2D/safety_margin_back2.npy')
        # defining filenames for saving the mismatch edges per state-input pair
        self.mismatch_list_forw_filename = os.path.join(current_path, 'inv_pend_2D/mismatch_list_forw.dat')
        self.mismatch_list_back_filename = os.path.join(current_path, 'inv_pend_2D/mismatch_list_back.dat')
        self.num_transitions_forw_filename = os.path.join(current_path, 'inv_pend_2D/num_transitions_forw.dat')
        self.R_analytical_filename = os.path.join(current_path, 'inv_pend_2D/R_analytical.dat')
        self.R_NN_filename = os.path.join(current_path, 'inv_pend_2D/R_NN.dat')
        # defining filenames for saving the controller and winning domain
        self.controller_filename = os.path.join(current_path, 'inv_pend_2D/controller.dat')
        self.winning_domain_filename = os.path.join(current_path, 'inv_pend_2D/won_domain.pkl')

    # Defining the functions
    def dynamics_forw(self, z, t, u):
        """Defining forward dynamics for the target system"""
        return [z[1], math.sin(z[0])]  # for the 2-d inverted pendulum example

    def dynamics_back(self, z, t, u):
        """Defining backward dynamics for the target system"""
        return [-z[1], -math.sin(z[0])]  # for the 2-d inverted pendulum example

    def Jacob_forw(self, inp):
        """Defining Jacobian of forward dynamics"""
        L = np.array([[0, 1], [1, 0]])  # for the 2-d inverted pendulum example
        return L


    def Jacob_back(self, inp):
        """Defining Jacobian of backward dynamics"""
        L = np.array([[0, 1], [1, 0]])  # for the 2-d inverted pendulum example
        return L


class tora:
    def __init__(self):
        self.X_range = np.array([[-2, 2], [-2, 2], [-2, 2], [-2, 2]])
        self.U_range = np.array([[-1.1, 1.1]])  # input space
        self.Goal_range = np.array([[-1, 1], [-2, 2], [-2, 2], [-2, 2]])  # target range
        self.Obs_range = np.array([[.2, .6], [.2, .6]])  # obstacle range
        self.sample_time = .5  # sampling time in seconds
        self.eta_x = np.array([[.2], [.2], [.2], [.2]])
        self.eta_u = np.array([[.2]])
        self.W = np.array([[0], [0], [0], [0]])
        self.Lip_const = np.array([2*self.sample_time, 2*self.sample_time, 2.1*self.sample_time, 1*self.sample_time])  # component-wise Lipschitz constant (not exatly Lipschitz constant: measuring the maximum changes in every component per sample time)
        self.nbins = 3  # used for computing the integral
        self.time_span = np.linspace(0, self.sample_time, self.nbins)  # used for computing the integral
        current_path = os.getcwd()
        self.alpha = 0.02  # factor of eta_x that will be used for shifting the reachable set box
        self.shift_no = 12  # shifting the index vectors to avoid negative values
        self.W = np.array([[0], [0], [0], [0]])
        current_path = os.getcwd()
        self.is_GB_NN_needed = 0  # set to 1 only if the Jacobian matrix is input-dependent
        # parallelization parameters
        self.length = 60
        self.num_tasks_per_step = 32 * self.length  # num_tasks_per_step = num_cpu *  length
        # learning related settings
        self.epochs_TS = 50
        self.epochs_GB = 200
        self.batch_size_TS = 64
        self.batch_size_GB = 1
        self.learning_rate = 0.0005
        # defining filenames for saving the transition system
        self.forw_inp_TS_filename = os.path.join(current_path, 'tora/forw_inp_TS.dat')
        self.forw_out_TS_filename = os.path.join(current_path, 'tora/forw_out_TS.dat')
        self.forw_out_TS_c_filename = os.path.join(current_path, 'tora/forw_out_TS_c.dat')
        self.forw_out_TS_r_filename = os.path.join(current_path, 'tora/forw_out_TS_r.dat')
        self.forw_out_TS_r1_filename = os.path.join(current_path, 'tora/forw_out_TS_r1.dat')
        self.forw_out_TS_r2_filename = os.path.join(current_path, 'tora/forw_out_TS_r2.dat')
        self.forw_sub_inp_TS_filename = os.path.join(current_path, 'tora/forw_sub_inp_TS.dat')
        self.forw_sub_out_TS_c_filename = os.path.join(current_path, 'tora/forw_sub_out_TS_c.dat')
        self.forw_sub_out_TS_r_filename = os.path.join(current_path, 'tora/forw_sub_out_TS_r.dat')
        self.back_inp_TS_filename = os.path.join(current_path, 'tora/back_inp_TS.dat')
        self.back_out_TS_filename = os.path.join(current_path, 'tora/back_out_TS.dat')
        self.back_out_TS_c_filename = os.path.join(current_path, 'tora/back_out_TS_c.dat')
        self.back_out_TS_r_filename = os.path.join(current_path, 'tora/back_out_TS_r.dat')
        self.back_out_TS_r1_filename = os.path.join(current_path, 'tora/back_out_TS_r1.dat')
        self.back_out_TS_r2_filename = os.path.join(current_path, 'tora/back_out_TS_r2.dat')
        # defining paths for saving the trained NNs
        self.checkpoint_path_TS_forw = os.path.join(current_path, 'tora/NN_forw/TS/')
        self.checkpoint_path_TS_back = os.path.join(current_path, 'tora/NN_back/TS/')
        self.checkpoint_path_GB_forw = os.path.join(current_path, 'tora/NN_forw/GB/')
        self.checkpoint_path_GB_back = os.path.join(current_path, 'tora/NN_back/GB/')
        self.checkpoint_path_GB1_forw = os.path.join(current_path, 'tora/NN_forw/GB1/')
        self.checkpoint_path_GB2_forw = os.path.join(current_path, 'tora/NN_forw/GB2/')
        self.checkpoint_path_GB1_back = os.path.join(current_path, 'tora/NN_back/GB1/')
        self.checkpoint_path_GB2_back = os.path.join(current_path, 'tora/NN_back/GB2/')
        # defining filenames for saving the approximation error vector
        self.e_vec_TS_forw_filename = os.path.join(current_path, 'tora/e_vec_TS_forw.dat')
        self.e_vec_TS_back_filename = os.path.join(current_path, 'tora/e_vec_TS_back.dat')
        self.e_vec_GB_forw_filename = os.path.join(current_path, 'tora/e_vec_GB_forw.dat')
        self.e_vec_GB_back_filename = os.path.join(current_path, 'tora/e_vec_GB_back.dat')
        self.e_vec_TS_forw1_filename = os.path.join(current_path, 'tora/e_vec_TS_forw1.dat')
        self.e_vec_TS_forw2_filename = os.path.join(current_path, 'tora/e_vec_TS_forw2.dat')
        self.e_vec_TS_back1_filename = os.path.join(current_path, 'tora/e_vec_TS_back1.dat')
        self.e_vec_TS_back2_filename = os.path.join(current_path, 'tora/e_vec_TS_back2.dat')
        # defining filenames for saving the safety margins
        self.safety_margin_FW_filename = os.path.join(current_path, 'tora/safety_margin_forw.npy')
        self.safety_margin_BW_filename = os.path.join(current_path, 'tora/safety_margin_back.npy')
        self.safety_margin_FW1_filename = os.path.join(current_path, 'tora/safety_margin_forw1.npy')
        self.safety_margin_FW2_filename = os.path.join(current_path, 'tora/safety_margin_forw2.npy')
        self.safety_margin_BW1_filename = os.path.join(current_path, 'tora/safety_margin_back1.npy')
        self.safety_margin_BW2_filename = os.path.join(current_path, 'tora/safety_margin_back2.npy')
        # defining filenames for saving the mismatch edges per state-input pair
        self.mismatch_list_forw_filename = os.path.join(current_path, 'tora/mismatch_list_forw.dat')
        self.mismatch_list_back_filename = os.path.join(current_path, 'tora/mismatch_list_back.dat')
        self.num_transitions_forw_filename = os.path.join(current_path, 'tora/num_transitions_forw.dat')
        self.R_analytical_filename = os.path.join(current_path, 'tora/R_analytical.dat')
        self.R_NN_filename = os.path.join(current_path, 'tora/R_NN.dat')
        # defining filenames for saving the controller and winning domain
        self.controller_filename = os.path.join(current_path, 'tora/controller.dat')
        self.winning_domain_filename = os.path.join(current_path, 'tora/won_domain.pkl')
    # Defining the functions
    def dynamics_forw(self, z, t, u):
        """Defining forward dynamics for the target system"""
        return [z[1], -z[0]+.1*math.sin(z[2]), z[3], u[0]]  # for the 2-d inverted pendulum example

    def dynamics_back(self, z, t, u):
        """Defining backward dynamics for the target system"""
        return [-z[1], z[0]-.1*math.sin(z[2]), -z[3], -u[0]]  # for the 2-d inverted pendulum example

    def Jacob_forw(self, inp):
        """Defining Jacobian of forward dynamics"""
        L = np.array([[0, 1, 0, 0], [1, 0, .1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])  # for the 2-d inverted pendulum example
        return L


    def Jacob_back(self, inp):
        """Defining Jacobian of backward dynamics"""
        L = np.array([[0, 1, 0, 0], [1, 0, .1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])  # for the 2-d inverted pendulum example
        return L


# unicycle(dim = 3+2)
class unicycle:
    def __init__(self):
        self.num_samples = 50   # number of samples to be taken for each state-input pair
        self.gamma = 0*np.ones(3)  # The added bias value
        self.X_range = np.array([[0, 5], [0, 5], [-1.6, 1.6]])  # state-space
        self.X_range_FW = np.array([[-1, 6], [-1, 6], [-2.2, 2.2]])  # state-space for the FW computations
        self.U_range = np.array([[-1.1, 1.1], [-1.1, 1.1]])  # input space
        self.Goal_range = np.array([[3.2, 4.8], [3.2, 4.8], [-math.pi / 4 - .5, math.pi / 4 + .5]])  # target range
        self.Obs_range = np.array([[.2, .6], [.2, .6], [-3, 3]])  # obstacle range
        self.smoothing_factor = 1
        self.sample_time = .3  # sampling time in seconds
        self.Lip_const = np.array([self.sample_time, self.sample_time, self.sample_time])  # component-wise Lipschitz constant (not exatly Lipschitz constant: measuring the maximum changes in every component per sample time)
        self.eta_x = np.array([[.2], [.2], [.2]])  # state-space discretization size
        self.eta_u = np.array([[.3], [.3]])  # input-space discretization size
        self.alpha = 0.1  # factor of eta_x that will be used for shifiting the reachable set box
        self.shift_no = 3  # shifting the index vectors to avoid negative values
        self.W = np.array([[0], [0], [0]])  # disturbance bounds
        self.nbins = 10  # used for computing the integral
        self.time_span = np.linspace(0, self.sample_time, self.nbins)  # used for computing the integral
        current_path = os.getcwd()
        # parallelization parameters
        self.length = 30
        self.num_tasks_per_step = 32*self.length  # num_tasks_per_step = num_cpu *  length
        # learning related settings
        self.epochs_TS = 20
        self.epochs_GB = 200
        self.batch_size_TS = 64
        self.batch_size_GB = 1
        self.learning_rate = 0.001
        # defining filenames for saving the transition system
        self.forw_inp_TS_filename = os.path.join(current_path, 'unicycle/forw_inp_TS.dat')
        self.forw_out_TS_filename = os.path.join(current_path, 'unicycle/forw_out_TS.dat')
        self.forw_out_TS_c_filename = os.path.join(current_path, 'unicycle/forw_out_TS_c.dat')
        self.forw_out_TS_r_filename = os.path.join(current_path, 'unicycle/forw_out_TS_r.dat')
        self.forw_out_TS_r1_filename = os.path.join(current_path, 'unicycle/forw_out_TS_r1.dat')
        self.forw_out_TS_r2_filename = os.path.join(current_path, 'unicycle/forw_out_TS_r2.dat')
        self.forw_sub_inp_TS_filename = os.path.join(current_path, 'unicycle/forw_sub_inp_TS.dat')
        self.forw_sub_out_TS_c_filename = os.path.join(current_path, 'unicycle/forw_sub_out_TS_c.dat')
        self.forw_sub_out_TS_r_filename = os.path.join(current_path, 'unicycle/forw_sub_out_TS_r.dat')
        self.back_inp_TS_filename = os.path.join(current_path, 'unicycle/back_inp_TS.dat')
        self.back_out_TS_filename = os.path.join(current_path, 'unicycle/back_out_TS.dat')
        self.back_out_TS_c_filename = os.path.join(current_path, 'unicycle/back_out_TS_c.dat')
        self.back_out_TS_r_filename = os.path.join(current_path, 'unicycle/back_out_TS_r.dat')
        self.back_out_TS_r1_filename = os.path.join(current_path, 'unicycle/back_out_TS_r1.dat')
        self.back_out_TS_r2_filename = os.path.join(current_path, 'unicycle/back_out_TS_r2.dat')
        # defining paths for saving the trained NNs
        self.checkpoint_path_TS_forw = os.path.join(current_path, 'unicycle/NN_forw/TS/')
        self.checkpoint_path_TS_back = os.path.join(current_path, 'unicycle/NN_back/TS/')
        # defining filenames for saving the approximation error vector
        self.e_vec_TS_forw_filename = os.path.join(current_path, 'unicycle/e_vec_TS_forw.dat')
        self.e_vec_TS_forw1_filename = os.path.join(current_path, 'unicycle/e_vec_TS_forw1.dat')
        self.e_vec_TS_forw2_filename = os.path.join(current_path, 'unicycle/e_vec_TS_forw2.dat')
        self.e_vec_TS_back_filename = os.path.join(current_path, 'unicycle/e_vec_TS_back.dat')
        self.e_vec_TS_back1_filename = os.path.join(current_path, 'unicycle/e_vec_TS_back1.dat')
        self.e_vec_TS_back2_filename = os.path.join(current_path, 'unicycle/e_vec_TS_back2.dat')
        # defining filenames for saving the safety margins
        self.safety_margin_FW1_filename = os.path.join(current_path, 'unicycle/safety_margin_forw1.npy')
        self.safety_margin_FW2_filename = os.path.join(current_path, 'unicycle/safety_margin_forw2.npy')
        self.safety_margin_BW1_filename = os.path.join(current_path, 'unicycle/safety_margin_back1.npy')
        self.safety_margin_BW2_filename = os.path.join(current_path, 'unicycle/safety_margin_back2.npy')
        # defining filenames for saving the mismatch edges per state-input pair
        self.mismatch_list_forw_filename = os.path.join(current_path, 'unicycle/mismatch_list_forw.dat')
        self.mismatch_list_back_filename = os.path.join(current_path, 'unicycle/mismatch_list_back.dat')
        self.num_transitions_forw_filename = os.path.join(current_path, 'unicycle/num_transitions_forw.dat')
        self.R_analytical_filename = os.path.join(current_path, 'unicycle/R_analytical.dat')
        self.R_NN_filename = os.path.join(current_path, 'unicycle/R_NN.dat')
        # defining filenames for saving the controller and winning domain
        self.controller_filename = os.path.join(current_path, 'unicycle/controller.dat')
        self.winning_domain_filename = os.path.join(current_path, 'unicycle/won_domain.pkl')

    # Defining the functions
    def dynamics_forw(self, z, t, u):
        """Defining forward dynamics for the target system"""
        return [u[0] * math.cos(z[2]), u[0] * math.sin(z[2]), u[1]]  # for the 3-d unicycle example


    def dynamics_back(self, z, t, u):
        """Defining backward dynamics for the target system"""
        return [-u[0] * math.cos(z[2]), -u[0] * math.sin(z[2]), -u[1]]  # for the 3-d unicycle example

    def Jacob_forw(self, inp):
        """Defining Jacobian of forward dynamics"""
        L = np.array([[0, 0, abs(inp[0])], [0, 0, abs(inp[0])], [0, 0, 0]])  # for the unicycle example
        return L


    def Jacob_back(self, inp):
        """Defining Jacobian of backward dynamics"""
        L = np.array([[0, 0, abs(-inp[0])], [0, 0, abs(-inp[0])], [0, 0, 0]])  # for the unicycle example
        return L

# 4-D vehicle (dim = 4+2)
class vehicle_4d:
    def __init__(self):
        self.num_samples = 10  # number of samples to be taken for each state-input pair
        self.gamma = 0 * np.ones(4)  # The added bias value
        self.X_range = np.array([[0, 5], [0, 5], [-1.6, 1.6], [-1.1, 1.1]])  # state-space
        self.X_range_FW = np.array([[-1, 6], [-1, 6], [-2.2, 2.2], [-1.5, 1.5]])  # state-space for the FW computations
        self.U_range = np.array([[-1.1, 1.1], [-1.1, 1.1]])  # input space
        self.Goal_range = np.array([[3.2, 4.8], [3.2, 4.8], [-1.1, 1.1], [-1.1, 1.1]])  # target range
        self.Obs_range = np.array([[.2, .6], [.2, .6], [-3, 3]])  # obstacle range
        self.smoothing_factor = 1
        self.sample_time = .3  # sampling time in seconds
        self.Lip_const = np.array([self.sample_time, self.sample_time,
                                   self.sample_time, self.sample_time])  # component-wise Lipschitz constant (not exatly Lipschitz constant: measuring the maximum changes in every component per sample time)
        self.eta_x = np.array([[.2], [.2], [.2], [.2]])  # state-space discretization size
        self.eta_u = np.array([[.3], [.3]])  # input-space discretization size
        self.shift_no = 3  # shifting the index vectors to avoid negative values
        self.W = np.array([[0], [0], [0], [0]])  # disturbance bounds
        self.nbins = 10  # used for computing the integral
        self.time_span = np.linspace(0, self.sample_time, self.nbins)  # used for computing the integral
        current_path = os.getcwd()
        # parallelization parameters
        self.length = 20
        self.num_tasks_per_step = 32 * self.length  # num_tasks_per_step = num_cpu *  length
        # learning related settings
        self.epochs_TS = 30
        self.epochs_GB = 200
        self.batch_size_TS = 64
        self.batch_size_GB = 1
        self.learning_rate = 0.001
        # defining filenames for saving the transition system
        self.forw_inp_TS_filename = os.path.join(current_path, '4D_car/forw_inp_TS.dat')
        self.forw_out_TS_filename = os.path.join(current_path, '4D_car/forw_out_TS.dat')
        self.forw_out_TS_c_filename = os.path.join(current_path, '4D_car/forw_out_TS_c.dat')
        self.forw_out_TS_r_filename = os.path.join(current_path, '4D_car/forw_out_TS_r.dat')
        self.forw_sub_inp_TS_filename = os.path.join(current_path, '4D_car/forw_sub_inp_TS.dat')
        self.forw_sub_out_TS_c_filename = os.path.join(current_path, '4D_car/forw_sub_out_TS_c.dat')
        self.forw_sub_out_TS_r_filename = os.path.join(current_path, '4D_car/forw_sub_out_TS_r.dat')
        self.back_inp_TS_filename = os.path.join(current_path, '4D_car/back_inp_TS.dat')
        self.back_out_TS_filename = os.path.join(current_path, '4D_car/back_out_TS.dat')
        self.back_out_TS_c_filename = os.path.join(current_path, '4D_car/back_out_TS_c.dat')
        self.back_out_TS_r_filename = os.path.join(current_path, '4D_car/back_out_TS_r.dat')
        # defining paths for saving the trained NNs
        self.checkpoint_path_TS_forw = os.path.join(current_path, '4D_car/NN_forw/TS/')
        self.checkpoint_path_TS_back = os.path.join(current_path, '4D_car/NN_back/TS/')
        # defining filenames for saving the approximation error vector
        self.e_vec_TS_forw_filename = os.path.join(current_path, '4D_car/e_vec_TS_forw.dat')
        self.e_vec_TS_back_filename = os.path.join(current_path, '4D_car/e_vec_TS_back.dat')
        self.e_vec_GB_forw_filename = os.path.join(current_path, '4D_car/e_vec_GB_forw.dat')
        self.e_vec_GB_back_filename = os.path.join(current_path, '4D_car/e_vec_GB_back.dat')
        # defining filenames for saving the safety margins
        self.safety_margin_FW_filename = os.path.join(current_path, '4D_car/safety_margin_forw.npy')
        self.safety_margin_BW_filename = os.path.join(current_path, '4D_car/safety_margin_back.npy')
        # defining filenames for saving the mismatch edges per state-input pair
        self.mismatch_list_forw_filename = os.path.join(current_path, '4D_car/mismatch_list_forw.dat')
        self.mismatch_list_back_filename = os.path.join(current_path, '4D_car/mismatch_list_back.dat')
        self.num_transitions_forw_filename = os.path.join(current_path, '4D_car/num_transitions_forw.dat')
        self.R_analytical_filename = os.path.join(current_path, '4D_car/R_analytical.dat')
        self.R_NN_filename = os.path.join(current_path, '4D_car/R_NN.dat')
        # defining filenames for saving the controller and winning domain
        self.controller_filename = os.path.join(current_path, '4D_car/controller.dat')
        self.winning_domain_filename = os.path.join(current_path, '4D_car/won_domain.pkl')

    # Defining the functions
    def dynamics_forw(self, z, t, u):
        """Defining forward dynamics for the target system"""
        return [z[3] * math.cos(z[2]), z[3] * math.sin(z[2]), u[0], u[1]]  # for the 4-d car example


    def dynamics_back(self, z, t, u):
        """Defining backward dynamics for the target system"""
        return [-z[3] * math.cos(z[2]), -z[3] * math.sin(z[2]), -u[0], -u[1]]  # for the 4-d car example

    def Jacob_forw(self, inp):
        """Defining Jacobian of forward dynamics"""
        L = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])  # for the 4-d car example
        #  L = np.array([[0, 0, math.sin(math.pi/4), 1], [0, 0, 1, math.sin(math.pi/4)], [0, 0, 0, 0], [0, 0, 0, 0]])  # for the 4-d car example with limited angle between -+pi/4
        return L


    def Jacob_back(self, inp):
        """Defining Jacobian of backward dynamics"""
        L = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])  # for the 4-d car example
        #  L = np.array([[0, 0, math.sin(math.pi/4), 1], [0, 0, 1, math.sin(math.pi/4)], [0, 0, 0, 0], [0, 0, 0, 0]])  # for the 4-d car example with limited angle between -+pi/4

        return L

# 5D car (dim = 5+2)
class vehicle_5d:
    def __init__(self):
        self.num_samples = 41  # number of samples to be taken for each state-input pair
        self.gamma = 0 * np.ones(3)  # The added bias value
        self.X_range = np.array([[0, 5], [0, 5], [-.9, .9], [-1.1, 1.1], [-1.1, 1.1]])  # state-space
        self.X_range_FW = np.array([[0, 5], [0, 5], [-.9, .9], [-1.1, 1.1], [-1.1, 1.1]])  # state-space
        self.U_range = np.array([[-1.1, 1.1], [-1.1, 1.1]])  # input space
        self.sample_time = 0.5  # sampling time in seconds
        self.eta_x = np.array([[.2], [.2], [.2], [.2], [.2]])  # state-space discretization size
        self.eta_u = np.array([[.3], [.3]])  # input-space discretization size
        self.W = np.array([[0], [0], [0], [0], [0]])  # disturbance bounds
        self.nbins = 100  # used for computing the integral
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
        # defining filenames for saving the transition system
        self.forw_inp_TS_filename = os.path.join(current_path, '5D_car/forw_inp_TS.dat')
        self.forw_out_TS_c_filename = os.path.join(current_path, '5D_car/forw_out_TS_c.dat')
        self.forw_out_TS_r_filename = os.path.join(current_path, '5D_car/forw_out_TS_r.dat')
        self.back_inp_TS_filename = os.path.join(current_path, '5D_car/back_inp_TS.dat')
        self.back_out_TS_c_filename = os.path.join(current_path, '5D_car/back_out_TS_c.dat')
        self.back_out_TS_r_filename = os.path.join(current_path, '5D_car/back_out_TS_r.dat')
        # defining filenames for saving the input-GB sets
        """self.discrete_inp_set_filename = os.path.join(current_path, '5D_car/discrete_inp_set.dat')
        self.forw_GB_filename = os.path.join(current_path, '5D_car/forw_GB.dat')
        self.back_GB_filename = os.path.join(current_path, '5D_car/back_GB.dat')"""
        # defining paths for saving the trained NNs...
        # defining filenames for saving the approximation error vector
        self.e_vec_TS_forw_filename = os.path.join(current_path, '5D_car/e_vec_TS_forw.dat')
        self.e_vec_TS_back_filename = os.path.join(current_path, '5D_car/e_vec_TS_back.dat')
        self.e_vec_GB_forw_filename = os.path.join(current_path, '5D_car/e_vec_GB_forw.dat')
        self.e_vec_GB_back_filename = os.path.join(current_path, '5D_car/e_vec_GB_back.dat')
        # defining filenames for saving the safety margins
        self.safety_margin_FW_filename = os.path.join(current_path, '5D_car/safety_margin_forw.npy')
        self.safety_margin_BW_filename = os.path.join(current_path, '5D_car/safety_margin_back.npy')
        # defining filenames for saving the controller and winning domain
        self.controller_filename = os.path.join(current_path, '5D_car/controller.dat')
        self.winning_domain_filename = os.path.join(current_path, '5D_car/won_domain.pkl')

    # Defining the functions
    def dynamics_forw(self, z, t, u):
        """Defining forward dynamics for the target system"""
        return [z[3]*math.cos(z[2]), z[3]*math.sin(z[2]), z[4], u[0], u[1]]  # for the 5-d car example


    def dynamics_back(self, z, t, u):
        """Defining backward dynamics for the target system"""
        return [-z[3]*math.cos(z[2]), -z[3]*math.sin(z[2]), -z[4], -u[0], -u[1]]  # for the 5-d car example

    def Jacob_forw(self, inp):
        """Defining Jacobian of forward dynamics"""
        L = np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # for the 5-d car example
        return L


    def Jacob_back(self, inp):
        """Defining Jacobian of backward dynamics"""
        L = np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # for the 5-d car example
        return L

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
        self.nbins = 100  # used for computing the integral
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
        self.nbins = 100  # used for computing the integral
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