from __future__ import division
import warnings
import numpy as np
import scipy as sp
from matplotlib import image
from random import randint
from scipy.misc import logsumexp
from helper_functions import image_to_matrix, matrix_to_image, \
                             flatten_image_matrix, unflatten_image_matrix, \
                             image_difference


warnings.simplefilter(action="ignore", category=FutureWarning)


def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    # TODO: finish this function
    original_dim = image_values.shape
    flatten_image = flatten_image_matrix(image_values)
    dim = flatten_image.shape
    indicator_indices = np.zeros((dim[0],1))
    distance = np.zeros((dim[0],k))
    if initial_means == None:
        means_indices = np.random.choice(dim[0], k, replace=False)
        means = flatten_image[means_indices,:].copy()
    else:
        means = initial_means
    previous_means = np.zeros(means.shape)
    while not np.array_equal(means, previous_means):
        indicator = np.zeros((dim[0],k))
        previous_means = means.copy()
        for i in range(k):
            distance[:,i] = np.sum(np.power(np.subtract(flatten_image, means[i,:]),2), axis=1, dtype=float)
        indicator_indices = np.argmin(distance,axis=1)
        indx = np.arange(dim[0]).reshape(-1,1)
        indicator_indices = indicator_indices.reshape(-1,1)
        indicator[indx,indicator_indices] = 1
        for i in range(k):
            means[i,:] = np.divide(np.sum(np.multiply(indicator[:,i].reshape(-1,1),flatten_image), axis=0, dtype=float), np.sum(indicator[:,i],dtype=float))
    
    for i in range(k):
        indices = np.where(indicator[:,i]==1)
        flatten_image[indices,:] = means[i,:]
    updated_image_values = unflatten_image_matrix(flatten_image,original_dim[1])
    return updated_image_values



def default_convergence(prev_likelihood, new_likelihood, conv_ctr,
                        conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.flatten_image = flatten_image_matrix(self.image_matrix)
        self.num_components = num_components
        if(means is None):
            self.means = np.zeros(num_components)
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)
        ind = np.where(self.flatten_image<0)
        self.flatten_image[ind] = 10e-9

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
        # TODO: finish this
        log_N = -np.add(0.5*np.log(2*np.pi*self.variances),np.exp(np.subtract(np.log(np.square(np.subtract(val,self.means))),np.log(2*self.variances))))
        return logsumexp(log_N,b=self.mixing_coefficients.reshape(1,-1))

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
        # TODO: finish this
        flatten_image = flatten_image_matrix(self.image_matrix)
        dim = flatten_image.shape
        indx = np.random.choice(dim[0],self.num_components,replace=0)
        self.means = flatten_image[indx]
        self.variances = np.ones(self.num_components,dtype=float)
        self.mixing_coefficients = np.multiply(np.ones(self.num_components,dtype=float),1/float(self.num_components))

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """
        # TODO: finish this
        N = self.flatten_image.shape[0]
        converge_counter = 0
        converge_condition = False
        self.means = np.float64(self.means)
        self.variances = np.float64(self.variances)
        self.mixing_coefficients = np.float64(self.mixing_coefficients)
        prev_likelihood = self.likelihood()
        while not converge_condition:
            log_N = -np.add(0.5*np.log(2*np.pi*self.variances),np.exp(np.subtract(np.log(np.square(np.subtract(self.flatten_image,self.means.reshape(1,-1)))),np.log(2*self.variances))))
            log_gamma_num = np.add(np.log(self.mixing_coefficients),log_N)
            log_gamma_den = logsumexp(log_N,b=self.mixing_coefficients.reshape(1,-1),axis=1).reshape(-1,1)
            log_gamma = np.subtract(log_gamma_num, log_gamma_den)
            self.means = np.sum(np.exp(np.subtract(np.add(log_gamma,np.log(self.flatten_image)),logsumexp(log_gamma,axis=0).reshape(1,-1))),axis=0)
            self.variances = np.sum(np.exp(np.subtract(np.add(log_gamma,np.log(np.square(np.subtract(self.flatten_image,self.means)))),logsumexp(log_gamma,axis=0).reshape(1,-1))),axis=0)
            self.mixing_coefficients = np.exp(np.subtract(logsumexp(log_gamma,axis=0).reshape(1,-1),np.log(N)))
            new_likelihood = self.likelihood()
            converge_counter,converge_condition=convergence_function(prev_likelihood, new_likelihood, converge_counter)
            prev_likelihood = new_likelihood
        
        

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # TODO: finish this
        dim = self.flatten_image.shape
        original_dim = self.image_matrix.shape
        indicator_indices = np.zeros((dim[0],1))
        indicator = np.zeros((dim[0],self.num_components))
        distance = np.square(np.subtract(self.flatten_image,self.means))
        indicator_indices = np.argmin(distance,axis=1)
        indx = np.arange(dim[0]).reshape(-1,1)
        indicator_indices = indicator_indices.reshape(-1,1)
        indicator[indx,indicator_indices] = 1
        means = self.means.copy()
        means = means.reshape(1,-1)
        for i in range(self.num_components):
            indices = np.where(indicator[:,i]==1)
            self.flatten_image[indices] = means[0,i]
        updated_image_values = unflatten_image_matrix(self.flatten_image,original_dim[1])
        return updated_image_values

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """
        # TODO: finish this
        self.means = np.array(self.means)
        self.variances = np.array(self.variances)
        self.mixing_coefficients = np.array(self.mixing_coefficients)
        log_N = -np.add(0.5*np.log(2*np.pi*self.variances),np.exp(np.subtract(np.log(np.square(np.subtract(self.flatten_image,self.means.reshape(1,-1)))),np.log(2*self.variances))))
        return np.sum(logsumexp(log_N,b=self.mixing_coefficients.reshape(1,-1),axis=1),axis=0)

    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # finish this
        prev_likelihood = self.likelihood()
        best_likelihood = prev_likelihood
        for iteration in range(iters):
            self.train_model(convergence_function=default_convergence)
            new_likelihood = self.likelihood()
            if new_likelihood > best_likelihood:
                best_means = self.means.copy()
                best_variances = self.variances.copy()
                best_mixing_coefficients = self.mixing_coefficients.copy()
                best_likelihood = new_likelihood
            self.initialize_training()
            prev_likelihood = self.likelihood()
        self.means = best_means
        self.variances = best_variances
        self.mixing_coefficients = best_mixing_coefficients
        imge = self.segment()
        img_shape = imge.shape
        imge = imge.reshape((img_shape[0],img_shape[1]))
        return imge
            


class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that
        you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient
         initializations too if that works well.]
        """
        # TODO: finish this
        dim = self.flatten_image.shape
        flatten_image = np.sort(self.flatten_image)
        image_mean = np.mean(flatten_image)
        sub_image_means = np.zeros((1,self.num_components))
        SDAM = np.sum(np.square(np.subtract(flatten_image,image_mean)))
        best_GVF = 0
        for i in range(2000):
            breaking_indices = np.random.choice(dim[0],self.num_components,replace=False)
            sub_image = np.split(flatten_image,breaking_indices)
            SDCM_ALL = 0
            for j in range(self.num_components):
                sub_image_means[0,j] = np.mean(sub_image[j])
                SDCM_ALL += np.sum(np.square(np.subtract(sub_image[j],sub_image_means[0,j])))
            GVF = (SDAM-SDCM_ALL)/float(SDAM)
            if (GVF > best_GVF) and (not np.isnan(sub_image_means).any()):
                best_GVF = GVF
                best_means = sub_image_means.copy()
        self.means = best_means
        self.variances = np.ones(self.num_components,dtype=float)
        self.mixing_coefficients = np.multiply(np.ones(self.num_components,dtype=float),1/float(self.num_components))



def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """
    # TODO: finish this function
    mean_condition = (0.9*np.abs(previous_variables[0]) < np.abs(new_variables[0])).all() and \
                     (np.abs(new_variables[0]) < 1.1*np.abs(previous_variables[0])).all()
    variance_condition = (0.9*np.abs(previous_variables[1]) < np.abs(new_variables[1])).all() and \
                     (np.abs(new_variables[1]) < 1.1*np.abs(previous_variables[1])).all()
    mixing_coeff_condition = (0.9*np.abs(previous_variables[2]) < np.abs(new_variables[2])).all() and \
                     (np.abs(new_variables[2]) < 1.1*np.abs(previous_variables[2])).all()
    if mean_condition and variance_condition and mixing_coeff_condition:
        conv_ctr += 1
    else:
        conv_ctr = 0
    return conv_ctr, conv_ctr > conv_ctr_cap

class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        # TODO: finish this function
        N = self.flatten_image.shape[0]
        converge_counter = 0
        converge_condition = False
        self.means = np.float64(self.means)
        self.variances = np.float64(self.variances)
        self.mixing_coefficients = np.float64(self.mixing_coefficients)
        prev_variables = [self.means.copy(), self.variances.copy(), self.mixing_coefficients.copy()]
        while not converge_condition:
            log_N = -np.add(0.5*np.log(2*np.pi*self.variances),np.exp(np.subtract(np.log(np.square(np.subtract(self.flatten_image,self.means.reshape(1,-1)))),np.log(2*self.variances))))
            log_gamma_num = np.add(np.log(self.mixing_coefficients),log_N)
            log_gamma_den = logsumexp(log_N,b=self.mixing_coefficients.reshape(1,-1),axis=1).reshape(-1,1)
            log_gamma = np.subtract(log_gamma_num, log_gamma_den)
            self.means = np.sum(np.exp(np.subtract(np.add(log_gamma,np.log(self.flatten_image)),logsumexp(log_gamma,axis=0).reshape(1,-1))),axis=0)
            self.variances = np.sum(np.exp(np.subtract(np.add(log_gamma,np.log(np.square(np.subtract(self.flatten_image,self.means)))),logsumexp(log_gamma,axis=0).reshape(1,-1))),axis=0)
            self.mixing_coefficients = np.exp(np.subtract(logsumexp(log_gamma,axis=0).reshape(1,-1),np.log(N)))
            new_variables = [self.means.copy(), self.variances.copy(), self.mixing_coefficients.copy()]
            converge_counter,converge_condition=convergence_function(prev_variables, new_variables, converge_counter)
            prev_variables = [self.means.copy(), self.variances.copy(), self.mixing_coefficients.copy()]


def bayes_info_criterion(gmm):
    # TODO: finish this function
    log_likelihood = gmm.likelihood()
    N = gmm.flatten_image.shape[0]
    return round(np.log(N)*3*gmm.num_components-2*log_likelihood,0)


def BIC_likelihood_model_test(image_matrix):
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel

    for testing purposes:
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]
    ]
    """
    # TODO: finish this method
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]]
    best_BIC = float("inf")
    best_likelihood = -float("inf")
    for i in range(6):
        k = i+2
        gmm = GaussianMixtureModel(image_matrix,k)
        gmm.initialize_training()
        gmm.means = np.array(comp_means[i]).reshape(1,-1)
        gmm.train_model()
        likelihood = gmm.likelihood()
        BIC = bayes_info_criterion(gmm)
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            num_components_like = k
        if BIC < best_BIC:
            best_BIC = BIC
            num_components_BIC = k

def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    # TODO: fill in bic and likelihood
    bic = 7
    likelihood = 7
    pairs = {
        'BIC': bic,
        'likelihood': likelihood
    }
    return pairs

def return_your_name():
    # return your name
    # TODO: finish this
    return "Dan Monga Kilanga"

def bonus(points_array, means_array):
    """
    Return the distance from every point in points_array
    to every point in means_array.

    returns:
    dists = numpy array of float
    """
    # TODO: fill in the bonus function
    # REMOVE THE LINE BELOW IF ATTEMPTING BONUS
#    raise NotImplementedError
#    dim_p = points_array.shape
#    dim_m = means_array.shape
#    points_array = points_array.reshape(dim_p[0],1,dim_p[1])
#    means_array = means_array.reshape(1,dim_m[0],dim_m[1])
#    print "here"
#    dists = np.sqrt(np.sum(np.square(np.subtract(points_array,means_array)),axis=2))
#    dists = np.sqrt(np.sum(np.subtract(np.add(np.square(points_array),np.square(means_array)),2*np.multiply(points_array,means_array)),axis=2))
    dists = np.sqrt(np.subtract(np.add(np.sum(np.square(points_array),axis=1,keepdims=True),np.sum(np.square(means_array),axis=1)), 2*np.dot(points_array,means_array.T)))
    return dists
