from iml.common import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data, convert_to_instance_with_index
from iml.explanations import AdditiveExplanation
from iml.links import convert_to_link, IdentityLink
from iml.datatypes import convert_to_data, DenseData
from scipy.special import binom
import numpy as np
import pandas as pd
import logging
import copy
import itertools
from sklearn.linear_model import LassoLarsIC, Lasso
from sklearn.cluster import KMeans
from tqdm import tqdm
from .explainer import Explainer

log = logging.getLogger('shap')


def kmeans(X, k, round_values=True):
    """ Summarize a dataset with k mean samples weighted by the number of data points they
    each represent.

    Parameters
    ----------
    X : numpy.array or pandas.DataFrame
        Matrix of data samples to summarize (# samples x # features)

    k : int
        Number of means to use for approximation.

    round_values : bool
        For all i, round the ith dimension of each mean sample to match the nearest value
        from X[:,i]. This ensures discrete features always get a valid value.

    Returns
    -------
    DenseData object.
    """

    group_names = [str(i) for i in range(X.shape[1])]
    if str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
        group_names = X.columns
        X = X.values
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    if round_values:
        for i in range(k):
            for j in range(X.shape[1]):
                ind = np.argmin(np.abs(X[:,j] - kmeans.cluster_centers_[i,j]))
                kmeans.cluster_centers_[i,j] = X[ind,j]

    return DenseData(kmeans.cluster_centers_, group_names, None, 1.0*np.bincount(kmeans.labels_))


class KernelExplainer(Explainer):
    """Uses the Kernel SHAP method to explain the output of any function.

    Kernel SHAP is a method that uses a special weighted linear regression
    to compute the importance of each feature. The computed importance values
    are Shapley values from game theory and also coefficents from a local linear
    regression.


    Parameters
    ----------
    model : function or iml.Model
        User supplied function that takes a matrix of samples (# samples x # features) and
        computes a the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    data : numpy.array or pandas.DataFrame or iml.DenseData
        The background dataset to use for integrating out features. To determine the impact
        of a feature, that feature is set to "missing" and the change in the model output
        is observed. Since most models aren't designed to handle arbitrary missing data at test
        time, we simulate "missing" by replacing the feature with the values it takes in the
        background dataset. So if the background dataset is a simple sample of all zeros, then
        we would approximate a feature being missing by setting it to zero. For small problems
        this background datset can be the whole training set, but for larger problems consider
        using a single reference value or using the kmeans function to summarize the dataset.

    link : "identity" or "logit"
        A generalized linear model link to connect the feature importance values to the model
        output. Since the feature importance values, phi, sum up to the model output, it often makes
        sense to connect them to the ouput with a link function where link(outout) = sum(phi).
        If the model output is a probability then the LogitLink link function makes the feature
        importance values have log-odds units.
    """

    def __init__(self, model, data, link=IdentityLink(), **kwargs):

        # convert incoming inputs to standardized iml objects
        self.link = convert_to_link(link)
        self.model = convert_to_model(model)
        self.keep_index = kwargs.get("keep_index", False)
        self.keep_index_ordered = kwargs.get("keep_index_ordered", False)
        self.data = convert_to_data(data, keep_index=self.keep_index)
        match_model_to_data(self.model, self.data)

        # enforce our current input type limitations
        assert isinstance(self.data, DenseData), "Shap explainer only supports the DenseData input currently."
        assert not self.data.transposed, "Shap explainer does not support transposed DenseData currently."

        # warn users about large background data sets
        if len(self.data.weights) > 100:
            log.warning("Using " + str(len(self.data.weights)) + " background data samples could cause " +
                        "slower run times. Consider using shap.kmeans(data, K) to summarize the background " +
                        "as K weighted samples.")

        # init our parameters
        self.N = self.data.data.shape[0]
        self.P = self.data.data.shape[1]
        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

        # find E_x[f(x)]
        if self.keep_index:
            model_null = self.model.f(self.data.convert_to_df())
        else:
            model_null = self.model.f(self.data.data)
        if isinstance(model_null, (pd.DataFrame, pd.Series)):
            model_null = model_null.values
        self.fnull = np.sum((model_null.T * self.data.weights).T, 0)
        self.expected_value = self.fnull

        # see if we have a vector output
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
        else:
            self.D = self.fnull.shape[0]

    def shap_values(self, X, **kwargs):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            A matrix of samples (# samples x # features) on which to explain the model's output.

        nsamples : "auto" or int
            Number of times to re-evaluate the model when explaining each prediction. More samples
            lead to lower variance estimates of the SHAP values.

        l1_reg : "auto" or float
            The l1 regularization to use for feature selection (the estimation procedure is based on
            a debiased lasso). Set this to zero to remove the feature selection step before estimation.

        Returns
        -------
        For models with a single output this returns a matrix of SHAP values
        (# samples x # features). Each row sums to the difference between the model output for that
        sample and the expected value of the model output (which is stored as expected_value
        attribute of the explainer). For models with vector outputs this returns a list
        of such matrices, one for each output.
        """

        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
            if self.keep_index:
                index_value = X.index.values
                index_name = X.index.name
                column_name = list(X.columns)
            X = X.values

        assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

        # single instance
        if len(X.shape) == 1:
            data = X.reshape((1, X.shape[0]))
            if self.keep_index:
                data = convert_to_instance_with_index(data, column_name, index_name, index_value)
            explanation = self.explain(data, **kwargs)

            # vector-output
            s = explanation.effects.shape
            if len(s) == 2:
                outs = [np.zeros(s[0]) for j in range(s[1])]
                for j in range(s[1]):
                    outs[j] = explanation.effects[:, j]
                    assert abs(self.link.f(self.expected_value[j]) - explanation.base_value[j]) < 1e-6
                return outs

            # single-output
            else:
                out = np.zeros(s[0])
                out[:] = explanation.effects
                assert abs(self.link.f(self.expected_value) - explanation.base_value) < 1e-6
                return out

        # explain the whole dataset
        elif len(X.shape) == 2:
            explanations = []
            for i in tqdm(range(X.shape[0]), disable=kwargs.get("silent", False)):
                data = X[i:i + 1, :]
                if self.keep_index:
                    data = convert_to_instance_with_index(data, column_name, index_value[i:i + 1], index_name)
                explanations.append(self.explain(data, **kwargs))

            # vector-output
            s = explanations[0].effects.shape
            if len(s) == 2:
                outs = [np.zeros((X.shape[0], s[0])) for j in range(s[1])]
                for i in range(X.shape[0]):
                    for j in range(s[1]):
                        outs[j][i] = explanations[i].effects[:, j]
                        assert abs(self.link.f(self.expected_value[j]) - explanations[i].base_value[j]) < 1e-6
                return outs

            # single-output
            else:
                out = np.zeros((X.shape[0], s[0]))
                for i in range(X.shape[0]):
                    out[i] = explanations[i].effects
                    assert abs(self.link.f(self.expected_value) - explanations[i].base_value) < 1e-6
                return out

    def explain(self, incoming_instance, **kwargs):
        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
        self.M = len(self.varyingFeatureGroups)

        # find f(x)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            model_out = self.model.f(instance.x)
        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values[0]
        self.fx = model_out[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then there no feature has an effect
        if self.M == 0:
            phi = np.zeros((len(self.data.groups), self.D))
            phi_var = np.zeros((len(self.data.groups), self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((len(self.data.groups), self.D))
            phi_var = np.zeros((len(self.data.groups), self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds[0],d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "auto")

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2**11

            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2 ** 30
            if self.M <= 30:
                self.max_samples = 2 ** self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.allocate()

            # weight the different subset sizes
            num_subset_sizes = np.int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = np.int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            log.debug("weight_vector = {0}".format(weight_vector))
            log.debug("num_subset_sizes = {0}".format(num_subset_sizes))
            log.debug("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
            log.debug("M = {0}".format(self.M))

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype='int64')
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):

                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes: nsubsets *= 2
                log.debug("subset_size = {0}".format(subset_size))
                log.debug("nsubsets = {0}".format(nsubsets))
                log.debug("self.nsamples*weight_vector[subset_size-1] = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1]))
                log.debug("self.nsamples*weight_vector[subset_size-1/nsubsets = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets))

                # see if we have enough samples to enumerate all subsets of this size
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes: w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype='int64')] = 1.0
                        self.addsample(instance.x, mask, w)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.addsample(instance.x, mask, w)
                else:
                    break
            log.info("num_full_subsets = {0}".format(num_full_subsets))

            # add random samples from what is left of the subset space
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug("samples_left = {0}".format(samples_left))
            if num_full_subsets != num_subset_sizes:
                weight_left = np.sum(weight_vector[num_full_subsets:])
                rand_sample_weight = weight_left / samples_left
                log.info("weight_left = {0}".format(weight_left))
                log.info("rand_sample_weight = {0}".format(rand_sample_weight))
                remaining_weight_vector = weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                log.info("remaining_weight_vector = {0}".format(remaining_weight_vector))
                log.info("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
                ind_set = np.arange(len(remaining_weight_vector))
                while samples_left > 0:
                    mask[:] = 0.0
                    np.random.shuffle(group_inds)
                    ind = np.random.choice(ind_set, 1, p=remaining_weight_vector)[0]
                    mask[group_inds[:ind + num_full_subsets + 1]] = 1.0
                    samples_left -= 1
                    self.addsample(instance.x, mask, rand_sample_weight)

                    # add the compliment sample
                    if samples_left > 0:
                        mask -= 1.0
                        mask[:] = np.abs(mask)
                        self.addsample(instance.x, mask, rand_sample_weight)
                        samples_left -= 1

            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            phi = np.zeros((len(self.data.groups), self.D))
            phi_var = np.zeros((len(self.data.groups), self.D))
            for d in range(self.D):
                vphi, vphi_var = self.solve(self.nsamples / self.max_samples, d)
                phi[self.varyingInds, d] = vphi
                phi_var[self.varyingInds, d] = vphi_var

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)
            phi_var = np.squeeze(phi_var, axis=1)

        # return the Shapley values along with variances of the estimates
        # note that if features were eliminated by l1 regression their
        # variance will be 0, even though they are not perfectly known
        return AdditiveExplanation(
            self.link.f(self.fnull if self.vector_out else self.fnull[0]),
            self.link.f(self.fx if self.vector_out else self.fx[0]),
            phi, phi_var, instance, self.link, self.model, self.data
        )

    def varying_groups(self, x):
        varying = np.zeros(len(self.data.groups))
        for i in range(0, len(self.data.groups)):
            inds = self.data.groups[i]
            num_mismatches = np.sum(np.abs(x[0, inds] - self.data.data[:, inds]) > 1e-7)
            varying[i] = num_mismatches > 0
        return np.nonzero(varying)[0]

    def allocate(self):
        self.synth_data = np.zeros((self.nsamples * self.N, self.P))
        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.y = np.zeros((self.nsamples * self.N, self.D))
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0
        if self.keep_index:
            self.synth_data_index = [None] * (self.nsamples * self.N)

    def addsample(self, x, m, w):
        offset = self.nsamplesAdded * self.N
        for i in range(self.N):
            if self.keep_index:
                self.synth_data_index[offset+i] = self.data.index_value[i]
            for j in range(self.M):
                for k in self.varyingFeatureGroups[j]:
                    if m[j] == 1.0:
                        self.synth_data[offset + i, k] = x[0, k]
                    else:
                        self.synth_data[offset + i, k] = self.data.data[i, k]

        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def run(self):
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N
        data = self.synth_data[self.nsamplesRun*self.N:self.nsamplesAdded*self.N,:]
        if self.keep_index:
            index = self.synth_data_index[self.nsamplesRun*self.N:self.nsamplesAdded*self.N]
            index = pd.DataFrame(index, columns=[self.data.index_name])
            data = pd.DataFrame(data, columns=self.data.group_names)
            data = pd.concat([index, data], axis=1).set_index(self.data.index_name)
            if self.keep_index_ordered:
                data = data.sort_index()
        modelOut = self.model.f(data)
        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        # if len(modelOut.shape) > 1:
        #     raise ValueError("The supplied model function should output a vector not a matrix!")
        self.y[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :] = np.reshape(modelOut, (num_to_run, self.D))

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = np.zeros(self.D)
            for j in range(0, self.N):
                eyVal += self.y[i * self.N + j, :] * self.data.weights[j]

            self.ey[i, :] = eyVal
            self.nsamplesRun += 1

    def solve(self, fraction_evaluated, dim):
        eyAdj = self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim])
        s = np.sum(self.maskMatrix, 1)

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M)
        log.debug("fraction_evaluated = {0}".format(fraction_evaluated))
        if (self.l1_reg not in ["auto", False, 0]) or (fraction_evaluated < 0.2 and self.l1_reg == "auto"):
            w_aug = np.hstack((self.kernelWeights * (self.M - s), self.kernelWeights * s))
            log.info("np.sum(w_aug) = {0}".format(np.sum(w_aug)))
            log.info("np.sum(self.kernelWeights) = {0}".format(np.sum(self.kernelWeights)))
            w_sqrt_aug = np.sqrt(w_aug)
            eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))))
            eyAdj_aug *= w_sqrt_aug
            mask_aug = np.transpose(w_sqrt_aug * np.transpose(np.vstack((self.maskMatrix, self.maskMatrix - 1))))
            var_norms = np.array([np.linalg.norm(mask_aug[:, i]) for i in range(mask_aug.shape[1])])

            if self.l1_reg == "auto":
                model = LassoLarsIC(criterion="aic")
            elif self.l1_reg == "bic" or self.l1_reg == "aic":
                model = LassoLarsIC(criterion=self.l1_reg)
            else:
                model = Lasso(alpha=self.l1_reg)

            model.fit(mask_aug, eyAdj_aug)
            nonzero_inds = np.nonzero(model.coef_)[0]

        if len(nonzero_inds) == 0:
            return np.zeros(self.M), np.ones(self.M)

        # eliminate one variable with the constraint that all features sum to the output
        eyAdj2 = eyAdj - self.maskMatrix[:, nonzero_inds[-1]] * (
                    self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))
        etmp = np.transpose(np.transpose(self.maskMatrix[:, nonzero_inds[:-1]]) - self.maskMatrix[:, nonzero_inds[-1]])
        log.debug("etmp[:4,:] {0}".format(etmp[:4, :]))

        # solve a weighted least squares equation to estimate phi
        tmp = np.transpose(np.transpose(etmp) * np.transpose(self.kernelWeights))
        tmp2 = np.linalg.inv(np.dot(np.transpose(tmp), etmp))
        w = np.dot(tmp2, np.dot(np.transpose(tmp), eyAdj2))
        log.debug("np.sum(w) = {0}".format(np.sum(w)))
        log.debug("self.link(self.fx) - self.link(self.fnull) = {0}".format(
            self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])))
        log.debug("self.fx = {0}".format(self.fx[dim]))
        log.debug("self.link(self.fx) = {0}".format(self.link.f(self.fx[dim])))
        log.debug("self.fnull = {0}".format(self.fnull[dim]))
        log.debug("self.link(self.fnull) = {0}".format(self.link.f(self.fnull[dim])))
        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])) - sum(w)
        log.info("phi = {0}".format(phi))

        # clean up any rounding errors
        for i in range(self.M):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi, np.ones(len(phi))
