from scipy.stats import multivariate_normal as mvn
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel, Hyperparameter, _check_length_scale
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from RBF import RBF
from scipy.linalg import block_diag

class RBF_groupwise(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Radial-basis function kernel (aka squared-exponential kernel).
    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length scale
    parameter :math:`l>0`, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:
    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)
    where :math:`l` is the length scale of the kernel and
    :math:`d(\\cdot,\\cdot)` is the Euclidean distance.
    For advice on how to set the length scale parameter, see e.g. [1]_.
    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    See [2]_, Chapter 4, Section 4.2, for further details of the RBF kernel.
    Read more in the :ref:`User Guide <gp_kernels>`.
    .. versionadded:: 0.18
    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.
    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_
    .. [2] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8354..., 0.03228..., 0.1322...],
           [0.7906..., 0.0652..., 0.1441...]])
    """

    def __init__(self, length_scale_group0=1.0, length_scale_group1=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale_group0 = length_scale_group0
        self.length_scale_group1 = length_scale_group1
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale_group0) and len(self.length_scale_group0) > 1

    @property
    def hyperparameter_length_scale_group0(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale_group0",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale_group0),
            )
        return Hyperparameter("length_scale_group0", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_length_scale_group1(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale_group1",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale_group1),
            )
        return Hyperparameter("length_scale_group1", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        p = X.shape[1] - 1
        length_scale_group0 = _check_length_scale(X, self.length_scale_group0)
        length_scale_group1 = _check_length_scale(X, self.length_scale_group1)
        if Y is None:
            X_groups = X[:, -1]
            X = X[:, :-1]
            X0 = X[X_groups == 0, :]
            X1 = X[X_groups == 1, :]
            K_X0X0 = RBF(length_scale=length_scale_group0)(X0)
            K_X1X1 = RBF(length_scale=length_scale_group1)(X1)
            K = block_diag(K_X0X0, K_X1X1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            
            X_groups = X[:, -1]
            Y_groups = Y[:, -1]
            X = X[:, :-1]
            Y = Y[:, :-1]
            diff_group_indicator = (np.expand_dims(X_groups, 1) - np.expand_dims(Y_groups, 0))**2
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            diff_group_scaling_term = diff_group_indicator * self.group_diff_param + 1
            dists /= diff_group_scaling_term
            K = np.exp(-0.5 * dists)
            K *= self.output_variance
            K /= ((diff_group_scaling_term)**(0.5 * p))

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale ** 2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )