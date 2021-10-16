from jax import vmap
import jax.numpy as jnp
import jax.random as random
import jax.scipy as scipy
from scipy.optimize import minimize
import numpy as onp
import warnings
from abc import ABC, abstractmethod
from .util import softplus, embed_distance_matrix


##############################
####### Abstract class #######
##############################
class Kernel(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def num_cov_params(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def kernel_vectorized(self):
        pass

    def cov_map(self, cov_func, xs, xs2=None):
        if xs2 is None:
            return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
        else:
            return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T

    def check_params(self, params):
        try:
            assert len(params.primal) == self.num_cov_params
        except:
            assert len(params) == self.num_cov_params

    def transform_params(self, params, log_params):
        params = jnp.array(params)
        if log_params:
            transf = lambda x: jnp.exp(params)
        else:
            transf = lambda x: params
        return transf(params)

##############################
############ RBF #############
##############################
class RBF(Kernel):
    num_cov_params = 2

    def __init__(self):
        super().__init__()

    def kernel_vectorized(self, x1, x2):
        return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2))

    def __call__(self, params, x1, x2=None, log_params=True):

        self.check_params(params)
        params = self.transform_params(params, log_params)
        output_scale = params[0]
        lengthscale = params[1]

        x1 = x1 / lengthscale
        if x2 is not None:
            x2 = x2 / lengthscale
        cov = output_scale * self.cov_map(self.kernel_vectorized, x1, x2)
        return cov

##############################
######## Matern 1/2 ##########
##############################
class Matern12(Kernel):
    num_cov_params = 2

    def __init__(self):
        super().__init__()

    def kernel_vectorized(self, x1, x2, lengthscale):
        return jnp.exp(-0.5 * jnp.sqrt(jnp.sum((x1 - x2) ** 2)) / lengthscale)

    def cov_map(self, cov_func, xs, lengthscale, xs2=None,):
        if xs2 is None:
            return vmap(
                lambda x: vmap(lambda y: cov_func(x, y, lengthscale=lengthscale))(xs)
            )(xs)
        else:
            return vmap(
                lambda x: vmap(lambda y: cov_func(x, y, lengthscale=lengthscale))(xs)
            )(xs2).T

    def __call__(self, params, x1, x2=None, log_params=True):

        self.check_params(params)
        params = self.transform_params(params, log_params)
        output_scale = params[0]
        lengthscale = params[1]

        cov = output_scale * self.cov_map(
            self.kernel_vectorized, xs=x1, xs2=x2, lengthscale=lengthscale
        )
        return cov.squeeze()


##############################
###### Multi-group RBF #######
##############################
class MultiGroupRBF(Kernel):
    num_cov_params = 3

    def __init__(self):
        super().__init__()

    def kernel_vectorized(self, x1, x2, group_embeddings1, group_embeddings2, group_diff_param):
        p = x1.shape[-1]
        dists = jnp.sum((x1 - x2) ** 2)
        group_dists = jnp.sum((group_embeddings1 - group_embeddings2) ** 2)

        cov = (
            1
            / (group_diff_param * group_dists + 1) ** (0.5 * p)
            * jnp.exp(-0.5 * dists / (group_diff_param * group_dists + 1))
        )
        return cov

    def cov_map(self, cov_func, xs, group_embeddings1, group_diff_param, xs2=None, group_embeddings2=None,):
        if xs2 is None:
            return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
        else:
            return vmap(
                lambda x, g1: vmap(
                    lambda y, g2: cov_func(x, y, g1, g2, group_diff_param=group_diff_param)
                )(xs, group_embeddings1)
            )(xs2, group_embeddings2).T

    def __call__(self, params, x1, groups1, group_distances, x2=None, groups2=None, log_params=True):

        self.check_params(params)
        params = self.transform_params(params, log_params)
        output_scale = params[0]
        group_diff_param = params[1]
        lengthscale = params[2]

        if not isinstance(groups1.flat[0], onp.integer):
            warnings.warn(
                "Casting group labels to integers. Make sure your group labels are ints to avoid undue casting!"
            )
            groups1 = groups1.astype(int)
            if groups2 is not None:
                groups2 = groups2.astype(int)

        assert onp.all(onp.diag(group_distances) == 0)

        ## Embed group distance matrix in Euclidean space for convenience.
        embedding = embed_distance_matrix(group_distances)

        x1 = x1 / (lengthscale)
        group_embeddings1 = jnp.array([embedding[xx] for xx in groups1])
        if x2 is not None:
            x2 = x2 / (lengthscale)
            group_embeddings2 = jnp.array([embedding[xx] for xx in groups2])
        else:
            x2 = x1
            group_embeddings2 = group_embeddings1

        cov = self.cov_map(
            self.kernel_vectorized,
            xs=x1,
            xs2=x2,
            group_embeddings1=group_embeddings1,
            group_embeddings2=group_embeddings2,
            group_diff_param=group_diff_param,
        )
        return output_scale * cov.squeeze()

##############################
### Multi-group Matern 1/2 ###
##############################
class MultiGroupMatern12(Kernel):
    num_cov_params = 4

    def __init__(self):
        super().__init__()

    def kernel_vectorized(self, 
                            x1,
                            x2,
                            group_embeddings1,
                            group_embeddings2,
                            lengthscale,
                            group_diff_param,
                            dependency_scale,):
        p = x1.shape[-1]
        dists = jnp.sqrt(jnp.sum((x1 - x2) ** 2))
        group_dists = jnp.sum((group_embeddings1 - group_embeddings2) ** 2)

        pre_exp_term = dependency_scale ** (0.5 * p) / (
            (group_diff_param * group_dists + 1) ** 0.5
            * (group_diff_param * group_dists + dependency_scale) ** (0.5 * p)
        )

        exp_term = jnp.exp(
            -lengthscale
            * dists
            * (
                (group_diff_param * group_dists + 1)
                / (group_diff_param * group_dists + dependency_scale)
            )
            ** 0.5
        )

        cov = pre_exp_term * exp_term
        return cov

    def cov_map(self, 
                cov_func,
                xs,
                group_embeddings1,
                lengthscale,
                group_diff_param,
                dependency_scale,
                xs2=None,
                group_embeddings2=None,):
        if xs2 is None:
            return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
        else:
            return vmap(
                lambda x, g1: vmap(
                    lambda y, g2: cov_func(
                        x,
                        y,
                        g1,
                        g2,
                        lengthscale=lengthscale,
                        group_diff_param=group_diff_param,
                        dependency_scale=dependency_scale,
                    )
                )(xs, group_embeddings1)
            )(xs2, group_embeddings2).T


    def __call__(self, params, x1, groups1, group_distances, x2=None, groups2=None, log_params=True):
        self.check_params(params)
        params = self.transform_params(params, log_params)
        output_scale = params[0]
        group_diff_param = params[1]
        lengthscale = params[2]
        dependency_scale = params[3]

        if not isinstance(groups1.flat[0], onp.integer):
            warnings.warn(
                "Casting group labels to integers. Make sure your group labels are ints to avoid undue casting!"
            )
            groups1 = groups1.astype(int)
            if groups2 is not None:
                groups2 = groups2.astype(int)

        assert onp.all(onp.diag(group_distances) == 0)

        ## Embed group distance matrix in Euclidean space for convenience.
        embedding = embed_distance_matrix(group_distances)

        group_embeddings1 = jnp.array([embedding[xx] for xx in groups1])
        if x2 is not None:
            group_embeddings2 = jnp.array([embedding[xx] for xx in groups2])
        else:
            x2 = x1
            group_embeddings2 = group_embeddings1

        cov = self.cov_map(
            self.kernel_vectorized,
            xs=x1,
            xs2=x2,
            group_embeddings1=group_embeddings1,
            group_embeddings2=group_embeddings2,
            lengthscale=lengthscale,
            group_diff_param=group_diff_param,
            dependency_scale=dependency_scale,
        )
        return output_scale * cov.squeeze()


if __name__ == "__main__":
    # kernel = Matern12()
    kernel = MultiGroupRBF()
    K = kernel(jnp.array([0., 0.]), onp.random.normal(size=(20, 1)))
    print(K.shape)
    import ipdb; ipdb.set_trace()






