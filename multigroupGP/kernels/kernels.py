from abc import ABC, abstractmethod
import warnings
from jax import vmap
import jax.numpy as jnp
import numpy as onp
from .util import embed_distance_matrix


CASTING_WARNING = "\n\n⚠️ Casting group labels to integers. Make sure your group labels are ints to avoid undue casting! ⚠️\n"

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

    @abstractmethod
    def store_params(self, params):
        pass

    def cov_map(self, cov_func, xs, xs2=None):
        if xs2 is None:
            return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T

    def check_params(self, params):
        try:
            assert len(params.primal) == self.num_cov_params
        except:
            assert len(params) == self.num_cov_params

    def transform_params(self, params, log_params):
        params = jnp.array(params)
        if log_params:

            def transf(x):
                return jnp.exp(params)

        else:

            def transf(x):
                return params

        return transf(params)


##############################
############ RBF #############
##############################
class RBF(Kernel):
    num_cov_params = 2

    def __init__(self, amplitude=1.0, lengthscale=1.0):
        super().__init__()
        self.params = {
            "amplitude": amplitude,
            "lengthscale": lengthscale,
        }
        self.is_fitted = False

    def kernel_vectorized(self, x1, x2):
        return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2))

    def __call__(self, x1, params=None, x2=None, log_params=True):

        if params is None:
            params = jnp.array(
                [
                    self.params["amplitude"],
                    self.params["lengthscale"],
                ]
            )

        self.check_params(params)
        params = self.transform_params(params, log_params)
        output_scale = params[0]
        lengthscale = params[1]

        x1 = x1 / lengthscale
        if x2 is not None:
            x2 = x2 / lengthscale
        cov = output_scale * self.cov_map(self.kernel_vectorized, x1, x2)
        return cov

    def create_param_dict(self, params):
        return {
            "amplitude": params[0],
            "lengthscale": params[1],
        }

    def store_params(self, params):
        self.params = self.create_param_dict(params)


##############################
######## Matern 1/2 ##########
##############################
class Matern12(Kernel):
    num_cov_params = 2

    def __init__(self, amplitude=1.0, lengthscale=1.0):
        super().__init__()
        self.params = {
            "amplitude": amplitude,
            "lengthscale": lengthscale,
        }
        self.is_fitted = False

    def kernel_vectorized(self, x1, x2, lengthscale):
        return jnp.exp(-0.5 * jnp.sqrt(jnp.sum((x1 - x2) ** 2)) / lengthscale)

    def cov_map(
        self,
        cov_func,
        xs,
        lengthscale,
        xs2=None,
    ):
        if xs2 is None:
            return vmap(
                lambda x: vmap(lambda y: cov_func(x, y, lengthscale=lengthscale))(xs)
            )(xs)
        return vmap(
            lambda x: vmap(lambda y: cov_func(x, y, lengthscale=lengthscale))(xs)
        )(xs2).T

    def __call__(self, x1, params=None, x2=None, log_params=True):

        if params is None:
            params = jnp.array(
                [
                    self.params["amplitude"],
                    self.params["lengthscale"],
                ]
            )

        self.check_params(params)
        params = self.transform_params(params, log_params)
        output_scale = params[0]
        lengthscale = params[1]

        cov = output_scale * self.cov_map(
            self.kernel_vectorized, xs=x1, xs2=x2, lengthscale=lengthscale
        )
        return cov.squeeze()

    def create_param_dict(self, params):
        return {
            "amplitude": params[0],
            "lengthscale": params[1],
        }

    def store_params(self, params):
        self.params = self.create_param_dict(params)


##############################
#### Hierarchical kernel #####
##############################
class HGPKernel(Kernel):
    num_cov_params = None

    def __init__(self, within_group_kernel, between_group_kernel):
        super().__init__()
        self.within_group_kernel = within_group_kernel
        self.between_group_kernel = between_group_kernel
        self.num_cov_params = (
            within_group_kernel.num_cov_params + between_group_kernel.num_cov_params
        )
        self.params = {
            "within_group": within_group_kernel.params,
            "between_group": between_group_kernel.params,
        }
        self.is_fitted = False

    def __call__(
        self, x1, groups1, params=None, x2=None, groups2=None, log_params=True
    ):

        if params is None:
            within_group_params = None
            between_group_params = None
        else:
            within_group_params = params[: self.within_group_kernel.num_cov_params]
            between_group_params = params[self.within_group_kernel.num_cov_params :]

        cov_within = self.within_group_kernel(
            x1, params=within_group_params, x2=x2, log_params=log_params
        )
        cov_between = self.between_group_kernel(
            x1, params=between_group_params, x2=x2, log_params=log_params
        )

        if x2 is None and groups2 is None:
            groups2 = groups1

        same_group_mask = (
            onp.expand_dims(groups1, 1) == onp.expand_dims(groups2, 0)
        ).astype(int)

        cov = cov_between + same_group_mask * cov_within
        return cov

    def store_params(self, params):
        within_group_params = params[: self.within_group_kernel.num_cov_params]
        between_group_params = params[self.within_group_kernel.num_cov_params :]
        self.params = {
            "within_group": self.within_group_kernel.create_param_dict(
                within_group_params
            ),
            "between_group": self.between_group_kernel.create_param_dict(
                between_group_params
            ),
        }

    def kernel_vectorized(self):
        pass


##############################
###### Multi-group RBF #######
##############################
class MultiGroupRBF(Kernel):
    num_cov_params = 3

    def __init__(self, amplitude=1.0, group_diff_param=1.0, lengthscale=1.0):
        super().__init__()
        self.params = {
            "amplitude": amplitude,
            "group_diff_param": group_diff_param,
            "lengthscale": lengthscale,
        }
        self.is_fitted = False

    def kernel_vectorized(
        self, x1, x2, group_embeddings1, group_embeddings2, group_diff_param
    ):
        p = x1.shape[-1]
        dists = jnp.sum((x1 - x2) ** 2)
        group_dists = jnp.sum((group_embeddings1 - group_embeddings2) ** 2)

        cov = (
            1
            / (group_diff_param * group_dists + 1) ** (0.5 * p)
            * jnp.exp(-0.5 * dists / (group_diff_param * group_dists + 1))
        )
        return cov

    def cov_map(
        self,
        cov_func,
        xs,
        group_embeddings1,
        group_diff_param,
        xs2=None,
        group_embeddings2=None,
    ):
        if xs2 is None:
            return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
        return vmap(
            lambda x, g1: vmap(
                lambda y, g2: cov_func(x, y, g1, g2, group_diff_param=group_diff_param)
            )(xs, group_embeddings1)
        )(xs2, group_embeddings2).T

    def __call__(
        self,
        x1,
        groups1,
        params=None,
        group_distances=None,
        x2=None,
        groups2=None,
        log_params=True,
    ):

        if params is None:
            params = jnp.array(
                [
                    self.params["amplitude"],
                    self.params["group_diff_param"],
                    self.params["lengthscale"],
                ]
            )

        self.check_params(params)
        params = self.transform_params(params, log_params)
        output_scale = params[0]
        group_diff_param = params[1]
        lengthscale = params[2]

        if group_distances is None:
            n_groups = len(jnp.unique(groups1))
            group_distances = jnp.ones(n_groups) - jnp.eye(n_groups)

        # if not isinstance(groups1.flat[0], jnp.integer):
        if not issubclass(groups1.dtype.type, jnp.integer):
            warnings.warn(CASTING_WARNING)
            groups1 = groups1.astype(int)
            if groups2 is not None:
                groups2 = groups2.astype(int)

        # assert jnp.all(jnp.diag(group_distances) == 0)

        # Embed group distance matrix in Euclidean space for convenience.
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

    def store_params(self, params):
        self.params = {
            "amplitude": params[0],
            "group_diff_param": params[1],
            "lengthscale": params[2],
        }


##############################
### Multi-group Matern 1/2 ###
##############################
class MultiGroupMatern12(Kernel):
    num_cov_params = 4

    def __init__(
        self, amplitude=1.0, group_diff_param=1.0, lengthscale=1.0, dependency_scale=1.0
    ):
        super().__init__()
        self.params = {
            "amplitude": amplitude,
            "group_diff_param": group_diff_param,
            "lengthscale": lengthscale,
            "dependency_scale": dependency_scale,
        }
        self.is_fitted = False

    def kernel_vectorized(
        self,
        x1,
        x2,
        group_embeddings1,
        group_embeddings2,
        lengthscale,
        group_diff_param,
        dependency_scale,
    ):
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

    def cov_map(
        self,
        cov_func,
        xs,
        group_embeddings1,
        lengthscale,
        group_diff_param,
        dependency_scale,
        xs2=None,
        group_embeddings2=None,
    ):
        if xs2 is None:
            return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
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

    def __call__(
        self,
        x1,
        groups1,
        params=None,
        group_distances=None,
        x2=None,
        groups2=None,
        log_params=True,
    ):

        if params is None:
            params = jnp.array(
                [
                    self.params["amplitude"],
                    self.params["group_diff_param"],
                    self.params["lengthscale"],
                    self.params["dependency_scale"],
                ]
            )

        self.check_params(params)
        params = self.transform_params(params, log_params)
        output_scale = params[0]
        group_diff_param = params[1]
        lengthscale = params[2]
        dependency_scale = params[3]

        if group_distances is None:
            n_groups = len(onp.unique(groups1))
            group_distances = onp.ones(n_groups) - onp.eye(n_groups)

        if not isinstance(groups1.flat[0], onp.integer):
            warnings.warn(CASTING_WARNING)
            groups1 = groups1.astype(int)
            if groups2 is not None:
                groups2 = groups2.astype(int)

        assert onp.all(onp.diag(group_distances) == 0)

        # Embed group distance matrix in Euclidean space for convenience.
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

    def store_params(self, params):
        self.params = {
            "amplitude": params[0],
            "a": params[1],
            "lengthscale": params[2],
            "dependency_scale": params[3],
        }


if __name__ == "__main__":
    # kernel = Matern12()
    kernel = MultiGroupRBF()
    K = kernel(jnp.array([0.0, 0.0]), onp.random.normal(size=(20, 1)))
    print(K.shape)
    import ipdb

    ipdb.set_trace()
