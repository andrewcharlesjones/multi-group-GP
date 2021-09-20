import autograd.numpy as np


def rbf_covariance(kernel_params, x, xp):
    """RBF covariance function
    
    Args:
        kernel_params (list): Kernel parameters in the following order: [output scale, length scales]
        x (np array): input 1
        xp (np array): input 2
    
    Returns:
        np array: covariance matrix
    """
    assert len(kernel_params) == 2
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1])

    diffs = np.expand_dims(x / lengthscales, 1) - np.expand_dims(xp / lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))


def multigroup_rbf_covariance(kernel_params, X1, X2, groups1, groups2, group_distances):

    assert len(kernel_params) == 3
    output_scale = np.exp(kernel_params[0])
    group_diff_param = np.exp(kernel_params[1])
    lengthscales = np.exp(kernel_params[2])

    assert X1.shape[1] == X2.shape[1]
    assert groups1.shape[1] == groups2.shape[1]
    assert group_distances.shape[1] == groups1.shape[1]

    p = X1.shape[1]
    n_groups = groups1.shape[1]
    # import ipdb; ipdb.set_trace()

    diffs = np.expand_dims(X1 / lengthscales, 1) - np.expand_dims(X2 / lengthscales, 0)
    dists = np.sum(diffs ** 2, axis=2)

    diff_group_indicator = (
        np.expand_dims(groups1, 1) - np.expand_dims(groups2, 0)
    ) ** 2
    diff_group_scaling_term = np.zeros(dists.shape)

    for ii in range(n_groups):

        for jj in range(ii):

            if ii == jj:
                continue

            curr_group_distance = group_distances[ii, jj]

            diff_group_scaling_term += (
                curr_group_distance * group_diff_param ** 2 + 1
            ) * (
                np.logical_and(
                    diff_group_indicator[:, :, ii] == 1,
                    diff_group_indicator[:, :, jj] == 1,
                )
            ).astype(
                int
            )

    samegroup_mask = (diff_group_scaling_term == 0).astype(int)
    diff_group_scaling_term += samegroup_mask

    dists /= diff_group_scaling_term

    K = output_scale * np.exp(-0.5 * dists)
    K /= (diff_group_scaling_term) ** (0.5 * p)

    return K


def multigroup_matern12_covariance(kernel_params, X1, X2, groups1, groups2, group_distances):

    assert len(kernel_params) == 4
    output_scale = np.exp(kernel_params[0])
    group_diff_param = np.exp(kernel_params[1])
    lengthscales = np.exp(kernel_params[2])
    separability_param = np.exp(kernel_params[3])

    assert X1.shape[1] == X2.shape[1]
    assert groups1.shape[1] == groups2.shape[1]
    assert group_distances.shape[1] == groups1.shape[1]

    p = X1.shape[1]
    n_groups = groups1.shape[1]

    diffs = np.expand_dims(X1 / lengthscales, 1) - np.expand_dims(X2 / lengthscales, 0)
    # dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    dists = np.linalg.norm(diffs, axis=2, ord=2)
    # import ipdb; ipdb.set_trace()

    diff_group_indicator = (
        np.expand_dims(groups1, 1) - np.expand_dims(groups2, 0)
    ) ** 2
    diff_group_scaling_term = np.zeros(dists.shape)
    premult_term1 = np.zeros(dists.shape)
    premult_term2 = np.zeros(dists.shape)

    for ii in range(n_groups):

        for jj in range(ii):

            if ii == jj:
                continue

            curr_group_distance = group_distances[ii, jj]

            pairwise_group_indicator_matrix = (
                np.logical_and(
                    diff_group_indicator[:, :, ii] == 1,
                    diff_group_indicator[:, :, jj] == 1,
                )
            ).astype(
                int
            )

            a_times_d_sq = curr_group_distance * group_diff_param ** 2

            diff_group_scaling_term += (
                a_times_d_sq + 1
            ) * pairwise_group_indicator_matrix / (
            a_times_d_sq + separability_param)

            premult_term1 += (
                a_times_d_sq + 1
            ) * pairwise_group_indicator_matrix

            premult_term2 += (
                a_times_d_sq + separability_param
            ) * pairwise_group_indicator_matrix

    samegroup_mask = (diff_group_scaling_term == 0).astype(int)
    divisor = separability_param * samegroup_mask + np.logical_not(samegroup_mask)
    diff_group_scaling_term += (samegroup_mask / divisor)
    premult_term1 += samegroup_mask
    premult_term2 += samegroup_mask * separability_param

    premult_term_full = output_scale * separability_param**(0.5 * p) / (premult_term1**0.5 * premult_term2**(p * 0.5))

    dists *= (diff_group_scaling_term**0.5)

    K = premult_term_full * np.exp(-dists)
    # import ipdb; ipdb.set_trace()

    return K


def matern12_covariance(kernel_params, x, xp):

    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    c = 1.0

    assert x.shape[1] == xp.shape[1]
    p = x.shape[1]

    diffs = (
        np.expand_dims(x / lengthscales, 1)
        - np.expand_dims(xp / lengthscales, 0)
    )


    dists = np.linalg.norm(diffs, axis=2, ord=2)

    exponential_term = np.exp(-0.5 * dists)

    K = output_scale * exponential_term

    return K


# def mg_matern12_covariance(kernel_params, x, xp):

#     output_scale = np.exp(kernel_params[0])
#     group_diff_param = np.exp(kernel_params[1])  # + 0.0001
#     lengthscales = np.exp(kernel_params[2:])
#     c = 1.0

#     assert x.shape[1] == xp.shape[1]
#     p = x.shape[1] - 1

#     x_groups = x[:, -1]
#     x = x[:, :-1]
#     xp_groups = xp[:, -1]
#     xp = xp[:, :-1]

#     diffs = (
#         np.expand_dims(x / lengthscales, 1)
#         - np.expand_dims(xp / lengthscales, 0)
#         + 1e-6
#     )
#     # import ipdb ;ipdb.set_trace()
#     # dists = np.sqrt(np.sum(diffs**2, axis=2))
#     dists = np.linalg.norm(diffs, axis=2, ord=2)

#     diff_group_indicator = (
#         np.expand_dims(x_groups, 1) - np.expand_dims(xp_groups, 0)
#     ) ** 2
#     diff_group_scaling_term = diff_group_indicator * group_diff_param ** 2
#     # dists /= diff_group_scaling_term

#     exponential_term = np.exp(
#         -(diff_group_scaling_term + 1)
#         / (output_scale * c ** (0.5 * p) + c) ** 0.5
#         * dists
#     )
#     premult_term = (
#         output_scale
#         * c ** (0.5 * p)
#         / (
#             (diff_group_scaling_term + 1) ** 0.5
#             * (diff_group_scaling_term + c) ** (0.5 * c)
#         )
#     )

#     # K = output_scale * c**(0.5 * p) * (1 + np.sqrt(3) * dists) * np.exp(-np.sqrt(3) * dists)
#     # K /= ((diff_group_scaling_term)**(0.5 * p))

#     K = premult_term * exponential_term
#     # import ipdb ;ipdb.set_trace()

#     return K


def hierarchical_multigroup_kernel(
    params, X1, X2, groups1, groups2, group_distances, within_group_kernel, between_group_kernel
):

    diff_group_indicator = (
        np.expand_dims(groups1, 1) - np.expand_dims(groups2, 0)
    ) ** 2
    # same_group_mask = (np.sum(diff_group_indicator, axis=2) == 0).astype(int)

    # import ipdb; ipdb.set_trace()
    n_params_total = len(params)
    assert (n_params_total - 1) % 1 == 0
    n_params_between = (n_params_total - 1) // 2
    between_group_params = params[:n_params_between]
    within_group_params = params[n_params_between:]

    K_between = between_group_kernel(between_group_params, X1, X2)
    K_within = within_group_kernel(within_group_params, X1, X2, groups1=groups1, groups2=groups2, group_distances=group_distances)

    K = K_between + K_within

    return K

def hgp_kernel(
    params, X1, X2, groups1, groups2, within_group_kernel, between_group_kernel
):

    diff_group_indicator = (
        np.expand_dims(groups1, 1) - np.expand_dims(groups2, 0)
    ) ** 2
    same_group_mask = (np.sum(diff_group_indicator, axis=2) == 0).astype(int)

    n_params_total = len(params)
    within_group_params = params[: n_params_total // 2]
    between_group_params = params[n_params_total // 2 :]

    K_within = within_group_kernel(within_group_params, X1, X2)
    K_between = between_group_kernel(between_group_params, X1, X2)

    K = K_between + same_group_mask * K_within

    return K
