import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from multigroupGP import GP, rbf_kernel

key = random.PRNGKey(1)

true_params = [
    jnp.zeros(1),  # Amplitude
    jnp.zeros(1),  # Lengthscale
]

noise_variance = 0.05
n = 50
p = 1
X = random.uniform(key, minval=-10, maxval=10, shape=(n, 1))

K_XX = (
    rbf_kernel(
        true_params,
        x1=X,
    )
    + noise_variance * jnp.eye(n)
)
Y = random.multivariate_normal(random.PRNGKey(12), jnp.zeros(n), K_XX)

Xtest = jnp.linspace(-10, 10, 200)[:, None]
gp = GP(kernel=rbf_kernel, key=key)
gp.fit(X, Y)
preds_mean = gp.predict(Xtest)

plt.figure(figsize=(10, 5))
plt.scatter(X.squeeze(), Y, color="black", label="Data")
plt.plot(Xtest, preds_mean, color="red", label="Predictions")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.savefig("./images/gp_preds.png")
plt.show()
