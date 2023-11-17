### Gamma Poisson Model
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

def model(data):
    # Prior distributions for the Gamma parameters
    alpha = pyro.sample("alpha", dist.Exponential(torch.tensor(1.0)))
    beta = pyro.sample("beta", dist.Exponential(torch.tensor(1.0)))

    # Gamma-Poisson likelihood
    with pyro.plate("data", len(data)):
        lambda_ = pyro.sample("lambda", dist.Gamma(alpha, beta))
        pyro.sample("obs", dist.Poisson(lambda_), obs=data)

def guide(data):
    # Variational parameters
    alpha_q = pyro.param("alpha_q", torch.tensor(1.0), constraint=dist.constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(1.0), constraint=dist.constraints.positive)
    pyro.sample("alpha", dist.Delta(alpha_q))
    pyro.sample("beta", dist.Delta(beta_q))

# Synthetic data (optional)
true_alpha = 2.0
true_beta = 0.5
data = dist.Gamma(true_alpha, true_beta).sample([100])

# SVI for MLE
svi = SVI(model, guide, Adam({"lr": 0.005}), loss=Trace_ELBO())

n_steps = 1000
for step in range(n_steps):
    svi.step(data)

# Extract the estimated parameters
alpha_est = pyro.param("alpha_q").item()
beta_est = pyro.param("beta_q").item()
print(f"Estimated Alpha: {alpha_est}, Estimated Beta: {beta_est}")