### Gamma Poisson Model
import numpy as np
from scipy.stats import poisson, gamma 
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(42)

# Define the Poisson likelihood function
def poisson_likelihood(params, data):
    lambd = params[0]
    log_likelihood = -np.sum(poisson.logpmf(data, mu=lambd))
    return log_likelihood

# Define the Gamma prior function
def gamma_prior(alpha, beta, lambd):
    return -gamma.logpdf(lambd, a=alpha, scale=1/beta)

# Define the MAP objective function
def map_objective(params, data):
    lambd = params[0]
    alpha, beta = params[1], params[2]

    # Negative log-likelihood
    likelihood_term = poisson_likelihood(params, data)

    # Negative log-prior
    prior_term = gamma_prior(alpha, beta, lambd)

    # Combine likelihood and prior
    map_objective = likelihood_term + prior_term

    return map_objective

df = pd.read_csv('./data/output_data.csv', index_col=0)
df = df[df['Province']=='BC']
y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))

# # Initial guess for the parameter
# initial_lambda = 1.0

# # Maximize the likelihood using the minimize function from scipy

# result = minimize(poisson_likelihood, [initial_lambda], args=(y,), method='L-BFGS-B')

# # Extract the MLE estimate of lambda
# mle_lambda = result.x[0]

# # Print the results
# print(f"Initial guess of lambda: {initial_lambda}")
# print(f"MLE estimate of lambda: {mle_lambda} which is also the Expectation of the Poisson distribution")
# print(f"Average of the data: {np.mean(y)}")


### MAP
initial_params = [1.0, 1.0, 1.0]    
result = minimize(map_objective, initial_params, args=(y,), method='L-BFGS-B')
lambda_map = result.x[0]
alpha_map = result.x[1]
beta_map = result.x[2]
print(f"MAP estimate of lambda: {lambda_map}")
print(f"MAP estimate of alpha: {alpha_map}")
print(f"MAP estimate of beta: {beta_map}")

posterior_samples = np.random.gamma(alpha_map + sum(y), 1 / (len(y) + beta_map), size=1000)

# Plot the posterior samples
# plt.hist(y, bins=30, density=True, alpha=0.5, color='green', label='Empirical distribution')
plt.hist(posterior_samples, bins=30, density=True, alpha=0.7, color='#ff7f0e', label='Posterior Samples')
plt.title('Posterior distribution of Gamma Poisson Model')
plt.xlabel('Sample')
plt.ylabel('Density')
plt.legend()
plt.savefig('./results/map_gamma_poisson.png')


# # True parameter value
# mle_lambda = np.mean(y)

# # Number of samples
# num_samples = 1000

# # Number of observations in each sample
# sample_size = 100

# # Generate samples from a Poisson distribution
# samples = np.random.poisson(mle_lambda, size=num_samples)

# # Calculate the mean of each sample
# sample_means = np.mean(samples, axis=1)

# # Plot the sampling distribution
# plt.hist(y, bins=30, density=True, alpha=0.7, label='Empirical distribution')
# plt.hist(samples, bins=30, density=True, alpha=0.7,color='#2ca02c', label='Sampling distribution')
# plt.axvline(mle_lambda, color='red', linestyle='dashed', linewidth=2, label='Mean')
# plt.title('Sampling Distribution of Poisson MLE')
# plt.xlabel('Samples')
# plt.ylabel('Density')
# plt.legend()
# plt.savefig('./results/poisson_sampling_distribution_cmp_BC.png')