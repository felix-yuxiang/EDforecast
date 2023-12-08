import numpy as np
from scipy.stats import poisson, gamma
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

df = pd.read_csv('./data/output_data.csv', index_col=0)
df = df[df['Province']=='BC']
y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))
data = y

# Define the Poisson likelihood function
def poisson_likelihood(params, data):
    lambd = params[0]
    log_likelihood = np.sum(poisson.logpmf(data, mu=lambd))
    return log_likelihood

# Define the Gamma prior function
def gamma_prior(params, alpha, beta):
    lambd = params[0]
    log_prior = gamma.logpdf(lambd, a=alpha, scale=1/beta)
    return log_prior

# Define the log posterior function
def log_posterior(params, data, alpha, beta):
    return poisson_likelihood(params, data) + gamma_prior(params, alpha, beta)

# Hyperparameters for the Gamma prior
alpha = 1.0273036877
beta = 0.5967806138508519

# Number of posterior samples
num_samples = 100

# # Sample from the posterior
# posterior_samples = np.zeros(num_samples)
# for i in range(num_samples):
#     # Sample lambda from the posterior Gamma distribution
#     lambda_sample = np.random.gamma(np.sum(data) + alpha, 1 / (len(data) + beta))
#     posterior_samples[i] = lambda_sample

# Generate predictive samples for new data points
# num_new_data_points = 200
# predictive_samples = np.zeros((num_samples, num_new_data_points))

# for i in range(num_samples):
#     # For each posterior sample of lambda, generate new data points
#     predictive_samples[i, :] = np.random.poisson(posterior_samples[i], size=num_new_data_points)

HIST_BINS = 30
# Plot the sampling distribution of the predictive posterior

def update(frame):
    ax.clear()


    # Sample from the posterior
    posterior_samples = np.zeros(num_samples)
    for i in range(num_samples):
        # Sample lambda from the posterior Gamma distribution
        lambda_sample = np.random.gamma(np.sum(data) + alpha, 1 / (len(data) + beta))
        posterior_samples[i] = lambda_sample

    num_new_data_points = len(y)
    predictive_samples = np.zeros((num_samples, num_new_data_points))
    # Simulate random samples from the population
    for i in range(num_samples):
            predictive_samples[i, :] = np.random.poisson(posterior_samples[i], size=num_new_data_points)
    samples = predictive_samples[frame, :]


    # Plot the sampling distribution
    ax.hist(y, bins=30, density = True, alpha=0.5, label='Empirical distribution')
    ax.hist(samples, bins=10, density = True ,label = "Predictive Posterior Samples", ec="yellow", fc="green")
    ax.legend()
    ax.set_title('Sampling Distribution of Predictive Posterior vs Data distribution')
    ax.set_xlabel('New Data Points vs Data points')
    ax.set_ylabel('Density')

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, 100, repeat=True, blit=False)

ani.save('./results/predictive_posterior.gif', writer='pillow', fps=5)

# plt.hist(y, bins=30, density=True, alpha=0.5, color='blue', label='Empirical distribution')
# plt.hist(predictive_samples.flatten(), bins=30, density=True, alpha=0.5, color='green', label='Predictive Posterior Samples')
# plt.title('Sampling Distribution of Predictive Posterior vs Data distribution')
# plt.xlabel('New Data Points vs Data points')
# plt.ylabel('Density')
# plt.legend()
# plt.savefig('./results/poisson_gamma_predictive_BC_cmp.png')