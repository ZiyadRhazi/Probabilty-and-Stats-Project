# 2. Mathematical Modeling & Calculus Derivations

## 2.1 The Bayesian Inference Problem
Our goal is to estimate the state vector $x_k$ (position and velocity) given noisy measurements $z_{1:k}$. By Bayes' Theorem:

$$P(x_k | z_{1:k}) = \frac{P(z_k | x_k) P(x_k | z_{1:k-1})}{\int P(z_k | x_k) P(x_k | z_{1:k-1}) dx_k}$$

Where:
* $P(z_k | x_k)$ is the **Likelihood** (Sensor Model).
* $P(x_k | z_{1:k-1})$ is the **Prior** (predicted from previous state).
* The denominator is the **Evidence** (marginal likelihood), calculated via integration over the state space.

## 2.2 Continuous Time Kinematics
We model the vehicle using continuous calculus-based kinematics:
$$\frac{dx}{dt} = v, \quad \frac{dv}{dt} = a$$

Integrating over time step $\Delta t$:
$$x(t + \Delta t) = \int_{t}^{t+\Delta t} (v(\tau)) d\tau = x(t) + v(t)\Delta t + \frac{1}{2}a \Delta t^2$$

## 2.3 Sensor Fusion Model
We assume Gaussian noise for sensors. For a sensor $i$ (e.g., LiDAR), the likelihood function is:
$$f(z_i | x) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left( -\frac{(z_i - x)^2}{2\sigma_i^2} \right)$$

In our PyMC model, we fuse multiple sensors by multiplying their likelihoods (assuming independence):
$$P(Z | x) \propto \prod_{sensor \in \{LiDAR, Radar, Camera\}} \mathcal{N}(z_{sensor} | x, \sigma_{sensor}^2)$$
