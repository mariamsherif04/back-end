# Influenza SVEIR PDE Model with Vaccination and Spatial Dynamics

## Abstract
This report addresses the numerical solution of a one-dimensional partial differential equation (PDE) model describing the spatial and temporal dynamics of an influenza epidemic with vaccination. The model follows the SVEIR compartmental structure, representing the populations of **Susceptible**, **Vaccinated**, **Exposed**, **Infected**, and **Recovered** individuals as functions of both space and time.

Originally solved using the **Method of Lines (MOL)**, we extend the analysis by applying a range of alternative methods, including several **finite difference schemes**, a **Strang splitting method**, and a **deep learning approach**. The goal is to compare the **accuracy and efficiency** of each method in modeling influenza spread with vaccination and diffusion.

## I. Introduction
Influenza transmission is governed by multiple biological processes including infection, recovery, immunity loss, and vaccination. While classical compartmental models typically address only the **temporal dynamics** of disease spread, **spatial heterogeneity**—such as population movement and localized outbreaks—can significantly influence epidemic behavior.

To capture these effects, this work models influenza using a **spatiotemporal SVEIR framework**, represented by a system of coupled nonlinear partial differential equations (PDEs). The model tracks the densities of Susceptible (S), Vaccinated (V), Exposed (E), Infected (I), and Recovered (R) individuals as functions of both time and space.

Originally solved using the **Method of Lines (MOL)**, which converts the PDE system into ODEs by discretizing space, the model provides insight into how vaccination and diffusion influence epidemic spread.

In this project, we extend the analysis by comparing four approaches:
- The **Explicit Finite Difference Method**
- The **Crank–Nicolson Method**
- The **Strang Splitting Method**
- A **Deep Learning model** trained to approximate PDE solutions

Our goal is to assess the **accuracy**, **stability**, and **practicality** of each method for modeling spatiotemporal disease dynamics.

## II. Literature Review
Recent research has explored various extensions of classical epidemic models to capture the spatial and temporal complexity of influenza transmission.

One notable study by **Chen and Xu (2019)** developed a mathematical model involving two virus strains and their corresponding vaccines. Their model incorporates **spatial diffusion** and **time delays** to more accurately reflect the incubation period and population movement. Formulated as a system of **delayed PDEs**, the study focused on analyzing the **global stability** of disease-free and endemic equilibria using **Lyapunov functionals**, with the theoretical results verified through numerical simulations.

Another relevant work by **Jang et al. (2020)** examined an **optimal control problem** for an SIR reaction–diffusion model under constraints on **vaccine availability**. The model accounts for spatial spread and realistic limitations on vaccination rates and coverage. The authors formulated a **constrained optimization problem** aimed at minimizing both the infection burden and vaccination costs. They used a **penalty method** to address inequality constraints and applied a **gradient-based algorithm** to solve the resulting system. Their numerical experiments compared optimal vaccination strategies in both homogeneous and heterogeneous spatial settings, highlighting the role of spatial structure in disease control planning.

These studies demonstrate the increasing use of **PDE-based models** in epidemiology, especially in analyzing the **spatial impact** of vaccination strategies and optimizing public health interventions.

Our work builds on this foundation by exploring **numerical and data-driven solutions** to a spatial SVEIR influenza model with vaccination and diffusion.
## III. Model Formulation

This study models the spread of influenza using a one-dimensional **reaction–diffusion system** based on the **SVEIR** framework. The model captures both **temporal and spatial dynamics** of disease transmission and vaccination.

Five state variables are tracked as functions of space \( x \in [-3, 3] \) and time \( t \geq 0 \):

- \( S(x, t) \): Susceptible population  
- \( V(x, t) \): Vaccinated population  
- \( E(x, t) \): Exposed (latent) population  
- \( I(x, t) \): Infected population  
- \( R(x, t) \): Recovered population  

### PDE System

The model is governed by a system of nonlinear **partial differential equations**, where each compartment evolves due to biological processes (infection, recovery, vaccination, etc.) and **spatial diffusion**:

\[
\frac{\partial S}{\partial t} = -\beta \beta_E E S - \beta \beta_I I S + \alpha I S - \phi S - rS + \delta R + \theta V + r + d_1 \frac{\partial^2 S}{\partial x^2} \tag{1}
\]

\[
\frac{\partial V}{\partial t} = -\beta \beta_E \beta_V E V - \beta \beta_I \beta_V I V + \alpha I V - rV - \theta V + \phi S + d_2 \frac{\partial^2 V}{\partial x^2} \tag{2}
\]

\[
\frac{\partial E}{\partial t} = \beta \beta_E E S + \beta \beta_I I S + \beta \beta_E \beta_V E V + \beta \beta_I \beta_V I V + \alpha I E - (r + \kappa + \sigma) E + d_3 \frac{\partial^2 E}{\partial x^2} \tag{3}
\]

\[
\frac{\partial I}{\partial t} = \sigma E - (r + \alpha + \gamma) I + \alpha I^2 + d_4 \frac{\partial^2 I}{\partial x^2} \tag{4}
\]

\[
\frac{\partial R}{\partial t} = \kappa E + \gamma I - r R - \delta R + \alpha I R + d_5 \frac{\partial^2 R}{\partial x^2} \tag{5}
\]

---

### Boundary and Initial Conditions

To simulate a **closed spatial domain**, Neumann (no-flux) boundary conditions are applied at \( x = \pm 3 \) for all variables:

\[
\frac{\partial U}{\partial x} \Big|_{x = -3}^{x = 3} = 0,\quad \text{for } U = S, V, E, I, R
\]

The **initial population distributions** are defined as:

\[
S(x,0) = 0.86 \exp\left(-\left(\frac{x}{1.4}\right)^2\right), \quad
V(x,0) = 0.10 \exp\left(-\left(\frac{x}{1.4}\right)^2\right)
\]

\[
E(x,0) = 0, \quad
I(x,0) = 0.04 \exp(-x^2), \quad
R(x,0) = 0
\]

---

### Parameter Values

| Parameter | Description                         | Value         |
|----------:|-------------------------------------|---------------|
| \( \beta \)      | Contact rate                         | 0.5140        |
| \( \beta_E \)    | Infectiousness of exposed            | 0.25          |
| \( \beta_I \)    | Infectiousness of infected           | 1             |
| \( \beta_V \)    | Vaccine protection factor            | 0.9           |
| \( \sigma^{-1} \) | Latent period (days)                | 2             |
| \( \gamma^{-1} \) | Recovery period (days)              | 5             |
| \( \delta^{-1} \) | Immunity loss duration (days)       | 365           |
| \( r \)          | Birthrate                            | \( 7.14 \times 10^{-5} \) |
| \( \alpha \)     | Flu-induced mortality rate           | \( 9.3 \times 10^{-6} \) |
| \( \kappa \)     | Latent recovery rate                 | \( 1.857 \times 10^{-4} \) |
| \( \theta^{-1} \)| Vaccine immunity duration (days)     | 365           |
| \( \phi \)       | Vaccination rate                     | 0.05          |
| \( d_1, d_2 \)   | Diffusivity for S, V                 | 0.05          |
| \( d_3 \)        | Diffusivity for E                    | 0.025         |
| \( d_4 \)        | Diffusivity for I                    | 0.001         |
| \( d_5 \)        | Diffusivity for R                    | 0             |
---

