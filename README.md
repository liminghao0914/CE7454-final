# CE7454-final Project: Multifidelity-DeepONet for Data Center Thermal Prediction

Welcome to the repository of our CE7454-final project, where we delve into the implementation and evaluation of a cutting-edge Multifidelity-DeepONet model. This model integrates high-fidelity Computational Fluid Dynamics (CFD) simulations with real-time, low-fidelity sensor data to enhance thermal prediction accuracy in data centers. Our extensive experiments demonstrate that the Multifidelity-DeepONet outperforms other baseline methods in predicting and managing thermal conditions within data centers.

## Project Overview

Data centers are crucial to modern digital infrastructure, requiring efficient thermal management for operational stability and hardware longevity. Accurate thermal prediction within these facilities is challenging due to complex layouts, varying operational loads, and environmental conditions. Our project aims to tackle these challenges by employing a Multifidelity-DeepONet, which combines high-fidelity and low-fidelity data sources for improved predictive accuracy.

### Key Features
- **Integration of Multifidelity Data**: Combines detailed CFD simulations (high-fidelity) and real-time sensor readings (low-fidelity) for comprehensive thermal mapping.
- **Advanced Modeling Techniques**: Utilizes Deep Operator Networks (DeepONet) extended to handle multifidelity data, enhancing prediction capabilities.
- **Real-World Application**: Focuses on practical scenarios in data centers, addressing the need for effective thermal management.

### Experimentation and Results
We conducted various numerical experiments, including Poisson Equation and Reaction-Diffusion System simulations, to validate the efficacy of our model. The results highlight significant improvements in prediction accuracy over standard DeepONet models.

## Repository Structure
This repository contains all the necessary code, data samples, and documentation for our project. Here's a quick overview:

- `src/`: Source code of the Multifidelity-DeepONet implementation.
- `dataset/`: Sample high-fidelity and low-fidelity data used in our experiments.
- `results/`: Output results and visualizations from our experiments.

## Visualizations
![Predicted solution of three types of trained DeepONets in the test dataset. Data-A and Data-B are two random selected samples.](/results/3diffusion_reaction_don.png)
*Figure: Predicted solution of three types of trained DeepONets in the test dataset. Data-A and Data-B are two random selected samples.*

![The interpolation (Figure 4a, Figure 4b) and extrapola-tion (Figure 4c, Figure 4d) performance analysis of multifidelity-DeepONet, where we set α = 0.5.](/results/comparison.png)
*Figure: Performance comparison between Multifidelity-DeepONet and standard DeepONet models.*

![Poisson equation: Source term f(x) and solution u(x)](/results/poisson.png)
*Figure: Learning poison equation with two types of dataset (small-range high-fidelity dataset and large-range low-fidelity dataset) for Normal and Multi-fidelity DeepONet.*

## Getting Started
To get started with our project, clone the repository and follow the instructions in our documentation. Ensure you have the necessary dependencies installed, as listed in `requirements.txt`.

## Contributions
We welcome contributions and suggestions to improve our project. Please feel free to submit issues and pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Our sincere thanks to CE7454 course staff and participants.
- Special thanks to all contributors and researchers whose works have inspired this project.

---
For more detailed information and specific instructions, please refer to our project documentation and code comments.

---

