# Generative Adversarial Networks (GANs)

## Overview
This repository contains implementations of various types of Generative Adversarial Networks (GANs) for different datasets. GANs are a class of machine learning frameworks introduced by Ian Goodfellow and his colleagues in 2014. They consist of two neural networks, a generator and a discriminator, which are trained simultaneously through a game-theoretic framework. The generator learns to produce data (e.g., images) that is indistinguishable from real data, while the discriminator learns to differentiate between real and fake data.

## Files
### 1. GAN.py
- **Description**: This file contains the implementation of a basic GAN for the MNIST dataset. It includes functions to build the generator and discriminator models, as well as training functions.
- **Models**: GAN model
- **Datasets**: MNIST

### 2. GAN_Model.py
- **Description**: Similar to WGAN_Model.py, this file implements a GAN using a custom model subclass with a custom training step. It includes functions to build the generator and critic models, as well as training functions.
- **Models**: GAN model
- **Datasets**: cars

### 3. GAN_Model_conditional.py
- **Description**: This file extends the GAN implementation to a conditional setting, where the generator takes both random noise and class labels as input. It includes functions to build conditional generator and critic models, as well as training functions.
- **Models**: Conditional GAN model
- **Datasets**: cats vs. dogs

### 4. WGAN_Model.py
- **Description**: This file implements a Wasserstein GAN (WGAN) using a custom model subclass with a custom training step. It includes functions to build the generator and critic models, as well as training functions.
- **Models**: WGAN model
- **Datasets**: cars

### 5. WGAN_Model_conditional.py
- **Description**: This file extends the WGAN implementation to a conditional setting, where the generator takes both random noise and class labels as input. It includes functions to build conditional generator and critic models, as well as training functions.
- **Models**: Conditional WGAN model
- **Datasets**: cats vs. dogs

## Usage
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the desired GAN implementation file (e.g., GAN.py, WGAN_Model.py) to train the model on the corresponding dataset.
4. Optionally, customize the model architecture, hyperparameters, and training process as needed.

## Datasets
- **MNIST**: A dataset of handwritten digits commonly used for training and testing machine learning models.
- **Custom datasets**: These are custom datasets used in the provided implementations, such as images of cars and a dataset of cats vs. dogs.

## Contributions
Contributions to this repository are welcome. If you have any suggestions, improvements, or additional implementations of GANs, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
