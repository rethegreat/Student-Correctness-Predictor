# Correctness-Predictor
A series of different machine learning model attempting to model if a student can correctly answer a diagnostic question based on their previous response. 

*Built by [Ivan Ye](https://github.com/ivanfye) and [Boaz Cheung](https://github.com/rethegreat)*

## Contents
- **k-Nearest Neighbor (kNN):** Analysis of kNN, comparing 'impute by user' and 'impute by item'. Includes accuracy data and discussion on assumptions and limitations.

- **Item Response Theory (IRT):** Mathematical formulations in IRT, training/validation log-likelihoods, and accuracy metrics.

- **Neural Networks:** Comparison of Alternating Least Squares and neural networks. Details on implementation and accuracy.

- **Ensemble Techniques:** Ensemble process, dataset creation, IRT model training, and performance evaluation.

- **Extensions and Modifications:** Improvements using deep neural networks, Leaky ReLU activation, and Adam optimizer. Discusses limitations like sample size and hyperparameter sensitivity.


## Extensions and Modifications
This section discusses the enhancement of the algorithm with a deep neural network, addressing potential underfitting in earlier models. Key points include:
- **Deep Neural Network Implementation:** To better capture complex data relationships, hidden layers are added with Leaky ReLU activation functions.
- **Adaptive Moment Estimation (Adam) Optimizer:** Used for a robust approach with an adaptive learning rate and momentum, aiding in handling noisy gradients.
- **Utilizing CUDA for Training Optimization:** Leveraging GPU's parallel computing capabilities, the CUDA model is employed to manage the increased computational demands of this more complex model.


## Limitations
- **Small Sample Size:** The model's complexity increases the risk of overfitting due to a limited data sample.
- **Overfitting Mitigation:** Strategies include sampling more data, employing bagging techniques, and using a L2 regularizer.
- **Hyper-Parameter Sensitivity:** Numerous parameters (alpha, k, learning rate, num epoch, lambda) require careful tuning to avoid sub-optimal solutions.
- **Vulnerability to Adversarial Attacks:** The deep neural network's sensitivity to data changes makes it prone to such attacks. Regularizing gradients and introducing randomness in training can help mitigate this.

