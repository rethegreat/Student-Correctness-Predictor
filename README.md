# Correctness-Predictor
A series of different machine learning model attempting to model if a student can correctly answer a diagnostic question based on their previous response. 

*Built by [Ivan Ye](https://github.com/ivanfye) and [Boaz Cheung](https://github.com/rethegreat)*

## Contents
- **k-Nearest Neighbor (kNN):** Analysis of kNN, comparing 'impute by user' and 'impute by item'. Includes accuracy data and discussion on assumptions and limitations.

- **Item Response Theory (IRT):** Mathematical formulations in IRT, training/validation log-likelihoods, and accuracy metrics.

- **Neural Networks:** Comparison of Alternating Least Squares and neural networks. Details on implementation and accuracy.

- **Ensemble Techniques:** Ensemble process, dataset creation, IRT model training, and performance evaluation.

- **Extensions and Modifications of the Neural Network:** Improvements using deep neural networks, Leaky ReLU activation, and Adam optimizer. Discusses limitations like sample size and hyperparameter sensitivity.

# Models
This section provides an examination of various machine learning models, such as k-Nearest Neighbor, Item Response Theory, Neural Networks, and Ensemble Techniques.

## k-Nearest Neighbour
kNN performs collaborative filtering using the other students' answers to predict whether the specific student can correctly answer some diagnostic questions. Both user-based and item-based collaborative filtering was used.
- **User-based assumption:** if student X has the same answers on other diagnositc questions as student Y, student X's correctness on specific diagnostic questions matches that of student Y.

- **Item-based assumption:** if question A is answered similarly by other sutdents as question B, question A's difficulty for specific students matches that of question B.

![knn](https://github.com/rethegreat/Correctness-Predictor/blob/main/src/img/knn_acc.png)

## Item Response Theory
IRT assigns each student an ability value, θ<sub>i</sub> for each student i, and each question a difficulty value, β<sub>j</sub> for each student j, to formulate a probability distribution.

## Ensemble
The ensemble process that was implemented was done through first creating 3 datasets from the given train data, each the same size as the original with replacement. The base model used was IRT, so there were 3 IRT models, each one trained on one of the bootstrapped training sets. Each IRT model then outputs its own θ and β which were each used to predict the correctness, using the given evaluate function, the average of which was taken.

# Extensions and Modifications of the neural network
This section discusses the enhancement of the algorithm with a deep neural network, addressing potential underfitting in earlier models. Key points include:
- **Deep Neural Network Implementation:** To better capture complex data relationships, hidden layers are added with Leaky ReLU activation functions.
- **Adaptive Moment Estimation (Adam) Optimizer:** Used for a robust approach with an adaptive learning rate and momentum, aiding in handling noisy gradients.
- **Utilizing CUDA for Training Optimization:** Leveraging GPU's parallel computing capabilities, the CUDA model is employed to manage the increased computational demands of this more complex model.


### Limitations
- **Small Sample Size:** The model's complexity increases the risk of overfitting due to a limited data sample.
- **Overfitting Mitigation:** Strategies include sampling more data, employing bagging techniques, and using a L2 regularizer.
- **Hyper-Parameter Sensitivity:** Numerous parameters (alpha, k, learning rate, num epoch, lambda) require careful tuning to avoid sub-optimal solutions.
- **Vulnerability to Adversarial Attacks:** The deep neural network's sensitivity to data changes makes it prone to such attacks. Regularizing gradients and introducing randomness in training can help mitigate this.

### Comparison figure with model and updated model
![Comparison figure](https://github.com/rethegreat/Correctness-Predictor/blob/main/src/img/model_comparison.png)

