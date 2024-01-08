import sys
sys.path.append('../')
from utils import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.

    # user | question | correct
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c = data['is_correct'][i]

        log_lklihood += c * np.log(sigmoid(theta[u] - beta[q])) + (1 - c) * np.log(1 - sigmoid(theta[u] - beta[q]))
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, iterations):
    """ Update theta and beta using gradient descent.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    for _ in range(iterations):
        d_theta = np.zeros_like(theta)
        d_beta = np.zeros_like(beta)
        for i, q in enumerate(data["question_id"]):
            u = data["user_id"][i]
            c = data['is_correct'][i]

            d_theta[u] += (c - sigmoid(theta[u]-beta[q])) 
            d_beta[q] += (-c + sigmoid(theta[u]-beta[q])) 

        theta += lr * d_theta
        beta += lr * d_beta

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.random.rand(len(set(data['user_id'])))
    beta = np.random.rand(len(set(data['question_id'])))
    val_acc_lst = []
    nllk = [[],[]]

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_valid = neg_log_likelihood(val_data, theta, beta)
        nllk[0].append(neg_lld_train)
        nllk[1].append(neg_lld_valid)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld_train, score))
        theta, beta = update_theta_beta(data, lr, theta, beta, iterations)

    return theta, beta, val_acc_lst, nllk


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.001
    iterations = 50
    iter = range(iterations)

    theta, beta, val_acc_list, nllk = irt(train_data, val_data, lr, iterations)
    valid_acc = evaluate(val_data,theta,beta)
    test_acc = evaluate(test_data, theta, beta)

    print("Final validation accuracy: " + str(valid_acc))
    print("Test accuracy: " + str(test_acc))
    plt.figure(1)
    plt.plot(iter, nllk[0], label='train')
    plt.plot(iter, nllk[1], label='validation')
    
    plt.xlabel("iteration")
    plt.ylabel("negative log-likelihood")
    plt.title("train and validation log-likelihoods by iteration")
    plt.legend()


    questions = [1, 2, 3]
    p = [[],[],[]]
    theta.sort()
    for n, j in enumerate(questions):
        p[n] = sigmoid(theta - beta[j])
    
    plt.figure(2)
    plt.plot(theta, p[0], label='j1')
    plt.plot(theta, p[1], label='j2')
    plt.plot(theta, p[2], label='j3')
    plt.ylabel("p(correct)")
    plt.xlabel("theta")
    plt.title("student ability vs p(correct)")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
