from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.utils.data

import numpy as np

import matplotlib.pyplot as plt

import time
import torch
import torch.optim as optim


def load_data(device='cpu', base_path="./data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
    WHERE:
    zero_train_matrix: 2D sparse matrix where missing entries are
    filled with 0.
    train_data: 2D sparse matrix
    valid_data: A dictionary {user_id: list,
    user_id: list, is_correct: list}
    test_data: A dictionary {user_id: list,
    user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    train_data = load_train_csv(base_path)
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    if device == 'cuda':
        zero_train_matrix = torch.FloatTensor(zero_train_matrix).cuda()
        train_matrix = torch.FloatTensor(train_matrix).cuda()
    else:
        zero_train_matrix = torch.FloatTensor(zero_train_matrix)
        train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, train_data, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, alpha, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        self.g = nn.Sequential(
            nn.Linear(num_question, k),
            nn.LeakyReLU(alpha),
            nn.Linear(k, k)
        )

        self.h = nn.Sequential(
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, num_question)
        )


    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        g = self.g(inputs)

        h = self.h(torch.reciprocal(1 + torch.exp(g)))
        out = torch.reciprocal(1 + torch.exp(h))

        return out


def train(model, lr, lamb, train_data, zero_train_data, train, valid_data, test_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    valid_accuracies = []
    test_acc = 0
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = torch.isnan(train_data[user_id].unsqueeze(0))
            target[0][nan_mask[0]] = output[0][nan_mask[0]]

            g_params = list(model.g.parameters())
            h_params = list(model.h.parameters())

            # Calculate regularization term
            reg_term = (lamb / 2.) * (torch.sum(g_params[0] ** 2) + torch.sum(h_params[0] ** 2))

            loss = torch.sum((output - target) ** 2.) + reg_term
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_accuracies.append(valid_acc)

        if num_epoch - 1 == epoch:
            test_acc = evaluate(model, zero_train_data, test_data)
            print("Test Acc: " + str(test_acc))

    # print_acc(valid_accuracies, num_epoch)
    return valid_accuracies, test_acc


def print_acc(valid_accuracies, num_epoch):
    indicies = list(range(1, num_epoch+1))
    plt.plot(indicies, valid_accuracies, marker='o', linestyle='-', color='r', label='Validation Accuracy')

    plt.title('Training and Validation Accuracies Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        zero_train_matrix, train_matrix, train_data, valid_data, test_data = load_data("cuda")
    else:
        device = torch.device("cpu")
        zero_train_matrix, train_matrix, train_data, valid_data, test_data = load_data()

    # Set model hyperparameters.
    start_time = time.time()
    k = 100
    alpha = .01
    model = AutoEncoder(len(train_matrix[0]), alpha, k)
    model.to(device)
    # Set optimization hyperparameters.
    lr = .0002
    num_epoch = 30
    lamb = .005

    acc, test_acc = train(model, lr, lamb, train_matrix, zero_train_matrix, train_data,
    valid_data, test_data, num_epoch)
    end_time = time.time()

    # Calculate the total training time
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")

    return training_time, acc, test_acc, num_epoch


if __name__ == "__main__":
    main()