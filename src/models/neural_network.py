import sys
sys.path.insert(0, '../')
from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import time
import numpy as np
import torch

import matplotlib.pyplot as plt

import torch.optim as optim


def load_data(base_path="../data"):
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
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)
    train_input = load_train_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, train_input, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # Implement the function as described in the docstring. #
        # Use sigmoid activations for f and g. #
        #####################################################################
        g = self.g(inputs)

        h = self.h(torch.reciprocal(1 + torch.exp(g)))
        out = torch.reciprocal(1 + torch.exp(h))

        #####################################################################
        # END OF YOUR CODE #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, train_input, valid_data, test_data, num_epoch):
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
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    valid_accuracies = []
    train_accuracies = []
    costs = []
    test_acc = 0
    for epoch in range(0, num_epoch):
        model.train()
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + (lamb/2.)*(model.get_weight_norm())
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        # train_acc = evaluate(model, zero_train_data, train_input)
        valid_accuracies.append(valid_acc)
        # train_accuracies.append(train_acc)
        costs.append(train_loss)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
        "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        if num_epoch - 1 == epoch:
            test_acc = evaluate(model, zero_train_data, test_data)
            print("Test Acc: " + str(test_acc))
    # print_loss(costs, num_epoch)
    # print_acc(valid_accuracies, train_accuracies, num_epoch)
    return valid_accuracies, test_acc
#####################################################################
# END OF YOUR CODE #
#####################################################################

def print_acc(valid_accuracies, train_accuracies, num_epoch):
    indicies = list(range(1, num_epoch+1))
    plt.plot(indicies, train_accuracies, marker='o', linestyle='-', color='b', label='Training Accuracy')
    plt.plot(indicies, valid_accuracies, marker='o', linestyle='-', color='r', label='Validation Accuracy')

    plt.title('Training and Validation Accuracies Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_loss(loss, num_epoch):
    indices = list(range(1, num_epoch+1))
    plt.plot(indices, loss, marker='o', linestyle='-', color='b')

    plt.title('Training Cost Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Cost')
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
    zero_train_matrix, train_matrix, train_input ,valid_data, test_data = load_data()

    #####################################################################
    # Try out 5 different k and select the best k using the #
    # validation set. #
    #####################################################################
    # Set model hyperparameters.
    start_time = time.time()
    k = 100
    model = AutoEncoder(len(train_matrix[0]), k)

    # Set optimization hyperparameters.
    lr = .01
    num_epoch = 30
    lamb = .01

    acc, test_acc = train(model, lr, lamb, train_matrix, zero_train_matrix, train_input,
    valid_data, test_data, num_epoch)

    end_time = time.time()

    # Calculate the total training time
    training_time = end_time - start_time

    print(f"Training time: {training_time} seconds")

    return training_time, acc, test_acc, num_epoch

#####################################################################
# END OF YOUR CODE #
#####################################################################


if __name__ == "__main__":
    main()

