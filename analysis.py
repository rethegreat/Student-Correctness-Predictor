import neural_network_b
import part_a.neural_network

import matplotlib.pyplot as plt

def main():
    nnb_time, nnb_acc, nnb_test_acc, num_epoch_b = neural_network_b.main()
    nn_time, nn_acc, nn_test_acc, num_epoch = part_a.neural_network.main()

    if num_epoch != num_epoch_b:
        print("please try again and compare with the same num_epoch")
    else:
        print_acc(nn_acc, nnb_acc, num_epoch)

    print("Running time of original model {} \t Running time of modified model {}".format(nn_time, nnb_time))

    print("Test accuracy of original model {} \t Test accuracy of modified model {}".format(nn_test_acc, nnb_test_acc))


def print_acc(accuracies, accuracies_b, num_epoch):
    indices = list(range(1, num_epoch + 1))
    plt.plot(indices, accuracies, marker='o', linestyle='-', color='b', label='Validation Accuracy of original model')
    plt.plot(indices, accuracies_b, marker='o', linestyle='-', color='r', label='Validation Accuracy of modified model')

    plt.title('Validation Accuracies of the two models Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()