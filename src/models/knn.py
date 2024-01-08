from sklearn.impute import KNNImputer
import sys
sys.path.append('../')
from utils import *
import matplotlib.pyplot as plt

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. 

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("User Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. 

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Item Validation Accuracy: {}".format(acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    k = [1,6,11,16,21,26]
    acc = [[],[]]
    k_star_user = (None, float('-inf'))
    k_star_item = (None, float('-inf'))

    for x in k:
        score_user = knn_impute_by_user(sparse_matrix, val_data, x)
        score_item = knn_impute_by_item(sparse_matrix, val_data, x)

        if score_user > k_star_user[1]:
            k_star_user = (x, score_user)
        if score_item > k_star_item[1]:
            k_star_item = (x, score_item)

        acc[0].append(score_user)
        acc[1].append(score_item)

    k_star_test_user = knn_impute_by_user(sparse_matrix, test_data, k_star_user[0])
    k_star_test_item = knn_impute_by_item(sparse_matrix, test_data, k_star_item[0])
   
    print("k* for user is", k_star_user[0], "with final test accuracy of", k_star_test_user)
    print("k* for item is", k_star_item[0], "with final test accuracy of", k_star_test_item)
    
    plt.plot(k, acc[0], label='user')
    plt.plot(k, acc[1], label='item')
    plt.xticks(k)
    
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.title("accuracy on the validation data by collaborative filtering")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
