from item_response import *
import numpy as np

np.random.seed(42)

def bagging(data, m):
    ans = []
    size = len(data['user_id']) 

    for i in range(m):
        ind = np.random.choice(size, size, replace=True)
        temp = {}
        for key in data:
            temp[key] = [data[key][j] for j in ind]
        
        ans.append(temp)

    return ans

def evaluate(data, theta_sets, beta_sets):
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
        p_a = 0
        for num in range(len(theta_sets)):
            x = (theta_sets[num][u] - beta_sets[num][q])
            p_a += sigmoid(x)
        pred.append((p_a / len(theta_sets)) >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    m = 3
    data_sets = bagging(train_data, m)

    lr = 0.001
    iterations = 25

    theta_sets = []
    beta_sets = []
    for data in data_sets:
        t,b,_,_ = irt(data, val_data, lr, iterations)
        theta_sets.append(t)
        beta_sets.append(b)

    valid_acc = evaluate(val_data,theta_sets,beta_sets)
    test_acc = evaluate(test_data,theta_sets,beta_sets)

    print("Final validation accuracy: " + str(valid_acc))
    print("Test accuracy: " + str(test_acc))

if __name__ == "__main__":
    main()