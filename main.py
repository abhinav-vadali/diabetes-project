import logistic_regression
import decision_tree
import preprocessing
import evaluation

if __name__ == '__main__':

    diabetes_path = '/Users/abhinavvadali/Documents/PycharmProjects/tensorflow/Dataset of Diabetes .csv'
    data = preprocessing.load_data(diabetes_path)

    X, y = preprocessing.isolate_target(data, 'CLASS')
    train_X, test_X, train_y, test_y = preprocessing.split_data(X, y)

    # Using one-hot-encoding to make categorical variables continuous
    train_X = preprocessing.encode_categorical(train_X)
    test_X = preprocessing.encode_categorical(test_X)

    # We can do the decision tree first because we don't need to scale for decision tree
    decision_tree_model = decision_tree.fit_model(train_X, train_y)
    predictions_decision_tree = decision_tree_model.predict(test_X)
    # Returning results of decision tree
    print("\nDecision Tree Summary:")
    print(evaluation.interpret_results(predictions_decision_tree, test_y))
    print("---------------------------------------------------")

    train_X = preprocessing.scale_data(train_X)
    test_X = preprocessing.scale_data(test_X)

    regression_model = logistic_regression.fit_model(train_X, train_y)
    predictions_regression = regression_model.predict(test_X)

    # Printing results of logistic regression
    print("\nLogistic Regression Summary:")
    print(evaluation.interpret_results(predictions_regression, test_y))
    print("---------------------------------------------------")
