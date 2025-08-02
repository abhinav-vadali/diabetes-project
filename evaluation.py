from sklearn.metrics import accuracy_score

def interpret_results(predictions, test_y):
    """Creates an interpretation of the predictions in a readable format

    Args:
        predictions: a list of predictions for the output feature made by the model
        test_y: a list containing the testing data from the output feature

    Returns: A string

    """
    num_yes = 0
    num_no = 0
    num_prediabetic = 0
    for i in predictions:
        if i == 'Y':
            num_yes += 1
        elif i == 'N':
            num_no += 1
        elif i == 'P':
            num_prediabetic += 1
    accuracy = accuracy_score(predictions, test_y) * 100
    return (f"The number of people that likely have diabetes (Y) is {num_yes} people"
            f"\nThe number of people that likely don't have diabetes (N) is: {num_no} people"
            f"\nThe number of people that are likely prediabetic (P) is {num_prediabetic} people"
            f"\nThe accuracy of the model is approximately {accuracy:.2f}%")