from fairlearn.datasets import fetch_boston


def load_training_data():
    """Loading the Boston dataset, a data source notoriously famous for containing some serious ethical issues"""
    X, y = fetch_boston(return_X_y=True)
    return X, y


def train_model(X, y):
    pass


def score_model():
    pass
