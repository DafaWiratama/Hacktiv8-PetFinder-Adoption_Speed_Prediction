from sklearn.model_selection import train_test_split


class Dataset:
    x = None
    y = None

    x_train, y_train = None, None
    x_val, y_val = None, None
    x_test, y_test = None, None

    def __init__(self, x, y, val_ratio=0.2, test_ratio=.5, stratify=True):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_ratio, stratify=y if stratify else None, random_state=13)
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=test_ratio, stratify=y_val if stratify else None, random_state=13)

        self.x, self.y = x, y
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test
