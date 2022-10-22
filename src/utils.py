import pickle
import pandas as pd
import src.config as cfg
from sklearn import preprocessing
import category_encoders as ce


def save_as_pickle(df: pd.DataFrame, path: str) -> None:
    df.to_pickle(path)

def load_as_pickle(path: str) -> pd.DataFrame:
    data_from_pickle = pd.read_pickle(path)
    return data_from_pickle
    
def save_model_as_pickle(model, path:str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
        
def load_model_as_pickle(path:str):
    model = None
    
    with open(path, "rb") as f:
        model = pickle.load(f)
    
    return model


def label_encoder(columns, train, test : pd.DataFrame):
    train_copy = train.copy()
    test_copy = test.copy()
    for col in columns:
        train_copy[col] = train_copy[col].astype(str)
        train_copy[col] = preprocessing.LabelEncoder().fit_transform(train_copy[col])
        if test.empty == False:
            test_copy[col] = test_copy[col].astype(str)
            test_copy[col] = preprocessing.LabelEncoder().fit_transform(test_copy[col])
    if test.empty == False:
        return train_copy, test_copy
    return test_copy


class TargetEncoder:
    def __init__(self, cat_cols, target_cols, num_classes=1):
        self.te = [ce.TargetEncoder(drop_invariant=False)] * num_classes
        self.cat_cols = cat_cols
        self.num_classes = num_classes
        self.target_cols = target_cols
        
        
    def fit(self, x, y):
        for target_num in range(self.num_classes):
            target_col = self.target_cols[target_num]
            self.te[target_num].fit(x[self.cat_cols], y[target_col])

        
    def transform(self, x, y=None):
        target_encoded_x = x.copy()
        for target_num in range(self.num_classes):
            target_col = self.target_cols[target_num]
            encoded_cols = self.te[target_num].transform(x[self.cat_cols])
            
            target_encoded_x[[col + "/" + target_col for col in self.cat_cols]] = encoded_cols

        target_encoded_x = target_encoded_x.drop(self.cat_cols, axis=1)
        return target_encoded_x
            
    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
        

class LabelEncoder:
    def __init__(self, columns):
        self.columns = columns
        
        
    def fit(self, x, y=None):
        train_copy = x.copy()
        for col in self.columns:
            train_copy[col] = train_copy[col].astype(str)
            train_copy[col] = preprocessing.LabelEncoder().fit_transform(train_copy[col])
        return train_copy

        
    def transform(self, x, y=None):
        return self.fit(x)
            
    def fit_transform(self, x, y=None):
        return self.fit(x)