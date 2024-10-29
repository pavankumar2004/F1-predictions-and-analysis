import numpy as np
class CustomLabelEncoder:
    """Custom implementation of LabelEncoder"""
    
    def __init__(self):
        self.classes_ = None
        self.class_dict = None
    
    def fit(self, y):
        self.classes_ = np.unique(y)
        self.class_dict = {c: i for i, c in enumerate(self.classes_)}
        return self
    
    def transform(self, y):
        return np.array([self.class_dict[yi] for yi in y])
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        inverse_dict = {i: c for c, i in self.class_dict.items()}
        return np.array([inverse_dict[yi] for yi in y])
