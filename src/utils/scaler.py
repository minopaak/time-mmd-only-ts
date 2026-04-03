import numpy as np

class StandardScalerNP:
    """Standard scaler using numpy"""
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        """Fit scaler on data"""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # 0으로 나누는 것을 방지
        self.std[self.std == 0] = 1.0
        return self
    
    def transform(self, data):
        """Transform data using fitted mean and std"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted yet")
        return (data - self.mean) / self.std
    
    def fit_transform(self, data):
        """Fit and transform in one step"""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data):
        """Inverse transform scaled data"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted yet")
        return data * self.std + self.mean
    
    def inverse_transform_feature(self, data, feature_idx):
        """Inverse transform a specific feature"""
        result = data.copy()
        result[..., feature_idx] = (data[..., feature_idx] * self.std[feature_idx] + 
                                     self.mean[feature_idx])
        return result