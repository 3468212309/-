import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class DataPreprocessAgent:
    def __init__(self, config):
        self.config = config

    def denoise(self, hyperspectral_array):
        # 简单移动平均去噪（示例）
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(hyperspectral_array, size=3)

    def extract_features(self, df_clean):
        # 计算12个关键植被指数
        indices = {}
        # NDVI, PRI, PSRI, etc.
        indices['NDVI'] = (df_clean['NIR'] - df_clean['RED']) / (df_clean['NIR'] + df_clean['RED'] + 1e-6)
        # ... 其余指数
        return pd.DataFrame(indices)

    def multivariate_analysis(self, features_df):
        pca = PCA(n_components=0.95)
        reduced = pca.fit_transform(features_df)
        return reduced, pca
