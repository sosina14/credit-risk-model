# -*- coding: utf-8 -*-
"""
data_processing.py — Credit Risk Model
Full feature engineering pipeline using sklearn Pipeline.
Author: Sosina Ayele
"""

import logging
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)
RANDOM_STATE = 42


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Compute customer-level aggregate features from transaction data."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        logger.info("Computing aggregate features...")
        agg = X.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            std_amount=('Amount', 'std'),
            min_amount=('Amount', 'min'),
            max_amount=('Amount', 'max'),
            total_value=('Value', 'sum'),
            avg_value=('Value', 'mean'),
            transaction_count=('TransactionId', 'count'),
            unique_products=('ProductId', 'nunique'),
            unique_channels=('ChannelId', 'nunique'),
            fraud_count=('FraudResult', 'sum'),
            fraud_rate=('FraudResult', 'mean'),
        ).reset_index()
        agg['std_amount'] = agg['std_amount'].fillna(0)
        return agg


class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Extract temporal features from TransactionStartTime."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        logger.info("Extracting temporal features...")
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(
            df['TransactionStartTime'], errors='coerce')
        df['tx_hour'] = df['TransactionStartTime'].dt.hour
        df['tx_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
        df['is_weekend'] = (df['tx_dayofweek'] >= 5).astype(int)
        return df.groupby('CustomerId').agg(
            avg_hour=('tx_hour', 'mean'),
            avg_dayofweek=('tx_dayofweek', 'mean'),
            weekend_ratio=('is_weekend', 'mean'),
        ).reset_index()


class RFMFeatures(BaseEstimator, TransformerMixin):
    """Compute RFM metrics per customer."""
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date
    def fit(self, X, y=None):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(
            df['TransactionStartTime'], errors='coerce')
        self.snapshot_date_ = (pd.to_datetime(self.snapshot_date)
            if self.snapshot_date else df['TransactionStartTime'].max())
        return self
    def transform(self, X):
        logger.info("Computing RFM features...")
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(
            df['TransactionStartTime'], errors='coerce')
        return df.groupby('CustomerId').agg(
            Recency=('TransactionStartTime',
                lambda x: (self.snapshot_date_ - x.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Value', 'sum'),  # use Value (absolute) for monetary
        ).reset_index()


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Label encode categorical columns."""
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols or [
            'ProductCategory', 'ChannelId', 'ProviderId', 'CurrencyCode']
        self.encoders_ = {}
    def fit(self, X, y=None):
        for col in self.cat_cols:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders_[col] = le
        return self
    def transform(self, X):
        logger.info("Encoding categorical features...")
        df = X.copy()
        for col, le in self.encoders_.items():
            if col in df.columns:
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else 'unknown')
                if 'unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'unknown')
                df[col + '_encoded'] = le.transform(df[col].astype(str))
        return df


class ProxyTargetBuilder(BaseEstimator, TransformerMixin):
    """
    Build is_high_risk proxy target using KMeans clustering on RFM.
    High-risk cluster = lowest Frequency AND lowest Monetary value.
    """
    def __init__(self, n_clusters=3, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        rfm_cols = ['Recency', 'Frequency', 'Monetary']
        rfm_data = X[rfm_cols].copy().fillna(0)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(rfm_data)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans_.fit(scaled)

        # Get cluster centers in original scale
        centers = pd.DataFrame(
            self.scaler_.inverse_transform(self.kmeans_.cluster_centers_),
            columns=rfm_cols
        )
        logger.info(f"Cluster centers:\n{centers.round(2)}")

        # High risk = LOW frequency + LOW monetary (disengaged customers)
        # Normalize each metric and compute composite risk score
        centers['norm_freq'] = (centers['Frequency'] - centers['Frequency'].min()) / \
                                (centers['Frequency'].max() - centers['Frequency'].min() + 1e-9)
        centers['norm_mon'] = (centers['Monetary'] - centers['Monetary'].min()) / \
                               (centers['Monetary'].max() - centers['Monetary'].min() + 1e-9)
        # Low frequency + low monetary = high risk
        centers['engagement_score'] = centers['norm_freq'] + centers['norm_mon']
        self.high_risk_cluster_ = int(centers['engagement_score'].idxmin())
        logger.info(
            f"High-risk cluster: {self.high_risk_cluster_} "
            f"(Freq={centers.loc[self.high_risk_cluster_, 'Frequency']:.1f}, "
            f"Mon={centers.loc[self.high_risk_cluster_, 'Monetary']:.1f})"
        )
        return self

    def transform(self, X):
        logger.info("Assigning proxy target labels...")
        df = X.copy()
        rfm_data = df[['Recency', 'Frequency', 'Monetary']].fillna(0)
        scaled = self.scaler_.transform(rfm_data)
        df['cluster'] = self.kmeans_.predict(scaled)
        df['is_high_risk'] = (df['cluster'] == self.high_risk_cluster_).astype(int)
        count = df['is_high_risk'].sum()
        logger.info(
            f"High-risk customers: {count:,} ({count/len(df)*100:.1f}%)"
        )
        return df


def build_features(raw_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Full end-to-end feature engineering pipeline.
    Loads raw transactions, engineers features, builds proxy target.
    Returns model-ready customer-level DataFrame.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Data file not found: {raw_path}")

    logger.info(f"Loading data from: {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw data: {df.shape}")

    # Feature engineering
    agg = AggregateFeatures().fit_transform(df)
    temporal = TemporalFeatures().fit_transform(df)

    rfm_builder = RFMFeatures()
    rfm_builder.fit(df)
    rfm = rfm_builder.transform(df)

    cat_encoder = CategoricalEncoder()
    cat_encoder.fit(df)
    df_enc = cat_encoder.transform(df)
    encoded_cols = [c for c in df_enc.columns if c.endswith('_encoded')]
    cat_agg = df_enc.groupby('CustomerId')[encoded_cols].agg(
        lambda x: x.mode()[0] if len(x) > 0 else 0).reset_index()

    # Merge all customer-level features
    customer_df = agg.merge(temporal, on='CustomerId', how='left')
    customer_df = customer_df.merge(rfm, on='CustomerId', how='left')
    customer_df = customer_df.merge(cat_agg, on='CustomerId', how='left')

    # Build proxy target
    proxy = ProxyTargetBuilder()
    proxy.fit(customer_df)
    customer_df = proxy.transform(customer_df)

    # Handle missing values
    num_cols = customer_df.select_dtypes(include=[np.number]).columns
    customer_df[num_cols] = customer_df[num_cols].fillna(
        customer_df[num_cols].median())

    logger.info(f"Final dataset: {customer_df.shape}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        customer_df.to_csv(output_path, index=False)
        logger.info(f"Saved to: {output_path}")

    return customer_df


if __name__ == "__main__":
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw/data.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed/features.csv"
    df = build_features(raw, out)
    print(f"\nShape: {df.shape}")
    print(f"High-risk rate: {df['is_high_risk'].mean()*100:.1f}%")
    print(f"\nTarget distribution:\n{df['is_high_risk'].value_counts()}")
    print(f"\nSample:\n{df.head()}")
