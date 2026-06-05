# -*- coding: utf-8 -*-
"""Unit tests for data_processing.py"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_processing import (
    AggregateFeatures, TemporalFeatures,
    RFMFeatures, ProxyTargetBuilder
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'CustomerId': ['C1','C1','C2','C2','C3'],
        'TransactionId': ['T1','T2','T3','T4','T5'],
        'Amount': [100, -50, 200, 300, 150],
        'Value': [100, 50, 200, 300, 150],
        'FraudResult': [0, 0, 1, 0, 0],
        'ProductId': ['P1','P2','P1','P3','P1'],
        'ChannelId': ['CH1','CH1','CH2','CH2','CH1'],
        'TransactionStartTime': [
            '2024-01-01','2024-01-15',
            '2024-02-01','2024-02-10','2024-03-01'
        ]
    })

def test_aggregate_features_columns(sample_df):
    """Test that AggregateFeatures returns expected columns."""
    result = AggregateFeatures().fit_transform(sample_df)
    expected = ['CustomerId','total_amount','avg_amount',
                'transaction_count','fraud_rate']
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"

def test_aggregate_features_count(sample_df):
    """Test that output has one row per customer."""
    result = AggregateFeatures().fit_transform(sample_df)
    assert len(result) == sample_df['CustomerId'].nunique()

def test_rfm_features_columns(sample_df):
    """Test RFM output has Recency, Frequency, Monetary columns."""
    builder = RFMFeatures()
    builder.fit(sample_df)
    result = builder.transform(sample_df)
    for col in ['Recency','Frequency','Monetary']:
        assert col in result.columns

def test_rfm_recency_non_negative(sample_df):
    """Test that Recency values are non-negative."""
    builder = RFMFeatures()
    builder.fit(sample_df)
    result = builder.transform(sample_df)
    assert (result['Recency'] >= 0).all()

def test_proxy_target_binary(sample_df):
    """Test that is_high_risk is binary (0 or 1)."""
    agg = AggregateFeatures().fit_transform(sample_df)
    rfm_b = RFMFeatures()
    rfm_b.fit(sample_df)
    rfm = rfm_b.transform(sample_df)
    merged = agg.merge(rfm, on='CustomerId')
    proxy = ProxyTargetBuilder(n_clusters=2)
    proxy.fit(merged)
    result = proxy.transform(merged)
    assert set(result['is_high_risk'].unique()).issubset({0, 1})

def test_temporal_features_columns(sample_df):
    """Test that temporal features returns expected columns."""
    result = TemporalFeatures().fit_transform(sample_df)
    for col in ['CustomerId','avg_hour','weekend_ratio']:
        assert col in result.columns