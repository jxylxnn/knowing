"""
Advanced Trainer for NBA Prediction Models.
Implements adversarial validation and feature selection for robust model training.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """
    Advanced trainer implementing:
    1. Adversarial Validation - Detects train/val distribution shift and reweights samples
    2. Feature Selection - Uses recursive feature elimination for optimal feature subset
    """
    
    def __init__(self, feature_cols: List[str], cat_features: Optional[List[str]] = None):
        """
        Initialize the AdvancedTrainer.
        
        Args:
            feature_cols: List of feature column names
            cat_features: List of categorical feature names
        """
        self.feature_cols = feature_cols
        self.cat_features = cat_features or []
        self.use_gpu = False
        self.adversarial_model = None
        self.selected_features: Optional[List[str]] = None
        
    def perform_adversarial_validation(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Adversarial Validation: Train a classifier to distinguish train from validation.
        
        If the classifier can easily distinguish them, it means there's a distribution shift.
        We use the classifier's predictions to upweight training samples that look like validation.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            
        Returns:
            np.ndarray: Sample weights for training data
        """
        logger.info("Performing Adversarial Validation...")
        
        # Prepare adversarial dataset
        # Label: 0 = train, 1 = validation
        train_adv = train_df[self.feature_cols].copy()
        train_adv['_is_val'] = 0
        
        val_adv = val_df[self.feature_cols].copy()
        val_adv['_is_val'] = 1
        
        adv_df = pd.concat([train_adv, val_adv], ignore_index=True)
        
        # Handle missing values
        X_adv = adv_df[self.feature_cols].fillna(0)
        y_adv = adv_df['_is_val']
        
        # Train adversarial classifier
        self.adversarial_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validate to check separability
        cv_scores = cross_val_score(self.adversarial_model, X_adv, y_adv, cv=5, scoring='roc_auc')
        mean_auc = np.mean(cv_scores)
        
        logger.info(f"Adversarial Validation AUC: {mean_auc:.4f}")
        
        if mean_auc > 0.7:
            logger.warning(
                f"High AUC ({mean_auc:.4f}) indicates significant train/val distribution shift. "
                "Applying adversarial reweighting."
            )
        else:
            logger.info(
                f"Low AUC ({mean_auc:.4f}) indicates good train/val alignment. "
                "Using uniform weights."
            )
            return np.ones(len(train_df))
        
        # Fit the full model
        self.adversarial_model.fit(X_adv, y_adv)
        
        # Get probability that each training sample looks like validation
        X_train = train_df[self.feature_cols].fillna(0)
        probs = self.adversarial_model.predict_proba(X_train)[:, 1]
        
        # Convert to weights: higher prob = more like validation = higher weight
        # We want to focus on training samples that resemble the validation set
        # Weight formula: w = prob / (1 - prob + eps), then normalize
        eps = 1e-6
        weights = probs / (1 - probs + eps)
        
        # Clip extreme weights
        weights = np.clip(weights, 0.1, 10.0)
        
        # Normalize to sum to N (same as uniform weights)
        weights = weights * len(weights) / weights.sum()
        
        logger.info(f"Adversarial weights - min: {weights.min():.3f}, max: {weights.max():.3f}, "
                   f"mean: {weights.mean():.3f}")
        
        return weights
    
    def select_best_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        min_features: int = 20,
        max_features: Optional[int] = None
    ) -> List[str]:
        """
        Select optimal features using Recursive Feature Elimination with Cross-Validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            min_features: Minimum number of features to keep
            max_features: Maximum number of features (None = no limit)
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting best features from {len(X.columns)} candidates...")
        
        # Handle missing values
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Quick filter: remove zero-variance features
        variances = X_clean.var()
        non_zero_var_cols = variances[variances > 1e-10].index.tolist()
        X_clean = X_clean[non_zero_var_cols]
        
        if len(non_zero_var_cols) < len(X.columns):
            logger.info(f"Removed {len(X.columns) - len(non_zero_var_cols)} zero-variance features")
        
        # If we have too many features, do a quick pre-selection with Random Forest
        if len(X_clean.columns) > 100:
            logger.info("Performing quick pre-selection with Random Forest importance...")
            
            rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
            # Convert regression target to classification buckets for feature selection
            y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
            
            try:
                rf.fit(X_clean, y_binned)
                importances = pd.Series(rf.feature_importances_, index=X_clean.columns)
                top_features = importances.nlargest(100).index.tolist()
                X_clean = X_clean[top_features]
                logger.info(f"Pre-selected top {len(top_features)} features by RF importance")
            except Exception as e:
                logger.warning(f"RF pre-selection failed: {e}. Using all features.")
        
        # RFECV for final selection
        try:
            from sklearn.linear_model import Ridge
            from sklearn.feature_selection import RFECV
            
            estimator = Ridge(alpha=1.0)
            
            # Limit CV folds for speed
            n_cv = min(5, len(X_clean) // 100) if len(X_clean) > 500 else 3
            
            rfecv = RFECV(
                estimator=estimator,
                step=max(1, len(X_clean.columns) // 20),  # Remove ~5% of features per step
                cv=n_cv,
                scoring='neg_mean_squared_error',
                min_features_to_select=min_features,
                n_jobs=-1
            )
            
            rfecv.fit(X_clean, y)
            
            selected_mask = rfecv.support_
            self.selected_features = X_clean.columns[selected_mask].tolist()
            
            logger.info(f"RFECV selected {len(self.selected_features)} features")
            
        except Exception as e:
            logger.warning(f"RFECV failed: {e}. Using all preprocessed features.")
            self.selected_features = X_clean.columns.tolist()
        
        # Enforce limits
        if max_features and len(self.selected_features) > max_features:
            self.selected_features = self.selected_features[:max_features]
            logger.info(f"Truncated to max {max_features} features")
        
        if len(self.selected_features) < min_features:
            # If we ended up with too few, use top features by variance
            additional = [c for c in non_zero_var_cols if c not in self.selected_features]
            additional = additional[:min_features - len(self.selected_features)]
            self.selected_features.extend(additional)
            logger.info(f"Padded to min {min_features} features")
        
        return self.selected_features
    
    def get_feature_importance_report(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Generate a feature importance report using multiple methods.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LassoCV
        
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        importance_df = pd.DataFrame(index=X_clean.columns)
        
        # 1. Random Forest importance
        try:
            rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
            rf.fit(X_clean, y)
            importance_df['rf_importance'] = rf.feature_importances_
        except Exception as e:
            logger.warning(f"RF importance failed: {e}")
            importance_df['rf_importance'] = 0
        
        # 2. Lasso coefficients
        try:
            lasso = LassoCV(cv=3, random_state=42, n_jobs=-1)
            lasso.fit(X_clean, y)
            importance_df['lasso_coef'] = np.abs(lasso.coef_)
        except Exception as e:
            logger.warning(f"Lasso importance failed: {e}")
            importance_df['lasso_coef'] = 0
        
        # 3. Correlation with target
        importance_df['correlation'] = X_clean.apply(lambda col: abs(col.corr(y)))
        
        # Aggregate score
        importance_df['aggregate_score'] = (
            importance_df['rf_importance'].rank(pct=True) * 0.4 +
            importance_df['lasso_coef'].rank(pct=True) * 0.3 +
            importance_df['correlation'].rank(pct=True) * 0.3
        )
        
        return importance_df.sort_values('aggregate_score', ascending=False)
