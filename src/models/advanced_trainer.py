"""
Advanced Trainer for NBA Prediction Models.
Implements adversarial validation and feature selection for robust model training.
GPU-accelerated implementations for faster training.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GPUAdversarialValidator:
    """GPU-accelerated adversarial validation using PyTorch MLP."""
    
    def __init__(self, input_dim: int, device=None):
        self.device = device
        self.model = None
        self.input_dim = input_dim
        self.is_fitted = False
        
        if device is None:
            try:
                from src.models.gpu_utils import get_device
                self.device = get_device()
            except ImportError:
                self.device = None
    
    def _init_model(self):
        """Lazy initialization of model."""
        if self.model is not None:
            return
        
        try:
            import torch
            import torch.nn as nn
            
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
            if self.device is not None and self.device.type == 'cuda':
                self.model = self.model.to(self.device)
                logger.info("GPUAdversarialValidator initialized on GPU")
            else:
                logger.info("GPUAdversarialValidator initialized on CPU")
                
        except ImportError:
            raise ImportError("PyTorch required for GPUAdversarialValidator")
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, 
            batch_size: int = 4096) -> 'GPUAdversarialValidator':
        """
        Train classifier to distinguish train from validation.
        
        Args:
            X: Feature matrix
            y: Labels (0=train, 1=val)
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            self
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        self._init_model()
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        if self.device is not None and self.device.type == 'cuda':
            X_tensor = X_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.model.train()
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.debug(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability that samples look like validation data."""
        import torch
        
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        if self.device is not None and self.device.type == 'cuda':
            X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy().flatten()
        
        return np.column_stack([1 - probs, probs])
    
    def get_sample_weights(self, X_train: np.ndarray) -> np.ndarray:
        """Get sample weights based on validation similarity."""
        probs = self.predict_proba(X_train)[:, 1]
        
        # Convert to weights: higher prob = more like validation = higher weight
        eps = 1e-6
        weights = probs / (1 - probs + eps)
        weights = np.clip(weights, 0.1, 10.0)
        weights = weights * len(weights) / weights.sum()
        
        return weights


class GPUFeatureSelector:
    """GPU-accelerated feature selection using L1 regularization."""
    
    def __init__(self, device=None):
        self.device = device
        self.model = None
        self.selected_features = None
        self.importance_scores = None
        
        if device is None:
            try:
                from src.models.gpu_utils import get_device
                self.device = get_device()
            except ImportError:
                pass
    
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        min_features: int = 20,
        threshold: float = 0.001
    ) -> List[str]:
        """
        Select features using L1-regularized linear model on GPU.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            min_features: Minimum number of features to keep
            threshold: Minimum importance threshold
        
        Returns:
            List of selected feature names
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        logger.info(f"GPU Feature Selection: Starting with {len(X.columns)} features")
        
        # Clean data
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Remove zero-variance features
        variances = X_clean.var()
        non_zero_var_cols = variances[variances > 1e-10].index.tolist()
        X_clean = X_clean[non_zero_var_cols]
        
        if len(non_zero_var_cols) < len(X.columns):
            logger.info(f"Removed {len(X.columns) - len(non_zero_var_cols)} zero-variance features")
        
        # Convert to tensors
        X_tensor = torch.tensor(X_clean.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)
        
        if self.device is not None and self.device.type == 'cuda':
            X_tensor = X_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)
        
        # L1-regularized linear model (Lasso-style)
        n_features = X_clean.shape[1]
        model = nn.Linear(n_features, 1, bias=True)
        
        if self.device is not None and self.device.type == 'cuda':
            model = model.to(self.device)
        
        # Use Adam with weight decay (proximal to L1)
        optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # Simple training loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            preds = model(X_tensor).squeeze()
            loss = criterion(preds, y_tensor)
            # Add explicit L1 penalty
            l1_penalty = 0.01 * torch.abs(model.weight).sum()
            total_loss = loss + l1_penalty
            total_loss.backward()
            optimizer.step()
        
        # Get feature importance from weights
        importance = torch.abs(model.weight).squeeze().cpu().detach().numpy()
        
        # Select features above threshold
        feature_names = X_clean.columns.tolist()
        
        # Sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Select top features
        n_select = max(min_features, (importance > threshold).sum())
        selected = importance_df.head(n_select)['feature'].tolist()
        
        self.selected_features = selected
        self.importance_scores = dict(zip(feature_names, importance))
        
        logger.info(f"GPU Feature Selection: Selected {len(selected)} features")
        
        return selected
    
    def get_importance_report(self) -> pd.DataFrame:
        """Get feature importance report."""
        if self.importance_scores is None:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.importance_scores.items()
        ]).sort_values('importance', ascending=False)


class GPUBlender:
    """GPU-accelerated ensemble blender using PyTorch."""
    
    def __init__(self, input_dim: int, device=None):
        self.device = device
        self.input_dim = input_dim
        self.model = None
        self.is_fitted = False
        
        if device is None:
            try:
                from src.models.gpu_utils import get_device
                self.device = get_device()
            except ImportError:
                pass
    
    def _init_model(self):
        """Initialize the linear model."""
        if self.model is not None:
            return
        
        import torch
        import torch.nn as nn
        
        self.model = nn.Linear(self.input_dim, 1, bias=True)
        
        if self.device is not None and self.device.type == 'cuda':
            self.model = self.model.to(self.device)
            logger.info(f"GPUBlender initialized on {self.device}")
        else:
            logger.info("GPUBlender initialized on CPU")
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 1e-2) -> 'GPUBlender':
        """Fit linear blender on GPU."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        self._init_model()
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        if self.device is not None and self.device.type == 'cuda':
            X_tensor = X_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.model(X_tensor).squeeze()
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using GPU blender."""
        import torch
        
        if not self.is_fitted:
            raise RuntimeError("Blender not fitted. Call fit() first.")
        
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        if self.device is not None and self.device.type == 'cuda':
            X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_tensor).squeeze().cpu().numpy()
        
        return preds
    
    @property
    def coef_(self) -> np.ndarray:
        """Get model coefficients (for compatibility with sklearn)."""
        import torch
        if self.model is None:
            return np.zeros(self.input_dim)
        return self.model.weight.cpu().detach().numpy().flatten()
    
    @property
    def intercept_(self) -> float:
        """Get model intercept (for compatibility with sklearn)."""
        import torch
        if self.model is None:
            return 0.0
        return self.model.bias.cpu().detach().numpy().item()


class AdvancedTrainer:
    """
    Advanced trainer implementing:
    1. Adversarial Validation - Detects train/val distribution shift and reweights samples
    2. Feature Selection - Uses L1 regularization for optimal feature subset
    
    Uses GPU acceleration when available.
    """
    
    def __init__(self, feature_cols: List[str], cat_features: Optional[List[str]] = None, use_gpu: bool = True):
        """
        Initialize the AdvancedTrainer.
        
        Args:
            feature_cols: List of feature column names
            cat_features: List of categorical feature names
            use_gpu: Whether to use GPU acceleration
        """
        self.feature_cols = feature_cols
        self.cat_features = cat_features or []
        self.use_gpu = use_gpu
        self.device = None
        self.adversarial_model = None
        self.selected_features: Optional[List[str]] = None
        
        if use_gpu:
            try:
                from src.models.gpu_utils import check_gpu_compatibility, get_device
                if check_gpu_compatibility():
                    self.device = get_device()
                    logger.info(f"AdvancedTrainer initialized with GPU: {self.device}")
                else:
                    logger.info("AdvancedTrainer initialized on CPU (GPU not available)")
            except ImportError:
                logger.info("AdvancedTrainer initialized on CPU (gpu_utils not found)")
    
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
        train_adv = train_df[self.feature_cols].copy()
        train_adv['_is_val'] = 0
        
        val_adv = val_df[self.feature_cols].copy()
        val_adv['_is_val'] = 1
        
        adv_df = pd.concat([train_adv, val_adv], ignore_index=True)
        
        # Handle missing values
        X_adv = adv_df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        y_adv = adv_df['_is_val'].values
        
        X_train = train_df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        
        # Use GPU model if available
        if self.use_gpu and self.device is not None and self.device.type == 'cuda':
            try:
                self.adversarial_model = GPUAdversarialValidator(
                    input_dim=X_adv.shape[1], 
                    device=self.device
                )
                self.adversarial_model.fit(X_adv, y_adv, epochs=50, batch_size=4096)
                
                # Get sample weights
                weights = self.adversarial_model.get_sample_weights(X_train)
                
                # Compute AUC manually
                probs = self.adversarial_model.predict_proba(X_adv)[:, 1]
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_adv, probs)
                
                logger.info(f"Adversarial Validation AUC (GPU): {auc:.4f}")
                
                if auc > 0.7:
                    logger.warning(
                        f"High AUC ({auc:.4f}) indicates significant train/val distribution shift. "
                        "Applying adversarial reweighting."
                    )
                else:
                    logger.info(
                        f"Low AUC ({auc:.4f}) indicates good train/val alignment. "
                        "Using uniform weights."
                    )
                    return np.ones(len(train_df))
                
                logger.info(f"Adversarial weights - min: {weights.min():.3f}, max: {weights.max():.3f}, "
                           f"mean: {weights.mean():.3f}")
                
                return weights
                
            except Exception as e:
                logger.warning(f"GPU adversarial validation failed: {e}. Falling back to CPU.")
        
        # CPU fallback using sklearn
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        self.adversarial_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validate to check separability
        cv_scores = cross_val_score(self.adversarial_model, X_adv, y_adv, cv=5, scoring='roc_auc')
        mean_auc = np.mean(cv_scores)
        
        logger.info(f"Adversarial Validation AUC (CPU): {mean_auc:.4f}")
        
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
        probs = self.adversarial_model.predict_proba(X_train)[:, 1]
        
        # Convert to weights
        eps = 1e-6
        weights = probs / (1 - probs + eps)
        weights = np.clip(weights, 0.1, 10.0)
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
        Select optimal features using GPU-accelerated L1 regularization.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            min_features: Minimum number of features to keep
            max_features: Maximum number of features (None = no limit)
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting best features from {len(X.columns)} candidates...")
        
        # Use GPU feature selector if available
        if self.use_gpu and self.device is not None and self.device.type == 'cuda':
            try:
                selector = GPUFeatureSelector(device=self.device)
                selected = selector.select_features(X, y, min_features=min_features)
                self.selected_features = selected
                self.importance_scores = selector.importance_scores
                return selected
            except Exception as e:
                logger.warning(f"GPU feature selection failed: {e}. Falling back to CPU.")
        
        # CPU fallback
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
            
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
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
            
            n_cv = min(5, len(X_clean) // 100) if len(X_clean) > 500 else 3
            
            rfecv = RFECV(
                estimator=estimator,
                step=max(1, len(X_clean.columns) // 20),
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