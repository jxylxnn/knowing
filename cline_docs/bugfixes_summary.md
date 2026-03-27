# NBA Prediction System - Critical Bug Fixes Summary

## Date: March 12, 2026

This document summarizes the critical bug fixes implemented during the forensic analysis of the NBA prediction system.

---

## CRITICAL FIXES IMPLEMENTED

### ISSUE #1: Data Leakage in Feature Engineering (CRITICAL - FIXED)

**Location:** `src/preprocessing/feature_engineer.py`, `_add_advanced_scoring_features()`

**Problem:** Features were computed using current game statistics without shifting, causing target leakage. The model learned from data it wouldn't have at prediction time.

**Fix Applied:**
- All game statistics (FGA, FTA, TOV, MIN, PTS, etc.) now use `.shift(1)` before feature computation
- Team possession estimates use rolling averages of shifted values
- Usage rate, efficiency metrics, and shot profiles all computed from historical data only

**Code Changes:**
```python
# Before (LEAKY):
player_poss = df['FGA'] + 0.44 * df['FTA'] + df['TOV']

# After (CORRECT):
shifted_fga = df.groupby('PLAYER_ID')['FGA'].shift(1)
shifted_fta = df.groupby('PLAYER_ID')['FTA'].shift(1)
shifted_tov = df.groupby('PLAYER_ID')['TOV'].shift(1)
player_poss = shifted_fga + 0.44 * shifted_fta + shifted_tov
```

---

### ISSUE #2: StackedEnsemble Time-Series Leakage (HIGH - FIXED)

**Location:** legacy ensemble model, `fit()` method

**Problem:** 
1. Player averages used `cumsum()` which included the current row, causing leakage
2. Time-series validation used `<=` instead of strict `<`, potentially allowing same-day games in both splits

**Fix Applied:**
```python
# Before (LEAKY):
player_cumsum = df_full.groupby('PLAYER_ID')[self.target_name].cumsum() - df_full[self.target_name]
player_avg = (player_cumsum / player_cumcount.replace(0, np.nan)).fillna(self.league_avg)

# After (CORRECT):
player_avg = df_full.groupby('PLAYER_ID')[self.target_name].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean()
).fillna(self.league_avg)

# Time-series validation fix:
# Before: assert train_max_date <= val_min_date
# After: assert train_max_date < val_min_date
```

---

### ISSUE #3: MultiOutputNN Uncertainty Calculation (MEDIUM - FIXED)

**Location:** legacy multi-output model, `predict()` method

**Problem:** Standard deviation was incorrectly computed using `var_` instead of `scale_`.

**Fix Applied:**
```python
# Before (INCORRECT):
stds = np.sqrt(vars_scaled) * np.sqrt(self.scaler_y.var_)

# After (CORRECT):
stds = np.sqrt(vars_scaled) * self.scaler_y.scale_
```

**Explanation:** When `y_scaled = (y - mean) / std`, the variance in scaled space needs to be multiplied by `std^2`, not `var`. The `scale_` attribute of StandardScaler is the standard deviation used in transformation.

---

### ISSUE #4: Quantile Calibration Method (MEDIUM - FIXED)

**Location:** `src/models/model_manager.py`, `_calibrate_quantile()`

**Problem:** The original implementation used a simple median shift, which is not correct for quantile calibration.

**Fix Applied:** Implemented proper coverage-based quantile calibration that:
- Checks actual coverage of predictions (what % of actuals fall below the prediction)
- Computes calibration shift based on percentile of residuals
- For P10: shifts to ensure ~10% of actuals are below the prediction
- For P90: shifts to ensure ~90% of actuals are below the prediction

---

### ISSUE #5: StackedEnsemble GPU Parameter Handling (MEDIUM - FIXED)

**Location:** legacy ensemble model

**Problem:** The `rf_proxy` model (which is XGBRegressor) wasn't receiving GPU parameters because the condition `if 'xgb' in name` didn't match `'rf_proxy'`.

**Fix Applied:**
```python
# Before:
if 'xgb' in name and self.use_gpu:
    model.set_params(device='cuda', tree_method='hist')

# After:
if self.use_gpu:
    if 'xgb' in name or name == 'rf_proxy':
        model.set_params(device='cuda', tree_method='hist')
```

---

## REMAINING ITEMS FOR FUTURE WORK

### Not Yet Fixed (Lower Priority)

1. **GNN Training on Aggregated Data** - The GNN model trains on player-level aggregated means, losing temporal information. Consider training on game-level sequences.

2. **Batch Prediction Feature Validation** - Add explicit validation that critical features exist before prediction.

3. **Epsilon Protection Standardization** - Create a consistent `EPS = 1e-7` constant for numerical stability throughout feature engineering.

4. **PositionalEncoding Dimension Mismatch** - For odd `d_model` values, there could be shape mismatches in the transformer.

---

## TESTING RECOMMENDATIONS

After these fixes, the following should be verified:

1. **Cross-validation scores should drop slightly** - This is expected because the model can no longer cheat with future information.

2. **Test set performance should improve** - Removing leakage means the model learns patterns that generalize better.

3. **Quantile intervals should be better calibrated** - Check that P10 intervals contain ~10% of actuals and P90 intervals contain ~90%.

4. **Uncertainty estimates should be more reliable** - The corrected std calculation should produce better calibrated prediction intervals.

---

## FILES MODIFIED

1. `src/preprocessing/feature_engineer.py` - Fixed data leakage in advanced scoring features
2. Legacy ensemble model - Fixed time-series leakage and GPU parameter handling
3. Legacy multi-output model - Fixed uncertainty calculation
4. `src/models/model_manager.py` - Fixed quantile calibration

---

## VERIFICATION

To verify these fixes work correctly:

```bash
# Run the test suite
python -m pytest tests/ -v

# Train a new model and compare performance
python train.py --mode standard

# Check that validation scores are realistic (not suspiciously high)
