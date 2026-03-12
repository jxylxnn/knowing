# Bug Fix Plan: GroupBy Series KeyError in Feature Engineering

## Executive Summary

A critical bug exists in [`_add_advanced_scoring_features()`](src/preprocessing/feature_engineer.py:1003) where computed pandas Series objects are incorrectly passed to `groupby()[...]` as column selectors instead of being used directly. This causes a `KeyError` when pandas tries to interpret the numeric values in the Series as column names.

---

## 1. Task Interpretation

### The Real Objective
Fix the `KeyError: 'Columns not found: 0.0, 0.5, 0.19789549120832817...'` error that occurs during training when feature engineering runs.

### Root Cause Analysis
The error occurs because of a **pandas API misuse pattern**:

```python
# INCORRECT - usage_raw is a Series, not a column name
df.groupby('PLAYER_ID')[usage_raw].transform(...)
```

When `usage_raw` is a pandas Series containing float values like `0.15`, `0.20`, etc., pandas interprets these values as column names to select. Since columns named `0.15`, `0.20` don't exist, it raises a KeyError listing all the "missing columns" (which are actually the values in the Series).

### Expected Behavior
The code should compute rolling statistics on the derived Series values, not try to select columns by those values.

---

## 2. System Context Analysis

### Affected File
- [`src/preprocessing/feature_engineer.py`](src/preprocessing/feature_engineer.py) - Method `_add_advanced_scoring_features()` at lines 1003-1127

### Data Flow
```
train.py:main()
  └── feature_engineer.create_features(merged_df)
        └── _add_advanced_scoring_features(df)  ← ERROR OCCURS HERE
              └── df.groupby('PLAYER_ID')[usage_raw].transform(...)  ← BUG
```

### Pattern Analysis
The codebase has **two patterns** for groupby operations:

1. **Correct Pattern** (used in most of the codebase):
   ```python
   # stat is a string column name like 'PTS', 'REB'
   df.groupby('PLAYER_ID')[stat].transform(lambda x: ...)
   ```

2. **Incorrect Pattern** (only in `_add_advanced_scoring_features`):
   ```python
   # usage_raw is a computed Series, NOT a column name
   df.groupby('PLAYER_ID')[usage_raw].transform(lambda x: ...)
   ```

---

## 3. Problem Diagnosis

### Affected Code Locations

| Line(s) | Variable | Feature Name | Bug Type |
|---------|----------|--------------|----------|
| 1058-1060 | `usage_raw` | `ROLL_USG_PCT_{window}` | Series passed as column selector |
| 1075-1077 | `reb_opp` | `ROLL_REB_OPPORTUNITY_{window}` | Series passed as column selector |
| 1087-1089 | `three_pt_freq` | `ROLL_3PT_FREQ_{window}` | Series passed as column selector |
| 1097-1099 | `ft_rate` | `ROLL_FT_RATE_{window}` | Series passed as column selector |
| 1109-1111 | `pts_share` | `ROLL_PTS_SHARE_{window}` | Series passed as column selector |
| 1123-1125 | `ts_pct` | `ROLL_TS_PCT_MOMENTUM_{window}` | Series passed as column selector |

### Why This Bug Occurs
The developer likely intended to:
1. Compute a derived metric (e.g., `usage_raw`)
2. Group by player and compute rolling statistics on that metric

But the syntax `df.groupby('PLAYER_ID')[series_variable]` is interpreted by pandas as "select columns named by the values in `series_variable`", not "use this Series for the operation".

### Why It Wasn't Caught Earlier
- This code path may not have been tested with data that triggers the feature
- The error only occurs when the DataFrame has specific columns (`FGA_TEAM`, `FTA_TEAM`, etc.) that enable these features
- Unit tests may not have covered this specific method

---

## 4. Architecture Proposal

### Solution Options

#### Option A: Add Temporary Columns (Recommended)
Add computed Series as temporary columns before groupby operations, then optionally remove them.

```python
# Add as temporary column
df['_tmp_usage_raw'] = usage_raw
for window in [5, 10]:
    new_cols[f'ROLL_USG_PCT_{window}'] = df.groupby('PLAYER_ID')['_tmp_usage_raw'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
# Column gets cleaned up by existing cleanup code at line 335-338
```

**Pros:**
- Minimal code change
- Consistent with existing codebase patterns
- Temporary columns are already cleaned up by existing code (lines 335-338)
- Easy to understand and maintain

**Cons:**
- Adds temporary columns to DataFrame (minor memory overhead)

#### Option B: Use apply() on Grouped Data
Use `groupby().apply()` with the Series passed directly.

```python
for window in [5, 10]:
    new_cols[f'ROLL_USG_PCT_{window}'] = (
        df.groupby('PLAYER_ID')
        .apply(lambda g: usage_raw.loc[g.index].shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
```

**Pros:**
- No temporary columns

**Cons:**
- More complex syntax
- Potential performance issues with `apply()`
- Index alignment can be tricky
- Less readable

#### Option C: Pre-compute and Store in DataFrame
Compute all derived metrics first, store them with meaningful names, then use standard column selection.

```python
# Compute all derived metrics first
df['USAGE_RATE_RAW'] = usage_raw
df['REB_OPPORTUNITY'] = reb_opp
# ... etc

# Then use standard pattern
for window in [5, 10]:
    new_cols[f'ROLL_USG_PCT_{window}'] = df.groupby('PLAYER_ID')['USAGE_RATE_RAW'].transform(...)
```

**Pros:**
- Most readable
- Derived features are preserved (could be useful for analysis)
- Follows existing codebase patterns exactly

**Cons:**
- More columns added to DataFrame
- Need to decide which to keep vs remove

### Recommended Solution: Option A

Option A is recommended because:
1. Minimal code changes required
2. Consistent with existing cleanup patterns in the codebase
3. Easy to review and maintain
4. Performance impact is negligible

---

## 5. Performance Considerations

### Memory Impact
Adding temporary columns adds O(n) memory where n = number of rows. This is negligible compared to the existing DataFrame size.

### Computation Impact
No additional computation overhead - the same calculations are performed, just stored temporarily.

### Best Practices Applied
- Use `shift(1)` to prevent data leakage (already implemented)
- Use `min_periods=1` for rolling windows (already implemented)
- Clean up temporary columns after use (existing pattern at lines 335-338)

---

## 6. Implementation Roadmap

### Step 1: Fix `usage_raw` (Lines 1057-1060)
```python
# BEFORE
for window in [5, 10]:
    new_cols[f'ROLL_USG_PCT_{window}'] = df.groupby('PLAYER_ID')[usage_raw].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# AFTER
df['_tmp_usage_raw'] = usage_raw
for window in [5, 10]:
    new_cols[f'ROLL_USG_PCT_{window}'] = df.groupby('PLAYER_ID')['_tmp_usage_raw'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
```

### Step 2: Fix `reb_opp` (Lines 1074-1077)
```python
# BEFORE
for window in [5, 10]:
    new_cols[f'ROLL_REB_OPPORTUNITY_{window}'] = df.groupby('PLAYER_ID')[reb_opp].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# AFTER
df['_tmp_reb_opp'] = reb_opp
for window in [5, 10]:
    new_cols[f'ROLL_REB_OPPORTUNITY_{window}'] = df.groupby('PLAYER_ID')['_tmp_reb_opp'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
```

### Step 3: Fix `three_pt_freq` (Lines 1086-1089)
```python
# BEFORE
for window in [10, 20]:
    new_cols[f'ROLL_3PT_FREQ_{window}'] = df.groupby('PLAYER_ID')[three_pt_freq].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# AFTER
df['_tmp_three_pt_freq'] = three_pt_freq
for window in [10, 20]:
    new_cols[f'ROLL_3PT_FREQ_{window}'] = df.groupby('PLAYER_ID')['_tmp_three_pt_freq'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
```

### Step 4: Fix `ft_rate` (Lines 1096-1099)
```python
# BEFORE
for window in [10]:
    new_cols[f'ROLL_FT_RATE_{window}'] = df.groupby('PLAYER_ID')[ft_rate].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# AFTER
df['_tmp_ft_rate'] = ft_rate
for window in [10]:
    new_cols[f'ROLL_FT_RATE_{window}'] = df.groupby('PLAYER_ID')['_tmp_ft_rate'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
```

### Step 5: Fix `pts_share` (Lines 1108-1111)
```python
# BEFORE
for window in [10]:
    new_cols[f'ROLL_PTS_SHARE_{window}'] = df.groupby('PLAYER_ID')[pts_share].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# AFTER
df['_tmp_pts_share'] = pts_share
for window in [10]:
    new_cols[f'ROLL_PTS_SHARE_{window}'] = df.groupby('PLAYER_ID')['_tmp_pts_share'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
```

### Step 6: Fix `ts_pct` (Lines 1122-1125)
```python
# BEFORE
for window in [5, 10]:
    new_cols[f'ROLL_TS_PCT_MOMENTUM_{window}'] = df.groupby('PLAYER_ID')[ts_pct].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# AFTER
df['_tmp_ts_pct'] = ts_pct
for window in [5, 10]:
    new_cols[f'ROLL_TS_PCT_MOMENTUM_{window}'] = df.groupby('PLAYER_ID')['_tmp_ts_pct'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
```

### Step 7: Verify Cleanup
Confirm that existing cleanup code (lines 335-338) removes all `_tmp_*` columns:
```python
tmp_cols = [c for c in df.columns if c.startswith('_tmp_')]
if tmp_cols:
    logger.debug(f"Removing {len(tmp_cols)} temporary columns")
    df = df.drop(columns=tmp_cols)
```

---

## 7. Risk Analysis

### Potential Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Temporary columns not cleaned up | Low | Medium | Existing cleanup code handles `_tmp_` prefix |
| Index misalignment | Low | High | Series computed from same DataFrame, indices match |
| Performance regression | Very Low | Low | Negligible memory overhead |
| Feature values change | None | N/A | Same computation, just stored temporarily |

### Regression Prevention
1. Add unit test for `_add_advanced_scoring_features()` method
2. Test with DataFrame containing required columns (`FGA_TEAM`, `FTA_TEAM`, etc.)
3. Verify output features have expected values

---

## 8. Testing Strategy

### Unit Test Requirements
```python
def test_add_advanced_scoring_features():
    """Test that _add_advanced_scoring_features works with team stats columns."""
    # Create test DataFrame with required columns
    df = pd.DataFrame({
        'PLAYER_ID': ['p1', 'p1', 'p1', 'p2', 'p2', 'p2'],
        'TEAM_ID': ['t1', 't1', 't1', 't2', 't2', 't2'],
        'OPPONENT_ID': ['t2', 't2', 't2', 't1', 't1', 't1'],
        'GAME_DATE': pd.date_range('2024-01-01', periods=6),
        'FGA': [10, 12, 8, 15, 11, 9],
        'FTA': [4, 3, 5, 2, 4, 3],
        'TOV': [2, 1, 3, 2, 1, 2],
        'MIN': [30, 28, 32, 35, 30, 28],
        'PTS': [15, 12, 18, 20, 14, 16],
        'FGA_TEAM': [80, 75, 82, 78, 80, 76],
        'FTA_TEAM': [25, 22, 28, 20, 24, 22],
        'TOV_TEAM': [12, 10, 14, 11, 12, 10],
        'PTS_TEAM': [100, 95, 105, 98, 102, 96],
        'OPP_FGA_ALLOWED': [78, 80, 76, 82, 75, 80],
        'OPP_FGM_ALLOWED': [35, 36, 34, 37, 33, 36],
        'FG3A': [5, 4, 6, 3, 5, 4],
        'FGA': [10, 12, 8, 15, 11, 9],  # Note: duplicate in test, should be unique
    })
    
    fe = FeatureEngineer()
    result = fe._add_advanced_scoring_features(df)
    
    # Verify new columns exist
    assert 'ROLL_USG_PCT_5' in result.columns
    assert 'ROLL_USG_PCT_10' in result.columns
    # ... more assertions
```

### Integration Test
Run full training pipeline to verify no errors occur.

---

## 9. Summary

### Bug
Six instances in `_add_advanced_scoring_features()` pass computed pandas Series objects to `groupby()[...]` as column selectors, causing KeyError.

### Fix
Add each computed Series as a temporary column (`_tmp_*` prefix) before the groupby operation, then use the column name string for selection.

### Impact
- **Files Changed**: 1 (`src/preprocessing/feature_engineer.py`)
- **Lines Changed**: ~30 lines
- **Risk Level**: Low
- **Testing Required**: Unit test for affected method

### Verification
After fix, training should complete without KeyError, and the following features will be correctly computed:
- `ROLL_USG_PCT_5`, `ROLL_USG_PCT_10`
- `ROLL_REB_OPPORTUNITY_5`, `ROLL_REB_OPPORTUNITY_10`
- `ROLL_3PT_FREQ_10`, `ROLL_3PT_FREQ_20`
- `ROLL_FT_RATE_10`
- `ROLL_PTS_SHARE_10`
- `ROLL_TS_PCT_MOMENTUM_5`, `ROLL_TS_PCT_MOMENTUM_10`