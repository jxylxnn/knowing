# Add LightGBM as Third Base Model — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement task-by-task.
> **Blockers resolved:** LightGBM 4.6.0 already in requirements.txt. Config scaffolding exists.

**Goal:** Add LightGBM as third base model to the CatBoost + Transformer ensemble, with per-target training, save/load, and 3-way blending.

**Architecture:** LightGBM mirrors the CatBoost trainer pattern — one trainer per target statistic, trained in parallel, saved/loaded from `models/` directory. The existing 2-model blend (inverse-MAE or Ridge) becomes a 3-model blend: CatBoost + LightGBM + Transformer. LightGBM uses leaf-wise tree growth (different inductive bias from CatBoost's symmetric trees), creating error diversity needed for ensemble gains.

**Design decisions:**
- LightGBM is tree-only (no quantile models) — CatBoost already provides P10/P90
- LightGBM uses RMSE loss by default (like CatBoost primary model)
- No multi-loss branch for LightGBM — keeps trainer simpler
- GPU support via `device='gpu'` parameter in LightGBM (different API from CatBoost's `task_type='GPU'`)
- Blend weights encoded as `{catboost, lightgbm, transformer}` per target

---

## Files impacted

**CREATE (1):**
- `src/training/lightgbm_trainer.py` — LightGBMTrainer class (~400 lines)

**MODIFY (8):**
- `config/default.yaml` — add `lightgbm:` config section
- `src/config/model_config.py` — enable lightgbm in model config, add tier scaling
- `src/training/pipeline.py` — add _train_lightgbm_parallel, lightgbm artifacts, 3-way blend
- `src/training/presets.py` — add lightgbm_enabled to preset
- `src/models/base.py` — add lightgbm load/save utilities
- `src/models/model_manager.py` — add lightgbm model storage and loading
- `src/pipeline/prediction_service.py` — blend lightgbm predictions into ensemble
- `src/evaluation/ensemble_optimizer.py` — expand tunable dims from 13 → 19 (add 6 lightgbm blend ratios)

**TEST (1):**
- `tests/test_lightgbm_trainer.py` — unit tests

---

## Blocker Analysis (Pre-Mortem)

1. **CRITICAL — Blend weight contract breakage:** Adding a third model changes the blend weight dict shape from `{catboost, transformer}` to `{catboost, lightgbm, transformer}`. All downstream consumers (prediction_service.py, model_manager.py, ensemble_optimizer.py, validate_blend_contract) must update simultaneously or they'll KeyError on `lightgbm`. **Fix:** Do all blend-weight-consuming changes in one commit after the trainer is merged.

2. **HIGH — LightGBM GPU vs CatBoost GPU contention:** If both train on GPU simultaneously, they'll fight for VRAM. CatBoost uses `task_type='GPU'` and LightGBM uses `device='gpu'`. Both allocate aggressively. **Fix:** When GPU training, serialize LightGBM after CatBoost (or train on CPU since LightGBM is fast on CPU). Set `max_workers=1` when GPU + LightGBM enabled.

3. **MEDIUM — LightGBM uses different feature expectations:** LightGBM handles categorical features differently than CatBoost. CatBoost accepts string column names in `cat_features`; LightGBM requires integer-encoded categoricals. **Fix:** The feature_selector already encodes categoricals to integers — pass encoded column indices to LightGBM.

4. **MEDIUM — Model stack metadata contract:** `model_stack_metadata.pkl` stores `model_count: 2`. Must bump to 3 when LightGBM enabled. Downstream code checks this. **Fix:** Update metadata save and the validation logic.

5. **LOW — Existing trained models won't have LightGBM weights:** Loading pre-existing models after code update will fail blend validation because blend_weights.pkl doesn't have `lightgbm` keys. **Fix:** `load_blend_weights_from_disk` should gracefully handle missing lightgbm keys by setting them to 0.0 (CatBoost+Transformer-only fallback for backward compat).

---

## Task 1: Add LightGBM config section

**Files:** `config/default.yaml`

**Step 1: Add lightgbm config**
Insert after the catboost section (after line ~348, before `lstm:`):

```yaml
lightgbm:
  enabled: true
  boosting_type: "gbdt"
  n_estimators: 500
  max_depth: -1
  learning_rate: 0.03
  num_leaves: 63
  feature_fraction: 0.75
  bagging_fraction: 0.75
  bagging_freq: 1
  min_child_samples: 20
  reg_alpha: 0.1
  reg_lambda: 0.1
  early_stopping_rounds: 80
  random_state: 42
  n_jobs: -1
  use_gpu: false
  verbose: -1
```

**Step 2: Add lightgbm_enabled to full preset**
In `training_presets.full`, add `lightgbm_enabled: true`.
In `training_presets.small`, add `lightgbm_enabled: false`.

**Verification:** `python -c "from src.config import load_config; c = load_config('config/default.yaml'); print(c.lightgbm)"` prints config dict.

---

## Task 2: Update model_config.py tier scaling for LightGBM

**Files:** `src/config/model_config.py`

**Step 1: Find the lightgbm config dict (line ~463)**
Change `'enabled': False` to `'enabled': True`.

**Step 2: Add tier-specific overrides**
In the tier scaling function (where CatBoost/Transformer params are scaled by model size), add LightGBM scaling:

```
S:  n_estimators=300,  num_leaves=31,  learning_rate=0.05
M:  n_estimators=500,  num_leaves=63,  learning_rate=0.03  (default)
L:  n_estimators=800,  num_leaves=127, learning_rate=0.02
XL: n_estimators=1200, num_leaves=255, learning_rate=0.015
```

**Verification:** `python -c "from src.config.model_config import get_model_config; cfg, _ = get_model_config(force_size='M'); print(cfg['lightgbm']['enabled'])"` prints True.

---

## Task 3: Create LightGBMTrainer class

**Files:** CREATE `src/training/lightgbm_trainer.py`

**Pattern:** Mirrors `CatBoostTrainer` in `src/training/catboost_trainer.py`. Key differences:
- Uses `lightgbm.LGBMRegressor` instead of `CatBoostRegressor`
- No multi_loss branch, no quantile branch (simpler)
- GPU via `device='gpu'` instead of `task_type='GPU'`
- Save format: `.txt` (LightGBM native) + `.joblib` (sklearn wrapper)
- Categorical features handled by passing integer column indices

**Class structure:**
```python
class LightGBMTrainer(BaseTrainer):
    """LightGBM trainer with per-target hyperparameter tuning."""
    
    TARGET_PROFILES = {
        'PTS': {'num_leaves': 95, 'n_estimators': 800, 'learning_rate': 0.02, 'min_child_samples': 8},
        'REB': {'num_leaves': 75, 'n_estimators': 600, 'learning_rate': 0.025, 'min_child_samples': 12},
        'AST': {'num_leaves': 75, 'n_estimators': 600, 'learning_rate': 0.025, 'min_child_samples': 12},
        'STL': {'num_leaves': 45, 'n_estimators': 500, 'learning_rate': 0.03, 'min_child_samples': 25},
        'BLK': {'num_leaves': 45, 'n_estimators': 500, 'learning_rate': 0.03, 'min_child_samples': 25},
        'TOV': {'num_leaves': 55, 'n_estimators': 500, 'learning_rate': 0.028, 'min_child_samples': 18},
    }
    
    def __init__(self, model_name, target, config, use_gpu=False, device=None, random_state=42):
        ...
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_cols = None
        self.cat_features = None  # list of integer column indices for LightGBM
    
    def _build_params(self) -> Dict[str, Any]:
        # Cross-validate: num_leaves < 2^max_depth if max_depth != -1
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            sample_weight=None, feature_cols=None, cat_features=None) -> TrainResult:
        # LGBMRegressor.fit(X, y, eval_set=[(X_val, y_val)], 
        #   callbacks=[lgb.early_stopping(80), lgb.log_evaluation(50)])
    
    def predict(self, X) -> np.ndarray:
        ...
    
    def save(self, path) -> None:
        # Save as: {target}_lgbm.joblib (sklearn wrapper with metadata)
        # OR: {target}_lgbm.txt (native LightGBM format)
        # + {target}_lgbm_metadata.joblib
    
    @classmethod
    def load(cls, path, target) -> 'LightGBMTrainer':
        ...
    
    @classmethod
    def missing_runtime_artifacts(cls, path, target) -> List[str]:
        ...
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        ...

def train_lightgbm_target(target, X_train, y_train, X_val, y_val, 
                           config, cat_features=None, sample_weight=None, 
                           use_gpu=False) -> Tuple[str, TrainResult]:
    """Train LightGBM for a single target (parallel-safe)."""
```

**Key pitfalls to handle:**
1. LightGBM `cat_feature` parameter takes integer column indices, not column names. Convert cat_features (column names) to indices.
2. LightGBM `device='gpu'` only works with `boosting_type='gbdt'` (not 'dart' or 'goss').
3. `num_leaves` must be <= 2^max_depth when max_depth != -1. Add a `_clamp_num_leaves` helper.
4. LightGBM's native save is `model.booster_.save_model('path.txt')`. Use joblib for the sklearn wrapper to preserve the full object.

**Verification:** `python -c "from src.training.lightgbm_trainer import LightGBMTrainer; print('import ok')"`

---

## Task 4: Write unit tests

**Files:** CREATE `tests/test_lightgbm_trainer.py`

**Tests:**
```python
class TestLightGBMTrainer:
    def test_init_defaults(self):
        trainer = LightGBMTrainer("test_pts", "PTS", config={})
        assert trainer.target == "PTS"
        assert not trainer.is_trained
    
    def test_fit_predict_small_data(self):
        # Create synthetic data: 200 samples, 10 features
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
        y = X['f0'] * 2 + X['f1'] * 3 + np.random.randn(200) * 0.5
        
        trainer = LightGBMTrainer("test", "PTS", {'n_estimators': 50, 'num_leaves': 15, 'verbose': -1})
        result = trainer.fit(X[:150], y[:150], X[50:], y[50:])
        
        assert trainer.is_trained
        assert 'mae' in result.metrics
        assert result.metrics['mae'] > 0
    
    def test_predict_returns_correct_shape(self):
        ...
    
    def test_save_and_load_roundtrip(self, tmp_path):
        ...
    
    def test_feature_importance(self):
        ...
    
    def test_constant_target_fallback(self):
        # All y values identical → ConstantRegressor fallback
        ...
    
    def test_per_target_tuning(self):
        trainer = LightGBMTrainer("test_stl", "STL", {'use_per_target_tuning': True, 'n_estimators': 100})
        params = trainer._build_params()
        assert params['num_leaves'] == 45  # STL profile
        assert params['min_child_samples'] == 25
```

**Verification:** `pytest tests/test_lightgbm_trainer.py -v`

---

## Task 5: Update presets.py

**Files:** `src/training/presets.py`

**Step 1: Add `lightgbm_enabled` to the `TrainingPreset` dataclass**
```python
@dataclass
class TrainingPreset:
    name: str
    default_mode: str
    default_model_size: str
    transformer_enabled: bool
    lightgbm_enabled: bool = False   # NEW
    recent_seasons: Optional[int] = None
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 10, 20, 50])
    enable_groups: List[str] = field(default_factory=list)
```

**Step 2: Update `resolve_training_preset` to read from config**
```python
preset_data = presets_dict.get(name, presets_dict.get('full', {}))
return TrainingPreset(
    name=name,
    ...
    lightgbm_enabled=preset_data.get('lightgbm_enabled', False),
    ...
)
```

**Verification:** `python -c "from src.training.presets import resolve_training_preset; p = resolve_training_preset('full', {'full': {'lightgbm_enabled': True}}); print(p.lightgbm_enabled)"`

---

## Task 6: Update TrainingPipeline (core integration)

**Files:** `src/training/pipeline.py`

This is the biggest change. Multiple sections to modify:

**Step 1: Import LightGBMTrainer**
```python
from src.training.lightgbm_trainer import (
    LightGBMTrainer, 
    ConstantRegressor,
    train_lightgbm_target,
)
```

**Step 2: Add lightgbm storage dicts to `__init__`**
```python
self.lightgbm_models: Dict[str, Any] = {}          # NEW
self.lightgbm_trainers: Dict[str, Any] = {}        # NEW
```

**Step 3: Add `_lightgbm_train_config` method**
```python
def _lightgbm_train_config(self) -> Dict[str, Any]:
    cfg = dict(self.model_config.get("lightgbm", {}))
    cfg["use_per_target_tuning"] = True
    return cfg
```

**Step 4: Add `_train_lightgbm_parallel` method**
Mirrors `_train_catboost_parallel` (lines 304-493) but uses `train_lightgbm_target` instead of `train_catboost_target`. Key differences:
- Uses `self.lightgbm_models[target]` and `self.lightgbm_trainers[f"lightgbm_{target}"]`
- On GPU: serialize after CatBoost (CPU-only for LightGBM when GPU training to avoid contention)
- Skip quantile models (CatBoost handles uncertainty)
- Call `self.experiment.log_model_metrics("lightgbm", result.metrics, target)`

**Step 5: Add `_save_lightgbm_artifacts` method**
Mirrors `_save_catboost_artifacts` (lines 495-532). Save format:
```
{target}_lgbm.joblib  (sklearn wrapper)
{target}_lgbm_metadata.joblib
```

**Step 6: Update blend weight computation**
Current signature: `_build_inverse_mae_weights(catboost_results, transformer_result)` → dict with `{catboost, transformer, catboost_mae, transformer_mae}`.

New signature: `_build_inverse_mae_weights(catboost_results, lightgbm_results, transformer_result)` → dict with `{catboost, lightgbm, transformer, catboost_mae, lightgbm_mae, transformer_mae}`.

Weight formula: for each target, `inv_cb = 1/cb_mae, inv_lgbm = 1/lgbm_mae, inv_tx = 1/tx_mae`, normalize to sum=1.

**Step 7: Update `_build_ridge_blend_weights`**
Current: Ridge on 2-column stack `[cb_preds, tx_preds]`.
New: Ridge on 3-column stack `[cb_preds, lgbm_preds, tx_preds]`.
With 3 columns, `ridge.coef_` length = 3 instead of 2.

**Step 8: Update `train()` method**
After CatBoost training (line 1001), add:
```python
if self.model_config.get("lightgbm", {}).get("enabled", False):
    logger.info("=== Training LightGBM Models ===")
    lightgbm_results = self._train_lightgbm_parallel(fit_df, val_df)
    results["lightgbm"] = lightgbm_results
else:
    lightgbm_results = {}
```

Update blend weight calls to pass `lightgbm_results`.

**Step 9: Update `_save_model_stack_metadata`**
Change `model_count` from `2 if transformer_enabled else 1` to include LightGBM count.
```python
model_count = 1  # CatBoost always
if transformer_enabled: model_count += 1
if lightgbm_enabled: model_count += 1
```

**Step 10: Update `_validate_runtime_artifact_contract`**
Add LightGBM artifact validation:
```python
for target in self.TARGETS:
    target_missing = LightGBMTrainer.missing_runtime_artifacts(self.models_dir, target)
    if target_missing:
        per_target_missing.setdefault(target, []).extend(target_missing)
```

**Step 11: Update `load_models()`**
Add LightGBM loading loop (mirrors CatBoost loading at lines 1079-1093):
```python
self.lightgbm_models = {}
for target in self.TARGETS:
    try:
        trainer = LightGBMTrainer.load(self.models_dir, target)
    except Exception:
        continue
    if trainer.model is not None:
        self.lightgbm_models[target] = trainer.model
```

**Verification:** Run full training in quick mode:
`python train.py --mode quick --preset small` (which disables lightgbm by default, so this should still work unchanged)
Then test with lightgbm: modify small preset to enable lightgbm temporarily → quick training completes without errors.

---

## Task 7: Update model_manager.py

**Files:** `src/models/model_manager.py`

**Step 1: Add lightgbm model storage**
```python
self.lightgbm_models: Dict[str, Any] = {}    # NEW
```

**Step 2: Update `validate_runtime_artifacts`**
Add LightGBM artifact check:
```python
from src.training.lightgbm_trainer import LightGBMTrainer
for target in self.targets:
    target_missing = LightGBMTrainer.missing_runtime_artifacts(self.models_dir, target)
    ...
```

**Step 3: Update model loading** (around line ~200 in prepare_data / load flow)
Load LightGBM models from disk similar to CatBoost loading.

**Verification:** `python -c "from src.models.model_manager import ModelManager; m = ModelManager(data_dir='data', models_dir='models'); print(m.lightgbm_models)"`

---

## Task 8: Update prediction_service.py for 3-way blend

**Files:** `src/pipeline/prediction_service.py`

**Step 1: Add LightGBM prediction in `_predict_with_ensemble`**
After the CatBoost predictions loop (after line ~181), add:
```python
# 2. LightGBM predictions
lgbm_preds = {}
for target in self.config.training.targets:
    if target not in self.pipeline.lightgbm_models:
        lgbm_preds[target] = None
        continue
    try:
        model = self.pipeline.lightgbm_models[target]
        lgbm_preds[target] = float(model.predict(X)[0])
    except Exception:
        lgbm_preds[target] = None
```

**Step 2: Update the blend (line ~190-203)**
Change from 2-way to 3-way:
```python
if transformer_preds is not None or any(v is not None for v in lgbm_preds.values()):
    for target in self.config.training.targets:
        blend_cfg = self.blend_weights.get(target, {})
        cb_w = float(blend_cfg.get('catboost', 1.0))
        lgbm_w = float(blend_cfg.get('lightgbm', 0.0))
        tx_w = float(blend_cfg.get('transformer', 0.0))
        intercept = float(blend_cfg.get('intercept', 0.0))
        
        blended = predictions[target] * cb_w + intercept
        if lgbm_preds.get(target) is not None and lgbm_w > 0:
            blended += lgbm_preds[target] * lgbm_w
        if transformer_preds is not None and tx_w > 0:
            tx_idx = self.config.training.targets.index(target)
            blended += float(transformer_preds[tx_idx]) * tx_w
        
        predictions[target] = float(blended)
```

**Verification:** Integration test or manual: train models, run `python query_prob.py`, verify predictions include LightGBM contributions in model_contributions dict.

---

## Task 9: Update ensemble_optimizer.py for extra blend dims

**Files:** `src/evaluation/ensemble_optimizer.py`

**Step 1: Bump tunable dims**
```python
# Old: 13 = 6 cb ratios + 6 intercepts + 1 mae blend
# New: 19 = 6 cb ratios + 6 lgbm ratios + 6 intercepts + 1 mae blend
_TUNABLE_DIMS = 19
```

**Step 2: Update `_weights_to_vector` and `_vector_to_weights`**
Add lightgbm blend ratio for each target to the encoding.

**Step 3: Update bounds**
```python
LGBM_BLEND_BOUNDS = (0.1, 0.9)  # lightgbm fraction per target
```

**Step 4: Update normalization**
3-way weights must now sum to ~1.0 across catboost + lightgbm + transformer for each target.

**Step 5: Update `_evaluate_weights`**
When computing ensemble prediction, blend 3 models instead of 2.

**Verification:** Unit test `tests/test_ensemble_optimizer.py` (if exists), or manual: enable self_optimization in config, run backtest, verify optimization completes.

---

## Task 10: Update train.py entry point

**Files:** `train.py`

**Step 1: Pass `lightgbm_enabled` from preset to pipeline**
After line 410 (`pipeline.model_config["transformer"]["enabled"] = ...`):
```python
pipeline.model_config["lightgbm"]["enabled"] = bool(preset.lightgbm_enabled)
```

**Step 2: Add `--no-lightgbm` CLI flag** (optional, for debugging)
```python
parser.add_argument('--no-lightgbm', action='store_true', 
                    help='Disable LightGBM even if preset enables it')
```

**Verification:** `python train.py --mode quick --preset small` runs without LightGBM. Update small preset temporarily to `lightgbm_enabled: true`, verify LightGBM trains.

---

## Task 11: Update model base utilities

**Files:** `src/models/base.py`

**Step 1: Update `validate_blend_contract`**
Check for lightgbm weight presence alongside transformer weight:
```python
has_lightgbm_weight = any(
    float(weights.get("lightgbm", 0.0)) > 0.0
    for key, weights in blend_weights.items()
    if key != "_method"
)
if has_lightgbm_weight and not lightgbm_models_loaded:
    # warn/error
```

**Step 2: Add `load_lightgbm_weights_from_disk` helper**
Or extend `load_blend_weights_from_disk` to handle backward compat with 2-model weights.

**Verification:** Load old 2-model blend_weights.pkl with updated code → no crash, lightgbm keys default to 0.0.

---

## Integration test (end-to-end)

After all tasks complete:

```bash
source venv/bin/activate

# 1. Quick smoke test (small preset, no lightgbm — backward compat)
python train.py --mode quick --preset small
# Expected: completes with CatBoost only, no errors

# 2. Full preset with LightGBM (quick mode)
python train.py --mode quick --preset full
# Expected: trains CatBoost + LightGBM + Transformer, 3-way blend weights saved

# 3. Verify saved artifacts
ls models/ | grep lgbm
# Expected: pts_lgbm.joblib, pts_lgbm_metadata.joblib, reb_lgbm.joblib, etc.

# 4. Verify blend weights have 3 keys
python -c "import joblib; bw = joblib.load('models/blend_weights.pkl'); print(bw['PTS'].keys())"
# Expected: dict_keys(['catboost', 'lightgbm', 'transformer', 'catboost_mae', 'lightgbm_mae', 'transformer_mae'])

# 5. Run prediction
python query_prob.py
# Expected: predictions work, model_contributions shows 3 models

# 6. Run full test suite
pytest tests/ -x -m "not slow"
```

---

## Expected improvements

Based on ensemble theory and comparable projects (chevyphillip/plus-ev-model uses RF+GB+Lasso ensembles):

- **MAE reduction:** 3-8% improvement on validation MAE from ensemble diversity
- **Calibration improvement:** 3-model blend produces better-calibrated predictions (wider error diversity means less overconfident predictions)
- **Training time increase:** ~20-30% (LightGBM is fast — leaf-wise growth converges faster than CatBoost's symmetric trees)
- **Inference time increase:** ~50% (3 model.predict() calls instead of 2, but LightGBM inference is fast)

---

## Rollback plan

If LightGBM degrades performance:
1. Set `lightgbm.enabled: false` in config/default.yaml
2. Set `lightgbm_enabled: false` in presets
3. The blend weight computation already handles 0-weight gracefully
4. Old blend_weights.pkl files still work (backward compat handled in load)
