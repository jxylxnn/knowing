# CatBoost GPU Callback Fix Plan

## Executive Summary

The training pipeline fails when attempting to use CatBoost with GPU acceleration due to CatBoost's limitation that **user-defined callbacks are not supported on GPU**. The current implementation tries to use a custom progress callback regardless of device, causing a `CatBoostError`. Additionally, the error handling fallback has a secondary bug where the callback's `after_iteration` method receives a malformed `info` object during exception handling.

---

## STEP 1 — TASK INTERPRETATION

### The Real Objective
Fix the CatBoost GPU training failure to enable successful model training on NVIDIA A100-SXM4-40GB GPU.

### Hidden Requirements
1. Maintain training progress visibility on both CPU and GPU
2. Preserve the rich logging functionality where possible
3. Handle graceful fallback from GPU to CPU when needed
4. Ensure the fix doesn't regress CPU training performance

### Expected Behavior
- GPU training should work without callbacks
- CPU training should retain progress callbacks
- Fallback mechanism should work correctly
- Training progress should still be logged (via alternative means on GPU)

### System Constraints
- CatBoost GPU does NOT support `callbacks` parameter
- CatBoost GPU has limited metric logging compared to CPU
- The A100 GPU (40GB VRAM) is powerful and should be utilized when possible

---

## STEP 2 — SYSTEM CONTEXT ANALYSIS

### Architecture Overview

```
train.py
    └── TrainingPipeline (src/training/pipeline.py)
            └── CatBoostTrainer (src/training/catboost_trainer.py)
                    └── CatBoostProgressCallback (custom callback)
```

### Error Flow Analysis

```
1. _train_single_model() called with use_gpu=True
2. CatBoostProgressCallback created
3. fit_kwargs['callbacks'] = [callback]
4. model.fit() called with task_type='GPU'
5. CatBoost raises: "User defined callbacks are not supported for GPU"
6. Exception handler catches error, tries fallback to CPU
7. During exception cleanup, callback.after_iteration() called
8. info object is SimpleNamespace without 'learn_error' attribute
9. AttributeError: 'types.SimpleNamespace' object has no attribute 'learn_error'
```

### Key Code Locations

| File | Lines | Purpose |
|------|-------|---------|
| [`catboost_trainer.py`](src/training/catboost_trainer.py:20) | 20-75 | `CatBoostProgressCallback` class |
| [`catboost_trainer.py`](src/training/catboost_trainer.py:274) | 274-331 | `_train_single_model()` method |
| [`catboost_trainer.py`](src/training/catboost_trainer.py:304) | 304-309 | Callback instantiation |
| [`catboost_trainer.py`](src/training/catboost_trainer.py:311) | 311-317 | fit_kwargs with callbacks |
| [`catboost_trainer.py`](src/training/catboost_trainer.py:319) | 319-329 | Exception handling fallback |

---

## STEP 3 — PROBLEM & WEAKNESS DETECTION

### Primary Issue: GPU Callback Incompatibility
**Location:** [`_train_single_model()`](src/training/catboost_trainer.py:274) lines 304-314

**Problem:** The code unconditionally creates and passes a callback to CatBoost, but CatBoost GPU does not support user-defined callbacks.

```python
# Current problematic code (lines 304-314)
callback = CatBoostProgressCallback(
    target=self.target,
    total_iterations=params['iterations'],
    log_every=50
)

fit_kwargs = {
    'eval_set': (X_val, y_val),
    'use_best_model': True,
    'callbacks': [callback]  # <-- FAILS ON GPU
}
```

### Secondary Issue: Callback Attribute Error
**Location:** [`CatBoostProgressCallback.after_iteration()`](src/training/catboost_trainer.py:33) lines 48-50

**Problem:** The callback assumes `info` has `learn_error` and `test_error` attributes, but during error handling, these may not exist.

```python
# Current problematic code (lines 48-50)
train_loss = info.learn_error[-1] if info.learn_error else 0
val_loss = info.test_error[-1] if info.test_error else 0
```

### Tertiary Issue: Incomplete Fallback
**Location:** [`_train_single_model()`](src/training/catboost_trainer.py:319) lines 319-329

**Problem:** The fallback to CPU still includes the callback in `fit_kwargs`, which would fail again if the callback were the issue.

---

## STEP 4 — BEST-IN-CLASS SOLUTIONS

### Option A: Conditional Callback (Recommended)
**Approach:** Only use callbacks when training on CPU. For GPU, rely on CatBoost's built-in `verbose` parameter.

**Pros:**
- Minimal code change
- Preserves CPU training experience
- No performance impact

**Cons:**
- Less detailed progress on GPU
- Need alternative progress tracking for GPU

### Option B: Polling-Based Progress (Alternative)
**Approach:** Use CatBoost's `eval_set` with periodic metric polling instead of callbacks.

**Pros:**
- Works on both CPU and GPU
- More portable

**Cons:**
- More complex implementation
- May not provide same granularity

### Option C: Hybrid Approach (Best)
**Approach:** 
1. Use callbacks on CPU
2. Use verbose logging on GPU
3. Add post-training metric logging for both

**Pros:**
- Best user experience on both platforms
- Maintains all functionality
- Clean separation of concerns

---

## STEP 5 — OPTIMAL ARCHITECTURE DESIGN

### Solution Architecture

```
_train_single_model()
    │
    ├── Check task_type (GPU vs CPU)
    │
    ├── IF CPU:
    │       └── Create callback, add to fit_kwargs
    │       └── Use verbose=False (callback handles logging)
    │
    ├── IF GPU:
    │       └── NO callback in fit_kwargs
    │       └── Use verbose=200 for built-in progress
    │
    └── Exception handling:
            └── Retry on CPU WITHOUT callback if original was GPU
            └── Proper attribute access in callback
```

### Code Changes Required

#### Change 1: Conditional Callback in `_train_single_model()`

**File:** [`src/training/catboost_trainer.py`](src/training/catboost_trainer.py:274)

```python
def _train_single_model(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_function: str,
    model_type: str,
    sample_weight: Optional[np.ndarray] = None,
) -> CatBoostRegressor:
    """Train a single CatBoost model with detailed progress tracking."""
    params = self._build_params(loss_function, model_type)
    
    task_type = 'GPU' if self.use_gpu else 'CPU'
    
    model_params = {**params, 'cat_features': self.cat_features or []}
    
    if task_type == 'GPU':
        model_params['task_type'] = 'GPU'
        model_params['devices'] = '0'
        logger.info(f"Using GPU acceleration for {self.target} {model_type} model")
    else:
        model_params['task_type'] = 'CPU'
        logger.info(f"Using CPU for {self.target} {model_type} model")
    
    # GPU does not support user-defined callbacks
    # Use verbose logging for GPU, custom callback for CPU
    use_callback = task_type == 'CPU'
    
    if use_callback:
        model_params['verbose'] = False
    else:
        model_params['verbose'] = 200  # Built-in progress every 200 iterations
    
    model = CatBoostRegressor(**model_params)
    
    fit_kwargs = {
        'eval_set': (X_val, y_val),
        'use_best_model': True,
    }
    
    # Only add callback for CPU training
    if use_callback:
        callback = CatBoostProgressCallback(
            target=self.target,
            total_iterations=params['iterations'],
            log_every=50
        )
        fit_kwargs['callbacks'] = [callback]
    
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except Exception as e:
        if task_type == 'GPU':
            logger.warning(f"GPU training failed ({e}), falling back to CPU")
            model_params['task_type'] = 'CPU'
            del model_params['devices']
            
            # Rebuild fit_kwargs for CPU fallback
            fallback_fit_kwargs = {
                'eval_set': (X_val, y_val),
                'use_best_model': True,
            }
            
            # Add callback for CPU fallback
            callback = CatBoostProgressCallback(
                target=self.target,
                total_iterations=params['iterations'],
                log_every=50
            )
            fallback_fit_kwargs['callbacks'] = [callback]
            
            if sample_weight is not None:
                fallback_fit_kwargs['sample_weight'] = sample_weight
            
            model = CatBoostRegressor(**model_params)
            model.fit(X_train, y_train, **fallback_fit_kwargs)
        else:
            raise
    
    return model
```

#### Change 2: Defensive Attribute Access in Callback

**File:** [`src/training/catboost_trainer.py`](src/training/catboost_trainer.py:33)

```python
def after_iteration(self, info):
    """Called after each iteration."""
    iteration = info.iteration
    
    # Calculate timing
    elapsed = time.time() - self.start_time
    if iteration > 0:
        time_per_iter = elapsed / iteration
        self.iter_times.append(time_per_iter)
        remaining_iters = self.total_iterations - iteration
        eta_seconds = remaining_iters * time_per_iter
    else:
        time_per_iter = 0
        eta_seconds = 0
    
    # Get metrics from info - use getattr for defensive access
    # CatBoost info object may have different attribute names depending on version
    learn_error = getattr(info, 'learn_error', None) or getattr(info, 'training_metrics', {}).get('learn_error', [])
    test_error = getattr(info, 'test_error', None) or getattr(info, 'validation_metrics', {}).get('test_error', [])
    
    train_loss = learn_error[-1] if learn_error and len(learn_error) > 0 else 0
    val_loss = test_error[-1] if test_error and len(test_error) > 0 else 0
    
    # Track best
    if val_loss < self.best_val_loss:
        self.best_val_loss = val_loss
        self.best_iteration = iteration
    
    # Create metrics object
    metrics = TrainingMetrics(
        target=self.target,
        model_type='catboost',
        iteration=iteration,
        total_iterations=self.total_iterations,
        train_loss=train_loss,
        val_loss=val_loss,
        best_iteration=self.best_iteration,
        best_val_loss=self.best_val_loss,
        time_per_iter=time_per_iter,
        eta_seconds=eta_seconds,
    )
    
    # Log via training logger
    self.training_logger.log_iteration(metrics)
    
    return True
```

---

## STEP 6 — PERFORMANCE CONSIDERATIONS

### GPU Training Performance
- **Before Fix:** Fails immediately, no training
- **After Fix:** Full GPU acceleration with A100
- **Expected Speedup:** 5-10x faster than CPU training

### Progress Logging Overhead
- **CPU with Callback:** Minimal overhead (~1-2%)
- **GPU with Verbose:** Built-in, no Python callback overhead
- **Trade-off:** Less detailed progress on GPU, but faster training

### Memory Considerations
- GPU training uses more VRAM but less system RAM
- Callback objects are small, negligible memory impact

---

## STEP 7 — IMPLEMENTATION ROADMAP

### Phase 1: Core Fix (Critical)
1. Modify `_train_single_model()` to conditionally use callbacks
2. Add defensive attribute access in `after_iteration()`
3. Fix fallback mechanism to properly rebuild fit_kwargs

### Phase 2: Testing
1. Test GPU training without callbacks
2. Test CPU training with callbacks
3. Test GPU→CPU fallback scenario
4. Verify progress logging works on both platforms

### Phase 3: Enhancement (Optional)
1. Add GPU-specific progress tracking using CatBoost's eval_result
2. Consider using `prediction_type` for additional metrics
3. Add metric logging after training completion

---

## STEP 8 — RISK ANALYSIS

### Risk 1: CatBoost Version Differences
**Impact:** Different CatBoost versions may have different callback behaviors
**Mitigation:** Use defensive `getattr()` calls, test on multiple versions

### Risk 2: Progress Visibility Loss on GPU
**Impact:** Users won't see detailed progress during GPU training
**Mitigation:** Use `verbose=200` for built-in logging, add post-training summary

### Risk 3: Fallback Still Failing
**Impact:** If GPU fails for other reasons, CPU fallback might also fail
**Mitigation:** Ensure fallback uses clean fit_kwargs without GPU-specific params

### Risk 4: Regression in CPU Training
**Impact:** Changes might break existing CPU training flow
**Mitigation:** Keep callback logic identical for CPU path, add tests

---

## STEP 9 — VALIDATION STEPS

### Unit Tests
```python
def test_cpu_training_with_callback():
    """Test that CPU training uses callbacks correctly."""
    
def test_gpu_training_without_callback():
    """Test that GPU training works without callbacks."""
    
def test_fallback_to_cpu():
    """Test that GPU failure falls back to CPU correctly."""
    
def test_callback_attribute_access():
    """Test defensive attribute access in callback."""
```

### Integration Tests
```python
def test_full_training_pipeline_gpu():
    """Test complete training pipeline on GPU."""
    
def test_full_training_pipeline_cpu():
    """Test complete training pipeline on CPU."""
```

### Manual Verification
1. Run training on GPU - should complete without errors
2. Check progress logs appear (verbose output)
3. Verify models are saved correctly
4. Confirm metrics are logged

---

## STEP 10 — FILES TO MODIFY

| File | Changes |
|------|---------|
| [`src/training/catboost_trainer.py`](src/training/catboost_trainer.py) | Main fix location |

### Summary of Changes

1. **Line ~287-300:** Add conditional logic for `use_callback` based on `task_type`
2. **Line ~304-317:** Move callback creation inside `if use_callback:` block
3. **Line ~311-317:** Build `fit_kwargs` conditionally
4. **Line ~319-329:** Rebuild `fit_kwargs` in fallback without GPU params
5. **Line ~48-50:** Use `getattr()` for defensive attribute access

---

## Conclusion

This fix addresses the core issue: CatBoost GPU does not support user-defined callbacks. The solution:

1. **Conditionally disables callbacks for GPU training** - using built-in `verbose` instead
2. **Fixes the callback's attribute access** - using defensive `getattr()` calls
3. **Properly handles GPU→CPU fallback** - rebuilding fit_kwargs cleanly

The implementation is minimal, focused, and maintains backward compatibility with CPU training while enabling GPU acceleration.