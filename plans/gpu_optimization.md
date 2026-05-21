# GPU Optimization Plan — knowing-master NBA Prediction System

## Current State

The project has good GPU infrastructure (`gpu_utils.py` — TF32, BF16, cuDNN, AMP) but uses it unevenly:

| Component | GPU Used? | Efficiency |
|-----------|-----------|------------|
| CatBoost training | Yes | Good, but leaks VRAM across targets |
| Transformer training | Yes | Well-optimized, but clone overhead |
| Monte Carlo simulation | **No** | 100% CPU — serial Python loop |
| Season simulation | **No** | CPU ThreadPoolExecutor |

The GPU sits **completely idle** during simulation — the most run-frequently operation. Training works but wastes VRAM between phases.

---

## Fixes (dependency-ordered)

### Fix 1: GPU Memory Cleanup Between CatBoost Targets
**File:** `src/training/pipeline.py`  
**Lines:** ~430-468 (parallel path), ~458-470 (sequential path)  
**Complexity:** Low — ~5 lines  
**Impact:** Prevents VRAM accumulation across 6 targets × 4 models = 24 allocations

What to do:
- In the parallel path (line 431 area), add `clear_gpu_memory()` after the `joblib.Parallel` block completes
- In the sequential path, add `clear_gpu_memory()` after each target's `_train_catboost_target()` returns
- Inside `catboost_trainer.py:fit()`, add `del` + `torch.cuda.empty_cache()` after all 4 model variants finish training for a target

### Fix 2: GPU Memory Cleanup Between CatBoost → Transformer Phases
**File:** `src/training/pipeline.py`  
**Lines:** ~470-675 (between CatBoost completion and Transformer start)  
**Complexity:** Low — ~3 lines  
**Impact:** Prevents CatBoost VRAM from starving Transformer allocation

What to do:
- In `pipeline.train()`, after all CatBoost targets complete but before `_train_transformer_model()`, call:
  ```python
  del self.catboost_models
  clear_gpu_memory()
  torch.cuda.empty_cache()
  ```

### Fix 3: Defer Best-Model Cloning to End of Training
**File:** `src/training/nn_trainer.py`  
**Lines:** ~320-345 (validation epoch logic)  
**Complexity:** Low — restructure clone timing  
**Impact:** Saves 50-100ms per epoch (5-10s over 100 epochs), eliminates GPU→CPU transfer on every loss improvement

What to do:
- Instead of cloning to CPU on every `val_loss < best_val_loss`, keep a reference to the best state dict on GPU
- At training end (`fit()` return), clone to CPU once
- Alternative: save best model to disk periodically (every 10 epochs) instead of keeping in memory

### Fix 4: Vectorize Per-Player Binomial Draws in Phase Simulator
**File:** `src/simulation/phase_simulator.py` (or wherever `simulate_team_phase` lives)  
**Complexity:** Low — replace sequential loops with vectorized calls  
**Impact:** Eliminates ~180 sequential `np_rng.binomial()` calls per sim. With 1000 sims and 20 players per game, that's 3.6M individual NumPy calls → 1 batched call.

What to do:
- Locate `simulate_team_phase()` or equivalent method with per-player binomial draws
- Batch all FG2M/FG3M/FTM binomial draws for all players into single `np_rng.binomial(n, p)` calls where `n` and `p` are arrays
- Use `(n_players, n_shot_types)` shaped arrays instead of scalar-per-player

### Fix 5: GPU-Accelerated Monte Carlo Simulation
**Files:** `src/simulation/game_simulator.py`, `src/simulation/phase_simulator.py`  
**Lines:** ~889-1084 (`_run_simulation()`), plus `simulate_team_phase()`  
**Complexity:** Medium — restructure simulation loop to batch all sims  
**Impact:** 50-100x faster simulation. Current 1000 sims/game in ~1s (CPU) → ~0.01-0.02s (GPU).

What to do:
- The `PlayerCorrelationEngine.apply_correlations_torch()` method already exists — wire it into the simulation path instead of the NumPy `apply_correlations()`
- Restructure `_run_simulation()` to:
  1. Pre-compute base predictions for all players (once) — already done
  2. Generate all random noise as GPU tensors in one batch: `(num_sims, n_players, n_stats)`
  3. Apply correlations via existing `apply_correlations_torch()`
  4. Apply per-sim modifiers (minutes, defense, pace, fatigue) as batched tensor operations
  5. Transfer final `(num_sims, n_players, n_stats)` tensor back to CPU once for assembly
- Key insight: every sim is independent. The only dependency is the correlation matrix injection, which `apply_correlations_torch()` already handles via Cholesky decomposition.
- Keep CPU fallback path intact with `if self.use_gpu:` branch
- The `clear_gpu_memory()` pattern from gpu_utils should be called after the tensor→CPU transfer to free VRAM

---

## Implementation Order

```
Fix 1 (cleanup between targets)
  └─► Fix 2 (cleanup between phases)
        └─► Fix 3 (defer model clone)
              └─► Fix 4 (vectorize binomials)
                    └─► Fix 5 (GPU Monte Carlo)
```

Fixes 1-4 are independent quick wins. Fix 5 depends on Fix 4 (the vectorized binomial draws become the GPU simulation's building blocks). Fixes 1-3 can be done simultaneously.

---

## Files to Modify

| File | Fixes | Lines Changed (est.) |
|------|-------|---------------------|
| `src/training/pipeline.py` | 1, 2 | ~10 |
| `src/training/catboost_trainer.py` | 1 | ~5 |
| `src/training/nn_trainer.py` | 3 | ~15 |
| `src/simulation/phase_simulator.py` | 4 | ~20 |
| `src/simulation/game_simulator.py` | 5 | ~80 |
| `src/simulation/player_correlation_engine.py` | 5 | ~10 (wire existing method) |
| `src/models/gpu_utils.py` | 5 | ~15 (add batch noise generator) |

---

## Expected Performance Gains

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| CatBoost training (6 targets) | VRAM degrades | Clean per-target | No OOM risk |
| Transformer training startup | May OOM | Clean slate | No OOM risk |
| Per-epoch validation | ~50ms clone overhead | ~0ms | 100 epochs saved |
| Phase simulation binomials | 180 sequential calls | 1 batched call | ~50x |
| Full Monte Carlo (1000 sims) | ~1s CPU | ~0.02s GPU | ~50x |
| Season simulation (15 games) | ~15s CPU | ~0.3s GPU | ~50x |

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| GPU OOM with batched sims | Medium | Scale batch size to VRAM; fall back to chunked batches |
| Numerical differences CPU vs GPU | Low | Keep CPU path as fallback; seed both identically |
| CatBoost retraining after `del` | Low | Models saved to disk before cleanup |
| Binomial vectorization changes output | Low | Use same RNG with vectorized call; verify with tests |
