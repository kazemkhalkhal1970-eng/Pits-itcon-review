# PITS Dataset — ITcon Review Repository

## Contents

1. `dataset_generator.py` — Python script that generates the full 1,200-cycle dataset
2. `excavator_action_dataset.npz` — Pre-generated dataset (19 MB, compressed)
3. `metadata.json` — Dataset metadata (soil parameters, classes, excavator specs)
4. `cycle_meta_train.json` — Per-cycle metadata for training set
5. `cycle_meta_test.json` — Per-cycle metadata for test set

## How to Reproduce

```bash
python3 dataset_generator.py
```

This will create the `excavator_action_dataset.npz` file in a `dataset/` folder.

## Dataset Summary

- **Train cycles**: 960 (160 per soil type × 6 soils)
- **Test cycles**: 240 (40 per soil type × 6 soils)
- **Soil types**: dry_sand, wet_sand, clay_loam, silty_loam, gravel, wet_clay
- **Action classes**: 7 (reach, penetrate, curl, lift, swing, dump, return)
- **Sample rate**: 30 Hz (subsampled to 10 Hz for training)
- **Signals per frame**: 8 (4 joint angles + 4 hydraulic pressures)
- **Soil features per cycle**: 6 (k_c, k_phi, n, c, phi, rho)

## Loading the Dataset

```python
import numpy as np
d = np.load('excavator_action_dataset.npz', allow_pickle=True)

# Training data
X_joints = d['X_joints_train']     # (960, 680, 4) — joint angles (rad)
X_press = d['X_press_train']        # (960, 680, 4) — hydraulic pressures (Pa)
Y = d['Y_train']                    # (960, 680) — action labels (0-6, -1 = padding)
soil_features = d['soil_features_train']  # (960, 6) — soil parameters
mask = d['mask_train']              # (960, 680) — valid frame mask (1=valid, 0=padding)

# Test data
X_joints_test = d['X_joints_test']  # (240, 680, 4)
X_press_test = d['X_press_test']    # (240, 680, 4)
Y_test = d['Y_test']                 # (240, 680)
```

## Physical Model

All signals are derived from:
- Bekker pressure-sinkage law: p = (k_c/b + k_phi) * z^n
- Mohr-Coulomb shear strength: tau = c + sigma * tan(phi)
- Soil parameters from Wong (2001), Theory of Ground Vehicles, 3rd ed.
