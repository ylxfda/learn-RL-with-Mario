# Test Scripts

Utility scripts for analyzing and visualizing DreamerV3 training data.

## extract_episode.py

Extract and visualize episode data from NPZ files saved during training.

### Usage

```bash
python test_scripts/extract_episode.py <npz_file> [-o output_dir]
```

### Arguments

- `npz_file`: Path to the NPZ episode file (e.g., `logdir/mario-debug/train_eps/env0-1266.npz`)
- `-o, --output`: Optional output directory (default: creates folder with same basename as NPZ file)

### Example

```bash
# Extract episode data
python test_scripts/extract_episode.py logdir/mario-debug/train_eps/env0-1266.npz

# This creates: logdir/mario-debug/train_eps/env0-1266/
# - image_1266.gif: Animated GIF of observations (1266 frames)
# - reward.csv: Reward values per timestep
# - action.csv: One-hot encoded actions per timestep
# - discount.csv: Discount factors per timestep
# - logprob.csv: Action log probabilities per timestep
# - is_first.csv: Episode start flags
# - is_terminal.csv: Episode end flags
```

### Output Format

- **Images**: Saved as animated GIF with frame count in filename (e.g., `image_1266.gif`)
- **Other data**: Saved as CSV files with timestep index
  - 1D arrays: Single column (e.g., `reward`)
  - 2D arrays: Multiple columns (e.g., `action_0`, `action_1`, ...)

### Requirements

- numpy
- PIL/Pillow

No pandas required!
