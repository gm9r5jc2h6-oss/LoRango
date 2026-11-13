# Data format and split discipline

The JSON file must contain exactly `train` and `calibration` arrays. Each row has a unique ID, a base prompt and a target prompt.

```json
{
  "id": "scene_001",
  "base_prompt": "a robot in a museum",
  "target_prompt": "a dog in a museum, same viewpoint and background"
}
```

The target prompt should preserve scene attributes that are not meant to change. For object replacement, explicitly retain viewpoint and background; for style changes, retain subject and composition.

The loader normalizes case and whitespace and rejects any base prompt, target
prompt, or `(base, target)` pair shared between train and calibration. IDs must
be globally unique. Seed values must be unique within each split and disjoint
across train/calibration. These checks enforce the artifact's use of the term
“held-out”; semantic near-duplicates still require human review.

Recommended minimum:

- train: 8 diverse scenes;
- calibration: 2 unseen scenes;
- train seeds: 2;
- calibration seeds: at least 1, disjoint from train seeds.

Paper validation and sealed-test prompts must not be placed in either split. The toolkit records the exact data-file hash in `RUN_METADATA.json`.
