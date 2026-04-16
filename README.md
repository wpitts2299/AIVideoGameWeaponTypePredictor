# Weapon Classification AI Prototype

This project trains a vision model to classify weapon images into one of these classes:

- `Automatic Rifle`
- `Bazooka`
- `Grenade Launcher`
- `Handgun`
- `Knife`
- `SMG`
- `Shotgun`
- `Sniper`
- `Sword`

The current kept model is the 3-epoch checkpoint.

## Project Files

### Python files

[`prepare_dataset.py`](c:/Users/wdp22/weapon_ai_prototype/prepare_dataset.py:1)
- Merges the two raw datasets, normalizes labels, and writes the generated dataset files used for training.
- Reads from `WPDataSet1` and `WPDataSet2`.
- Writes `train.json`, `val.json`, and `dataset_stats.json`.

[`train_model.py`](c:/Users/wdp22/weapon_ai_prototype/train_model.py:1)
- Trains the classifier and saves a `.pth` checkpoint.
- Uses a Vision Transformer (`vit_b_16`) from `torchvision`.
- Supports:
  - normal training on `train.json` and `val.json`
  - full K-fold cross-validation
  - single seeded fold-split training

[`evaluate_model.py`](c:/Users/wdp22/weapon_ai_prototype/evaluate_model.py:1)
- Evaluates a saved checkpoint on `val.json`.
- Prints overall metrics and per-class metrics.
- Writes evaluation results to a JSON stats file.

[`predict_weapon.py`](c:/Users/wdp22/weapon_ai_prototype/predict_weapon.py:1)
- Loads a saved checkpoint and predicts class probabilities for one image.
- Used for one-off testing on images such as the files in `TestGamingWeaponImages`.

### JSON files

[`train.json`](c:/Users/wdp22/weapon_ai_prototype/train.json:1)
- Generated training split.
- Each entry contains:
  - `image_path`
  - `label`

[`val.json`](c:/Users/wdp22/weapon_ai_prototype/val.json:1)
- Generated validation split.
- Used during evaluation and during standard training mode.

[`dataset_stats.json`](c:/Users/wdp22/weapon_ai_prototype/dataset_stats.json:1)
- Summary of the merged dataset.
- Includes:
  - total/train/val sample counts
  - per-class counts
  - source dataset counts
  - skipped item counts

[`training_stats.json`](c:/Users/wdp22/weapon_ai_prototype/training_stats.json:1)
- Training summary for the default 3-epoch model file `weapon_model.pth`.
- Includes:
  - device used
  - classes
  - class counts
  - best epoch
  - best validation metrics
  - confusion matrix
  - epoch-by-epoch history

[`training_stats_epoch3.json`](c:/Users/wdp22/weapon_ai_prototype/training_stats_epoch3.json:1)
- Same training summary as above, but tied to the explicitly named 3-epoch checkpoint file `weapon_model_epoch3.pth`.

[`evaluation_stats_epoch3.json`](c:/Users/wdp22/weapon_ai_prototype/evaluation_stats_epoch3.json:1)
- Saved evaluation results for the 3-epoch model on `val.json`.
- Includes:
  - accuracy
  - precision
  - recall
  - F1
  - per-class report
  - confusion matrix

### Model files

[`weapon_model.pth`](c:/Users/wdp22/weapon_ai_prototype/weapon_model.pth)
- Default model checkpoint used by the scripts if no other checkpoint is specified.

[`weapon_model_epoch3.pth`](c:/Users/wdp22/weapon_ai_prototype/weapon_model_epoch3.pth)
- Explicitly named 3-epoch checkpoint.

Note: `weapon_model.pth` and `weapon_model_epoch3.pth` are currently identical copies of the same trained model.

## How the Dataset Was Built

The final training dataset was created by merging two raw sources:

- [`WPDataSet1`](c:/Users/wdp22/weapon_ai_prototype/WPDataSet1)
- [`WPDataSet2`](c:/Users/wdp22/weapon_ai_prototype/WPDataSet2)

Dataset preparation details:

- `WPDataSet1` labels were read from the CVAT XML tags.
- `WPDataSet2` labels and train/validation split came from `metadata.csv`.
- Labels were normalized into a single shared class list.
- `WPDataSet1` used a stratified split with random seed `42`.
- `WPDataSet2` kept its provided train/val split.

Final merged dataset:

- Total samples: `734`
- Train samples: `587`
- Validation samples: `147`

Per-class totals:

- `Automatic Rifle`: `74`
- `Bazooka`: `67`
- `Grenade Launcher`: `88`
- `Handgun`: `90`
- `Knife`: `72`
- `SMG`: `87`
- `Shotgun`: `78`
- `Sniper`: `89`
- `Sword`: `89`

Source totals:

- `WPDataSet1`: `20`
- `WPDataSet2`: `714`

## How the Current Model Was Trained

The kept model is the 3-epoch model.

Training setup:

- Architecture: `torchvision.models.vit_b_16`
- Pretrained weights: `ViT_B_16_Weights.DEFAULT`
- Image size: `224 x 224`
- Batch size: `32`
- Optimizer: `AdamW`
- Learning rate: `1e-4`
- Epochs run: `3`
- Device used during training: `cuda`

Training behavior:

- Training used `train.json`
- Validation used `val.json`
- The best checkpoint was selected based on validation accuracy
- Best checkpoint occurred at epoch `2`

Epoch summary:

1. Epoch 1: train accuracy `0.5349`, validation accuracy `0.7483`
2. Epoch 2: train accuracy `0.8739`, validation accuracy `0.8707`
3. Epoch 3: train accuracy `0.9591`, validation accuracy `0.8435`

## Current Model Stats

Overall validation metrics for the kept 3-epoch model:

- Accuracy: `0.8707`
- Precision: `0.8834`
- Recall: `0.8707`
- F1: `0.8700`
- Validation samples: `147`

Per-class results:

- `Automatic Rifle`: precision `0.7333`, recall `0.7333`, F1 `0.7333`
- `Bazooka`: precision `0.9286`, recall `1.0000`, F1 `0.9630`
- `Grenade Launcher`: precision `0.6400`, recall `0.8889`, F1 `0.7442`
- `Handgun`: precision `0.9444`, recall `0.9444`, F1 `0.9444`
- `Knife`: precision `1.0000`, recall `1.0000`, F1 `1.0000`
- `SMG`: precision `0.8333`, recall `0.5882`, F1 `0.6897`
- `Shotgun`: precision `1.0000`, recall `0.7500`, F1 `0.8571`
- `Sniper`: precision `0.8947`, recall `0.9444`, F1 `0.9189`
- `Sword`: precision `1.0000`, recall `1.0000`, F1 `1.0000`

Areas where the model is strongest:

- `Knife`
- `Sword`
- `Bazooka`
- `Handgun`
- `Sniper`

Weaker classes in this validation run:

- `SMG`
- `Automatic Rifle`
- `Grenade Launcher`

## Typical Commands

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Rebuild the dataset JSON files:

```powershell
python prepare_dataset.py
```

Train the default model:

```powershell
python train_model.py
```

Evaluate the default model:

```powershell
python evaluate_model.py
```

Predict from one image using the default model:

```powershell
python predict_weapon.py "C:\path\to\image.png"
```

Predict using the explicitly named 3-epoch checkpoint:

```powershell
.\.venv\Scripts\python.exe -c "import torch; import torch.nn as nn; from PIL import Image; from torchvision import transforms; from torchvision.models import vit_b_16; MODEL_PATH=r'C:\Users\wdp22\weapon_ai_prototype\weapon_model_epoch3.pth'; IMAGE_PATH=r'C:\path\to\image.png'; DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); checkpoint=torch.load(MODEL_PATH, map_location=DEVICE); classes=checkpoint['classes']; model=vit_b_16(weights=None); in_features=model.heads.head.in_features; model.heads.head=nn.Linear(in_features, len(classes)); model.load_state_dict(checkpoint['model_state_dict']); model=model.to(DEVICE); model.eval(); transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]); img=Image.open(IMAGE_PATH).convert('RGB'); x=transform(img).unsqueeze(0).to(DEVICE); out=model(x); probs=torch.softmax(out, dim=1).squeeze(0).detach().cpu().numpy(); order=probs.argsort()[::-1]; print(f'Model: {MODEL_PATH}'); print(f'Device: {DEVICE}'); print(f'Top Prediction: {classes[order[0]]} ({probs[order[0]]*100:.2f}%)'); print(); print('All Probabilities:'); [print(f'{classes[i]}: {probs[i]*100:.2f}%') for i in order]"
```
