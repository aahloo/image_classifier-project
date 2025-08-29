# Image Classifier Project

A deep learning image classifier that identifies flower species using PyTorch and pre-trained VGG architectures.

## 🌸 What It Does

- **Classifies 102 flower species** with high accuracy
- **Pre-trained VGG backbone** (VGG13/VGG16) with custom classifier
- **GPU acceleration** for training and inference
- **Top-K predictions** with confidence scores
- **Command line interface** for easy deployment

## 📁 Project Structure

```
├── Image_Classifier.ipynb    # Jupyter notebook with model development
├── train.py                  # Command line training script
├── predict.py                # Command line prediction script
├── test_app.py               # Automated testing script
├── cat_to_name.json          # Flower species name mapping
└── README.md                 # This file
```

**What the test script does:**
- Trains a model with minimal epochs (quick test)
- Saves a checkpoint file
- Tests prediction on a sample image
- Validates output format and accuracy
- Reports success/failure for each component

**Expected output:**
```
=== Testing Training ===
Running: python train.py flowers/ --epochs 1 --hidden_units 256
✅ Training completed successfully!

=== Testing Prediction ===
Running: python predict.py flowers/test/10/image_07104.jpg checkpoint.pth --top_k 3
✅ Prediction completed successfully!
```

## 📊 Jupyter Notebook

- **Data exploration** and preprocessing
- **Model architecture** design and training
- **Validation** and accuracy testing  
- **Visualization** of predictions with matplotlib
- **Sanity checks** with sample images

## 🔧 Command Line Options

### train.py
| Option | Default | Description |
|--------|---------|-------------|
| `data_dir` | - | Path to dataset (required) |
| `--save_dir` | `.` | Checkpoint save directory |
| `--arch` | `vgg13` | Architecture (vgg13, vgg16) |
| `--learning_rate` | `0.001` | Learning rate |
| `--hidden_units` | `512` | Hidden layer size |
| `--epochs` | `3` | Training epochs |
| `--gpu` | `False` | Use GPU acceleration |

### predict.py
| Option | Default | Description |
|--------|---------|-------------|
| `input` | - | Input image path (required) |
| `checkpoint` | - | Model checkpoint (required) |
| `--top_k` | `1` | Number of predictions |
| `--category_names` | - | JSON file with flower names |
| `--gpu` | `False` | Use GPU for inference |

## 🎯 Model Performance

- **Training accuracy:** ~95%+ on flower dataset
- **Validation accuracy:** ~85%+ (varies by hyperparameters)
- **Inference time:** < 1 second per image (GPU)
- **Model size:** ~500MB (including VGG weights)

## 📋 Requirements

- Python 3.7+
- PyTorch
- torchvision  
- PIL (Pillow)
- NumPy
- Matplotlib
- JSON

Install with: `pip install torch torchvision pillow numpy matplotlib`

## 📚 Dataset

Expects data in this structure:
```
flowers/
├── train/
│   ├── 1/
│   ├── 2/
│   └── ...
├── valid/
│   ├── 1/
│   ├── 2/
│   └── ...
└── test/
    ├── 1/
    ├── 2/
    └── ...
```

Each numbered folder represents a flower class (1-102).

## 🔍 Example Usage

```bash
# Train a model
python train.py flowers/ --epochs 5 --gpu

# Predict top 3 most likely flowers
python predict.py flowers/test/15/image_06351.jpg checkpoint.pth \
    --top_k 3 --category_names cat_to_name.json

# Output:
# Top 3 predictions:
# 1. yellow iris: 0.8934
# 2. bearded iris: 0.0512  
# 3. dutch iris: 0.0324
```

