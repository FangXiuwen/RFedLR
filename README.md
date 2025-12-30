# Towards Robust Parameter-Efficient Fine-Tuning for Federated Learning

This repository provides resources for the following paper:

> [**Towards Robust Parameter-Efficient Fine-Tuning for Federated Learning**]([NeurIPS Poster Towards Robust Parameter-Efficient Fine-Tuning for Federated Learning](https://neurips.cc/virtual/2025/loc/san-diego/poster/119736))  

> Xiuwen Fang, [Mang Ye](https://marswhu.github.io/index.html)  

> *NeurIPS 2025*

## 🏗️ Project Structure

```
RFedLR/
├── Dataset/                    # Dataset utilities and data loading
│   ├── __init__.py
│   ├── cifar.py
│   ├── init_dataset.py
│   ├── utils.py               # Data loading, partitioning, and preprocessing
│   └── sampling.py            # IID and non-IID data sampling methods
├── HHF/                       # Main training algorithms
│   ├── RFedLR.py              # Main federated learning implementation
│   └── myoptim.py             # Custom optimizers (SelectiveBackPropSGD)
└── Network/                   # Neural network models
    └── Models_Def/
        └── lora.py            # LoRA implementation for Vision Transformer
```

## 📋 Requirements

### Dependencies

```bash
# Core ML libraries
torch>=1.9.0
torchvision>=0.10.0
timm>=0.6.0
numpy>=1.21.0

# Data processing
pandas>=1.3.0
PIL>=8.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Additional utilities
einops>=0.4.0
safetensors>=0.3.0

# For downloading models from Hugging Face
huggingface_hub>=0.10.0
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested with 4 GPUs)

- **Memory**: At least 16GB RAM recommended

- **Storage**: ~2GB for datasets and models

## 📦 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/FangXiuwen/RFedLR
cd RFedLR
```

2. **Install dependencies**:

```bash
pip install torch torchvision timm numpy pandas matplotlib seaborn einops safetensors pillow huggingface_hub
```

## 📁 Dataset Setup

### CIFAR-100 Dataset

**Important**: You need to download the CIFAR-100 dataset and place it in the `Dataset` folder.

1. **Create the dataset directory**:

```bash
mkdir -p Dataset/cifar_100
```

2. **Download CIFAR-100**:

The dataset will be automatically downloaded when you first run the code, or you can manually download it:

```bash
# The code will automatically download CIFAR-100 to Dataset/cifar_100/

# when you run the training script for the first time
```

3. **Verify dataset structure**:

```
Dataset/
└── cifar_100/
    ├── cifar-100-python/
    │   ├── train
    │   ├── test
    │   └── meta
    └── ...
```

## 🤖 Model Setup

### Vision Transformer Pretrained Model

**Important**: You need to download the ViT base model and place it in the `Network/Models_Def` folder.

1. **Download the pretrained ViT model**:

You can download the ViT base model from Hugging Face: [https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

1. Go to [https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

2. Click on "Files and versions"

3. Download `pytorch_model.bin`

4. Rename it to `vit_base_patch16_224.bin`

5. Place it in `Network/Models_Def/` folder

2. **Verify model file**:

```bash
ls -la Network/Models_Def/vit_base_patch16_224.bin
```

The file should be approximately 330MB in size (86.6M parameters as shown on the [Hugging Face model page](https://huggingface.co/google/vit-base-patch16-224)).

## 🏃‍♂️ Usage

### Basic Training

Run the federated learning training with default parameters:

```bash
cd HHF
python RFedLR.py
```

### Configuration

Key parameters can be modified in `RFedLR.py`:

```python
# Training Configuration
N_Participants = 5              # Number of federated clients
CommunicationEpoch = 40         # Number of communication rounds
TrainBatchSize = 256           # Training batch size
TestBatchSize = 512            # Testing batch size

# Noise Configuration
Noise_type = 'symmetric'        # ['pairflip', 'symmetric', None]
Noise_rate = 0.6               # Noise rate (0.0 to 1.0)

# Data Configuration
Private_Dataset_Name = 'cifar100'
Data_Partition = 'noniid'       # ['iid', 'noniid']
Noniid_Dirichlet_Beta = 0.5    # Non-IID distribution parameter

# LoRA Configuration
# In the model initialization:
# r=4, alpha=4 for LoRA parameters
```

### Output

The training will generate:

- **Model checkpoints**: Saved in `Model_Storage/` directory

- **Training logs**: Console output and log files

- **Performance metrics**: Training loss and test accuracy CSV files

## 🔧 Advanced Configuration

### Multi-GPU Setup

The code automatically detects and uses available GPUs:

```python
device_ids = [0,1,2,3]  # Modify based on your GPU setup
```

### Custom Noise Types

Implement custom noise in `Dataset/init_dataset.py`:

- Symmetric noise: Random label flipping

- Pairflip noise: Structured label confusion

### Hyperparameter Tuning

Key hyperparameters to adjust:

- `learning_rate`: 0.01 for LoRA, 0.001 for full fine-tuning

- `Robust_Ratio`: 0.2 (ratio of parameters to keep robust)

- `Importance_Weight`: 0.4 (balance between importance and data size)

## 📚 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{nips2025rfedlr,
  title={Towards Robust Parameter-Efficient Fine-Tuning for Federated Learning},
  author={Fang, Xiuwen and Ye, Mang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
