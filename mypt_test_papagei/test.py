import sys
sys.path.append("/home/ajhz380/projects/papagei-foundation-model")

# Import Necessary Packages:
import numpy as np
import torch
from linearprobing.utils import resample_batch_signal, load_model_without_module_prefix
from preprocessing.ppg import preprocess_one_ppg_signal
from segmentations import waveform_to_segments
from torch_ecg._preprocessors import Normalize
from models.resnet import ResNet1DMoE

# Load the PaPaGei-S Model:
## Define Model Configuration
model_config = {
    'base_filters': 32,
    'kernel_size': 3,
    'stride': 2,
    'groups': 1,
    'n_block': 18,
    'n_classes': 512, # Embedding dimension
    'n_experts': 3
}

## Initialize Model
model = ResNet1DMoE(
    in_channels=1,
    base_filters=model_config['base_filters'],
    kernel_size=model_config['kernel_size'],
    stride=model_config['stride'],
    groups=model_config['groups'],
    n_block=model_config['n_block'],
    n_classes=model_config['n_classes'],
    n_experts=model_config['n_experts']
)

## Load Pre-trained Weights
model_path = "weights/papagei_s.pt" # Ensure this path is correct
model = load_model_without_module_prefix(model, model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval() # Set model to evaluation mode
print(f"Model loaded on {device}")

# Pre-process a PPG Signal:
## Example PPG Signal
fs = 500  # Original sampling frequency in Hz
fs_target = 125 # Target sampling frequency in Hz
segment_duration_seconds = 10 # Duration of each segment in seconds
signal_duration_seconds = 60 # Total duration of the example signal

signal = np.random.randn(signal_duration_seconds * fs) # Example: 60s signal at 500Hz
print(f"Original PPG dimensions: {signal.shape}")

## Clean and segment the signal
signal_processed, _, _, _ = preprocess_one_ppg_signal(waveform=signal, frequency=fs)

segment_length_original_fs = fs * segment_duration_seconds
segmented_signals = waveform_to_segments(
    waveform_name='ppg', # Can be any name, not strictly used in this function
    segment_length=segment_length_original_fs,
    clean_signal=signal_processed
)

## Resample segments
resampled_segments = resample_batch_signal(
    segmented_signals, 
    fs_original=fs, 
    fs_target=fs_target, 
    axis=-1
)
print(f"After segmentation and resampling: {resampled_segments.shape}") # (num_segments, segment_length_target_fs)

## Convert to PyTorch Tensor
signal_tensor = torch.Tensor(resampled_segments).unsqueeze(dim=1).to(device) # (num_segments, 1, segment_length_target_fs)

# Extract Embeddings:
with torch.inference_mode():
    outputs = model(signal_tensor)
    # PaPaGei-S returns a tuple (embeddings, expert_outputs, gating_weights)
    # We are interested in the first element: embeddings
    embeddings = outputs[0].cpu().detach().numpy()
print(f"Embedding dimensions: {embeddings.shape}") # (num_segments, n_classes)