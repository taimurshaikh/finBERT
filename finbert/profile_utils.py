import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.ao.quantization

def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def get_profiler_activities(device):
    """Get appropriate profiler activities based on device"""
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def print_profiler_results(prof, device, title="Profiling Results"):
    """Pretty print profiler results"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")
    
    print("By CPU Time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    if device.type == "cuda":
        print("\nBy CUDA Time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    print(f"\n{'='*80}\n")


def setup_nltk_data():
    """Download necessary NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass


def print_device_info(device):
    """Print device information"""
    print(f"\n{'='*80}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    elif device.type == "mps":
        print("Note: MPS profiling shows CPU time only. Actual GPU execution time not separately tracked.")
    print(f"{'='*80}\n")


def quantize_int8_model(model, calibration_loader=None, device='cuda'):
    """
    Apply INT8 post-training quantization using TorchAO for GPU inference.
    
    Args:
        model: The model to quantize
        calibration_loader: Optional data loader for calibration (not used for dynamic quantization)
        device: Device to run quantization on
    
    Returns:
        Quantized model ready for GPU inference
    """
    try:
        from torchao.quantization import quantize_, int8_dynamic_activation_int4_weight
        
        # Move model to GPU for quantization
        model = model.to(device)
        model.eval()
        
        # Apply dynamic INT8 activation with INT4 weight quantization
        # This provides good compression with minimal accuracy loss
        quantize_(model, int8_dynamic_activation_int4_weight())
        
        print("✓ Applied TorchAO INT8 dynamic quantization")
        return model
        
    except ImportError:
        print("⚠ TorchAO not available, falling back to torch.ao.quantization")
        # Fallback to standard PyTorch dynamic quantization (CPU-optimized)
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model.cpu(),
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        print("✓ Applied torch.ao dynamic quantization (CPU-optimized)")
        return quantized_model.to(device)


print("✓ Helper utilities loaded")
