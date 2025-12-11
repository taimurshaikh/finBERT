from __future__ import absolute_import, division, print_function

import random

import pandas as pd
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
    TensorDataset)
from torch.optim import AdamW
# PyTorch
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.ao.quantization


from tqdm import tqdm
from tqdm import trange
from nltk.tokenize import sent_tokenize
from finbert.utils import *
import numpy as np
import logging

from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
from .profile_utils import get_model_size_mb, get_profiler_activities, print_profiler_results, setup_nltk_data, print_device_info
from finbert.finbert import FinBert

import time

class ProfiledFinBert(FinBert):
    """Extended FinBert class with profiling instrumentation.
    
    Note: GPU-specific profiling (ProfilerActivity.CUDA) only works with NVIDIA CUDA devices.
    For MPS (Apple Silicon), only CPU profiling is available, though actual computation runs on GPU.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.profile_results = {}
    
    def train(self, train_examples, model):
        """
        Trains the model with profiling instrumentation.
        """
        validation_examples = self.get_data('validation')
        global_step = 0
        self.validation_losses = []
        
        # Training
        train_dataloader = self.get_loader(train_examples, 'train')
        model.train()
        step_number = len(train_dataloader)
        
        # Setup profiler - CUDA profiling only works with NVIDIA GPUs, not MPS
        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        
        print("\\n" + "="*80)
        print("Starting Profiled Training")
        print(f"Device: {self.device}")
        print(f"Profiling activities: {activities}")
        if self.device.type == "mps":
            print("Note: MPS profiling shows CPU time only. Actual GPU execution time not separately tracked.")
        print("="*80 + "\\n")
        
        i = 0

        scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False
        ) as prof:
            
            for epoch in trange(int(self.config.num_train_epochs), desc="Epoch"):
                model.train()
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                
                for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                    
                    # Gradual unfreezing logic
                    if (self.config.gradual_unfreeze and i == 0):
                        for param in model.bert.parameters():
                            param.requires_grad = False
                    
                    if (step % (step_number // 3)) == 0:
                        i += 1
                    
                    if (self.config.gradual_unfreeze and i > 1 and i < self.config.encoder_no):
                        for k in range(i - 1):
                            try:
                                for param in model.bert.encoder.layer[self.config.encoder_no - 1 - k].parameters():
                                    param.requires_grad = True
                            except:
                                pass
                    
                    if (self.config.gradual_unfreeze and i > self.config.encoder_no + 1):
                        for param in model.bert.embeddings.parameters():
                            param.requires_grad = True
                    
                    # Data loading profiling
                    with record_function("data_transfer"):
                        batch = tuple(t.to(self.device) for t in batch)
                        input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch
                    
                    # Forward pass profiling
                    with record_function("forward_pass"):
                        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                            logits = model(input_ids, attention_mask, token_type_ids)[0]
                    
                    # Loss calculation profiling
                    with record_function("loss_calculation"):
                        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                            weights = self.class_weights.to(self.device)
                            if self.config.output_mode == "classification":
                                loss_fct = CrossEntropyLoss(weight=weights)
                                loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                            elif self.config.output_mode == "regression":
                                loss_fct = MSELoss()
                                loss = loss_fct(logits.view(-1), label_ids.view(-1))
                            
                            if self.config.gradient_accumulation_steps > 1:
                                loss = loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass profiling
                    with record_function("backward_pass"):
                        if self.config.use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    
                    # Optimizer step profiling
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        with record_function("optimizer_step"):
                            if self.config.fp16:
                                lr_this_step = self.config.learning_rate * warmup_linear(
                                    global_step / self.num_train_optimization_steps, self.config.warm_up_proportion)
                                for param_group in self.optimizer.param_groups:
                                    param_group['lr'] = lr_this_step
                            
                            if self.config.use_amp:
                                scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                scaler.step(self.optimizer)
                                scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                self.optimizer.step()
                                
                            self.scheduler.step()
                            self.optimizer.zero_grad()
                            global_step += 1
                    
                    # Only profile first epoch to save time
                    if epoch == 0 and step >= 20:
                        break
                
                # Break after first epoch for profiling
                if epoch == 0:
                    print("\\n" + "="*80)
                    print("Profiling complete for first epoch (20 steps)")
                    print("Continuing full training without profiling...")
                    print("="*80 + "\\n")
                    break
        
        # Print profiler results
        print("\\n" + "="*80)
        print("PROFILING RESULTS - Training")
        print("="*80 + "\\n")
        
        print("\\nBy CPU Time:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        
        if self.device.type == "cuda":
            print("\\nBy CUDA Time:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        print("\\n" + "="*80 + "\\n")
        
        # Store results
        self.profile_results['training'] = prof.key_averages()
        
        # Continue with full training without profiling
        for epoch in trange(int(self.config.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                
                if (self.config.gradual_unfreeze and i == 0):
                    for param in model.bert.parameters():
                        param.requires_grad = False
                
                if (step % (step_number // 3)) == 0:
                    i += 1
                
                if (self.config.gradual_unfreeze and i > 1 and i < self.config.encoder_no):
                    for k in range(i - 1):
                        try:
                            for param in model.bert.encoder.layer[self.config.encoder_no - 1 - k].parameters():
                                param.requires_grad = True
                        except:
                            pass
                
                if (self.config.gradual_unfreeze and i > self.config.encoder_no + 1):
                    for param in model.bert.embeddings.parameters():
                        param.requires_grad = True
                
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch
                
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    logits = model(input_ids, attention_mask, token_type_ids)[0]
                    weights = self.class_weights.to(self.device)
                    
                    if self.config.output_mode == "classification":
                        loss_fct = CrossEntropyLoss(weight=weights)
                        loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                    elif self.config.output_mode == "regression":
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), label_ids.view(-1))
                    
                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps
                
                if self.config.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        lr_this_step = self.config.learning_rate * warmup_linear(
                            global_step / self.num_train_optimization_steps, self.config.warm_up_proportion)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    
                    if self.config.use_amp:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
            
            # Validation
            validation_loader = self.get_loader(validation_examples, phase='eval')
            model.eval()
            
            valid_loss, valid_accuracy = 0, 0
            nb_valid_steps, nb_valid_examples = 0, 0
            
            for input_ids, attention_mask, token_type_ids, label_ids, agree_ids in tqdm(validation_loader, desc="Validating"):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                agree_ids = agree_ids.to(self.device)
                
                with torch.no_grad():
                    logits = model(input_ids, attention_mask, token_type_ids)[0]
                    
                    if self.config.output_mode == "classification":
                        loss_fct = CrossEntropyLoss(weight=weights)
                        tmp_valid_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                    elif self.config.output_mode == "regression":
                        loss_fct = MSELoss()
                        tmp_valid_loss = loss_fct(logits.view(-1), label_ids.view(-1))
                    
                    valid_loss += tmp_valid_loss.mean().item()
                    nb_valid_steps += 1
            
            valid_loss = valid_loss / nb_valid_steps
            self.validation_losses.append(valid_loss)
            print("Validation losses: {}".format(self.validation_losses))
            
            if valid_loss == min(self.validation_losses):
                try:
                    os.remove(self.config.model_dir / ('temporary' + str(best_model)))
                except:
                    print('No best model found')
                torch.save({'epoch': str(epoch), 'state_dict': model.state_dict()},
                           self.config.model_dir / ('temporary' + str(epoch)))
                best_model = epoch
        
        # Save the trained model
        checkpoint = torch.load(self.config.model_dir / ('temporary' + str(best_model)))
        model.load_state_dict(checkpoint['state_dict'])
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(self.config.model_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(self.config.model_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        os.remove(self.config.model_dir / ('temporary' + str(best_model)))
        
        return model




def profile_inference(text, model, write_to_csv=False, path=None, variant_name="unknown", use_gpu=False, gpu_name='cuda:0', batch_size=5, use_amp=False):
    """
    Profile model inference performance.
    
    Args:
        model: Model to profile
        text: Text to analyze
        variant_name: Name of the model variant
        use_gpu: Whether to use GPU
        gpu_name: GPU device name
        batch_size: Batch size for inference
    
    Returns:
        results_df: DataFrame with predictions
        metrics: Dictionary with performance metrics
    """
    from nltk.tokenize import sent_tokenize
    from finbert.utils import InputExample, convert_examples_to_features, softmax, chunks, get_device
    
    # Setup NLTK
    setup_nltk_data()
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Device selection
    if use_gpu:
        device = get_device(no_cuda=False)
        if device.type == "cuda" and gpu_name.startswith("cuda:"):
            device = torch.device(gpu_name)
    else:
        device = torch.device("cpu")
    
    print_device_info(device)
    
    # Check if model is already on device (e.g., BitsAndBytes quantized models)
    is_quantized = hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit
    is_quantized = is_quantized or (hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit)
    
    # Only move model if it's not already quantized and placed
    if not is_quantized:
        model = model.to(device)
    else:
        print(f"✓ Model already quantized and placed on device (skipping .to() call)")
        # For quantized models, get the actual device from model
        if hasattr(model, 'device'):
            device = model.device
        elif hasattr(model, 'hf_device_map'):
            # BitsAndBytes models have device_map
            device = torch.device('cuda:0')  # Usually on cuda:0
    
    
    label_list = ['positive', 'negative', 'neutral']
    label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    result = pd.DataFrame(columns=['sentence', 'logit', 'prediction', 'sentiment_score'])
    
    # Setup profiler
    activities = get_profiler_activities(device)
    
    total_inference_time = 0
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        
        with record_function("sentence_tokenization"):
            sentences = sent_tokenize(text)
        
        for batch in chunks(sentences, batch_size):
            with record_function("create_examples"):
                examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]
            
            with record_function("convert_to_features"):
                features = convert_examples_to_features(examples, label_list, 64, tokenizer)
            
            with record_function("prepare_tensors"):
                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
                all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
                all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(device)
            
            with torch.no_grad():
                # Remove the model_to_device profiling section for quantized models
                if not is_quantized:
                    with record_function("model_to_device"):
                        model = model.to(device)
                
                with record_function("inference_forward"):
                    start_time = time.time()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(input_ids=all_input_ids, attention_mask=all_attention_mask, token_type_ids=all_token_type_ids)[0]
                    total_inference_time += time.time() - start_time
                
                with record_function("postprocess_results"):
                    logits = softmax(np.array(logits.cpu()))
                    sentiment_score = pd.Series(logits[:, 0] - logits[:, 1])
                    predictions = np.squeeze(np.argmax(logits, axis=1))
                    
                    batch_result = {
                        'sentence': batch,
                        'logit': list(logits),
                        'prediction': predictions,
                        'sentiment_score': sentiment_score
                    }
                    
                    batch_result = pd.DataFrame(batch_result)
                    result = pd.concat([result, batch_result], ignore_index=True)
    
    # Print profiler results
    print_profiler_results(prof, device, title=f"Inference Profiling - {variant_name}")
    
    result['prediction'] = result.prediction.apply(lambda x: label_dict[x])
    if write_to_csv:
        result.to_csv(path, sep=',', index=False)
    
    # Extract profiler timings (prefer CUDA times when available on CUDA devices)
    ka = prof.key_averages()

    def _event_time_ms(events, key, prefer_cuda):
        for e in events:
            if e.key == key:
                # Some profiler builds may not have cuda_time_total attribute; guard access
                cuda_ms = getattr(e, "cuda_time_total", 0) / 1000.0 if getattr(e, "cuda_time_total", 0) else 0.0
                cpu_ms = getattr(e, "cpu_time_total", 0) / 1000.0 if getattr(e, "cpu_time_total", 0) else 0.0
                if prefer_cuda and cuda_ms:
                    return cuda_ms
                return cpu_ms
        return 0.0

    prefer_cuda = (device.type == "cuda")
    tokenization_time_ms = _event_time_ms(ka, "sentence_tokenization", prefer_cuda)
    inference_forward_time_ms = _event_time_ms(ka, "inference_forward", prefer_cuda)
    model_to_device_time_ms = _event_time_ms(ka, "model_to_device", prefer_cuda)

    metrics = {
        'variant': variant_name,
        'total_sentences': len(sentences),
        'inference_time_ms': total_inference_time * 1000,
        'time_per_sentence_ms': (total_inference_time * 1000) / len(sentences) if len(sentences) else 0.0,
        'device': str(device),
        'is_quantized': is_quantized,
        'tokenization_time_ms': tokenization_time_ms,
        'inference_forward_time_ms': inference_forward_time_ms,
        'model_to_device_time_ms': model_to_device_time_ms,
    }
    
    print(f"\nInference Summary:")
    print(f"  Total sentences: {metrics['total_sentences']}")
    print(f"  Total inference time: {metrics['inference_time_ms']:.2f} ms")
    print(f"  Time per sentence: {metrics['time_per_sentence_ms']:.2f} ms")
    if is_quantized:
        print(f"  ✓ Quantized model profiled successfully")
    
    return result, metrics


