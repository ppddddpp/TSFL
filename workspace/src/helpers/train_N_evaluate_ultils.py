import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

def train_client_locally(model: nn.Module, dataset: Subset, \
                        current_round: int, epochs: int = 3, device: str = "cpu") -> nn.Module:
    if len(dataset) < 10:
        print(f"Skipping client with {len(dataset)} samples (too few for stable training).")
        return model
    
    model.train()
    model.to(device)
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    current_lr = 0.1 * (0.99 ** (current_round - 1))
    optimizer = optim.SGD(model.parameters(), lr=current_lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    device_type = device.type if isinstance(device, torch.device) else str(device).split(':')[0]
    is_cuda = device_type == 'cuda'
    scaler = torch.amp.GradScaler(device=device_type, enabled=is_cuda)

    model_dtype = next(model.parameters()).dtype

    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            data = data.to(model_dtype)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device_type, enabled=is_cuda):
                output = model(data)
                loss = criterion(output, target)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    return model

def compute_client_update(global_model: nn.Module, local_model: nn.Module) -> dict:
    update_delta = {}
    global_dict = global_model.state_dict()
    local_dict = local_model.state_dict()
    
    is_corrupted = False

    for key in global_dict.keys():
        if 'num_batches_tracked' in key:
            continue
        delta = local_dict[key].cpu() - global_dict[key].cpu()
        if torch.isnan(delta).any() or torch.isinf(delta).any():
            is_corrupted = True
            break
            
    if is_corrupted:
        print("[WARNING] Client update corrupted (NaN/Inf detected).")
        for key in global_dict.keys():
            if 'num_batches_tracked' in key:
                continue
            update_delta[key] = torch.zeros_like(global_dict[key].cpu(), dtype=global_dict[key].dtype)
    else:
        for key in global_dict.keys():
            if 'num_batches_tracked' in key:
                continue
            delta = local_dict[key].cpu() - global_dict[key].cpu()
            update_delta[key] = delta.to(global_dict[key].dtype)
            
    return update_delta

def evaluate_model(model: nn.Module, test_images: torch.Tensor, test_labels: torch.Tensor, num_classes: int, device: str) -> dict:
    model.eval()
    
    total_samples = len(test_labels)
    chunk_size = max(1000, total_samples // 10) 
    
    correct = 0
    
    precision_calc = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    recall_calc = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    f1_calc = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)

    device_type = device.type if isinstance(device, torch.device) else str(device).split(':')[0]
    is_cuda = device_type == 'cuda'

    model_dtype = next(model.parameters()).dtype
    
    with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=is_cuda):
        for i in range(0, total_samples, chunk_size):
            img_chunk = test_images[i : i + chunk_size]
            label_chunk = test_labels[i : i + chunk_size]

            img_chunk = img_chunk.to(model_dtype)
            
            outputs = model(img_chunk)
            preds = outputs.argmax(dim=1)
            
            correct += (preds == label_chunk).sum().item()
            
            precision_calc.update(preds, label_chunk)
            recall_calc.update(preds, label_chunk)
            f1_calc.update(preds, label_chunk)

    return {
        "Accuracy": 100.0 * correct / total_samples,
        "Precision": precision_calc.compute().item() * 100.0,
        "Recall": recall_calc.compute().item() * 100.0,
        "F1_Score": f1_calc.compute().item() * 100.0
    }