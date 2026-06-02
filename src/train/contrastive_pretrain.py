import os
import sys
import torch
import hydra
from omegaconf import DictConfig
import torch.optim as optim
from lightly.loss import DINOLoss
from lightly.models.utils import update_momentum

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data.transforms import get_dino_transforms
from src.data.dataset import get_pretrain_dataloader
from src.models.backbones.swinv2_dino import DINO

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"========== Starting DINO Pretraining ==========")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # 1. Data Setup
    print("Initializing Data Transforms and Loader...")
    transform = get_dino_transforms(cfg.data.image_size, cfg.data.local_crop_size)
    dataloader = get_pretrain_dataloader(
        data_dir=cfg.data.root_path,
        transform=transform,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )
    
    # 2. Model Setup
    print(f"Initializing DINO Model with backbone: {cfg.experiment.model.backbone_name}")
    model = DINO(
        backbone_name=cfg.experiment.model.backbone_name,
        input_dim=cfg.experiment.model.input_dim,
        hidden_dim=cfg.experiment.model.projection_hidden_dim,
        bottleneck_dim=cfg.experiment.model.projection_bottleneck_dim,
        out_dim=cfg.experiment.model.projection_out_dim,
        pretrained=cfg.experiment.model.pretrained,
        dynamic_img_size=cfg.experiment.model.dynamic_img_size,
    ).to(device)
    
    # 3. Loss & Optimizer Setup
    criterion = DINOLoss(
        output_dim=cfg.experiment.model.projection_out_dim,
        warmup_teacher_temp_epochs=cfg.experiment.training.warmup_teacher_temp_epochs
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.experiment.training.learning_rate, 
        weight_decay=cfg.experiment.training.weight_decay
    )
    
    # Ensure save directory exists
    os.makedirs(cfg.experiment.training.save_path, exist_ok=True)
    
    # 4. Training Loop
    print("Beginning Training Loop...")
    epochs = cfg.experiment.training.epochs
    momentum = cfg.experiment.training.momentum_teacher

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for batch_idx, (views, _, _) in enumerate(dataloader):
            # Exponential Moving Average (EMA) update for Teacher
            update_momentum(model.student_backbone, model.teacher_backbone, m=momentum)
            update_momentum(model.student_head, model.teacher_head, m=momentum)
            
            # Move multiple views to device
            views = [view.to(device) for view in views]
            
            # Teacher processes 2 global views
            teacher_out = [model.forward_teacher(view) for view in views[:2]]
            
            # Student processes all views (global + local)
            student_out = [model.forward_student(view) for view in views]
            
            # Compute cross-view prediction loss
            loss = criterion(teacher_out, student_out, epoch=epoch)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            
            # Cancel gradients for the last layer of student head during first epoch
            model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
            
            optimizer.step()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | DINO Loss: {avg_loss:.4f}")
        
    # 5. Save final model
    save_file = os.path.join(cfg.experiment.training.save_path, "dino_pretrained_backbone.pth")
    torch.save(model.student_backbone.state_dict(), save_file)
    print(f"Training Complete! Backbone weights saved to {save_file}")

if __name__ == "__main__":
    main()
