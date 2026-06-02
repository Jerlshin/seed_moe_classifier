import copy
import torch
import torch.nn as nn
import timm

from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad

class DINO(nn.Module):
    """
    The Self-Supervised DINO architecture containing both Student and Teacher networks.
    """
    def __init__(
        self,
        backbone_name: str,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        out_dim: int,
        pretrained: bool = False,
        dynamic_img_size: bool = True,
    ):
        super().__init__()
        
        # 1. Initialize Student
        self.student_backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=dynamic_img_size,
        )
        self.student_head = DINOProjectionHead(
            input_dim, 
            hidden_dim, 
            bottleneck_dim, 
            out_dim, 
            freeze_last_layer=1
        )
        
        # 2. Initialize Teacher (Exact copy of student)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim, 
            hidden_dim, 
            bottleneck_dim, 
            out_dim
        )
        
        # 3. Stop gradients for the teacher (Updated via EMA, not backprop)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        """
        Forward pass for downstream tasks (returns flat features).
        """
        y = self.student_backbone(x)
        return y

    def forward_student(self, x):
        y = self.student_backbone(x)
        return self.student_head(y)

    def forward_teacher(self, x):
        y = self.teacher_backbone(x)
        return self.teacher_head(y)
