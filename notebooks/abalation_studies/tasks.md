1. Baseline comparison

* Seed Type Level (4)
* Fine-Grained Sub-Variety (27)
    * Rice (13)
    * Millet (8)
    * Amaranthus (3)
    * Mustard

* Metrics: Acc, Prec, Recall, F1
* Model Complexity: Total Trainable Parameters (M) and FLPS (GigaFLOPs)
* Inference Efficiency: Inference latency per image (ms)

* Models: resnet50, vit_b_16

* Unified Multi-Task Approach (Single Model Instance)
    * Shared Backbone: Pass your image through the backbone to get a single feature map (ResNet-50) or CLS token embedding (ViT-B/16).
    * Coarse Head: A fully connected layer mapping from the backbone feature dimension to 4 classes.
    * Fine-Grained Head: A fully connected layer mapping from the identical backbone feature dimension to 27 classes.
    * Loss Objective: Train the network using a combined loss function: ‭$L_{\text{total}} = 0.5 \cdot L_{\text{coarse}} + 0.5 \cdot L_{\text{fine}}$‬‭‬‭‬‭‬‭‬.


* Approach: Shared backbone (ResNet 50 Feature Map or ViT)
    - Coarse Head: 4 classes
    - Fine-Grained Head: 27 classes

    