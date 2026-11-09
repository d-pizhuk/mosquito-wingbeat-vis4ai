# Phase I AI Interaction Record – Model Setup

## AI Interaction

**User request:** Develop a ResNet-50 model with attention for the WINGBEATS dataset, using Modal for training and Jupyter notebooks for monitoring. The goal is to achieve good accuracy and support visualizations, with a budget of around $30.

**AI contribution:** Generated code for dataset preparation, ResNet-50 with CBAM attention, training, evaluation, Modal integration, notebooks, and tests.

**Main design decisions:**
- Use ImageNet-pretrained ResNet-50 with CBAM attention.
- Train the model using the full WINGBEATS training dataset.
- Prepare data on CPUs and train on an NVIDIA A100 GPU.
- Split data into training (70%), validation (10%), and test (20%) sets.
- Save checkpoints and support attention visualizations for Phase II.

**Verification:** The implementation was initially tested locally using synthetic data. We successfully prepared the WINGBEATS dataset, verified training on Modal, measured GPU performance, and tested model evaluation.

**Issues and limitations:** We fixed a Modal disk configuration error. Duplicate recordings were detected and removed. Attention maps also need careful interpretation, as they do not necessarily explain why the model makes a prediction.

## Next Steps

Both pairs will use the same trained model and dataset but investigate its behavior separately. The AI-aided pair will document further AI interactions, suggestions, corrections, and findings.