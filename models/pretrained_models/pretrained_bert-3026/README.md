# Trained BERT model

Trained on 362,986 samples from 2024, with 30 epochs and early stopping activated at epoch 17 with validation MSE loss of 0.00438.

## Architecture

**AISBERT** - BERT-based model for AIS trajectory prediction using Masked Trajectory Modeling (MTM)

### Components:

**1. Input Processing**
- 4-dimensional AIS feature vectors (lat, lon, speed, heading)
- Special tokens: `[CLS]` = [-1,-1,-1,-1], `[SEP]` = [-2,-2,-2,-2], `[PAD]` = [0,0,0,0]
- Max sequence length: 512 positions

**2. AISEmbeddings Layer**
- Linear projection: 4D features → 256D hidden space
- Positional embeddings (learned, up to 512 positions)
- LayerNorm + Dropout (0.1)

**3. BERT Encoder**
- 4 transformer layers
- 4 attention heads per layer
- Hidden size: 256
- Dropout: 0.1

**4. Prediction Head**
- Linear layer: 256D → 4D (reconstructs masked features)
- Loss: MSE on masked positions

**5. Training Configuration**
- Masking: 15% of tokens randomly masked
- Batch size: 512
- Epochs: 30

The model learns trajectory representations by reconstructing randomly masked timesteps, similar to BERT's MLM but for continuous AIS features instead of discrete tokens.
