# NOTES

## Approach

The solution uses two complementary approaches.

1. Handcrafted acoustic features
   - Energy
   - Pitch
   - Voicing
   - Spectral features
   - MFCC statistics

2. MFCC-based CNN
   - 40 MFCC coefficients
   - Fixed input size: 40 × 150
   - Lightweight CNN classifier

Both approaches are fully causal and only use audio before pause_start.

Future work is to ensemble both models for improved probability estimation.

## Libraries

- numpy
- librosa
- soundfile
- scikit-learn
- torch

## Dataset

English + Hindi combined.

Total samples:
496

## Best Results

Extra Trees:
AUC ≈ 0.667

CNN:
AUC ≈ 0.743