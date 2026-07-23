# RUNLOG

## Experiment 1
- Baseline silence-only endpointing provided in starter code.
- Performance expected to rely entirely on silence timeout.

---

## Experiment 2
- Added handcrafted prosodic features:
  - Energy statistics
  - Pitch statistics
  - Voicing features
  - Context features
- Trained:
  - Logistic Regression
  - Random Forest
  - HistGradientBoosting
- Best model:
  - Random Forest
- AUC ≈ 0.62

---

## Experiment 3
- Added:
  - MFCC statistics
  - Delta MFCC
  - Delta-Delta MFCC
  - Spectral centroid
  - Spectral bandwidth
  - Spectral rolloff
  - Spectral flatness
  - Zero Crossing Rate
- Total handcrafted features: 108
- Combined English and Hindi datasets.
- Added ExtraTrees classifier.
- Best model:
  - Extra Trees
- Validation AUC ≈ 0.667

---

## Experiment 4
- Built MFCC image dataset.
- Shape:
  (496,1,40,150)
- Trained lightweight CNN.
- Binary Cross Entropy Loss.
- AdamW optimizer.
- Early stopping enabled.
- Best validation AUC ≈ 0.743

---

## Future Work
- Ensemble CNN + Extra Trees.
- Tune feature importance.
- Improve predict.py.