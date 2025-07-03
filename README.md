# Proof of Concept: FoD Detection in SAFRAN Aircraft Nacelles Using Acoustic Signatures

## Objective
This notebook presents a proof of concept (PoC) for detecting **Foreign Object Defects (FoD)** in aircraft nacelles **after assembly**. The nacelle is rotated during testing, and the goal is to **detect and classify impact noises** while **minimizing false positives**.

## Detection Challenges
FoD are categorized into three difficulty levels based on their acoustic profile:

- **Obvious noises**: e.g., tools (cutters, brushes) falling
- **Subtle noises**: e.g., rivets dropping
- **Silent or near-inaudible**: e.g., chips, tape fragments

## Approach
We apply a PCA-based anomaly detection framework relying on audio indicators, divided into two main feature categories:

1. **Impulse Noise Detection**:
   - **Spectral Crest**  
   - **Temporal Kurtosis**  
   - **Spectral Flux**

2. **Ultrasonic Impact Energy**:
   - Captures high-frequency characteristics to help discriminate between object types

The methodology aims to combine these indicators to highlight **anormal acoustic events** corresponding to potential FoD, while maintaining a low false positive rate.

[Related deliverables]{https://drive.google.com/file/d/1r5iWEC_TMMzrGO0J_41L4Kg1oS_k__O1/view?usp=drive_link}

---

*This notebook includes data exploration, feature computation, PCA modeling.*

