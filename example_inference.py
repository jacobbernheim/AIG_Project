"""
Minimal example: predict on a random 2048bp sequence using existing classes.
"""
import numpy as np
from src.model_utils import ZeroShotScorer, ZeroShotScoreWeights
from src.genome_model import GenomeModel

# ============================================================
# 1. LOAD MODEL
# ============================================================
print("Loading model...")
genome_model = GenomeModel(organism="mouse")
print(f"Model info: {genome_model.get_model_info()}")
genome_model.load_model()

# ============================================================
# 2. GENERATE RANDOM 2048bp SEQUENCE
# ============================================================
SEQ_LEN = 2048
np.random.seed(42)
bases = ['A', 'C', 'G', 'T']
sequence = ''.join(np.random.choice(bases, size=SEQ_LEN))
print(f"\nSequence: {sequence[:50]}... ({len(sequence)} bp)")

# ============================================================
# 3. PREDICT
# ============================================================
print("\nRunning prediction...")
tracks = ["dnase", "chip_histone", "chip_tf"]
raw_outputs = genome_model.predict_on_sequence_raw(
    sequence,
    tracks=tracks,
    resolution=128,
)

print(f"\nReturned tracks: {list(raw_outputs.keys())}")
for track_name, arr in raw_outputs.items():
    print(f"  '{track_name}': shape={arr.shape}, "
          f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

# ============================================================
# 4. SCORE
# ============================================================
print("\nScoring...")
scorer = ZeroShotScorer(
    weights=ZeroShotScoreWeights(),
    signal_threshold=0.0,
)
result = scorer.score(raw_outputs)

print(f"\n{'='*60}")
print(f"Zero-Shot Score: {result.raw_score:.6f}")
print(f"Components: {result.component_values}")
print(f"{'='*60}")
