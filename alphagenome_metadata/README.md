---
license: other
library_name: alphagenome-pytorch
tags:
- genomics
- biology
- dna
- deep-learning
- regulatory-genomics
- chromatin-accessibility
- gene-expression
pipeline_tag: other
---

# AlphaGenome PyTorch

A PyTorch port of [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0), the DNA sequence model from Google DeepMind that predicts hundreds of genomic tracks at single base-pair resolution from sequences up to 1M bp.

This is an accessible, readable, and hackable implementation for integrating into existing PyTorch pipelines, fine-tuning on custom datasets, and building on top of.

## Model Details

- **Parameters**: 450M
- **Input**: One-hot encoded DNA sequence
- **Organisms**: Human, Mouse
- **Weights**: Converted from the official JAX checkpoint

## Download Weights

Available weight files:
- `model_all_folds.safetensors` - trained on all data (recommended)
- `model_fold_0.safetensors` through `model_fold_3.safetensors` - individual CV folds

```bash
# Using Hugging Face CLI
hf download gtca/alphagenome_pytorch model_all_folds.safetensors --local-dir .

# Or using Python
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('gtca/alphagenome_pytorch', 'model_all_folds.safetensors', local_dir='.')"
```

## Usage

```python
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor
import pyfaidx

model = AlphaGenome.from_pretrained("model_all_folds.safetensors")

with pyfaidx.Fasta("hg38.fa") as genome:
    sequence = str(genome["chr1"][1_000_000:1_131_072])

dna_onehot = sequence_to_onehot_tensor(sequence).unsqueeze(0)

preds = model.predict(dna_onehot, organism_index=0)  # 0=human, 1=mouse

# Access predictions by head name and resolution:
# - preds['atac'][1]: 1bp resolution, shape (batch, 131072, 256)
# - preds['atac'][128]: 128bp resolution, shape (batch, 1024, 256)
```

## Model Outputs

| Head | Tracks | Resolutions | Description |
|------|--------|-------------|-------------|
| atac | 256 | 1bp, 128bp | Chromatin accessibility |
| dnase | 384 | 1bp, 128bp | DNase-seq |
| procap | 128 | 1bp, 128bp | Transcription initiation |
| cage | 640 | 1bp, 128bp | 5' cap RNA |
| rnaseq | 768 | 1bp, 128bp | RNA expression |
| chip_tf | 1664 | 128bp | TF binding |
| chip_histone | 1152 | 128bp | Histone modifications |
| contact_maps | 28 | 64x64 | 3D chromatin contacts |
| splice_sites | 5 | 1bp | Splice site classification (D+, A+, D−, A−, None) |
| splice_junctions | 734 | pairwise | Junction read counts |
| splice_site_usage | 734 | 1bp | Splice site usage fraction |

## Installation

```bash
pip install alphagenome-pytorch
```

## License

The weights were ported from the weights [provided by Google DeepMind](https://www.kaggle.com/models/google/alphagenome). Those weights were created by Google DeepMind and are the property of Google LLC.

The model parameters, output, and any derivatives thereof remain subject to Google DeepMind’s AlphaGenome Model Terms (https://deepmind.google.com/science/alphagenome/model-terms).

[The model code](https://github.com/genomicsxai/alphagenome-pytorch) is released under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

These licensing terms are consistent with the terms for [the reference code](https://github.com/google-deepmind/alphagenome_research) and [the model weights](https://www.kaggle.com/models/google/alphagenome).

## Links

- [GitHub Repository](https://github.com/genomicsxai/alphagenome-pytorch)
- [Reference JAX Implementation](https://github.com/google-deepmind/alphagenome_research) (by Google DeepMind)
- [AlphaGenome Paper](https://www.nature.com/articles/s41586-025-10014-0)
- [AlphaGenome Documentation](https://www.alphagenomedocs.com/)

## Citation

```bibtex
@article{avsec2026alphagenome,
  title={Advancing regulatory variant effect prediction with AlphaGenome},
  author={Avsec, {\v{Z}}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and others},
  journal={Nature},
  volume={649},
  number={8099},
  pages={1206--1218},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```