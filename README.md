# K-NN Graph Coarse-Graining (Master’s Thesis Code)  <!-- #TODO: optionally tweak the title -->

> **TL;DR**: Code and experiments from my master’s thesis *“K-Nearest Neighbor graph coarse-graining from the perspective of dimensionality reduction.”*  
> It studies how shrinking k-NN graphs and condensing their information affect embeddings (e.g., t-SNE), and highlights links between HSNE and spectral **Kron reduction**, with ties to clustering.

---

## Overview

This repository accompanies my master’s thesis on graph coarse-graining viewed through the lens of dimensionality reduction. The work:

- investigates how compressing a k-NN graph changes low-dimensional embeddings (t-SNE and multiscale variants),
- draws a conceptual connection between **HSNE** (Pezzotti et al.) and **Kron reduction** (Schur complement on Laplacians),
- and discusses how “condensing” graphs relate to **clustering**.

📄 **Thesis PDF**: [`Masterarbeit_finalized.pdf`](./Masterarbeit_finalized.pdf)

---

## Important Content
- [Instantiating](./coarse_graining.py)
- [Coarse graining](./OOP_Multilevel_tsne.py)
- [Connecting_nodes](./OOP_Connecting.py)
- [Sampling_nodes](./OOP_Sampling.py)
- [Dataloader](./dataloader.py)


> These scripts are used for shrinking graphs and are the main work horses in this thesis.

---

## Installation

Create an environment and install dependencies.  
pip install -r requirements.txt


## Related Work (Pointers)

1. [HSNE (Hierarchical Stochastic Neighbor Embedding), Pezzotti et al.](https://doi.org/10.1111/cgf.12878)
2. [Kron reduction of graphs, Dörfler & Bullo.](https://doi.org/10.1109/TCSI.2012.2215780)
3. [Graph reduction with spectral and cut guarantees, Loukas (JMLR 2019).](https://www.jmlr.org/papers/volume20/18-680/18-680.pdf)

## Cite
```
@mastersthesis{christ_graph_coarsen_dr,
  title  = {K-Nearest Neighbor Graph Coarse-Graining from the Perspective of Dimensionality Reduction},
  author = {Moritz Christ},
  school = {Eberhard Karls Universität Tübingen},
  year   = {2025},
}
```

## Acknowledgements

Thanks to my advisor Dr. Dmitry Kobak and the whole team at the Hertie Instiute for Brain Health and AI.
