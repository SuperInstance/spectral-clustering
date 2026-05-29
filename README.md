# spectral-clustering

**Spectral clustering in pure Rust — partition graphs into communities using Laplacian eigenvalues.**

Zero-dependency implementation of Ng-Jordan-Weiss normalized spectral clustering, ratio cut, and modularity-based methods. Includes spectral embedding for visualization and quality metrics for evaluation.

## What This Gives You

- **Normalized cut clustering** — Ng-Jordan-Weiss algorithm: project onto Fiedler vectors, k-means in eigenvector space
- **Ratio cut** — Unnormalized spectral clustering via combinatorial Laplacian
- **Modularity** — Newman's modularity maximization for community detection
- **Spectral embedding** — Project nodes into 2D/3D eigenvector space for visualization
- **Quality metrics** — Conductance, silhouette score for cluster evaluation
- **Zero dependencies** — Jacobi eigenvalues and k-means++ built from scratch

## Quick Start

```rust
use spectral_clustering::SpectralCluster;

// From a flat row-major adjacency matrix
let adj = vec![
    0.0, 1.0, 1.0, 0.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 1.0, 0.0,
];
let sc = SpectralCluster::from_adjacency(adj);

// Cluster into 2 groups
let clusters = sc.cluster(2, "normalized");
// Two communities: {0,1} and {2,3,4}

// Or from edges
let sc = SpectralCluster::from_edges(4, &[(0,1,1.0), (1,2,1.0), (2,3,1.0), (0,3,1.0)]);

// Spectral embedding for visualization
let embedding = sc.spectral_embedding(2);

// Quality metrics
let modularity = sc.modularity(&clusters);
let conductance = sc.conductance(&clusters[0]);
let silhouette = sc.silhouette(&clusters);
```

## Methods

| Method | Laplacian | Best For |
|--------|-----------|----------|
| `"normalized"` | Normalized (Ng-Jordan-Weiss) | Balanced clusters |
| `"unnormalized"` | Combinatorial | Ratio cut optimization |
| `"modularity"` | Modularity matrix | Community detection |

## How It Fits

Part of the SuperInstance spectral ecosystem:

- **[spectral-graph-core](https://github.com/SuperInstance/spectral-graph-core)** — Eigenvalues, Laplacian construction, Fiedler vectors
- **spectral-clustering** — Partitioning via spectral methods (this repo)
- **[spectral-graph-v2](https://github.com/SuperInstance/spectral-graph-v2)** — CR, Fibonacci growth, adaptive thresholds

## Testing

```bash
cargo test
```

## Installation

```toml
[dependencies]
spectral-clustering = { git = "https://github.com/SuperInstance/spectral-clustering" }
```

## License

MIT

Part of the [SuperInstance](https://github.com/SuperInstance) ecosystem.
