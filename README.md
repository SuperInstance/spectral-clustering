# spectral-clustering

Spectral clustering using graph Laplacian eigenvalues in Rust.

## Features

- **Normalized cut**: Optimal k-way partition using Fiedler vectors (Ng-Jordan-Weiss)
- **Ratio cut**: Unnormalized spectral clustering
- **Spectral embedding**: Project nodes into eigenvector space for visualization
- **Modularity**: Newman's modularity for community detection
- **Cluster quality metrics**: Conductance, silhouette score

## Usage

```rust
use spectral_clustering::SpectralCluster;

// From a flat row-major adjacency matrix
let adj = vec![/* n*n entries */];
let sc = SpectralCluster::from_adjacency(adj);

// Cluster into k groups
let clusters = sc.cluster(2, "normalized");

// Or from edges
let sc = SpectralCluster::from_edges(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 1.0)]);

// Spectral embedding for visualization
let embedding = sc.spectral_embedding(2);

// Quality metrics
let modularity = sc.modularity(&clusters);
let conductance = sc.conductance(&clusters[0]);
let silhouette = sc.silhouette(&clusters);
```

## Methods

- `"normalized"` — Ng-Jordan-Weiss normalized spectral clustering
- `"unnormalized"` — Ratio cut (unnormalized Laplacian)
- `"modularity"` — Modularity-based spectral clustering

## Internal Implementation

- Jacobi eigenvalue decomposition (cyclic sweeps) for symmetric matrices
- K-means++ clustering in eigenvector space
