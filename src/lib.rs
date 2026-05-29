mod matrix;
mod jacobi;
mod kmeans;

use matrix::Matrix;

/// Spectral clustering algorithm over an undirected graph.
pub struct SpectralCluster {
    n: usize,
    adj: Matrix,
}

impl SpectralCluster {
    /// Create a new spectral clusterer from an adjacency matrix.
    ///
    /// `adj` is a flat row-major `Vec<f64>` of length n*n.
    pub fn from_adjacency(adj: Vec<f64>) -> Self {
        let n = (adj.len() as f64).sqrt() as usize;
        assert_eq!(n * n, adj.len(), "Adjacency matrix must be square");
        let m = Matrix::new(n, n, adj);
        // Symmetrize
        let mut sym = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let v = (m.at(i, j) + m.at(j, i)) / 2.0;
                sym[i * n + j] = v;
            }
        }
        Self {
            n,
            adj: Matrix::new(n, n, sym),
        }
    }

    /// Convenience: build from a list of edges `(i, j, weight)`.
    pub fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut adj = vec![0.0; n * n];
        for &(i, j, w) in edges {
            adj[i * n + j] += w;
            adj[j * n + i] += w;
        }
        Self::from_adjacency(adj)
    }

    /// Cluster into `k` groups using the given method.
    ///
    /// Methods: "normalized" (Ng-Jordan-Weiss), "unnormalized" (ratio cut), "modularity".
    ///
    /// Returns `Vec<Vec<usize>>` — k clusters, each a list of node indices.
    pub fn cluster(&self, k: usize, method: &str) -> Vec<Vec<usize>> {
        assert!(k >= 1 && k <= self.n, "k must be between 1 and n");
        if k == 1 {
            return vec![(0..self.n).collect()];
        }

        let embed = match method {
            "normalized" => {
                let lap = self.normalized_laplacian();
                let (_, vecs) = jacobi::eigen_decompose(&lap, k);
                // Row-normalize each eigenvector row (Ng-Jordan-Weiss)
                self.row_normalize(&vecs)
            }
            "unnormalized" => {
                let lap = self.unnormalized_laplacian();
                let (_, vecs) = jacobi::eigen_decompose(&lap, k);
                vecs
            }
            "modularity" => {
                let mod_mat = self.modularity_matrix();
                let (_, vecs) = jacobi::eigen_decompose(&mod_mat, k);
                vecs
            }
            _ => panic!("Unknown method: {}. Use 'normalized', 'unnormalized', or 'modularity'.", method),
        };

        // Run k-means on the rows of the embedding
        let labels = kmeans::kmeans(&embed, k, 50);
        let mut clusters: Vec<Vec<usize>> = vec![vec![]; k];
        for (i, &label) in labels.iter().enumerate() {
            clusters[label].push(i);
        }
        // Remove empty clusters
        clusters.retain(|c| !c.is_empty());
        clusters
    }

    /// Project nodes into `dim`-dimensional eigenvector space.
    pub fn spectral_embedding(&self, dim: usize) -> Vec<Vec<f64>> {
        let lap = self.normalized_laplacian();
        let (_, vecs) = jacobi::eigen_decompose(&lap, dim);
        // vecs is n×dim, return as Vec<Vec<f64>>
        (0..self.n)
            .map(|i| (0..dim).map(|j| vecs[i * dim + j]).collect())
            .collect()
    }

    /// Compute Newman's modularity for a given partition.
    pub fn modularity(&self, partition: &[Vec<usize>]) -> f64 {
        let m_total: f64 = self.adj.data.iter().sum::<f64>() / 2.0;
        if m_total == 0.0 {
            return 0.0;
        }
        let degrees: Vec<f64> = (0..self.n)
            .map(|i| (0..self.n).map(|j| self.adj.at(i, j)).sum())
            .collect();

        let mut q = 0.0;
        for community in partition {
            for &i in community {
                for &j in community {
                    q += self.adj.at(i, j) - degrees[i] * degrees[j] / (2.0 * m_total);
                }
            }
        }
        q / (2.0 * m_total)
    }

    /// Compute conductance of a single cluster.
    ///
    /// Conductance = (edges leaving the cluster) / (total edges incident to the cluster)
    pub fn conductance(&self, cluster: &[usize]) -> f64 {
        if cluster.is_empty() {
            return 0.0;
        }
        let cluster_set: std::collections::HashSet<usize> = cluster.iter().copied().collect();
        let mut cut: f64 = 0.0;
        let mut vol: f64 = 0.0;
        for &i in cluster {
            for j in 0..self.n {
                let w = self.adj.at(i, j);
                vol += w;
                if !cluster_set.contains(&j) {
                    cut += w;
                }
            }
        }
        if vol == 0.0 {
            return 0.0;
        }
        cut / vol
    }

    /// Compute mean silhouette score for a partition.
    pub fn silhouette(&self, partition: &[Vec<usize>]) -> f64 {
        if partition.len() <= 1 {
            return 0.0;
        }
        let n = self.n;
        // Precompute all-pairs shortest path distances using adjacency as distance
        let distances = self.compute_distances();

        // Assign each node to its cluster index
        let mut label = vec![0usize; n];
        for (ci, cluster) in partition.iter().enumerate() {
            for &node in cluster {
                label[node] = ci;
            }
        }

        let mut total: f64 = 0.0;
        for i in 0..n {
            let ci = label[i];
            // a(i): mean distance to same-cluster nodes
            let same: Vec<usize> = partition[ci]
                .iter()
                .filter(|&&x| x != i)
                .copied()
                .collect();
            let a = if same.is_empty() {
                0.0
            } else {
                same.iter().map(|&j| distances[i * n + j]).sum::<f64>() / same.len() as f64
            };

            // b(i): min mean distance to other clusters
            let mut b = f64::INFINITY;
            for (other_ci, other_cluster) in partition.iter().enumerate() {
                if other_ci == ci || other_cluster.is_empty() {
                    continue;
                }
                let mean_d: f64 =
                    other_cluster.iter().map(|&j| distances[i * n + j]).sum::<f64>()
                        / other_cluster.len() as f64;
                if mean_d < b {
                    b = mean_d;
                }
            }
            if b == f64::INFINITY {
                b = 0.0;
            }

            let sil = if same.is_empty() {
                0.0
            } else if a < b {
                1.0 - a / b
            } else if a > b {
                b / a - 1.0
            } else {
                0.0
            };
            total += sil;
        }
        total / n as f64
    }

    // --- Internal helpers ---

    fn degree_matrix(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| (0..self.n).map(|j| self.adj.at(i, j)).sum())
            .collect()
    }

    fn unnormalized_laplacian(&self) -> Matrix {
        let n = self.n;
        let deg = self.degree_matrix();
        let mut lap = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                lap[i * n + j] = if i == j {
                    deg[i]
                } else {
                    -self.adj.at(i, j)
                };
            }
        }
        Matrix::new(n, n, lap)
    }

    fn normalized_laplacian(&self) -> Matrix {
        let n = self.n;
        let deg = self.degree_matrix();
        let mut lap = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j && deg[i] > 0.0 {
                    lap[i * n + j] = 1.0;
                } else if deg[i] > 0.0 && deg[j] > 0.0 {
                    lap[i * n + j] = -self.adj.at(i, j) / (deg[i].sqrt() * deg[j].sqrt());
                }
            }
        }
        Matrix::new(n, n, lap)
    }

    fn modularity_matrix(&self) -> Matrix {
        let n = self.n;
        let m_total: f64 = self.adj.data.iter().sum::<f64>() / 2.0;
        let deg = self.degree_matrix();
        let mut b = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let expected = if m_total > 0.0 {
                    deg[i] * deg[j] / (2.0 * m_total)
                } else {
                    0.0
                };
                b[i * n + j] = self.adj.at(i, j) - expected;
            }
        }
        Matrix::new(n, n, b)
    }

    fn row_normalize(&self, vecs: &[f64]) -> Vec<f64> {
        let n = self.n;
        let dim = vecs.len() / n;
        let mut out = vec![0.0; vecs.len()];
        for i in 0..n {
            let norm: f64 = (0..dim).map(|j| vecs[i * dim + j].powi(2)).sum::<f64>().sqrt();
            for j in 0..dim {
                out[i * dim + j] = if norm > 1e-15 {
                    vecs[i * dim + j] / norm
                } else {
                    vecs[i * dim + j]
                };
            }
        }
        out
    }

    fn compute_distances(&self) -> Vec<f64> {
        let n = self.n;
        // Use shortest path (unweighted hop count based on nonzero adjacency)
        let mut dist = vec![f64::INFINITY; n * n];
        for i in 0..n {
            dist[i * n + i] = 0.0;
            for j in 0..n {
                if self.adj.at(i, j) > 0.0 {
                    dist[i * n + j] = 1.0;
                }
            }
        }
        // Floyd-Warshall
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let d = dist[i * n + k] + dist[k * n + j];
                    if d < dist[i * n + j] {
                        dist[i * n + j] = d;
                    }
                }
            }
        }
        // Replace infinities with n as f64 (disconnected)
        for d in dist.iter_mut() {
            if *d == f64::INFINITY {
                *d = n as f64;
            }
        }
        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a block matrix: two communities of `half` nodes each,
    /// with intra-community probability p_in and inter-community probability p_out.
    pub(crate) fn two_communities_adj(half: usize, p_in: f64, p_out: f64) -> Vec<f64> {
        let n = half * 2;
        let mut adj = vec![0.0; n * n];
        // Deterministic: connect all pairs with weight
        for i in 0..n {
            for j in (i + 1)..n {
                let same = (i < half) == (j < half);
                let w = if same { p_in } else { p_out };
                adj[i * n + j] = w;
                adj[j * n + i] = w;
            }
        }
        adj
    }

    #[test]
    fn test_two_clear_communities() {
        // Two communities, strong intra, weak inter
        let adj = two_communities_adj(10, 1.0, 0.01);
        let sc = SpectralCluster::from_adjacency(adj);
        let clusters = sc.cluster(2, "normalized");

        assert_eq!(clusters.len(), 2);
        // Check that the partition is close to the ground truth
        let c0: std::collections::HashSet<usize> = clusters[0].iter().copied().collect();
        let ground_truth_0: std::collections::HashSet<usize> = (0..10).collect();
        let ground_truth_1: std::collections::HashSet<usize> = (10..20).collect();

        let overlap0 = c0.intersection(&ground_truth_0).count();
        let overlap1 = c0.intersection(&ground_truth_1).count();
        // Should match one of the ground truth communities well
        let purity = overlap0.max(overlap1) as f64 / 10.0;
        assert!(purity > 0.8, "Purity too low: {}", purity);
    }

    #[test]
    fn test_single_cluster() {
        let adj = two_communities_adj(5, 1.0, 1.0); // Complete graph
        let sc = SpectralCluster::from_adjacency(adj);
        let clusters = sc.cluster(1, "normalized");
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 10);
    }

    #[test]
    fn test_complete_graph_conductance() {
        let n = 8;
        let adj = vec![1.0; n * n]; // Complete graph (including self-loops, doesn't matter for conductance)
        let sc = SpectralCluster::from_adjacency(adj);
        let cluster: Vec<usize> = (0..4).collect();
        let c = sc.conductance(&cluster);
        // In a complete graph, conductance of any k-subset = (k*(n-k)) / (k*n) = (n-k)/n
        let expected = (n - 4) as f64 / n as f64;
        assert!((c - expected).abs() < 0.01, "Conductance: {} vs expected {}", c, expected);
    }

    #[test]
    fn test_modularity_ground_truth() {
        let adj = two_communities_adj(10, 1.0, 0.01);
        let sc = SpectralCluster::from_adjacency(adj);
        let gt = vec![(0..10).collect::<Vec<_>>(), (10..20).collect::<Vec<_>>()];
        let mod_gt = sc.modularity(&gt);

        // Random partition
        let rand_part = vec![
            vec![0, 2, 5, 7, 11, 13, 16, 19],
            vec![1, 3, 4, 6, 8, 9, 10, 12, 14, 15, 17, 18],
        ];
        let mod_rand = sc.modularity(&rand_part);

        assert!(
            mod_gt > mod_rand,
            "Ground truth modularity ({}) should exceed random ({})",
            mod_gt,
            mod_rand
        );
    }

    #[test]
    fn test_silhouette_well_separated() {
        let adj = two_communities_adj(5, 1.0, 0.0);
        let sc = SpectralCluster::from_adjacency(adj);
        let partition = vec![(0..5).collect::<Vec<_>>(), (5..10).collect::<Vec<_>>()];
        let sil = sc.silhouette(&partition);
        assert!(sil > 0.5, "Silhouette should be > 0.5 for well-separated clusters, got {}", sil);
    }

    #[test]
    fn test_spectral_embedding_clusters() {
        let adj = two_communities_adj(10, 1.0, 0.01);
        let sc = SpectralCluster::from_adjacency(adj);
        let embed = sc.spectral_embedding(2);
        assert_eq!(embed.len(), 20);
        assert_eq!(embed[0].len(), 2);

        // Nodes 0..10 should cluster together in embedding space
        // Compute centroids
        let mut c0 = vec![0.0, 0.0];
        let mut c1 = vec![0.0, 0.0];
        for i in 0..10 {
            c0[0] += embed[i][0];
            c0[1] += embed[i][1];
        }
        for i in 10..20 {
            c1[0] += embed[i][0];
            c1[1] += embed[i][1];
        }
        c0[0] /= 10.0; c0[1] /= 10.0;
        c1[0] /= 10.0; c1[1] /= 10.0;

        // Nodes should be closer to their own centroid than the other
        let mut correct = 0;
        for i in 0..20 {
            let centroid = if i < 10 { &c0 } else { &c1 };
            let other = if i < 10 { &c1 } else { &c0 };
            let d_own = (embed[i][0] - centroid[0]).powi(2) + (embed[i][1] - centroid[1]).powi(2);
            let d_other = (embed[i][0] - other[0]).powi(2) + (embed[i][1] - other[1]).powi(2);
            if d_own <= d_other {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / 20.0;
        assert!(accuracy > 0.8, "Embedding clustering accuracy: {}", accuracy);
    }

    #[test]
    fn test_unnormalized_method() {
        let adj = two_communities_adj(8, 1.0, 0.01);
        let sc = SpectralCluster::from_adjacency(adj);
        let clusters = sc.cluster(2, "unnormalized");
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters.iter().map(|c| c.len()).sum::<usize>(), 16);
    }

    #[test]
    fn test_modularity_method() {
        let adj = two_communities_adj(8, 1.0, 0.01);
        let sc = SpectralCluster::from_adjacency(adj);
        let clusters = sc.cluster(2, "modularity");
        assert!(!clusters.is_empty());
        assert_eq!(clusters.iter().map(|c| c.len()).sum::<usize>(), 16);
    }

    #[test]
    fn test_conductance_isolated_cluster() {
        let adj = two_communities_adj(5, 1.0, 0.0);
        let sc = SpectralCluster::from_adjacency(adj);
        let cluster: Vec<usize> = (0..5).collect();
        let c = sc.conductance(&cluster);
        assert!((c).abs() < 0.01, "Isolated cluster should have ~0 conductance, got {}", c);
    }

    #[test]
    fn test_from_edges() {
        let sc = SpectralCluster::from_edges(
            4,
            &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 1.0)],
        );
        assert_eq!(sc.n, 4);
        let clusters = sc.cluster(2, "normalized");
        assert_eq!(clusters.iter().map(|c| c.len()).sum::<usize>(), 4);
    }
}

