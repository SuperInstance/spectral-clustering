/// Simple k-means clustering with k-means++ initialization.
///
/// `data` is n×dim row-major. Returns a label for each point.

use rand::Rng;

pub fn kmeans(data: &[f64], k: usize, max_iter: usize) -> Vec<usize> {
    let n = data.len() / k;
    let dim = k;
    if n == 0 || k == 0 {
        return vec![];
    }
    if k == 1 {
        return vec![0; n];
    }
    if n <= k {
        let mut labels = vec![0usize; n];
        for i in 0..n {
            labels[i] = i;
        }
        return labels;
    }

    let mut rng = rand::thread_rng();

    // K-means++ initialization
    let mut centroids: Vec<f64> = vec![0.0; k * dim];
    // Pick first centroid randomly
    let first = rng.gen_range(0..n);
    for d in 0..dim {
        centroids[d] = data[first * dim + d];
    }

    let mut dist_sq = vec![f64::INFINITY; n];
    for c_idx in 1..k {
        // Update distances
        for i in 0..n {
            let d = (0..dim)
                .map(|d| (data[i * dim + d] - centroids[c_idx.saturating_sub(1) * dim + d]).powi(2))
                .sum::<f64>();
            if d < dist_sq[i] {
                dist_sq[i] = d;
            }
        }
        // Weighted random selection
        let total: f64 = dist_sq.iter().sum();
        if total == 0.0 {
            // All points coincide; just pick random
            let idx = rng.gen_range(0..n);
            for d in 0..dim {
                centroids[c_idx * dim + d] = data[idx * dim + d];
            }
            continue;
        }
        let mut r = rng.gen_range(0.0..total);
        let mut chosen = 0;
        for i in 0..n {
            r -= dist_sq[i];
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }
        for d in 0..dim {
            centroids[c_idx * dim + d] = data[chosen * dim + d];
        }
    }

    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign each point to nearest centroid
        let mut changed = false;
        for i in 0..n {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for c in 0..k {
                let dist = (0..dim)
                    .map(|d| (data[i * dim + d] - centroids[c * dim + d]).powi(2))
                    .sum::<f64>();
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            if labels[i] != best {
                changed = true;
                labels[i] = best;
            }
        }

        if !changed {
            break;
        }

        // Update centroids
        let mut counts = vec![0usize; k];
        centroids.fill(0.0);
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for d in 0..dim {
                centroids[c * dim + d] += data[i * dim + d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c * dim + d] /= counts[c] as f64;
                }
            }
        }
    }

    labels
}
