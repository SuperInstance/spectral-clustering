use crate::matrix::Matrix;

/// Jacobi eigenvalue decomposition for symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues are sorted ascending,
/// and eigenvectors are returned as a flat n×k row-major array (column j is eigenvector j).
pub fn eigen_decompose(mat: &Matrix, k: usize) -> (Vec<f64>, Vec<f64>) {
    let n = mat.rows;
    assert!(n == mat.cols, "Matrix must be square");
    let k = k.min(n);

    let mut a = mat.data.clone();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // Jacobi cyclic sweeps
    let max_sweeps = 200;
    let tol = 1e-14;

    for _sweep in 0..max_sweeps {
        // Check convergence: max off-diagonal
        let mut off_norm = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_norm += a[i * n + j].powi(2);
            }
        }
        if off_norm < tol {
            break;
        }

        // One sweep: iterate over all upper-triangular (i, j) pairs
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-16 {
                    continue;
                }

                let app = a[p * n + p];
                let aqq = a[q * n + q];

                // Compute Jacobi rotation angle
                let tau = (aqq - app) / (2.0 * apq);
                // t = sign(tau) / (|tau| + sqrt(1 + tau^2))
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update matrix A in-place
                // First update row/col p and q
                for i in 0..n {
                    if i == p || i == q {
                        continue;
                    }
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    let new_aip = c * aip - s * aiq;
                    let new_aiq = s * aip + c * aiq;
                    a[i * n + p] = new_aip;
                    a[p * n + i] = new_aip;
                    a[i * n + q] = new_aiq;
                    a[q * n + i] = new_aiq;
                }

                let new_app = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                let new_aqq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                a[p * n + p] = new_app;
                a[q * n + q] = new_aqq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update eigenvector matrix V
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = c * vip - s * viq;
                    v[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }

    // Extract eigenvalues (diagonal of A)
    let mut eigenvalues: Vec<(f64, usize)> = (0..n)
        .map(|i| (a[i * n + i], i))
        .collect();

    // Sort by eigenvalue ascending
    eigenvalues.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Extract k smallest eigenvectors
    let mut vals = Vec::with_capacity(k);
    let mut vecs = vec![0.0; n * k];
    for (idx, &(val, col)) in eigenvalues.iter().take(k).enumerate() {
        vals.push(val);
        for row in 0..n {
            vecs[row * k + idx] = v[row * n + col];
        }
    }

    (vals, vecs)
}
