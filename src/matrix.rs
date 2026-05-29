/// Simple row-major matrix.
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    /// Make symmetric by averaging upper and lower triangles.
    pub fn symmetrize(&mut self) {
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                let avg = (self.at(i, j) + self.at(j, i)) / 2.0;
                self.set(i, j, avg);
                self.set(j, i, avg);
            }
        }
    }

    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.rows {
            for j in 0..self.cols {
                if (self.at(i, j) - self.at(j, i)).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }
}
