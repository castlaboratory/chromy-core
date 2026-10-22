//! A size measure that serves several estimates at once.
//!
//! A survey that must estimate more than one total has no single "size":
//! a unit may be large for one variable and negligible for another. The
//! maximal measure takes, for each unit, the largest of its shares across
//! the auxiliary variables, each share first scaled by how much sample
//! that variable is to command. Selecting proportional to the maximum
//! guarantees every variable at least its intended coverage (Folsom,
//! Potter and Williams, 1987).

use crate::error::Error;

/// The maximal size measure over several auxiliary variables.
///
/// `x` is the matrix of non-negative auxiliary values, column-major, with
/// `n_rows` units per column; `n_j` gives the relative sample size each
/// variable is to command, one positive value per column. Columns that sum
/// to zero carry no information and are ignored. Units whose every share
/// is zero receive the smallest positive measure, so the frame keeps full
/// coverage.
pub fn mpps(x: &[f64], n_rows: usize, n_j: &[f64]) -> Result<Vec<f64>, Error> {
    if n_rows == 0 || x.len() % n_rows != 0 {
        return Err(Error::Input(format!(
            "a matrix of {} values cannot have {} rows",
            x.len(),
            n_rows
        )));
    }
    let n_cols = x.len() / n_rows;
    if n_j.len() != n_cols {
        return Err(Error::Input(format!(
            "{} columns but {} sample-size factors",
            n_cols,
            n_j.len()
        )));
    }
    for &v in x {
        if !v.is_finite() || v < 0.0 {
            return Err(Error::Input(format!(
                "auxiliary values must be finite and non-negative; found {v}"
            )));
        }
    }
    for &f in n_j {
        if !f.is_finite() || f <= 0.0 {
            return Err(Error::Input(format!(
                "sample-size factors must be positive; found {f}"
            )));
        }
    }

    let mut col_sums = vec![0.0_f64; n_cols];
    for c in 0..n_cols {
        for r in 0..n_rows {
            col_sums[c] += x[c * n_rows + r];
        }
    }
    let used: Vec<usize> = (0..n_cols).filter(|&c| col_sums[c] > 0.0).collect();
    if used.is_empty() {
        return Err(Error::Input(
            "every auxiliary column sums to zero; no size can be measured".into(),
        ));
    }

    let mean_factor = {
        let mut s = 0.0;
        for &c in &used {
            s += n_j[c];
        }
        s / used.len() as f64
    };

    let mut size = vec![0.0_f64; n_rows];
    for r in 0..n_rows {
        let mut best = f64::NEG_INFINITY;
        for &c in &used {
            let share = x[c * n_rows + r] / col_sums[c] * (n_j[c] / mean_factor);
            if share > best {
                best = share;
            }
        }
        size[r] = best;
    }

    let smallest_positive = size
        .iter()
        .copied()
        .filter(|&s| s > 0.0)
        .fold(f64::INFINITY, f64::min);
    if !smallest_positive.is_finite() {
        return Err(Error::Input(
            "every unit has measure zero; no size can be assigned".into(),
        ));
    }
    for s in &mut size {
        if *s <= 0.0 {
            *s = smallest_positive;
        }
    }
    Ok(size)
}
