//! From sizes to expected hits, and the allocation that equalises CVs.
//!
//! A size measure is not yet a design. [`expected_hits`] turns sizes into
//! the expected hits the selection consumes: proportional to size, with
//! units so large that proportionality would push them past one resolved
//! into certainties by iteration — or, under minimum replacement, left
//! above one and allowed several hits. [`eqcv`] closes the loop for
//! multivariate designs: it searches for the per-variable sample sizes
//! whose maximal measure gives every auxiliary variable the same
//! anticipated coefficient of variation, so no estimate is quietly
//! sacrificed to another.

use crate::error::Error;
use crate::size::mpps;

/// Expected hits proportional to size.
///
/// `n` is the sample size. With `min_replacement` false (the usual case),
/// units whose proportional share reaches one are made certainties and the
/// remainder is re-shared until no unit crosses one; the result is a vector
/// of inclusion probabilities in `[0, 1]`, and `n` may not exceed the
/// number of units with positive size. With `min_replacement` true the
/// expected hits are simply `n` times the size shares, values above one
/// meaning several hits, which is what Chromy's procedure is for.
pub fn expected_hits(size: &[f64], n: u64, min_replacement: bool) -> Result<Vec<f64>, Error> {
    for &s in size {
        if !s.is_finite() || s < 0.0 {
            return Err(Error::Input(format!(
                "sizes must be finite and non-negative; found {s}"
            )));
        }
    }
    let total: f64 = size.iter().sum();
    if total <= 0.0 {
        return Err(Error::Input("the sizes sum to zero".into()));
    }
    if n < 1 {
        return Err(Error::Input("the sample size must be at least one".into()));
    }
    let positive = size.iter().filter(|&&s| s > 0.0).count() as u64;
    if !min_replacement && n > positive {
        return Err(Error::Input(format!(
            "a sample of {n} without replacement needs at least {n} units of \
             positive size; the frame has {positive}"
        )));
    }

    let n_f = n as f64;
    if min_replacement {
        return Ok(size.iter().map(|&s| n_f * s / total).collect());
    }
    if n == positive {
        return Ok(size.iter().map(|&s| f64::from(s > 0.0)).collect());
    }

    let n_units = size.len();
    let mut certain = vec![false; n_units];
    let mut lambda = vec![0.0_f64; n_units];
    loop {
        let n_certain = certain.iter().filter(|&&c| c).count();
        let remaining = n_f - n_certain as f64;
        if remaining <= 0.0 && certain.iter().any(|&c| !c) {
            return Err(Error::Internal(
                "certainty resolution consumed the whole sample".into(),
            ));
        }
        let rest_total: f64 = size
            .iter()
            .zip(&certain)
            .filter(|&(_, &c)| !c)
            .map(|(&s, _)| s)
            .sum();
        for i in 0..n_units {
            lambda[i] = if certain[i] {
                1.0
            } else if remaining > 0.0 {
                remaining * size[i] / rest_total
            } else {
                0.0
            };
        }
        let mut grew = false;
        for i in 0..n_units {
            if !certain[i] && lambda[i] >= 1.0 {
                certain[i] = true;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    Ok(lambda.iter().map(|&l| l.min(1.0)).collect())
}

/// How [`eqcv`] is to search.
#[derive(Debug, Clone, Copy)]
pub struct EqcvOptions {
    /// The most iterations the search may take.
    pub max_iterations: usize,
    /// Convergence: the largest anticipated CV may exceed the smallest by
    /// at most this relative amount.
    pub tolerance: f64,
    /// Damping of each multiplicative step, in `(0, 1]`; one is undamped.
    pub damping: f64,
}

impl Default for EqcvOptions {
    fn default() -> Self {
        EqcvOptions {
            max_iterations: 300,
            tolerance: 1e-3,
            damping: 0.5,
        }
    }
}

/// The outcome of an [`eqcv`] search.
#[derive(Debug, Clone)]
pub struct EqcvAllocation {
    /// The per-variable sample-size factors, one per column of `x`,
    /// normalised to geometric mean one over the informative columns.
    pub n_j: Vec<f64>,
    /// The anticipated CV of each informative column at the returned
    /// factors, in column order.
    pub cv: Vec<f64>,
    /// Which columns were informative (summed to more than zero).
    pub used: Vec<bool>,
    /// Iterations actually taken.
    pub iterations: usize,
    /// Whether the CVs met the tolerance. A false is a report, not an
    /// error: the factors returned are still the best the search reached.
    pub converged: bool,
}

/// Search for the sample-size factors that equalise anticipated CVs.
///
/// `x` is the auxiliary matrix, column-major with `n_rows` units, and `n`
/// the sample size (at most the number of units). At each step the
/// anticipated CV of the total of every informative column is computed
/// under the design that [`mpps`] and [`expected_hits`] would produce, and
/// the factors move against the spread of the CVs, damped, until the
/// spread falls within tolerance.
pub fn eqcv(
    x: &[f64],
    n_rows: usize,
    n: u64,
    options: EqcvOptions,
) -> Result<EqcvAllocation, Error> {
    if n_rows == 0 || x.len() % n_rows != 0 {
        return Err(Error::Input(format!(
            "a matrix of {} values cannot have {} rows",
            x.len(),
            n_rows
        )));
    }
    if n as usize > n_rows {
        return Err(Error::Input(format!(
            "equalising CVs needs n at most the number of units; n = {n} \
             exceeds {n_rows}. With n beyond the frame use minimum \
             replacement and equal or fixed factors"
        )));
    }
    let n_cols = x.len() / n_rows;

    let mut col_sums = vec![0.0_f64; n_cols];
    for c in 0..n_cols {
        for r in 0..n_rows {
            col_sums[c] += x[c * n_rows + r];
        }
    }
    let used_cols: Vec<usize> = (0..n_cols).filter(|&c| col_sums[c] > 0.0).collect();
    if used_cols.is_empty() {
        return Err(Error::Input(
            "every auxiliary column sums to zero; there is nothing to equalise".into(),
        ));
    }

    // The shares each informative column assigns to each unit, fixed for
    // the whole search.
    let mut shares = vec![0.0_f64; n_rows * used_cols.len()];
    for (uc, &c) in used_cols.iter().enumerate() {
        for r in 0..n_rows {
            shares[uc * n_rows + r] = x[c * n_rows + r] / col_sums[c];
        }
    }

    let mut n_j = vec![1.0_f64; n_cols];

    let anticipated_cv = |n_j: &[f64]| -> Result<Vec<f64>, Error> {
        let size = mpps(x, n_rows, n_j)?;
        let pi = expected_hits(&size, n, false)?;
        let a: Vec<f64> = pi.iter().map(|&p| p * (1.0 - p)).collect();
        let a_total: f64 = a.iter().sum();
        if a_total == 0.0 {
            return Ok(vec![0.0; used_cols.len()]);
        }
        let mut cv = vec![0.0_f64; used_cols.len()];
        for uc in 0..used_cols.len() {
            let mut bias = 0.0_f64;
            for r in 0..n_rows {
                bias += (1.0 - pi[r]) * shares[uc * n_rows + r];
            }
            let centre = bias / a_total;
            let mut ss = 0.0_f64;
            for r in 0..n_rows {
                let d = shares[uc * n_rows + r] / pi[r] - centre;
                ss += a[r] * d * d;
            }
            cv[uc] = ss.sqrt();
        }
        Ok(cv)
    };

    let geometric_mean = |values: &[f64]| -> f64 {
        let mut s = 0.0_f64;
        for &v in values {
            s += v.ln();
        }
        (s / values.len() as f64).exp()
    };

    let spread_ok = |cv: &[f64], tol: f64| -> bool {
        let pos: Vec<f64> = cv.iter().copied().filter(|&v| v > 0.0).collect();
        if pos.is_empty() {
            return true;
        }
        let max = pos.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = pos.iter().copied().fold(f64::INFINITY, f64::min);
        max / min - 1.0 < tol
    };

    let mut cv = anticipated_cv(&n_j)?;
    let mut iterations = 0_usize;
    while iterations < options.max_iterations {
        if spread_ok(&cv, options.tolerance) {
            break;
        }
        let positive: Vec<f64> = cv.iter().copied().filter(|&v| v > 0.0).collect();
        let g = geometric_mean(&positive);
        for (uc, &c) in used_cols.iter().enumerate() {
            if cv[uc] > 0.0 {
                n_j[c] *= (cv[uc] / g).powf(options.damping);
            }
        }
        let norm = geometric_mean(&used_cols.iter().map(|&c| n_j[c]).collect::<Vec<_>>());
        for f in &mut n_j {
            *f /= norm;
        }
        cv = anticipated_cv(&n_j)?;
        iterations += 1;
    }

    let converged = spread_ok(&cv, options.tolerance);
    let used: Vec<bool> = (0..n_cols).map(|c| col_sums[c] > 0.0).collect();
    Ok(EqcvAllocation {
        n_j,
        cv,
        used,
        iterations,
        converged,
    })
}
