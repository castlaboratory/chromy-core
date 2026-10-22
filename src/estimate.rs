//! Estimation that matches the design.
//!
//! Under minimum replacement a unit may be selected more than once, and
//! the estimator that fits is the multiplicity (Hansen–Hurwitz form)
//! estimator: each unit's value divided by its *effective* expected hits —
//! the ones the selection actually used — and weighted by the hits it
//! received. For the variance there is no unbiased answer under a
//! systematic-like design; the successive-difference estimator is the
//! working convention, computed over the units whose selection was
//! actually random, in list order, so that it reads the local variation
//! the ordering left behind.

use crate::error::Error;

/// The totals and their successive-difference variance.
#[derive(Debug, Clone)]
pub struct Estimate {
    /// The estimated total of each study variable, in column order.
    pub total: Vec<f64>,
    /// How many sampled units had a randomly decided hit — the sample the
    /// variance is computed over.
    pub random_draws: usize,
    /// The successive-difference variance of each total: zero when no hit
    /// was random, `NaN` when exactly one was (one observation estimates
    /// no variance).
    pub sd_variance: Vec<f64>,
}

/// The multiplicity estimator with successive-difference variance.
///
/// `y` is the matrix of study variables for the *frame*, column-major with
/// `n_rows` units. The remaining slices describe the sample, one entry per
/// sampled unit: `units` the 0-based frame indices, `positions` the
/// 0-based positions in the selection list (the order the variance reads),
/// `expected_hits` the effective expected hits, `hits` the realised hits.
pub fn multiplicity_estimate(
    y: &[f64],
    n_rows: usize,
    units: &[usize],
    positions: &[usize],
    expected_hits: &[f64],
    hits: &[u64],
) -> Result<Estimate, Error> {
    if n_rows == 0 || y.len() % n_rows != 0 {
        return Err(Error::Input(format!(
            "a matrix of {} values cannot have {} rows",
            y.len(),
            n_rows
        )));
    }
    let n_cols = y.len() / n_rows;
    let m = units.len();
    if positions.len() != m || expected_hits.len() != m || hits.len() != m {
        return Err(Error::Input(
            "units, positions, expected hits and hits must have one entry per \
             sampled unit"
                .into(),
        ));
    }
    for &v in y {
        if !v.is_finite() {
            return Err(Error::Input(format!(
                "study variables must be finite; found {v}"
            )));
        }
    }
    for &u in units {
        if u >= n_rows {
            return Err(Error::Input(format!(
                "sampled unit {u} is beyond the {n_rows} rows of the study matrix"
            )));
        }
    }
    for (s, &e) in expected_hits.iter().enumerate() {
        if !e.is_finite() || e <= 0.0 {
            return Err(Error::Input(format!(
                "expected hits must be positive on the sample; entry {s} is {e}"
            )));
        }
    }

    // The sample in list order: the successive differences are to read
    // neighbours of the ordered frame, not of whatever order the caller
    // stored the sample in.
    let mut by_list: Vec<usize> = (0..m).collect();
    by_list.sort_by_key(|&s| positions[s]);

    let mut total = vec![0.0_f64; n_cols];
    for c in 0..n_cols {
        for &s in &by_list {
            total[c] += y[c * n_rows + units[s]] / expected_hits[s] * hits[s] as f64;
        }
    }

    // The units whose hit was random: realised hits one above the floor of
    // the expected hits. Fixed hits are certainties and contribute no
    // variance.
    let random: Vec<usize> = by_list
        .iter()
        .copied()
        .filter(|&s| hits[s] as i64 - expected_hits[s].floor() as i64 == 1)
        .collect();
    let draws = random.len();

    let sd_variance: Vec<f64> = (0..n_cols)
        .map(|c| {
            if draws == 0 {
                0.0
            } else if draws == 1 {
                f64::NAN
            } else {
                let mut ss = 0.0_f64;
                for w in random.windows(2) {
                    let a = y[c * n_rows + units[w[0]]] / expected_hits[w[0]];
                    let b = y[c * n_rows + units[w[1]]] / expected_hits[w[1]];
                    ss += (b - a) * (b - a);
                }
                draws as f64 / (2.0 * (draws as f64 - 1.0)) * ss
            }
        })
        .collect();

    Ok(Estimate {
        total,
        random_draws: draws,
        sd_variance,
    })
}
