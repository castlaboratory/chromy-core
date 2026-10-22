//! The sequential selection itself: probability minimum replacement.
//!
//! Chromy's procedure walks the ordered list once. It carries the
//! fractional parts of the expected hits as a running remainder; each time
//! the remainder crosses an integer, one more selection is available, and a
//! single conditional draw per unit decides whether that selection lands
//! here or later. The conditional probabilities are built so that each
//! unit's realised hits are either the floor or the ceiling of its expected
//! hits — never further — which is what "minimum replacement" means.
//!
//! The arithmetic is done with compensated (Kahan) summation, and the
//! inevitable floating-point drift is not ignored but *assigned*: a single
//! unit absorbs the difference between the sum of expected hits and the
//! integer sample size, and the function reports which unit, by how much,
//! and every near-boundary crossing it had to settle. The effective
//! expected hits it returns are the ones the selection actually used, and
//! the ones estimation must divide by.

use crate::error::Error;

/// Tolerance for a remainder that lands just under an integer boundary:
/// four units in the last place of 1.0.
const TAU: f64 = 4.0 * f64::EPSILON;

/// The outcome of a [`pmr`] selection.
#[derive(Debug, Clone)]
pub struct Selection {
    /// Realised hits per unit, in list order. Always the floor or the
    /// ceiling of the unit's effective expected hits, and summing exactly
    /// to the sample size.
    pub hits: Vec<u64>,
    /// The expected hits the selection actually used, after the
    /// floating-point drift was assigned. These are the denominators of
    /// the sampling weights.
    pub effective_lambda: Vec<f64>,
    /// The integer sample size the expected hits summed to.
    pub sample_size: u64,
    /// The floating-point drift between the sum of expected hits and the
    /// sample size, before it was assigned.
    pub eps_sum: f64,
    /// The unit (list position) that absorbed the drift, if any was needed.
    pub corrected_unit: Option<usize>,
    /// How many crossings sat within [`TAU`] of an integer boundary and
    /// were settled by borrowing from the units that follow.
    pub adjusted_crossings: usize,
    /// The largest absolute difference between a unit's effective and
    /// nominal expected hits.
    pub max_adjustment: f64,
}

/// Select with probability minimum replacement down a list.
///
/// `lambda` holds the expected hits of each unit in list order — any
/// non-negative values; a value above one means the unit may be selected
/// more than once. `uniforms` holds one draw in `[0, 1)` per unit, from the
/// caller's generator: the selection is a pure function of its arguments,
/// so a seed in R or Python reproduces it exactly. `priority` breaks the
/// tie when more than one unit could absorb the floating-point drift;
/// lower wins. Pass `0..n` when indifferent.
///
/// The expected hits must sum to within `1e-8` of an integer — the sample
/// size. The function refuses anything else: silently rescaling the
/// caller's probabilities would detach the weights from the design.
pub fn pmr(lambda: &[f64], uniforms: &[f64], priority: &[usize]) -> Result<Selection, Error> {
    let n_units = lambda.len();
    if uniforms.len() != n_units {
        return Err(Error::Input(format!(
            "{} expected hits but {} uniforms; one draw per unit",
            n_units,
            uniforms.len()
        )));
    }
    if priority.len() != n_units {
        return Err(Error::Input(format!(
            "{} expected hits but {} priorities",
            n_units,
            priority.len()
        )));
    }
    if n_units == 0 {
        return Err(Error::Input("an empty list cannot be sampled".into()));
    }
    for (i, &l) in lambda.iter().enumerate() {
        if !l.is_finite() || l < 0.0 {
            return Err(Error::Input(format!(
                "expected hits must be finite and non-negative; unit {i} has {l}"
            )));
        }
    }
    for (i, &u) in uniforms.iter().enumerate() {
        if !(0.0..1.0).contains(&u) {
            return Err(Error::Input(format!(
                "uniforms must lie in [0, 1); draw {i} is {u}"
            )));
        }
    }

    // The sample size, by compensated summation: the expected hits are
    // trusted to sum to an integer, and the drift is measured, not assumed.
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for &x in lambda {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            comp += (sum - t) + x;
        } else {
            comp += (x - t) + sum;
        }
        sum = t;
    }
    let n = (sum + comp).round_ties_even();
    let eps_sum = (sum - n) + comp;
    if eps_sum.abs() >= 1e-8 {
        return Err(Error::Drift { eps: eps_sum });
    }
    let n = n as u64;

    let k: Vec<f64> = lambda.iter().map(|l| l.floor()).collect();
    let mut r: Vec<f64> = lambda.iter().zip(&k).map(|(l, ki)| l - ki).collect();

    // Assign the drift to one unit: the largest fractional part that can
    // take it, ties broken by priority. One unit moves by eps; every other
    // unit's probability is untouched.
    let mut corrected_unit = None;
    if eps_sum != 0.0 {
        let candidate = (0..n_units)
            .filter(|&i| {
                if eps_sum > 0.0 {
                    r[i] > eps_sum
                } else {
                    r[i] > 0.0 && r[i] - eps_sum < 1.0
                }
            })
            .min_by(|&a, &b| {
                r[b].partial_cmp(&r[a])
                    .unwrap()
                    .then(priority[a].cmp(&priority[b]))
            });
        let j = candidate.ok_or(Error::NoAbsorber { eps: eps_sum })?;
        r[j] -= eps_sum;
        corrected_unit = Some(j);
    }

    // The walk. `frac`/`frac_comp` carry the running remainder with its
    // compensation; `debt` carries what a near-boundary adjustment borrowed
    // from the units ahead; the integer state is exact in i64.
    let mut hits_cum_floor = vec![0_i64; n_units]; // V_I: selections available before the draw
    let mut remainder_after = vec![0.0_f64; n_units]; // V_F
    let mut taken = vec![0_i64; n_units]; // T: selections made up to and including unit i
    let mut hits = vec![0_i64; n_units];
    let mut frac = 0.0_f64;
    let mut frac_comp = 0.0_f64;
    let mut debt = 0.0_f64;
    let mut adjusted_crossings = 0_usize;

    for i in 0..n_units {
        let (i_prev, f_prev, t_prev) = if i == 0 {
            (0_i64, 0.0_f64, 0_i64)
        } else {
            (hits_cum_floor[i - 1], remainder_after[i - 1], taken[i - 1])
        };

        let repay = debt.min(r[i]);
        r[i] -= repay;
        debt -= repay;

        let mut s = frac + r[i];
        let mut cs = if frac.abs() >= r[i].abs() {
            frac_comp + ((frac - s) + r[i])
        } else {
            frac_comp + ((r[i] - s) + frac)
        };
        let mut s_eff = s + cs;

        // A remainder within TAU of 1 is a crossing in all but rounding:
        // treat it as one, and borrow the missing sliver from the units
        // that follow, so no unit is stranded with an unusable remainder.
        if i < n_units - 1 && (1.0 - TAU..1.0).contains(&s_eff) {
            debt += 1.0 - s_eff;
            r[i] += 1.0 - s_eff;
            s = 1.0;
            cs = 0.0;
            s_eff = 1.0;
            adjusted_crossings += 1;
        }

        let crossing: i64;
        if i == n_units - 1 {
            // The last unit closes the books: whatever remainder is left
            // must round to its crossing, and its fractional part is
            // rewritten so that the effective expected hits sum exactly.
            crossing = s_eff.round_ties_even() as i64;
            r[i] = crossing as f64 - f_prev;
            frac = 0.0;
            frac_comp = 0.0;
        } else {
            crossing = i64::from(s_eff >= 1.0);
            frac = s - crossing as f64;
            frac_comp = cs;
        }

        hits_cum_floor[i] = i_prev + k[i] as i64 + crossing;
        remainder_after[i] = (frac + frac_comp).max(0.0);

        // Chromy's conditional probability that the pending selection has
        // already been taken, given the integer state so far.
        let p_taken = if t_prev == i_prev {
            if crossing == 0 {
                r[i] / (1.0 - f_prev)
            } else {
                0.0
            }
        } else if crossing == 0 {
            1.0
        } else if f_prev > 0.0 {
            remainder_after[i] / f_prev
        } else {
            0.0
        };

        taken[i] = hits_cum_floor[i] + i64::from(uniforms[i] < p_taken);
        hits[i] = taken[i] - t_prev;
    }

    let effective_lambda: Vec<f64> = k.iter().zip(&r).map(|(ki, ri)| ki + ri).collect();
    let lost = lambda
        .iter()
        .zip(&effective_lambda)
        .filter(|&(&l, &e)| l > 0.0 && e <= 0.0)
        .count();
    if lost > 0 {
        return Err(Error::LostProbability { units: lost });
    }

    let total: i64 = hits.iter().sum();
    if total < 0 || total as u64 != n {
        return Err(Error::Internal(format!(
            "realised hits sum to {total}, expected {n}"
        )));
    }
    let hits: Vec<u64> = hits.iter().map(|&h| h as u64).collect();

    let max_adjustment = lambda
        .iter()
        .zip(&effective_lambda)
        .map(|(&l, &e)| (e - l).abs())
        .fold(0.0_f64, f64::max);

    Ok(Selection {
        hits,
        effective_lambda,
        sample_size: n,
        eps_sum,
        corrected_unit,
        adjusted_crossings,
        max_adjustment,
    })
}
