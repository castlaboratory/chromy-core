//! The full procedure: a random head, the rotated list, the selection.
//!
//! Run from a fixed first unit, a sequential procedure gives the units
//! near the head a subtly different treatment from the units near the
//! tail. Chromy's remedy, kept by SAS, is to draw the head itself with
//! probability proportional to size, rotate the list so it starts there,
//! and only then select; the procedure becomes invariant to where the
//! frame happened to begin.
//!
//! The head draw is the one point where this crate does not consume a
//! uniform the same way everywhere: a binding that wants to reproduce its
//! host language's sampler (R's `sample.int`, say) should draw the head
//! itself and call [`select_from`]; [`select`] draws it by cumulative
//! inversion from a single uniform.

use crate::error::Error;
use crate::pmr::{pmr, Selection};

/// Which units may be drawn as the head of the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Head {
    /// Every unit: the selection randomises all hits.
    All,
    /// Only units with a fractional expected hit: integer hits are taken
    /// as fixed and only the remainder is randomised.
    Residual,
}

/// The outcome of a full selection.
#[derive(Debug, Clone)]
pub struct Selected {
    /// Where the rotated list starts: a 0-based position in `order`.
    pub head: usize,
    /// The frame units in selection order: `order` rotated to the head.
    pub list: Vec<usize>,
    /// Realised hits, one per position of `list`.
    pub hits: Vec<u64>,
    /// Effective expected hits per *frame unit* (original indexing),
    /// after any floating-point adjustment.
    pub effective_lambda: Vec<f64>,
    /// The inner selection's accounting, when one was run; `None` when
    /// every expected hit was an integer and nothing was random.
    pub accounting: Option<Selection>,
}

fn check_order(order: &[usize], n_units: usize) -> Result<(), Error> {
    if order.len() != n_units {
        return Err(Error::Input(format!(
            "{} expected hits but an order of {} positions",
            n_units,
            order.len()
        )));
    }
    let mut seen = vec![false; n_units];
    for &u in order {
        if u >= n_units || seen[u] {
            return Err(Error::Input(
                "the order must be a permutation of the frame".into(),
            ));
        }
        seen[u] = true;
    }
    Ok(())
}

/// The list positions eligible to be the head, in list order.
pub fn eligible_positions(lambda: &[f64], order: &[usize], head: Head) -> Vec<usize> {
    match head {
        Head::All => (0..order.len()).collect(),
        Head::Residual => (0..order.len())
            .filter(|&p| lambda[order[p]] != lambda[order[p]].floor())
            .collect(),
    }
}

/// How many uniforms [`select_from`] will consume, beyond the head.
pub fn required_uniforms(lambda: &[f64], order: &[usize], head: Head) -> usize {
    match head {
        Head::All => order.len(),
        Head::Residual => eligible_positions(lambda, order, head).len(),
    }
}

/// Draw a position by cumulative inversion: the first weight whose running
/// sum exceeds `u` times the total.
pub fn head_position(weights: &[f64], u: f64) -> Result<usize, Error> {
    if !(0.0..1.0).contains(&u) {
        return Err(Error::Input(format!(
            "the head uniform must lie in [0, 1); it is {u}"
        )));
    }
    let mut total = 0.0_f64;
    for &w in weights {
        if !w.is_finite() || w < 0.0 {
            return Err(Error::Input(format!(
                "head weights must be finite and non-negative; found {w}"
            )));
        }
        total += w;
    }
    if total <= 0.0 {
        return Err(Error::Input("the head weights sum to zero".into()));
    }
    let target = u * total;
    let mut cum = 0.0_f64;
    for (p, &w) in weights.iter().enumerate() {
        cum += w;
        if cum > target {
            return Ok(p);
        }
    }
    Ok(weights.len() - 1)
}

/// Select with the head already drawn.
///
/// `start` is a 0-based position in `order` — one of the
/// [`eligible_positions`] — from the caller's own draw. `uniforms` feeds
/// the sequential selection, [`required_uniforms`] of them: one per unit
/// under [`Head::All`], one per fractional unit under [`Head::Residual`].
pub fn select_from(
    lambda: &[f64],
    order: &[usize],
    start: usize,
    uniforms: &[f64],
    head: Head,
) -> Result<Selected, Error> {
    let n_units = lambda.len();
    check_order(order, n_units)?;

    let eligible = eligible_positions(lambda, order, head);
    if eligible.is_empty() {
        // Every expected hit is an integer: the selection is the expected
        // hits themselves, and nothing is random.
        let hits = order
            .iter()
            .map(|&u| {
                if lambda[u] < 0.0 || !lambda[u].is_finite() {
                    Err(Error::Input(format!(
                        "expected hits must be finite and non-negative; unit {u} \
                         has {}",
                        lambda[u]
                    )))
                } else {
                    Ok(lambda[u] as u64)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Selected {
            head: 0,
            list: order.to_vec(),
            hits,
            effective_lambda: lambda.to_vec(),
            accounting: None,
        });
    }
    if !eligible.contains(&start) {
        return Err(Error::Input(format!(
            "the head position {start} is not eligible"
        )));
    }

    // Rotate so the list starts at the head; `positions` remembers where
    // each listed unit sat in the unrotated order, which is the priority
    // the drift correction respects.
    let positions: Vec<usize> = (start..n_units).chain(0..start).collect();
    let list: Vec<usize> = positions.iter().map(|&p| order[p]).collect();

    let mut effective_lambda = lambda.to_vec();
    let (hits, accounting) = match head {
        Head::All => {
            let listed: Vec<f64> = list.iter().map(|&u| lambda[u]).collect();
            let selection = pmr(&listed, uniforms, &positions)?;
            for (i, &u) in list.iter().enumerate() {
                effective_lambda[u] = selection.effective_lambda[i];
            }
            (selection.hits.clone(), selection)
        }
        Head::Residual => {
            let fractional: Vec<usize> = (0..n_units)
                .filter(|&i| lambda[list[i]] != lambda[list[i]].floor())
                .collect();
            let listed: Vec<f64> = fractional.iter().map(|&i| lambda[list[i]]).collect();
            let priority: Vec<usize> = fractional.iter().map(|&i| positions[i]).collect();
            let selection = pmr(&listed, uniforms, &priority)?;
            let mut hits = vec![0_u64; n_units];
            for i in 0..n_units {
                if lambda[list[i]] == lambda[list[i]].floor() {
                    hits[i] = lambda[list[i]] as u64;
                }
            }
            for (s, &i) in fractional.iter().enumerate() {
                hits[i] = selection.hits[s];
                effective_lambda[list[i]] = selection.effective_lambda[s];
            }
            (hits, selection)
        }
    };

    Ok(Selected {
        head: start,
        list,
        hits,
        effective_lambda,
        accounting: Some(accounting),
    })
}

/// Select with the head drawn here, by cumulative inversion.
///
/// `u_head` draws the head among the eligible positions with probability
/// proportional to expected hits; `uniforms` feeds the selection as in
/// [`select_from`]. A binding that must reproduce its host language's own
/// head sampler should call [`select_from`] instead.
pub fn select(
    lambda: &[f64],
    order: &[usize],
    head: Head,
    u_head: f64,
    uniforms: &[f64],
) -> Result<Selected, Error> {
    check_order(order, lambda.len())?;
    let eligible = eligible_positions(lambda, order, head);
    if eligible.is_empty() {
        return select_from(lambda, order, 0, uniforms, head);
    }
    let weights: Vec<f64> = eligible.iter().map(|&p| lambda[order[p]]).collect();
    let start = eligible[head_position(&weights, u_head)?];
    select_from(lambda, order, start, uniforms, head)
}
