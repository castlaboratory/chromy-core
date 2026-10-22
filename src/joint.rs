//! Second-order inclusion probabilities, exactly.
//!
//! Chromy's procedure run down a fixed list is the same design as ordered
//! pivotal sampling (Chauvet and Le Gleut, 2021, Theorem 1), and for that
//! design the joint inclusion probabilities have closed forms (their
//! Theorem 3): the list splits into *microstrata* at the integer
//! boundaries of the cumulative expected hits; selections in different
//! microstrata are nearly independent, coupled only through the
//! cross-border units, and the coupling decays as the product of the
//! factors `c_i = a_i b_i / ((1 - a_i)(1 - b_i))` of the boundaries in
//! between. With the random head the design is the mixture of the `N`
//! circular rotations, the rotation starting at unit `k` carrying weight
//! `pi_k / n` (their eq. 5.3) — which is precisely the head this crate
//! draws.
//!
//! With these probabilities the Sen–Yates–Grundy variance of the
//! Horvitz–Thompson total is exact, not approximated: the one piece of
//! Chromy's method that, until now, open software simply did not have.
//!
//! The formulas hold for the without-replacement design (every expected
//! hit at most one); a minimum-replacement design with hits above one has
//! joint *moments*, not joint inclusion probabilities, and is refused.

use crate::error::Error;

/// How near an integer the cumulative expected hits must land to be taken
/// as exactly on the boundary.
const BOUNDARY_TOL: f64 = 1e-9;

/// The microstratum structure of one ordered list.
struct Strata {
    /// For each unit (list position): the microstratum it opens into —
    /// interior units belong to exactly one, a cross-border unit `k_m`
    /// is recorded with its boundary index `m` (1-based).
    kind: Vec<UnitKind>,
    /// `b_m = V_{k_m} - m` for boundary `m` (1-based; index 0 unused).
    b: Vec<f64>,
    /// Prefix products of the coupling factors: `prefix[j]` is the
    /// product of `c_1..c_j`, with the positions of exact zeros tracked
    /// so that a ratio never divides by zero.
    c_prefix: Vec<f64>,
    zeros_up_to: Vec<usize>,
}

#[derive(Clone, Copy, PartialEq)]
enum UnitKind {
    /// Interior to microstratum `i` (1-based).
    Interior(usize),
    /// The cross-border unit of boundary `m`: closes microstratum `m`,
    /// opens `m + 1`.
    Cross(usize),
}

impl Strata {
    fn build(pi: &[f64], n: u64) -> Strata {
        let n_units = pi.len();
        let mut kind = vec![UnitKind::Interior(1); n_units];
        let mut b = vec![0.0_f64; n as usize + 1];
        let mut c = vec![0.0_f64; n as usize];

        // The walk, with compensated summation; a boundary is crossed
        // whenever the integer part advances. A unit that lands exactly on
        // an integer is not cross-border: it closes its microstratum from
        // the inside (joint probability zero with its stratum mates), and
        // the boundary decouples the strata (c = 0).
        let mut v = 0.0_f64;
        let mut comp = 0.0_f64;
        let mut v_eff = 0.0_f64; // the snapped running value
        let mut boundary = 0_usize; // boundaries crossed so far
        for (k, &p) in pi.iter().enumerate() {
            let v_prev = v_eff;
            let t = v + p;
            if v.abs() >= p.abs() {
                comp += (v - t) + p;
            } else {
                comp += (p - t) + v;
            }
            v = t;
            let mut v_now = v + comp;
            if (v_now - v_now.round()).abs() < BOUNDARY_TOL {
                v_now = v_now.round();
            }
            v_eff = v_now;
            let next_int = (boundary + 1) as f64;
            if v_now >= next_int && boundary + 1 < n as usize {
                boundary += 1;
                let a = next_int - v_prev;
                let bm = v_now - next_int;
                if bm <= BOUNDARY_TOL {
                    // Exact landing: interior to the stratum it closes.
                    kind[k] = UnitKind::Interior(boundary);
                    b[boundary] = 0.0;
                    c[boundary] = 0.0;
                } else {
                    kind[k] = UnitKind::Cross(boundary);
                    b[boundary] = bm;
                    c[boundary] = if a <= BOUNDARY_TOL {
                        0.0
                    } else {
                        a * bm / ((1.0 - a) * (1.0 - bm))
                    };
                }
            } else {
                kind[k] = UnitKind::Interior(boundary + 1);
            }
        }

        // Prefix products with zero tracking: c_range(i, j) multiplies
        // c_i..c_{j-1}.
        let m = c.len(); // n boundaries' slots, 1..n-1 real
        let mut c_prefix = vec![1.0_f64; m + 1];
        let mut zeros_up_to = vec![0_usize; m + 1];
        for t in 1..=m.saturating_sub(1) {
            let ct = c[t];
            zeros_up_to[t] = zeros_up_to[t - 1] + usize::from(ct == 0.0);
            c_prefix[t] = if ct == 0.0 {
                c_prefix[t - 1]
            } else {
                c_prefix[t - 1] * ct
            };
        }
        if m > 0 {
            zeros_up_to[m] = zeros_up_to[m.saturating_sub(1)];
            c_prefix[m] = c_prefix[m.saturating_sub(1)];
        }
        Strata {
            kind,
            b,
            c_prefix,
            zeros_up_to,
        }
    }

    /// The coupling `c(i, j)` across boundaries `i..j-1`, 1 when `i == j`.
    fn coupling(&self, i: usize, j: usize) -> f64 {
        if i >= j {
            return 1.0;
        }
        if self.zeros_up_to[j - 1] > self.zeros_up_to[i - 1] {
            return 0.0;
        }
        self.c_prefix[j - 1] / self.c_prefix[i - 1]
    }

    /// The joint inclusion probability of the units at list positions
    /// `k < l`, with first-order probabilities `pk`, `pl` (Theorem 3).
    fn joint(&self, k: usize, l: usize, pk: f64, pl: f64) -> f64 {
        if pk == 0.0 || pl == 0.0 {
            return 0.0;
        }
        match (self.kind[k], self.kind[l]) {
            (UnitKind::Interior(i), UnitKind::Interior(j)) => {
                if i == j {
                    0.0
                } else {
                    pk * pl * (1.0 - self.coupling(i, j))
                }
            }
            (UnitKind::Cross(m), UnitKind::Interior(j)) => {
                // k = k_m opens microstratum i = m + 1.
                let i = m + 1;
                let bm = self.b[m];
                let correction = if pk >= 1.0 || bm >= 1.0 - BOUNDARY_TOL {
                    0.0
                } else {
                    bm * (1.0 - pk) / (pk * (1.0 - bm)) * self.coupling(i, j)
                };
                pk * pl * (1.0 - correction)
            }
            (UnitKind::Interior(i), UnitKind::Cross(m)) => {
                // l = k_m closes microstratum m: as k_{j-1} with j = m + 1.
                let j = m + 1;
                let bm = self.b[m];
                let correction = if pl >= 1.0 || bm <= BOUNDARY_TOL {
                    0.0
                } else {
                    (1.0 - pl) * (1.0 - bm) / (pl * bm) * self.coupling(i, j)
                };
                pk * pl * (1.0 - correction)
            }
            (UnitKind::Cross(mk), UnitKind::Cross(ml)) => {
                let i = mk + 1;
                let j = ml + 1;
                let bi = self.b[mk];
                let bj = self.b[ml];
                let correction =
                    if pk >= 1.0 || pl >= 1.0 || bi >= 1.0 - BOUNDARY_TOL || bj <= BOUNDARY_TOL {
                        0.0
                    } else {
                        bi * (1.0 - bj) * (1.0 - pk) * (1.0 - pl) / (pk * pl * bj * (1.0 - bi))
                            * self.coupling(i, j)
                    };
                pk * pl * (1.0 - correction)
            }
        }
    }
}

fn validate(pi: &[f64], order: &[usize]) -> Result<u64, Error> {
    if order.len() != pi.len() {
        return Err(Error::Input(format!(
            "{} probabilities but an order of {} positions",
            pi.len(),
            order.len()
        )));
    }
    for (i, &p) in pi.iter().enumerate() {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(Error::Input(format!(
                "joint inclusion probabilities are defined for a without-replacement \
                 design: every probability in [0, 1]; unit {i} has {p}"
            )));
        }
    }
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for &p in pi {
        let t = sum + p;
        if sum.abs() >= p.abs() {
            comp += (sum - t) + p;
        } else {
            comp += (p - t) + sum;
        }
        sum = t;
    }
    let n = (sum + comp).round_ties_even();
    if ((sum - n) + comp).abs() >= 1e-6 {
        return Err(Error::Drift {
            eps: (sum - n) + comp,
        });
    }
    if n < 1.0 {
        return Err(Error::Input(
            "the probabilities sum to zero; there is no design".into(),
        ));
    }
    Ok(n as u64)
}

/// Joint inclusion probabilities of chosen units, exactly.
///
/// `pi` holds the first-order inclusion probabilities in frame order
/// (each in `[0, 1]`, summing to the sample size); `order` the listing
/// (0-based permutation); `units` the frame indices whose pairwise
/// probabilities are wanted — typically the sampled units. With
/// `randomized_head` false the list starts at its first unit; with true
/// the design is the mixture over the random head this crate draws, the
/// rotation starting at each unit weighted by its probability over the
/// sample size (Chauvet and Le Gleut, eq. 5.3).
///
/// Returns the `m x m` matrix row-major, `m = units.len()`, with the
/// first-order probabilities on the diagonal. Cost is `O(N m^2)` with the
/// randomized head, `O(N + m^2)` without.
pub fn joint_inclusion(
    pi: &[f64],
    order: &[usize],
    units: &[usize],
    randomized_head: bool,
) -> Result<Vec<f64>, Error> {
    let n = validate(pi, order)?;
    let n_units = pi.len();
    let mut seen = vec![false; n_units];
    for &u in order {
        if u >= n_units || seen[u] {
            return Err(Error::Input(
                "the order must be a permutation of the frame".into(),
            ));
        }
        seen[u] = true;
    }
    for &u in units {
        if u >= n_units {
            return Err(Error::Input(format!(
                "unit {u} is beyond the {n_units} units of the frame"
            )));
        }
    }

    let m = units.len();
    let mut matrix = vec![0.0_f64; m * m];
    for (r, &u) in units.iter().enumerate() {
        matrix[r * m + r] = pi[u];
    }
    if m < 2 {
        return Ok(matrix);
    }

    // Position of each frame unit in the (unrotated) list.
    let mut position = vec![0_usize; n_units];
    for (p, &u) in order.iter().enumerate() {
        position[u] = p;
    }

    let listed: Vec<f64> = order.iter().map(|&u| pi[u]).collect();

    let accumulate = |rotation_start: usize, weight: f64, matrix: &mut [f64]| {
        let rotated: Vec<f64> = (0..n_units)
            .map(|p| listed[(p + rotation_start) % n_units])
            .collect();
        let strata = Strata::build(&rotated, n);
        for r in 0..m {
            for s in (r + 1)..m {
                let pr = (position[units[r]] + n_units - rotation_start) % n_units;
                let ps = (position[units[s]] + n_units - rotation_start) % n_units;
                let (k, l) = if pr < ps { (pr, ps) } else { (ps, pr) };
                let value = strata.joint(k, l, rotated[k], rotated[l]);
                matrix[r * m + s] += weight * value;
                matrix[s * m + r] += weight * value;
            }
        }
    };

    if randomized_head {
        let n_f = n as f64;
        for (start, &p_start) in listed.iter().enumerate() {
            let weight = p_start / n_f;
            if weight > 0.0 {
                accumulate(start, weight, &mut matrix);
            }
        }
    } else {
        accumulate(0, 1.0, &mut matrix);
    }
    Ok(matrix)
}
