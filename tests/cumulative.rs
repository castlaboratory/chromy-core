//! A second, independent formulation of the selection as an oracle.
//!
//! The crate's `pmr` carries the running remainder as state, as the
//! reference implementation does. The same procedure can instead be read
//! off the *cumulative* expected hits: split each partial sum into integer
//! and fractional parts and express Chromy's transition probabilities in
//! those terms. The two formulations were written independently — this one
//! follows Clara Ribeiro de Santana's R implementation of the method — and
//! must give the same hits from the same uniforms; a disagreement is a bug
//! in one of them, whichever it is.

use chromy_core::{expected_hits, pmr};

/// SplitMix64, as in the other test files.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn uniforms(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform()).collect()
    }
}

/// The cumulative formulation: hits from the partial sums of the expected
/// hits, one conditional draw per unit.
fn pmr_cumulative(expected: &[f64], uniforms: &[f64]) -> Vec<i64> {
    let n_units = expected.len();
    let total_expected: f64 = expected.iter().sum();
    let n = total_expected.round_ties_even() as i64;

    let mut cumulative: Vec<f64> = Vec::with_capacity(n_units);
    let mut running = 0.0_f64;
    for &e in expected {
        running += e;
        cumulative.push(running);
    }
    // Snap values a hair's breadth from an integer, and force the end.
    for c in &mut cumulative {
        if (*c - c.round()).abs() < 1e-12 {
            *c = c.round();
        }
    }
    cumulative[n_units - 1] = n as f64;

    let mut hits = vec![0_i64; n_units];
    let mut total_prev = 0_i64;
    let mut int_prev = 0_i64;
    let mut frac_prev = 0.0_f64;

    for i in 0..n_units {
        let integer = cumulative[i].floor() as i64;
        let fraction = cumulative[i] - integer as f64;

        let total = if total_prev == int_prev {
            if fraction == 0.0 || frac_prev > fraction {
                integer
            } else {
                let chance = (fraction - frac_prev) / (1.0 - frac_prev);
                integer + i64::from(uniforms[i] < chance)
            }
        } else if fraction == 0.0 {
            integer
        } else if fraction > frac_prev {
            integer + 1
        } else {
            let chance = fraction / frac_prev;
            integer + i64::from(uniforms[i] < chance)
        };

        hits[i] = total - total_prev;
        total_prev = total;
        int_prev = integer;
        frac_prev = fraction;
    }
    hits
}

#[test]
fn the_two_formulations_agree() {
    let mut rng = Rng(20261022);
    for trial in 0..300 {
        let n_units = 10 + (trial % 70);
        let n = 1 + (trial % 19) as u64;
        let size: Vec<f64> = (0..n_units).map(|_| rng.uniform() * 5.0 + 0.05).collect();
        let lambda = expected_hits(&size, n, true).unwrap();
        let uniforms = rng.uniforms(n_units);
        let priority: Vec<usize> = (0..n_units).collect();

        let state_form = pmr(&lambda, &uniforms, &priority).unwrap();
        let cumulative_form = pmr_cumulative(&lambda, &uniforms);

        let state: Vec<i64> = state_form.hits.iter().map(|&h| h as i64).collect();
        assert_eq!(
            state, cumulative_form,
            "trial {trial}: the state and cumulative formulations disagree"
        );
    }
}
