//! The joint inclusion probabilities against the design itself.
//!
//! Theorem 3 of Chauvet and Le Gleut (2021) gives the closed forms; the
//! selector in this crate *is* the design they describe. So the exact
//! matrix must satisfy the fixed-size identities to machine precision,
//! and Monte Carlo frequencies from the crate's own selection must
//! converge to it — for the fixed list and for the random head alike.

use chromy_core::{expected_hits, joint_inclusion, pmr, select, Head};

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

fn identity_checks(pi: &[f64], matrix: &[f64], n: u64, what: &str) {
    let m = pi.len();
    for k in 0..m {
        for l in 0..m {
            let v = matrix[k * m + l];
            assert!(
                (-1e-12..=1.0 + 1e-12).contains(&v),
                "{what}: pi[{k},{l}] = {v} out of range"
            );
            assert!(
                (v - matrix[l * m + k]).abs() < 1e-12,
                "{what}: asymmetry at ({k},{l})"
            );
            if k != l {
                assert!(
                    v <= pi[k].min(pi[l]) + 1e-9,
                    "{what}: pi[{k},{l}] = {v} exceeds min of marginals"
                );
            }
        }
        let row: f64 = (0..m).filter(|&l| l != k).map(|l| matrix[k * m + l]).sum();
        let expected = (n as f64 - 1.0) * pi[k];
        assert!(
            (row - expected).abs() < 1e-8,
            "{what}: row {k} sums to {row}, expected {expected}"
        );
    }
}

#[test]
fn the_fixed_size_identities_hold() {
    let mut rng = Rng(1);
    for trial in 0..40 {
        let n_units = 6 + (trial % 25);
        let n = 2 + (trial % 5) as u64;
        let size: Vec<f64> = (0..n_units).map(|_| rng.uniform() * 4.0 + 0.1).collect();
        let pi = expected_hits(&size, n, false).unwrap();
        let order: Vec<usize> = (0..n_units).collect();
        let units: Vec<usize> = (0..n_units).collect();

        for &randomized in &[false, true] {
            let matrix = joint_inclusion(&pi, &order, &units, randomized).unwrap();
            identity_checks(&pi, &matrix, n, &format!("trial {trial} rand {randomized}"));
        }
    }
}

#[test]
fn a_certainty_is_jointly_included_with_everyone() {
    let pi = vec![1.0, 0.4, 0.6, 0.5, 0.5];
    let order: Vec<usize> = (0..5).collect();
    let units: Vec<usize> = (0..5).collect();
    for &randomized in &[false, true] {
        let matrix = joint_inclusion(&pi, &order, &units, randomized).unwrap();
        for l in 1..5 {
            assert!(
                (matrix[l] - pi[l]).abs() < 1e-10,
                "certainty pair with unit {l}: {} vs {}",
                matrix[l],
                pi[l]
            );
        }
        identity_checks(&pi, &matrix, 3, "certainty");
    }
}

#[test]
fn an_exact_stratum_has_zero_joint_probability_inside() {
    // The first microstratum closes exactly at the second unit: the two
    // can never be selected together down the fixed list.
    let pi = vec![0.5, 0.5, 0.7, 0.3, 0.6, 0.4];
    let order: Vec<usize> = (0..6).collect();
    let matrix = joint_inclusion(&pi, &order, &(0..6).collect::<Vec<_>>(), false).unwrap();
    assert!(
        matrix[1].abs() < 1e-12,
        "exact stratum: pi_12 = {}",
        matrix[1]
    );
    identity_checks(&pi, &matrix, 3, "exact stratum");
}

#[test]
fn monte_carlo_agrees_down_the_fixed_list() {
    let mut rng = Rng(7);
    let size = [5.0, 2.0, 4.0, 1.0, 3.0, 2.5, 1.5, 4.5];
    let pi = expected_hits(&size, 4, false).unwrap();
    let n_units = pi.len();
    let order: Vec<usize> = (0..n_units).collect();
    let units: Vec<usize> = (0..n_units).collect();
    let exact = joint_inclusion(&pi, &order, &units, false).unwrap();

    let replications = 800_000_usize;
    let priority: Vec<usize> = (0..n_units).collect();
    let mut counts = vec![0_u64; n_units * n_units];
    for _ in 0..replications {
        let s = pmr(&pi, &rng.uniforms(n_units), &priority).unwrap();
        for k in 0..n_units {
            if s.hits[k] > 0 {
                for l in (k + 1)..n_units {
                    if s.hits[l] > 0 {
                        counts[k * n_units + l] += 1;
                    }
                }
            }
        }
    }
    for k in 0..n_units {
        for l in (k + 1)..n_units {
            let observed = counts[k * n_units + l] as f64 / replications as f64;
            let truth = exact[k * n_units + l];
            let se = (truth * (1.0 - truth) / replications as f64).sqrt();
            assert!(
                (observed - truth).abs() < 4.5 * se.max(2e-4),
                "fixed list pair ({k},{l}): observed {observed}, exact {truth}"
            );
        }
    }
}

#[test]
fn monte_carlo_agrees_with_the_random_head() {
    let mut rng = Rng(11);
    let size = [5.0, 2.0, 4.0, 1.0, 3.0, 2.5, 1.5, 4.5];
    let pi = expected_hits(&size, 4, false).unwrap();
    let n_units = pi.len();
    let order: Vec<usize> = (0..n_units).collect();
    let units: Vec<usize> = (0..n_units).collect();
    let exact = joint_inclusion(&pi, &order, &units, true).unwrap();

    let replications = 800_000_usize;
    let mut counts = vec![0_u64; n_units * n_units];
    for _ in 0..replications {
        let u_head = rng.uniform();
        let uniforms = rng.uniforms(n_units);
        let s = select(&pi, &order, Head::All, u_head, &uniforms).unwrap();
        let mut included = vec![false; n_units];
        for (p, &u) in s.list.iter().enumerate() {
            if s.hits[p] > 0 {
                included[u] = true;
            }
        }
        for k in 0..n_units {
            if included[k] {
                for l in (k + 1)..n_units {
                    if included[l] {
                        counts[k * n_units + l] += 1;
                    }
                }
            }
        }
    }
    for k in 0..n_units {
        for l in (k + 1)..n_units {
            let observed = counts[k * n_units + l] as f64 / replications as f64;
            let truth = exact[k * n_units + l];
            let se = (truth * (1.0 - truth) / replications as f64).sqrt();
            assert!(
                (observed - truth).abs() < 4.5 * se.max(2e-4),
                "random head pair ({k},{l}): observed {observed}, exact {truth}"
            );
        }
    }
}

#[test]
fn a_minimum_replacement_design_is_refused() {
    assert!(joint_inclusion(&[1.4, 0.6], &[0, 1], &[0, 1], false).is_err());
    assert!(joint_inclusion(&[0.5, 0.6], &[0, 1], &[0, 1], false).is_err());
}
