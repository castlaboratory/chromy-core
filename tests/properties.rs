//! What the method promises, checked by simulation.
//!
//! The golden tests pin the crate to the reference implementation; these
//! pin it to the method itself: hits within one of the expected hits, the
//! sample size always exact, expected hits recovered in expectation, and
//! the serpentine order never jumping.

use chromy_core::*;

/// SplitMix64: a tiny deterministic generator for the tests, so the crate
/// itself stays free of dependencies and the tests free of flakiness.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniforms(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform()).collect()
    }
}

/// Expected hits that sum exactly to an integer, built from random sizes.
fn random_lambda(rng: &mut Rng, n_units: usize, n: u64) -> Vec<f64> {
    let size: Vec<f64> = (0..n_units).map(|_| rng.uniform() + 0.01).collect();
    expected_hits(&size, n, true).unwrap()
}

#[test]
fn hits_stay_within_one_of_the_expected_hits() {
    let mut rng = Rng(1);
    for trial in 0..200 {
        let n_units = 5 + (trial % 60);
        let n = 1 + (trial % 17) as u64;
        let lambda = random_lambda(&mut rng, n_units, n);
        let uniforms = rng.uniforms(n_units);
        let priority: Vec<usize> = (0..n_units).collect();
        let s = pmr(&lambda, &uniforms, &priority).unwrap();

        assert_eq!(s.hits.iter().sum::<u64>(), n, "the sample size is exact");
        for (i, &h) in s.hits.iter().enumerate() {
            let lo = s.effective_lambda[i].floor() as u64;
            let hi = s.effective_lambda[i].ceil() as u64;
            assert!(
                h == lo || h == hi,
                "unit {i}: {h} hits against expected {}",
                s.effective_lambda[i]
            );
        }
    }
}

#[test]
fn expected_hits_are_recovered_in_expectation() {
    let mut rng = Rng(2);
    let lambda = vec![0.3, 1.7, 0.5, 0.2, 0.8, 1.5];
    let n: u64 = lambda.iter().sum::<f64>().round() as u64;
    assert_eq!(n, 5);
    let priority: Vec<usize> = (0..lambda.len()).collect();

    let replications = 200_000;
    let mut mean = vec![0.0_f64; lambda.len()];
    for _ in 0..replications {
        let s = pmr(&lambda, &rng.uniforms(lambda.len()), &priority).unwrap();
        for (m, &h) in mean.iter_mut().zip(&s.hits) {
            *m += h as f64;
        }
    }
    for (i, m) in mean.iter().enumerate() {
        let observed = m / replications as f64;
        // Three standard errors of a binomial-like proportion.
        let se = (lambda[i].fract() * (1.0 - lambda[i].fract()) / replications as f64).sqrt();
        assert!(
            (observed - lambda[i]).abs() < 3.5 * se.max(1e-4),
            "unit {i}: observed {observed}, expected {}",
            lambda[i]
        );
    }
}

#[test]
fn certainty_resolution_sums_to_the_sample_size() {
    let mut rng = Rng(3);
    for trial in 0..100 {
        let n_units = 10 + (trial % 50);
        let mut size: Vec<f64> = (0..n_units).map(|_| rng.uniform() * 10.0).collect();
        size[0] = 500.0; // force a certainty
        let n = 2 + (trial % (n_units - 2)) as u64;
        let lambda = expected_hits(&size, n, false).unwrap();
        let total: f64 = lambda.iter().sum();
        assert!(
            (total - n as f64).abs() < 1e-9,
            "trial {trial}: probabilities sum to {total}, not {n}"
        );
        assert!(lambda.iter().all(|&l| (0.0..=1.0).contains(&l)));
    }
}

#[test]
fn the_serpentine_order_never_jumps_within_a_stratum() {
    // Two strata of sizes; within each stratum the serpentine must run
    // monotonically, upward in one, downward in the next.
    let stratum: Vec<String> = ["a", "a", "a", "a", "b", "b", "b", "b"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let size = vec![4.0, 2.0, 3.0, 1.0, 30.0, 10.0, 40.0, 20.0];
    let order = serpentine(&[Key::Categorical(&stratum), Key::Numeric(&size)]).unwrap();

    let listed: Vec<f64> = order.iter().map(|&u| size[u]).collect();
    assert_eq!(&listed[..4], &[1.0, 2.0, 3.0, 4.0], "first stratum ascends");
    assert_eq!(
        &listed[4..],
        &[40.0, 30.0, 20.0, 10.0],
        "second stratum descends: no seam at the boundary"
    );
}

#[test]
fn a_full_selection_is_reproducible_and_exact() {
    let mut rng = Rng(4);
    let n_units = 40;
    let size: Vec<f64> = (0..n_units).map(|_| rng.uniform() * 5.0 + 0.1).collect();
    let lambda = expected_hits(&size, 12, true).unwrap();
    let order: Vec<usize> = (0..n_units).collect();

    let u_head = rng.uniform();
    let uniforms = rng.uniforms(required_uniforms(&lambda, &order, Head::All));
    let a = select(&lambda, &order, Head::All, u_head, &uniforms).unwrap();
    let b = select(&lambda, &order, Head::All, u_head, &uniforms).unwrap();

    assert_eq!(a.hits, b.hits, "the same uniforms give the same sample");
    assert_eq!(a.hits.iter().sum::<u64>(), 12);
    assert_eq!(a.list.len(), n_units);

    // Residual mode fixes the integer part and randomises the rest.
    let lambda_mixed: Vec<f64> = lambda.iter().map(|&l| l + 1.0).collect(); // all have a fixed hit
    let needed = required_uniforms(&lambda_mixed, &order, Head::Residual);
    let c = select(
        &lambda_mixed,
        &order,
        Head::Residual,
        rng.uniform(),
        &rng.uniforms(needed),
    )
    .unwrap();
    for (p, &u) in c.list.iter().enumerate() {
        assert!(
            c.hits[p] >= lambda_mixed[u].floor() as u64,
            "a fixed hit was lost at position {p}"
        );
    }
}

#[test]
fn refusals_are_loud() {
    assert!(
        pmr(&[0.5, 0.6], &[0.1, 0.2], &[0, 1]).is_err(),
        "drift beyond tolerance"
    );
    assert!(
        pmr(&[0.5, 0.5], &[0.1], &[0, 1]).is_err(),
        "missing uniforms"
    );
    assert!(
        pmr(&[-0.5, 1.5], &[0.1, 0.2], &[0, 1]).is_err(),
        "negative expected hits"
    );
    assert!(
        expected_hits(&[1.0, 1.0], 3, false).is_err(),
        "n beyond the frame"
    );
    assert!(
        expected_hits(&[0.0, 0.0], 1, false).is_err(),
        "sizes sum to zero"
    );
    assert!(
        mpps(&[1.0, 2.0, 3.0], 2, &[1.0]).is_err(),
        "a ragged matrix"
    );
    assert!(
        head_position(&[0.0, 0.0], 0.5).is_err(),
        "weights sum to zero"
    );
}
