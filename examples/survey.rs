//! A small establishment survey, end to end: a frame with two auxiliary
//! measures, the equal-CV allocation, the maximal size, the serpentine
//! order, the selection, and the estimate.
//!
//! Run with `cargo run --example survey`.

use chromy_core::*;

/// SplitMix64, so the example is reproducible without a dependency.
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

fn main() -> Result<(), Error> {
    let mut rng = Rng(20261022);

    // A frame of 200 establishments in four regions, with employment and
    // revenue as auxiliary measures and revenue the study variable.
    let n_units = 200;
    let regions = ["North", "East", "South", "West"];
    let region: Vec<String> = (0..n_units)
        .map(|i| regions[i % regions.len()].to_string())
        .collect();
    let employment: Vec<f64> = (0..n_units)
        .map(|_| (rng.uniform() * 50.0).ceil() + 1.0)
        .collect();
    // Revenue is skewed and only loosely tied to employment, so the two
    // measures genuinely disagree about which units are large.
    let revenue: Vec<f64> = employment
        .iter()
        .map(|&e| {
            let u = rng.uniform();
            20.0 + 10.0 * e * u + 4000.0 * u * u * u
        })
        .collect();

    // One sample to serve both measures: the factors that equalise the
    // anticipated CVs, then the maximal size they imply.
    let mut x = employment.clone();
    x.extend_from_slice(&revenue);
    let allocation = eqcv(&x, n_units, 40, EqcvOptions::default())?;
    println!(
        "equal-CV factors: {:?} ({} iterations)",
        allocation.n_j, allocation.iterations
    );
    let size = mpps(&x, n_units, &allocation.n_j)?;
    let lambda = expected_hits(&size, 40, false)?;

    // Serpentine order within region by size, head drawn at random, and
    // the sequential selection down the rotated list.
    let order = serpentine(&[Key::Categorical(&region), Key::Numeric(&size)])?;
    let u_head = rng.uniform();
    let uniforms = rng.uniforms(required_uniforms(&lambda, &order, Head::All));
    let sample = select(&lambda, &order, Head::All, u_head, &uniforms)?;

    let drawn: Vec<usize> = (0..n_units).filter(|&p| sample.hits[p] > 0).collect();
    println!("selected {} establishments", drawn.len());

    // The estimate of total revenue, with its successive-difference SE.
    let units: Vec<usize> = drawn.iter().map(|&p| sample.list[p]).collect();
    let hits: Vec<u64> = drawn.iter().map(|&p| sample.hits[p]).collect();
    let expected: Vec<f64> = units.iter().map(|&u| sample.effective_lambda[u]).collect();
    let estimate = multiplicity_estimate(&revenue, n_units, &units, &drawn, &expected, &hits)?;

    let truth: f64 = revenue.iter().sum();
    println!(
        "total revenue: estimated {:.0}, true {:.0}, SE {:.0}",
        estimate.total[0],
        truth,
        estimate.sd_variance[0].sqrt()
    );
    Ok(())
}
