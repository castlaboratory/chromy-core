# chromy-core

[![crates.io](https://img.shields.io/crates/v/chromy-core.svg)](https://crates.io/crates/chromy-core)
[![docs.rs](https://img.shields.io/docsrs/chromy-core)](https://docs.rs/chromy-core)
[![license: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

Chromy's method of sequential random sampling, in the open. The procedure
selects units one at a time down an ordered list, each with probability
proportional to its size, so that every unit's realised number of selections
differs from its expected number by less than one — *probability minimum
replacement* (Chromy, 1979). It is how many large establishment and household
surveys draw their samples, and how `PROC SURVEYSELECT METHOD=PPS_SEQ` works;
until now it lived only inside proprietary software and one Stata module.

This crate is the numerical engine of an ecosystem: the R package
[`chromy`](https://github.com/castlaboratory/chromy) and the Python package
[`pychromy`](https://github.com/castlaboratory/pychromy) bind it. It has no
dependencies, and it takes its randomness as arguments — slices of uniforms —
so a caller in any language reproduces a selection exactly from its own
generator and seed.

## What it does

- **`pmr`** — the sequential selection itself, deterministic given the
  uniforms, with compensated summation and an explicit account of every
  floating-point adjustment: which unit absorbed the drift, by how much, and
  every near-boundary crossing it settled. The effective expected hits it
  reports are the denominators estimation must use.
- **`serpentine` / `nested`** — hierarchical sorting of the frame, so that
  neighbouring units are alike and the systematic character of the selection
  buys variance. The serpentine reverses direction at every group boundary;
  categorical keys rank by byte order, never by locale.
- **`mpps`** — a maximal size measure over several auxiliary variables
  (Folsom, Potter and Williams, 1987), for surveys that must serve more than
  one estimate.
- **`expected_hits`** — from sizes to expected hits, certainty units resolved
  by iteration, or left above one under minimum replacement.
- **`eqcv`** — the allocation that equalises anticipated coefficients of
  variation across the auxiliary variables.
- **`select` / `select_from`** — the full procedure with a random head, as
  SAS runs it.
- **`multiplicity_estimate`** — the estimator that matches the design, with
  the successive-difference variance.

## A selection, in full

```rust
use chromy_core::*;

// Expected hits proportional to size, 12 selections over 40 units.
let size: Vec<f64> = (1..=40).map(|i| i as f64).collect();
let lambda = expected_hits(&size, 12, true)?;

// Serpentine order within stratum by size.
let stratum: Vec<String> = (0..40).map(|i| format!("s{}", i / 10)).collect();
let order = serpentine(&[Key::Categorical(&stratum), Key::Numeric(&size)])?;

// The uniforms come from the caller: the selection is a pure function.
let uniforms: Vec<f64> = your_rng(required_uniforms(&lambda, &order, Head::All));
let sample = select(&lambda, &order, Head::All, your_rng_one(), &uniforms)?;

assert_eq!(sample.hits.iter().sum::<u64>(), 12);
```

`cargo run --example survey` walks a small establishment survey end to end:
two auxiliary measures, the equal-CV allocation, the maximal size, the
serpentine order, the selection, and the estimate with its standard error.

## Fidelity

The crate was distilled from a reference R implementation and is tested
against it bit for bit: `tests/golden.rs` replays fixed inputs through every
function and demands the integer outputs exactly and the floating-point
outputs to near machine precision. `tests/properties.rs` checks the method's
own promises by simulation — hits within one of the expected hits, the
sample size always exact, expected hits recovered in expectation, the
serpentine never jumping.

Errors are loud on purpose. Expected hits that do not sum to within `1e-8`
of an integer are refused, not rescaled: a selection whose inclusion
probabilities nobody knows estimates nothing.

## References

- Chromy, J. R. (1979). Sequential sample selection methods. *Proceedings of
  the American Statistical Association, Survey Research Methods Section*,
  401–406.
- Chauvet, G. and Le Gleut, R. (2021). Properties of Chromy's sampling
  procedure. *Survey Methodology*, 47(2).
- Folsom, R. E., Potter, F. J. and Williams, S. R. (1987). Notes on a
  composite size measure for self-weighting samples in multiple domains.
  *Proceedings of the American Statistical Association, Section on Survey
  Research Methods*, 792–796.

## License

GPL-3.0-or-later. © the CAST Lab authors; see the R package's DESCRIPTION
for the full list.
