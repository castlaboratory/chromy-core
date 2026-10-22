//! Chromy's method of sequential random sampling, as a library.
//!
//! Chromy (1979) selects units one at a time down an ordered list, each with
//! probability proportional to its size, so that the realised hits of every
//! unit differ from its expected hits by less than one — *probability
//! minimum replacement*. The procedure is the backbone of many large
//! establishment and household surveys, and of `PROC SURVEYSELECT
//! METHOD=PPS_SEQ`; until now it lived only inside proprietary software and
//! one Stata module. This crate is the method in the open, with the
//! apparatus that surrounds it in practice:
//!
//! - [`pmr`]: the sequential selection itself, deterministic given the
//!   uniforms, with compensated summation and an explicit account of every
//!   floating-point adjustment it had to make;
//! - [`serpentine`] and [`nested`]: hierarchical sorting of the frame, so
//!   that neighbouring units are alike and the systematic character of the
//!   selection buys variance;
//! - [`mpps`]: a maximal size measure over several auxiliary variables, for
//!   surveys that must serve more than one estimate;
//! - [`expected_hits`] and [`eqcv`]: from sizes to expected hits, with
//!   certainty units resolved by iteration, and the allocation that
//!   equalises anticipated coefficients of variation across variables;
//! - [`select`] / [`select_from`]: the full procedure with a random head,
//!   as SAS runs it;
//! - [`multiplicity_estimate`]: the estimator that matches the design, with
//!   the successive-difference variance.
//!
//! The crate has no dependencies and takes its randomness as arguments —
//! a slice of uniforms — so a caller in R, Python or Rust reproduces a
//! selection exactly from its own generator and seed. The reference for
//! every numerical choice is the implementation these functions were
//! distilled from, tested bit for bit in `tests/golden.rs`.
//!
//! Chromy, J. R. (1979). Sequential sample selection methods. *Proceedings
//! of the American Statistical Association, Survey Research Methods
//! Section*, 401–406. Chauvet, G. and Le Gleut, R. (2021). Properties of
//! Chromy's sampling procedure. *Survey Methodology* 47(2).

mod allocate;
mod error;
mod estimate;
mod order;
mod pmr;
mod select;
mod size;

pub use allocate::{eqcv, expected_hits, EqcvAllocation, EqcvOptions};
pub use error::Error;
pub use estimate::{multiplicity_estimate, Estimate};
pub use order::{nested, serpentine, Key};
pub use pmr::{pmr, Selection};
pub use select::{
    eligible_positions, head_position, required_uniforms, select, select_from, Head, Selected,
};
pub use size::mpps;
