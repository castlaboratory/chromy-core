use std::fmt;

/// Why a procedure refused to run, or could not finish.
///
/// Refusals are loud on purpose: a sampling procedure that silently mends
/// its input produces a sample whose inclusion probabilities nobody knows,
/// and the weights built from them estimate nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The input cannot define a valid selection — a negative size, a
    /// uniform outside `[0, 1)`, mismatched lengths. The message says which.
    Input(String),

    /// The expected hits summed too far from an integer to be a sample
    /// size: the drift exceeds the procedure's tolerance of `1e-8`, so the
    /// caller's expected hits do not add to a whole number of selections.
    Drift { eps: f64 },

    /// The floating-point drift could not be absorbed: no unit has a
    /// fractional expected hit large enough to take the correction.
    NoAbsorber { eps: f64 },

    /// After the corrections, units with positive expected hits were left
    /// with no probability of selection at all. Their weights would be
    /// infinite; the selection is refused instead.
    LostProbability { units: usize },

    /// The procedure's own accounting failed — realised hits that do not
    /// add to the sample size. This is a bug in the crate, not in the
    /// input; please report it.
    Internal(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Input(what) => write!(f, "invalid input: {what}"),
            Error::Drift { eps } => write!(
                f,
                "the expected hits sum {eps:e} away from an integer, beyond the 1e-8 \
                 tolerance; they do not define a sample size"
            ),
            Error::NoAbsorber { eps } => write!(
                f,
                "no unit can absorb the floating-point drift of {eps:e}: every \
                 fractional expected hit is too small"
            ),
            Error::LostProbability { units } => write!(
                f,
                "{units} unit(s) with positive expected hits were left with no \
                 probability of selection"
            ),
            Error::Internal(what) => write!(f, "internal accounting error: {what}"),
        }
    }
}

impl std::error::Error for Error {}
