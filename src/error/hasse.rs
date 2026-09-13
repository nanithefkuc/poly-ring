//! The error surface of the Hasse derivative, jet, and multiplicity plans.
//!
//! Every struct variant carries the offending value *and* the limit it
//! violated, following the ecosystem convention: hand-rolled `Display`,
//! `std::error::Error` under `std`, and no error describing a zero
//! field-element result (`inv(0) == 0` is `fgf`'s total-function convention).

use core::fmt;

use super::{ConfigError, PolynomialError, ProductError};

/// Failure while constructing or running a Hasse plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HasseError {
    /// A buffer length does not match the geometry it was declared for.
    LengthMismatch {
        /// Name of the argument whose length was wrong.
        argument: &'static str,
        /// Length the geometry requires.
        expected: usize,
        /// Length the caller supplied.
        actual: usize,
    },
    /// An input polynomial exceeded the plan's prepared coefficient bound.
    CoefficientCapacityExceeded {
        /// Maximum coefficient count the plan was built for.
        maximum: usize,
        /// Coefficient count the caller supplied.
        actual: usize,
    },
    /// A batch exceeded the scratch's prepared lane capacity.
    BatchCapacityExceeded {
        /// Maximum lane count the scratch was built for.
        maximum: usize,
        /// Lane count the caller supplied.
        actual: usize,
    },
    /// Scratch built for a different plan geometry.
    ///
    /// Compatibility is a property of the required buffer layout and
    /// lengths, never of point values: scratch holds no point-dependent
    /// state, and two different points with identical layout requirements
    /// can share scratch safely.
    ScratchMismatch,
    /// A derived length or byte count cannot be represented by `usize`.
    GeometryOverflow {
        /// Name of the overflowing dimension.
        context: &'static str,
    },
    /// Storage for a validated geometry could not be reserved.
    AllocationFailed {
        /// Name of the storage that failed to reserve.
        context: &'static str,
    },
    /// Supporting polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// A prepared product engine step failed.
    Product(ProductError),
}

impl From<PolynomialError> for HasseError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<ProductError> for HasseError {
    fn from(error: ProductError) -> Self {
        Self::Product(error)
    }
}

impl From<ConfigError> for HasseError {
    fn from(error: ConfigError) -> Self {
        Self::Product(ProductError::Config(error))
    }
}

impl fmt::Display for HasseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                argument,
                expected,
                actual,
            } => write!(formatter, "{argument} expects {expected}, found {actual}"),
            Self::CoefficientCapacityExceeded { maximum, actual } => write!(
                formatter,
                "{actual} coefficients exceed the prepared maximum of {maximum}"
            ),
            Self::BatchCapacityExceeded { maximum, actual } => write!(
                formatter,
                "batch of {actual} lanes exceeds the prepared maximum of {maximum}"
            ),
            Self::ScratchMismatch => {
                formatter.write_str("scratch does not match this plan's geometry")
            }
            Self::GeometryOverflow { context } => {
                write!(formatter, "{context} exceeds the address space")
            }
            Self::AllocationFailed { context } => {
                write!(formatter, "{context} could not be reserved")
            }
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Product(error) => error.fmt(formatter),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HasseError {}
