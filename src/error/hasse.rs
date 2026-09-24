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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn geometry_variants_report_their_values_and_limits() {
        let length = HasseError::LengthMismatch {
            argument: "derivative output",
            expected: 5,
            actual: 3,
        };
        let rendered_length = format!("{length}");
        assert!(
            rendered_length.contains("derivative output")
                && rendered_length.contains('5')
                && rendered_length.contains('3')
        );

        let capacity = HasseError::CoefficientCapacityExceeded {
            maximum: 8,
            actual: 12,
        };
        assert!(format!("{capacity}").contains('8') && format!("{capacity}").contains("12"));

        let batch = HasseError::BatchCapacityExceeded {
            maximum: 2,
            actual: 5,
        };
        assert!(format!("{batch}").contains('2') && format!("{batch}").contains('5'));

        let overflow = HasseError::GeometryOverflow {
            context: "jet slot bytes",
        };
        assert!(format!("{overflow}").contains("jet slot bytes"));

        let allocation = HasseError::AllocationFailed {
            context: "jet recursion slots",
        };
        assert!(format!("{allocation}").contains("jet recursion slots"));

        // Distinct failures stay distinct through the rendering.
        let rendered = [
            rendered_length,
            format!("{capacity}"),
            format!("{batch}"),
            format!("{overflow}"),
            format!("{allocation}"),
            format!("{}", HasseError::ScratchMismatch),
        ];
        let mut distinct = rendered.clone().to_vec();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), rendered.len());
        assert_eq!(length, length);
        assert_ne!(length, capacity);
    }

    #[test]
    fn wrapped_causes_are_preserved() {
        let polynomial = HasseError::from(PolynomialError::DivisionByZero);
        assert_eq!(
            polynomial,
            HasseError::Polynomial(PolynomialError::DivisionByZero)
        );
        assert_ne!(
            polynomial,
            HasseError::Product(ProductError::Polynomial(PolynomialError::DivisionByZero))
        );

        let product = HasseError::from(ProductError::Polynomial(PolynomialError::NonExactDivision));
        assert_eq!(
            product,
            HasseError::Product(ProductError::Polynomial(PolynomialError::NonExactDivision))
        );

        // A config cause travels through the prepared-product layer.
        let config = ConfigError::ScratchTooSmall {
            context: "prepared lane capacity",
            required: 4,
            available: 1,
        };
        assert_eq!(
            HasseError::from(config),
            HasseError::Product(ProductError::Config(config))
        );
        assert!(format!("{}", HasseError::from(config)).contains('4'));
    }

    #[cfg(feature = "std")]
    #[test]
    fn hasse_errors_expose_no_hidden_source_chain() {
        let errors: Vec<Box<dyn std::error::Error>> = vec![
            Box::new(HasseError::ScratchMismatch),
            Box::new(HasseError::LengthMismatch {
                argument: "jet output",
                expected: 3,
                actual: 5,
            }),
            Box::new(HasseError::Polynomial(PolynomialError::DivisionByZero)),
        ];
        for error in &errors {
            assert!(error.source().is_none());
        }
        assert!(format!("{}", errors[1]).contains('5'));
    }
}
