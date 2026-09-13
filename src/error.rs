//! Hand-rolled error enums, one per failure domain.
//!
//! Every struct variant carries the offending value *and* the limit it
//! violated. `Display` is implemented manually with inline-captured arguments;
//! [`std::error::Error`] is implemented under `std`. No error describes a zero
//! field-element result: `inv(0) == 0` is a total-function convention inherited
//! from `fgf`, and only genuine geometry violations, division by the zero
//! polynomial, and non-exact or ill-conditioned division are errors.

use core::fmt;

#[cfg(feature = "fft")]
use butterfly_fft::error::{PlanError, TransformLengthError};
#[cfg(feature = "fft")]
use butterfly_fft::ntt::NttError;

/// Failure while validating a geometry or reserving its storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// A required count is zero.
    ZeroParameter {
        /// Name of the zero-valued parameter.
        parameter: &'static str,
    },
    /// A requested point set is larger than the field that must hold it.
    FieldCapacityExceeded {
        /// Number of requested points.
        points: usize,
        /// Number of elements in the field.
        field_order: u128,
    },
    /// A derived length or byte count cannot be represented by `usize`.
    GeometryOverflow {
        /// Name of the overflowing dimension.
        context: &'static str,
    },
    /// A byte buffer length does not match the geometry it was declared for.
    BufferLength {
        /// Name of the buffer whose length was wrong.
        context: &'static str,
        /// Length the geometry requires, in bytes.
        expected: usize,
        /// Length the caller supplied, in bytes.
        actual: usize,
    },
    /// Reusable scratch was built for a smaller geometry than requested.
    ScratchTooSmall {
        /// Name of the scratch buffer that is too small.
        context: &'static str,
        /// Capacity the request needs.
        required: usize,
        /// Capacity the scratch was built with.
        available: usize,
    },
    /// Storage for a validated geometry could not be reserved.
    AllocationFailed {
        /// Name of the storage that failed to reserve.
        context: &'static str,
        /// Number of elements the reservation needed.
        elements: usize,
        /// Size of one element in bytes.
        element_size: usize,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroParameter { parameter } => {
                write!(formatter, "{parameter} must be nonzero")
            }
            Self::FieldCapacityExceeded {
                points,
                field_order,
            } => write!(
                formatter,
                "{points} points exceed the field capacity of {field_order} elements"
            ),
            Self::GeometryOverflow { context } => {
                write!(formatter, "{context} exceeds the address space")
            }
            Self::AllocationFailed {
                context,
                elements,
                element_size,
            } => write!(
                formatter,
                "{context} could not reserve {elements} elements of {element_size} bytes"
            ),
            Self::BufferLength {
                context,
                expected,
                actual,
            } => write!(
                formatter,
                "{context} expects {expected} bytes, found {actual}"
            ),
            Self::ScratchTooSmall {
                context,
                required,
                available,
            } => write!(
                formatter,
                "{context} needs capacity {required}, scratch holds {available}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ConfigError {}

/// Failure during polynomial arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PolynomialError {
    /// Checked coefficient geometry or allocation failed.
    Config(ConfigError),
    /// Polynomial division was requested with the zero divisor.
    DivisionByZero,
    /// A division expected to have zero remainder did not.
    NonExactDivision,
    /// A truncated power series inversion was requested for a polynomial
    /// whose constant coefficient is zero (not invertible modulo `x^t`).
    ZeroConstantTerm {
        /// Name of the operation that required a unit constant term.
        context: &'static str,
    },
}

impl From<ConfigError> for PolynomialError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl fmt::Display for PolynomialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::DivisionByZero => formatter.write_str("polynomial division by zero"),
            Self::NonExactDivision => formatter.write_str("polynomial division was not exact"),
            Self::ZeroConstantTerm { context } => write!(
                formatter,
                "{context} requires a nonzero constant coefficient"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PolynomialError {}

/// Failure during a batched polynomial product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProductError {
    /// Checked storage geometry or allocation failed.
    Config(ConfigError),
    /// Supporting polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// The requested transform domain cannot be constructed.
    #[cfg(feature = "fft")]
    Plan(PlanError),
    /// A conversion or transform buffer has inconsistent geometry.
    #[cfg(feature = "fft")]
    Transform(TransformLengthError),
    /// A multiplicative transform plan could not be built or executed.
    #[cfg(feature = "fft")]
    Ntt(NttError),
}

impl From<ConfigError> for ProductError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<PolynomialError> for ProductError {
    fn from(error: PolynomialError) -> Self {
        match error {
            PolynomialError::Config(config) => Self::Config(config),
            other => Self::Polynomial(other),
        }
    }
}

#[cfg(feature = "fft")]
impl From<PlanError> for ProductError {
    fn from(error: PlanError) -> Self {
        Self::Plan(error)
    }
}

#[cfg(feature = "fft")]
impl From<TransformLengthError> for ProductError {
    fn from(error: TransformLengthError) -> Self {
        Self::Transform(error)
    }
}
#[cfg(feature = "fft")]
impl From<NttError> for ProductError {
    fn from(error: NttError) -> Self {
        Self::Ntt(error)
    }
}
impl fmt::Display for ProductError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            #[cfg(feature = "fft")]
            Self::Plan(error) => error.fmt(formatter),
            #[cfg(feature = "fft")]
            Self::Ntt(error) => error.fmt(formatter),
            #[cfg(feature = "fft")]
            Self::Transform(error) => error.fmt(formatter),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ProductError {}

/// Failure while isolating roots over the coefficient field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RootError {
    /// Supporting polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Accelerated polynomial multiplication failed.
    Product(ProductError),
    /// The field is not represented as a supported binary extension field.
    UnsupportedField {
        /// Number of elements in the field.
        field_order: u128,
        /// Bytes in the stable element representation.
        element_bytes: usize,
    },
    /// The zero bivariate polynomial has every bounded polynomial as a root.
    ZeroBivariatePolynomial,
    /// A polynomial passed to the linearized solver carries a coefficient at
    /// a degree that is not a power of two.
    NotLinearized {
        /// The offending degree.
        degree: usize,
    },
    /// A caller-provided extraction resource limit was reached.
    ResourceLimitExceeded {
        /// Name of the bounded resource.
        resource: &'static str,
        /// Amount required to continue extraction.
        required: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// A factor known to split into distinct linear factors could not be split.
    FactorizationInvariant {
        /// Static explanation of the violated invariant.
        reason: &'static str,
    },
}

impl From<PolynomialError> for RootError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<ProductError> for RootError {
    fn from(error: ProductError) -> Self {
        Self::Product(error)
    }
}

impl From<ConfigError> for RootError {
    fn from(error: ConfigError) -> Self {
        Self::Polynomial(PolynomialError::Config(error))
    }
}

impl fmt::Display for RootError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Product(error) => error.fmt(formatter),
            Self::UnsupportedField {
                field_order,
                element_bytes,
            } => write!(
                formatter,
                "field order {field_order} with {element_bytes}-byte elements is not a supported binary field representation"
            ),
            Self::ZeroBivariatePolynomial => {
                formatter.write_str("the zero bivariate polynomial has every polynomial as a root")
            }
            Self::NotLinearized { degree } => write!(
                formatter,
                "coefficient at degree {degree} is not at a power-of-two degree"
            ),
            Self::ResourceLimitExceeded {
                resource,
                required,
                limit,
            } => write!(
                formatter,
                "{resource} requires {required}, exceeding the root-extraction limit {limit}"
            ),
            Self::FactorizationInvariant { reason } => {
                write!(
                    formatter,
                    "polynomial root-extraction invariant failed: {reason}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RootError {}

/// Failure while constructing or matching an evaluation domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DomainError {
    /// General checked geometry or allocation failure.
    Config(ConfigError),
    /// Two arbitrary evaluation points are equal.
    DuplicatePoint {
        /// Index of the first occurrence.
        first: usize,
        /// Index of the duplicate occurrence.
        second: usize,
    },
    /// A claimed additive subspace or coset size is not a power of two.
    NotSubspace {
        /// The size that is not a subspace size.
        size: usize,
        /// Smallest power-of-two bound the size exceeds, when it is too large.
        limit: usize,
    },
    /// A butterfly-fft plan could not represent the requested domain.
    #[cfg(feature = "fft")]
    TransformPlan(PlanError),
    /// A point set / value vector length does not match the domain size.
    LengthMismatch {
        /// Length required by the domain.
        expected: usize,
        /// Length found in the input.
        found: usize,
    },
}

impl From<ConfigError> for DomainError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

#[cfg(feature = "fft")]
impl From<PlanError> for DomainError {
    fn from(error: PlanError) -> Self {
        Self::TransformPlan(error)
    }
}

impl fmt::Display for DomainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::DuplicatePoint { first, second } => write!(
                formatter,
                "evaluation points at indices {first} and {second} are equal"
            ),
            Self::NotSubspace { size, limit } => {
                write!(
                    formatter,
                    "size {size} is not a subspace size (limit {limit})"
                )
            }
            #[cfg(feature = "fft")]
            Self::TransformPlan(error) => error.fmt(formatter),
            Self::LengthMismatch { expected, found } => write!(
                formatter,
                "input has {found} entries, but the evaluation domain requires {expected}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DomainError {}

/// Failure during evaluation or interpolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvalError {
    /// Supporting polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// The point set or value vector violates the domain geometry.
    Domain(DomainError),
}

impl From<PolynomialError> for EvalError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<DomainError> for EvalError {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

impl From<ConfigError> for EvalError {
    fn from(error: ConfigError) -> Self {
        Self::Polynomial(PolynomialError::Config(error))
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Domain(error) => error.fmt(formatter),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EvalError {}
