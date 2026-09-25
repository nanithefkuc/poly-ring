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
use butterfly_fft::error::{PlanError, TransformError};
#[cfg(feature = "fft")]
use butterfly_fft::ntt::NttError;

mod hasse;

pub use hasse::HasseError;

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
    /// A quotient-ring plan was requested for a nonzero constant modulus.
    ConstantModulus,
    /// A residue has no multiplicative inverse modulo the plan's modulus.
    NotInvertibleModulo,
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
            Self::ConstantModulus => {
                formatter.write_str("a polynomial quotient requires a positive-degree modulus")
            }
            Self::NotInvertibleModulo => {
                formatter.write_str("polynomial is not invertible modulo the modulus")
            }
            Self::ZeroConstantTerm { context } => write!(
                formatter,
                "{context} requires a nonzero constant coefficient"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PolynomialError {}

/// Failure during finite-field polynomial factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FactorizationError {
    /// Supporting polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// The zero polynomial has no finite irreducible factorization.
    ZeroPolynomial,
    /// Equal-degree factorization requires a positive irreducible degree.
    ZeroFactorDegree,
    /// The input to a distinct- or equal-degree stage has repeated factors.
    NotSquareFree,
    /// The input contains an irreducible factor of a different degree.
    NotEqualDegree {
        /// Degree requested for every irreducible factor.
        factor_degree: usize,
    },
    /// An internal factorization identity failed.
    Invariant {
        /// Static description of the violated identity.
        reason: &'static str,
    },
}

impl From<PolynomialError> for FactorizationError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<ConfigError> for FactorizationError {
    fn from(error: ConfigError) -> Self {
        Self::Polynomial(PolynomialError::Config(error))
    }
}

impl fmt::Display for FactorizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Polynomial(error) => error.fmt(formatter),
            Self::ZeroPolynomial => {
                formatter.write_str("the zero polynomial has no finite factorization")
            }
            Self::ZeroFactorDegree => {
                formatter.write_str("equal-degree factorization requires a positive degree")
            }
            Self::NotSquareFree => {
                formatter.write_str("factorization stage requires a square-free polynomial")
            }
            Self::NotEqualDegree { factor_degree } => write!(
                formatter,
                "polynomial contains an irreducible factor whose degree is not {factor_degree}"
            ),
            Self::Invariant { reason } => {
                write!(
                    formatter,
                    "polynomial factorization invariant failed: {reason}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FactorizationError {}

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
    Transform(TransformError),
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
impl From<TransformError> for ProductError {
    fn from(error: TransformError) -> Self {
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
    /// Supporting finite-field factorization failed.
    Factorization(FactorizationError),
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

impl From<FactorizationError> for RootError {
    fn from(error: FactorizationError) -> Self {
        Self::Factorization(error)
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
            Self::Factorization(error) => error.fmt(formatter),
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

/// Failure while constructing or running a Hermite interpolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HermiteError {
    /// Checked geometry or allocation failed.
    Config(ConfigError),
    /// Supporting polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// A prepared jet translation failed.
    Hasse(HasseError),
    /// A buffer length does not match the geometry it was declared for.
    LengthMismatch {
        /// Length the geometry requires.
        expected: usize,
        /// Length the caller supplied.
        actual: usize,
    },
    /// Two distinct entries carry the same point and both weights are
    /// positive, so their interpolation constraints conflict.
    DuplicatePoint {
        /// Index of the first occurrence.
        first: usize,
        /// Index of the second occurrence.
        second: usize,
    },
}

impl From<ConfigError> for HermiteError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<PolynomialError> for HermiteError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<HasseError> for HermiteError {
    fn from(error: HasseError) -> Self {
        Self::Hasse(error)
    }
}

impl fmt::Display for HermiteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Hasse(error) => error.fmt(formatter),
            Self::LengthMismatch { expected, actual } => write!(
                formatter,
                "Hermite interpolation expects {expected} jet values, found {actual}"
            ),
            Self::DuplicatePoint { first, second } => write!(
                formatter,
                "points {first} and {second} repeat with positive multiplicity"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HermiteError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    fn zero_parameter() -> ConfigError {
        ConfigError::ZeroParameter {
            parameter: "evaluation-domain length",
        }
    }

    fn capacity_exceeded() -> ConfigError {
        ConfigError::FieldCapacityExceeded {
            points: 300,
            field_order: 256,
        }
    }

    fn geometry_overflow() -> ConfigError {
        ConfigError::GeometryOverflow {
            context: "multivariate sum term count",
        }
    }

    fn buffer_length() -> ConfigError {
        ConfigError::BufferLength {
            context: "bivariate packed row",
            expected: 8,
            actual: 7,
        }
    }

    fn scratch_too_small() -> ConfigError {
        ConfigError::ScratchTooSmall {
            context: "prepared lane capacity",
            required: 4,
            available: 1,
        }
    }

    fn allocation_failed() -> ConfigError {
        ConfigError::AllocationFailed {
            context: "polynomial coefficients",
            elements: 64,
            element_size: 1,
        }
    }

    #[test]
    fn config_errors_report_their_values_and_limits() {
        // Every struct variant carries the offending value and the limit, and
        // the rendering preserves them: distinct failures stay distinct.
        let rendered = [
            format!("{}", zero_parameter()),
            format!("{}", capacity_exceeded()),
            format!("{}", geometry_overflow()),
            format!("{}", buffer_length()),
            format!("{}", scratch_too_small()),
            format!("{}", allocation_failed()),
        ];
        assert!(rendered[0].contains("evaluation-domain length"));
        assert!(rendered[1].contains("300") && rendered[1].contains("256"));
        assert!(rendered[2].contains("multivariate sum term count"));
        assert!(
            rendered[3].contains("bivariate packed row")
                && rendered[3].contains('8')
                && rendered[3].contains('7')
        );
        assert!(
            rendered[4].contains("prepared lane capacity")
                && rendered[4].contains('4')
                && rendered[4].contains('1')
        );
        assert!(rendered[5].contains("polynomial coefficients") && rendered[5].contains("64"));
        let mut distinct = rendered.to_vec();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), rendered.len());
        assert_eq!(zero_parameter(), zero_parameter());
        assert_ne!(zero_parameter(), capacity_exceeded());
    }

    #[test]
    fn polynomial_errors_preserve_their_cause() {
        let from_config = PolynomialError::from(capacity_exceeded());
        assert_eq!(from_config, PolynomialError::Config(capacity_exceeded()));
        assert!(format!("{from_config}").contains("300"));

        assert_eq!(
            format!("{}", PolynomialError::DivisionByZero),
            format!("{}", PolynomialError::DivisionByZero)
        );
        assert_ne!(
            PolynomialError::DivisionByZero,
            PolynomialError::NonExactDivision
        );
        let zero_constant = PolynomialError::ZeroConstantTerm {
            context: "series inverse",
        };
        assert_eq!(zero_constant, zero_constant);
        assert!(format!("{zero_constant}").contains("series inverse"));
        let rendered = [
            format!("{from_config}"),
            format!("{}", PolynomialError::DivisionByZero),
            format!("{}", PolynomialError::NonExactDivision),
            format!("{zero_constant}"),
        ];
        let mut distinct = rendered.clone().to_vec();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), rendered.len());
    }

    #[test]
    fn product_errors_preserve_their_cause() {
        // A config cause stays a config cause; any other polynomial failure
        // is preserved as a polynomial failure.
        assert_eq!(
            ProductError::from(capacity_exceeded()),
            ProductError::Config(capacity_exceeded())
        );
        assert_eq!(
            ProductError::from(PolynomialError::Config(capacity_exceeded())),
            ProductError::Config(capacity_exceeded())
        );
        assert_eq!(
            ProductError::from(PolynomialError::DivisionByZero),
            ProductError::Polynomial(PolynomialError::DivisionByZero)
        );
        let rendered = [
            format!("{}", ProductError::Config(capacity_exceeded())),
            format!(
                "{}",
                ProductError::Polynomial(PolynomialError::NonExactDivision)
            ),
        ];
        assert!(rendered[0].contains("300"));
        assert_ne!(rendered[0], rendered[1]);
    }

    #[cfg(feature = "fft")]
    #[test]
    fn product_errors_preserve_transform_failures() {
        use butterfly_fft::ntt::NttPlan;
        use butterfly_fft::transform::TransformPlan;
        use fgf::{Gf8B, Goldilocks};

        // An oversized additive plan names the log size and the cap.
        let plan_error = TransformPlan::<Gf8B>::new(512).unwrap_err();
        assert_eq!(
            plan_error,
            PlanError::DomainTooLarge {
                log_size: 9,
                cap: 8
            }
        );
        assert_eq!(
            ProductError::from(plan_error),
            ProductError::Plan(plan_error)
        );
        let rendered_plan = format!("{}", ProductError::Plan(plan_error));
        assert!(rendered_plan.contains('9') && rendered_plan.contains('8'));

        // A short execution buffer names both lengths.
        let plan = TransformPlan::<Gf8B>::new(8).expect("plan");
        let mut short = vec![<Gf8B as fgf::field::Field>::Elem::ZERO; 3];
        let transform_error = plan.forward(&mut short).unwrap_err();
        assert!(matches!(
            transform_error,
            TransformError::BufferLength { .. }
        ));
        assert_eq!(
            ProductError::from(transform_error),
            ProductError::Transform(transform_error)
        );
        let rendered_transform = format!("{}", ProductError::Transform(transform_error));
        assert!(rendered_transform.contains('8') && rendered_transform.contains('3'));

        // A non-power-of-two NTT size is rejected before any table exists.
        let ntt_error = NttPlan::<Goldilocks>::new(3).unwrap_err();
        assert!(matches!(ntt_error, NttError::InvalidSize { .. }));
        assert_eq!(ProductError::from(ntt_error), ProductError::Ntt(ntt_error));
        assert!(format!("{}", ProductError::Ntt(ntt_error)).contains('3'));
    }

    #[test]
    fn root_errors_preserve_their_cause() {
        assert_eq!(
            RootError::from(PolynomialError::DivisionByZero),
            RootError::Polynomial(PolynomialError::DivisionByZero)
        );
        assert_eq!(
            RootError::from(ProductError::Polynomial(PolynomialError::NonExactDivision)),
            RootError::Product(ProductError::Polynomial(PolynomialError::NonExactDivision))
        );
        assert_eq!(
            RootError::from(capacity_exceeded()),
            RootError::Polynomial(PolynomialError::Config(capacity_exceeded()))
        );

        let unsupported = RootError::UnsupportedField {
            field_order: 2_147_483_647,
            element_bytes: 4,
        };
        assert!(format!("{unsupported}").contains("2147483647"));
        assert!(format!("{unsupported}").contains('4'));
        let not_linearized = RootError::NotLinearized { degree: 3 };
        assert!(format!("{not_linearized}").contains('3'));
        let limited = RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein work items",
            required: 7,
            limit: 2,
        };
        let rendered_limited = format!("{limited}");
        assert!(
            rendered_limited.contains("Roth–Ruckenstein work items")
                && rendered_limited.contains('7')
                && rendered_limited.contains('2')
        );
        assert_ne!(
            RootError::ZeroBivariatePolynomial,
            RootError::NotLinearized { degree: 0 }
        );
        let invariant = RootError::FactorizationInvariant {
            reason: "the trace basis did not separate distinct roots",
        };
        assert!(format!("{invariant}").contains("the trace basis did not separate distinct roots"));
        let rendered = [
            format!("{}", RootError::Polynomial(PolynomialError::DivisionByZero)),
            format!("{unsupported}"),
            format!("{not_linearized}"),
            format!("{limited}"),
            format!("{}", RootError::ZeroBivariatePolynomial),
            format!("{invariant}"),
        ];
        let mut distinct = rendered.clone().to_vec();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), rendered.len());
    }

    #[test]
    fn domain_errors_report_their_values_and_limits() {
        assert_eq!(
            DomainError::from(capacity_exceeded()),
            DomainError::Config(capacity_exceeded())
        );
        let duplicate = DomainError::DuplicatePoint {
            first: 1,
            second: 4,
        };
        assert!(format!("{duplicate}").contains('1') && format!("{duplicate}").contains('4'));
        let not_subspace = DomainError::NotSubspace {
            size: 24,
            limit: 32,
        };
        assert!(
            format!("{not_subspace}").contains("24") && format!("{not_subspace}").contains("32")
        );
        let mismatch = DomainError::LengthMismatch {
            expected: 12,
            found: 3,
        };
        assert!(format!("{mismatch}").contains("12") && format!("{mismatch}").contains('3'));
        assert_ne!(duplicate, not_subspace);
    }

    #[cfg(feature = "fft")]
    #[test]
    fn domain_errors_preserve_plan_failures() {
        use butterfly_fft::transform::TransformPlan;
        use fgf::Gf8B;

        let plan_error = TransformPlan::<Gf8B>::new(512).unwrap_err();
        assert_eq!(
            DomainError::from(plan_error),
            DomainError::TransformPlan(plan_error)
        );
        assert!(format!("{}", DomainError::TransformPlan(plan_error)).contains('9'));
    }

    #[test]
    fn eval_errors_preserve_their_cause() {
        assert_eq!(
            EvalError::from(PolynomialError::DivisionByZero),
            EvalError::Polynomial(PolynomialError::DivisionByZero)
        );
        assert_eq!(
            EvalError::from(DomainError::DuplicatePoint {
                first: 0,
                second: 2
            }),
            EvalError::Domain(DomainError::DuplicatePoint {
                first: 0,
                second: 2
            })
        );
        assert_eq!(
            EvalError::from(scratch_too_small()),
            EvalError::Polynomial(PolynomialError::Config(scratch_too_small()))
        );
        assert_ne!(
            EvalError::Polynomial(PolynomialError::DivisionByZero),
            EvalError::Domain(DomainError::DuplicatePoint {
                first: 0,
                second: 2
            })
        );
        assert!(
            format!(
                "{}",
                EvalError::Domain(DomainError::LengthMismatch {
                    expected: 5,
                    found: 2
                })
            )
            .contains('5')
        );
    }

    #[test]
    fn hermite_errors_preserve_their_cause() {
        use super::hasse::HasseError;

        assert_eq!(
            HermiteError::from(geometry_overflow()),
            HermiteError::Config(geometry_overflow())
        );
        assert_eq!(
            HermiteError::from(PolynomialError::NonExactDivision),
            HermiteError::Polynomial(PolynomialError::NonExactDivision)
        );
        let hasse = HasseError::CoefficientCapacityExceeded {
            maximum: 8,
            actual: 12,
        };
        assert_eq!(HermiteError::from(hasse), HermiteError::Hasse(hasse));
        assert!(format!("{}", HermiteError::Hasse(hasse)).contains("12"));
        let mismatch = HermiteError::LengthMismatch {
            expected: 6,
            actual: 4,
        };
        assert!(format!("{mismatch}").contains('6') && format!("{mismatch}").contains('4'));
        let duplicate = HermiteError::DuplicatePoint {
            first: 0,
            second: 1,
        };
        assert!(format!("{duplicate}").contains('0') && format!("{duplicate}").contains('1'));
        assert_ne!(
            mismatch,
            HermiteError::DuplicatePoint {
                first: 6,
                second: 4
            }
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn errors_expose_no_hidden_source_chain() {
        // Every error is a flat leaf: the payload travels in the variant
        // (asserted above through `Display`), never in a chained source.
        let errors: Vec<Box<dyn std::error::Error>> = vec![
            Box::new(capacity_exceeded()),
            Box::new(PolynomialError::DivisionByZero),
            Box::new(ProductError::Polynomial(PolynomialError::NonExactDivision)),
            Box::new(RootError::ZeroBivariatePolynomial),
            Box::new(RootError::NotLinearized { degree: 3 }),
            Box::new(DomainError::NotSubspace { size: 6, limit: 8 }),
            Box::new(EvalError::Polynomial(PolynomialError::DivisionByZero)),
            Box::new(HermiteError::LengthMismatch {
                expected: 6,
                actual: 4,
            }),
        ];
        for error in &errors {
            assert!(error.source().is_none());
        }
        assert!(format!("{}", errors[0]).contains("300"));
        assert!(format!("{}", errors[5]).contains('6'));
        assert!(format!("{}", errors[7]).contains('6'));
    }
}
