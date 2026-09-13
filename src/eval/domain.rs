//! Validated arbitrary, additive-subspace, and affine-coset domains.
//!
//! The domain type is total in both build configurations: all three domain
//! kinds exist with and without the `fft` feature, with identical point
//! order (transform index order — a frozen wire property). With the feature,
//! subspace and coset domains carry a `butterfly-fft` plan and evaluate
//! through the transform; without it, the same constructors enumerate the
//! identical points and every operation falls back to the arbitrary-point
//! paths over them.

use alloc::vec::Vec;

use fgf::kernel::FieldKernels;

#[cfg(feature = "fft")]
use butterfly_fft::core::kernel::ButterflyKernels;
#[cfg(feature = "fft")]
use butterfly_fft::core::transform::TransformPlan;
#[cfg(feature = "fft")]
use butterfly_fft::shifted::ShiftedPlan;

use crate::error::{ConfigError, DomainError, EvalError};
use crate::eval::multipoint::MultipointScratch;
use crate::poly::Polynomial;

#[cfg(feature = "fft")]
use crate::eval::transform::TransformScratch;

/// Evaluation implementation selected by a validated domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationBackend {
    /// Scalar Horner evaluation at arbitrary points.
    Horner,
    /// Subproduct-tree evaluation at arbitrary points.
    SubproductTree,
    /// butterfly-fft evaluation over an additive subspace.
    ButterflyFftAdditive,
    /// butterfly-fft evaluation over an affine coset.
    ButterflyFftAffineCoset,
}

#[cfg(feature = "fft")]
mod sealed {
    use super::EvaluationDomain;
    use alloc::vec::Vec;

    use butterfly_fft::core::kernel::ButterflyKernels;
    use butterfly_fft::core::transform::TransformPlan;
    use butterfly_fft::shifted::ShiftedPlan;

    #[derive(Clone, Debug)]
    pub(super) enum DomainKind<F: ButterflyKernels> {
        Arbitrary,
        Subspace { plan: TransformPlan<F> },
        Affine { plan: ShiftedPlan<F> },
    }

    impl<F: ButterflyKernels> EvaluationDomain<F> {
        pub(super) fn from_parts(points: Vec<F::Elem>, kind: DomainKind<F>) -> Self {
            Self { points, kind }
        }

        pub(super) fn kind(&self) -> &DomainKind<F> {
            &self.kind
        }
    }
}

#[cfg(not(feature = "fft"))]
mod sealed {
    use super::EvaluationDomain;
    use alloc::vec::Vec;

    use fgf::kernel::FieldKernels;

    #[derive(Clone, Debug)]
    pub(super) enum DomainKind<F: FieldKernels> {
        Arbitrary,
        Subspace {
            #[allow(dead_code)]
            basis: Vec<F::Elem>,
        },
        Affine {
            #[cfg(feature = "fft")]
            shift: F::Elem,
            #[allow(dead_code)]
            basis: Vec<F::Elem>,
        },
    }

    impl<F: FieldKernels> EvaluationDomain<F> {
        pub(super) fn from_parts(points: Vec<F::Elem>, kind: DomainKind<F>) -> Self {
            Self { points, kind }
        }

        pub(super) fn kind(&self) -> &DomainKind<F> {
            &self.kind
        }
    }
}

use sealed::DomainKind;

/// Distinct evaluation points with an optional butterfly-fft execution plan.
///
/// With the `fft` feature the field bound is the transform's
/// `ButterflyKernels`; without it, any `FieldKernels` field works and every
/// operation runs the arbitrary-point paths.
#[cfg(feature = "fft")]
#[derive(Clone, Debug)]
pub struct EvaluationDomain<F: ButterflyKernels> {
    points: Vec<F::Elem>,
    kind: DomainKind<F>,
}

/// Distinct evaluation points over any field, always on the
/// arbitrary-point paths.
#[cfg(not(feature = "fft"))]
#[derive(Clone, Debug)]
pub struct EvaluationDomain<F: FieldKernels> {
    points: Vec<F::Elem>,
    kind: DomainKind<F>,
}

#[cfg(feature = "fft")]
impl<F: ButterflyKernels> EvaluationDomain<F> {
    /// Construct a domain from arbitrary distinct points.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] for an empty support, a support larger than
    /// the field, duplicate points, or a failed reservation.
    pub fn arbitrary(points: Vec<F::Elem>) -> Result<Self, DomainError> {
        validate_points::<F>(&points)?;
        Ok(Self::from_parts(points, DomainKind::Arbitrary))
    }

    /// Evaluation points in transform order.
    #[must_use]
    pub fn points(&self) -> &[F::Elem] {
        &self.points
    }

    /// Number of evaluation points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the domain contains no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Evaluate `polynomial` at every point of the domain.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] when a supporting buffer cannot be reserved.
    pub fn evaluate(
        &self,
        polynomial: &Polynomial<F>,
        scratch: &mut DomainScratch<F>,
    ) -> Result<Vec<F::Elem>, EvalError> {
        let mut values = Vec::new();
        self.evaluate_into(polynomial, scratch, &mut values)?;
        Ok(values)
    }

    /// Interpolate the minimal-degree polynomial through this domain's
    /// points and `values`.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] for a length mismatch and when a supporting
    /// buffer cannot be reserved.
    pub fn interpolate(
        &self,
        values: &[F::Elem],
        scratch: &mut DomainScratch<F>,
    ) -> Result<Polynomial<F>, EvalError> {
        let mut output = Polynomial::zero();
        self.interpolate_into(values, scratch, &mut output)?;
        Ok(output)
    }

    /// Evaluation backend selected by this domain and build configuration.
    #[must_use]
    pub fn backend(&self) -> EvaluationBackend {
        match self.kind() {
            DomainKind::Arbitrary => arbitrary_backend(self.points.len()),
            DomainKind::Subspace { .. } => EvaluationBackend::ButterflyFftAdditive,
            DomainKind::Affine { .. } => EvaluationBackend::ButterflyFftAffineCoset,
        }
    }

    fn interpolate_arbitrary(
        &self,
        values: &[F::Elem],
        output: &mut Polynomial<F>,
    ) -> Result<(), EvalError> {
        if values.len() != self.len() {
            return Err(EvalError::Domain(DomainError::LengthMismatch {
                expected: self.len(),
                found: values.len(),
            }));
        }
        if self.len() <= crate::eval::newton::MODULE_INTERPOLATION_CROSSOVER {
            crate::eval::newton::interpolate_newton_into(&self.points, values, output)
        } else {
            let interpolant = crate::eval::multipoint::interpolate_lagrange(&self.points, values)?;
            output.assign_from(&interpolant);
            Ok(())
        }
    }
}

#[cfg(not(feature = "fft"))]
impl<F: FieldKernels> EvaluationDomain<F> {
    /// Construct a domain from arbitrary distinct points.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] for an empty support, a support larger than
    /// the field, duplicate points, or a failed reservation.
    pub fn arbitrary(points: Vec<F::Elem>) -> Result<Self, DomainError> {
        validate_points::<F>(&points)?;
        Ok(Self::from_parts(points, DomainKind::Arbitrary))
    }

    /// Evaluation points in transform order.
    #[must_use]
    pub fn points(&self) -> &[F::Elem] {
        &self.points
    }

    /// Number of evaluation points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the domain contains no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Evaluate `polynomial` at every point of the domain.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] when a supporting buffer cannot be reserved.
    pub fn evaluate(
        &self,
        polynomial: &Polynomial<F>,
        scratch: &mut DomainScratch<F>,
    ) -> Result<Vec<F::Elem>, EvalError> {
        let mut values = Vec::new();
        self.evaluate_into(polynomial, scratch, &mut values)?;
        Ok(values)
    }

    /// Interpolate the minimal-degree polynomial through this domain's
    /// points and `values`.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] for a length mismatch and when a supporting
    /// buffer cannot be reserved.
    pub fn interpolate(
        &self,
        values: &[F::Elem],
        scratch: &mut DomainScratch<F>,
    ) -> Result<Polynomial<F>, EvalError> {
        let mut output = Polynomial::zero();
        self.interpolate_into(values, scratch, &mut output)?;
        Ok(output)
    }

    /// Evaluation backend selected by this domain and build configuration.
    #[must_use]
    pub fn backend(&self) -> EvaluationBackend {
        match self.kind() {
            DomainKind::Arbitrary | DomainKind::Subspace { .. } | DomainKind::Affine { .. } => {
                arbitrary_backend(self.points.len())
            }
        }
    }

    fn interpolate_arbitrary(
        &self,
        values: &[F::Elem],
        output: &mut Polynomial<F>,
    ) -> Result<(), EvalError> {
        if values.len() != self.len() {
            return Err(EvalError::Domain(DomainError::LengthMismatch {
                expected: self.len(),
                found: values.len(),
            }));
        }
        if self.len() <= crate::eval::newton::MODULE_INTERPOLATION_CROSSOVER {
            crate::eval::newton::interpolate_newton_into(&self.points, values, output)
        } else {
            let interpolant = crate::eval::multipoint::interpolate_lagrange(&self.points, values)?;
            output.assign_from(&interpolant);
            Ok(())
        }
    }
}

#[cfg(feature = "fft")]
impl<F: ButterflyKernels> EvaluationDomain<F> {
    /// Construct the default bit-basis additive subspace of `size` points.
    ///
    /// Under the default basis, point `i` is the field element whose stable
    /// representation holds the value `i`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] when `size` is not a power of two, exceeds
    /// the field, or the plan cannot be built.
    pub fn additive_subspace(size: usize) -> Result<Self, DomainError> {
        let log_size = validate_subspace_size::<F>(size)?;
        let basis = default_bit_basis::<F>(log_size);
        Self::additive_subspace_with_basis(size, &basis)
    }

    /// Construct an additive subspace from an explicit ordered basis prefix.
    ///
    /// Point `i` of the domain is the XOR of `basis[j]` over the set bits
    /// `j` of `i` — the order the `butterfly-fft` plan produces.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] when `size` is not a power of two, exceeds
    /// the field, the basis prefix is too short or dependent, or the plan
    /// cannot be built.
    pub fn additive_subspace_with_basis(
        size: usize,
        basis: &[F::Elem],
    ) -> Result<Self, DomainError> {
        validate_basis::<F>(size, basis)?;
        let plan = TransformPlan::<F>::with_basis(size, basis)?;
        let points = enumerate_subspace::<F>(basis, size)?;
        Ok(Self::from_parts(points, DomainKind::Subspace { plan }))
    }

    /// Construct a default bit-basis affine coset `shift + V`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] when `size` is not a power of two, exceeds
    /// the field, or the plan cannot be built.
    pub fn affine_coset(size: usize, shift: F::Elem) -> Result<Self, DomainError> {
        let log_size = validate_subspace_size::<F>(size)?;
        let basis = default_bit_basis::<F>(log_size);
        Self::affine_coset_with_basis(size, &basis, shift)
    }

    /// Construct an affine coset `shift + span(basis[..log_size(size)])`
    /// from an explicit ordered basis prefix.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] when `size` is not a power of two, exceeds
    /// the field, the basis prefix is too short or dependent, or the plan
    /// cannot be built.
    pub fn affine_coset_with_basis(
        size: usize,
        basis: &[F::Elem],
        shift: F::Elem,
    ) -> Result<Self, DomainError> {
        validate_basis::<F>(size, basis)?;
        let plan = ShiftedPlan::<F>::from_elements(size, basis, shift)?;
        let mut points = enumerate_subspace::<F>(basis, size)?;
        for point in &mut points {
            *point = fgf::field::Elem::add(shift, *point);
        }
        Ok(Self::from_parts(points, DomainKind::Affine { plan }))
    }

    /// The butterfly-fft plan for subspace and coset domains.
    #[must_use]
    pub fn transform_plan(&self) -> Option<&TransformPlan<F>> {
        match self.kind() {
            DomainKind::Arbitrary => None,
            DomainKind::Subspace { plan } => Some(plan),
            DomainKind::Affine { plan } => Some(plan.plan()),
        }
    }

    /// Write the evaluations at every point into `values`.
    ///
    /// Dispatches on the domain kind: transform paths for subspace and
    /// coset domains, Horner or the subproduct tree for arbitrary ones.
    /// Every backend produces the values in point order.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] when a supporting buffer cannot be reserved.
    pub fn evaluate_into(
        &self,
        polynomial: &Polynomial<F>,
        scratch: &mut DomainScratch<F>,
        values: &mut Vec<F::Elem>,
    ) -> Result<(), EvalError> {
        match self.kind() {
            DomainKind::Arbitrary => {
                evaluate_arbitrary_into(polynomial, &self.points, &mut scratch.multipoint, values)
            }
            DomainKind::Subspace { plan } => crate::eval::transform::evaluate_subspace_into(
                polynomial,
                plan,
                &mut scratch.transform,
                values,
            )
            .map_err(EvalError::Polynomial),
            DomainKind::Affine { plan } => crate::eval::transform::evaluate_coset_into(
                polynomial,
                plan,
                &mut scratch.transform,
                values,
            )
            .map_err(EvalError::Polynomial),
        }
    }

    /// Write the interpolant through this domain's points and `values` into
    /// `output`.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] for a length mismatch and when a supporting
    /// buffer cannot be reserved.
    pub fn interpolate_into(
        &self,
        values: &[F::Elem],
        scratch: &mut DomainScratch<F>,
        output: &mut Polynomial<F>,
    ) -> Result<(), EvalError> {
        match self.kind() {
            DomainKind::Arbitrary => self.interpolate_arbitrary(values, output),
            DomainKind::Subspace { plan } => crate::eval::transform::interpolate_subspace_into(
                plan,
                values,
                &mut scratch.transform,
                output,
            )
            .map_err(EvalError::Polynomial),
            DomainKind::Affine { plan } => crate::eval::transform::interpolate_subspace_into(
                plan.plan(),
                values,
                &mut scratch.transform,
                output,
            )
            .map_err(EvalError::Polynomial),
        }
    }
}

#[cfg(not(feature = "fft"))]
impl<F: FieldKernels> EvaluationDomain<F> {
    /// Construct the default bit-basis additive subspace of `size` points.
    ///
    /// Under the default basis, point `i` is the field element whose stable
    /// representation holds the value `i`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] when `size` is not a power of two, exceeds
    /// the field, or the basis is invalid.
    pub fn additive_subspace(size: usize) -> Result<Self, DomainError> {
        let log_size = validate_subspace_size::<F>(size)?;
        let basis = default_bit_basis::<F>(log_size);
        Self::additive_subspace_with_basis(size, &basis)
    }

    /// Construct an additive subspace from an explicit ordered basis prefix.
    ///
    /// Point `i` of the domain is the XOR of `basis[j]` over the set bits
    /// `j` of `i`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] when `size` is not a power of two, exceeds
    /// the field, or the basis prefix is too short or dependent.
    pub fn additive_subspace_with_basis(
        size: usize,
        basis: &[F::Elem],
    ) -> Result<Self, DomainError> {
        validate_basis::<F>(size, basis)?;
        let points = enumerate_subspace::<F>(basis, size)?;
        Ok(Self::from_parts(
            points,
            DomainKind::Subspace {
                basis: basis.to_vec(),
            },
        ))
    }

    /// Construct a default bit-basis affine coset `shift + V`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] when `size` is not a power of two, exceeds
    /// the field, or the basis is invalid.
    pub fn affine_coset(size: usize, shift: F::Elem) -> Result<Self, DomainError> {
        let log_size = validate_subspace_size::<F>(size)?;
        let basis = default_bit_basis::<F>(log_size);
        Self::affine_coset_with_basis(size, &basis, shift)
    }

    /// Construct an affine coset `shift + span(basis[..log2(size)])` from an
    /// explicit ordered basis prefix.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] when `size` is not a power of two, exceeds
    /// the field, or the basis prefix is too short or dependent.
    pub fn affine_coset_with_basis(
        size: usize,
        basis: &[F::Elem],
        shift: F::Elem,
    ) -> Result<Self, DomainError> {
        validate_basis::<F>(size, basis)?;
        let mut points = enumerate_subspace::<F>(basis, size)?;
        for point in &mut points {
            *point = fgf::field::Elem::add(shift, *point);
        }
        Ok(Self::from_parts(
            points,
            DomainKind::Affine {
                basis: basis.to_vec(),
            },
        ))
    }

    /// Write the evaluations at every point into `values`, on the
    /// arbitrary-point paths.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] when a supporting buffer cannot be reserved.
    pub fn evaluate_into(
        &self,
        polynomial: &Polynomial<F>,
        scratch: &mut DomainScratch<F>,
        values: &mut Vec<F::Elem>,
    ) -> Result<(), EvalError> {
        evaluate_arbitrary_into(polynomial, &self.points, &mut scratch.multipoint, values)
    }

    /// Write the interpolant through this domain's points and `values` into
    /// `output`, on the arbitrary-point paths.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] for a length mismatch and when a supporting
    /// buffer cannot be reserved.
    pub fn interpolate_into(
        &self,
        values: &[F::Elem],
        scratch: &mut DomainScratch<F>,
        output: &mut Polynomial<F>,
    ) -> Result<(), EvalError> {
        let _ = scratch;
        self.interpolate_arbitrary(values, output)
    }
}

/// Caller-owned reusable storage for domain evaluation and interpolation.
#[derive(Debug)]
pub struct DomainScratch<F: FieldKernels> {
    multipoint: MultipointScratch<F>,
    #[cfg(feature = "fft")]
    transform: TransformScratch,
}

impl<F: FieldKernels> DomainScratch<F> {
    /// Construct empty reusable domain scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            multipoint: MultipointScratch::new(),
            #[cfg(feature = "fft")]
            transform: TransformScratch::new(),
        }
    }
}

impl<F: FieldKernels> Default for DomainScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

fn arbitrary_backend(points: usize) -> EvaluationBackend {
    if points >= crate::eval::multipoint::MULTIPOINT_EVAL_CROSSOVER {
        EvaluationBackend::SubproductTree
    } else {
        EvaluationBackend::Horner
    }
}

fn evaluate_arbitrary_into<F: FieldKernels>(
    polynomial: &Polynomial<F>,
    points: &[F::Elem],
    scratch: &mut MultipointScratch<F>,
    values: &mut Vec<F::Elem>,
) -> Result<(), EvalError> {
    if points.len() < crate::eval::multipoint::MULTIPOINT_EVAL_CROSSOVER {
        values.clear();
        values.extend(
            points
                .iter()
                .copied()
                .map(|point| polynomial.evaluate(point)),
        );
        Ok(())
    } else {
        crate::eval::multipoint::evaluate_multipoint_into(polynomial, points, scratch, values)
            .map_err(EvalError::Polynomial)
    }
}

/// Validate an arbitrary support: nonempty, within field capacity,
/// pairwise distinct.
fn validate_points<F: FieldKernels>(points: &[F::Elem]) -> Result<(), DomainError> {
    if points.is_empty() {
        return Err(DomainError::Config(ConfigError::ZeroParameter {
            parameter: "evaluation-domain length",
        }));
    }
    if points.len() as u128 > F::ORDER {
        return Err(DomainError::Config(ConfigError::FieldCapacityExceeded {
            points: points.len(),
            field_order: F::ORDER,
        }));
    }
    if let Some((first, second)) = crate::eval::find_duplicate::<F>(points) {
        return Err(DomainError::DuplicatePoint { first, second });
    }
    Ok(())
}

/// Validate a subspace size: a power of two within the field capacity.
fn validate_subspace_size<F: FieldKernels>(size: usize) -> Result<u32, DomainError> {
    if size == 0 || !size.is_power_of_two() {
        return Err(DomainError::NotSubspace {
            size,
            limit: F::ORDER.min(usize::MAX as u128) as usize,
        });
    }
    if size as u128 > F::ORDER {
        return Err(DomainError::Config(ConfigError::FieldCapacityExceeded {
            points: size,
            field_order: F::ORDER,
        }));
    }
    Ok(size.trailing_zeros())
}

/// Validate an explicit basis prefix for a subspace of `size` points:
/// long enough and GF(2)-independent.
fn validate_basis<F: FieldKernels>(size: usize, basis: &[F::Elem]) -> Result<(), DomainError> {
    let log_size = size.trailing_zeros() as usize;
    debug_assert!(size.is_power_of_two());
    if basis.len() < log_size {
        return Err(DomainError::NotSubspace {
            size,
            limit: 1 << log_size,
        });
    }
    // GF(2) independence over the stable little-endian keys: reduce each
    // basis element against the pivot rows built so far; an element that
    // reduces to zero without creating a pivot is dependent.
    let mut pivots: Vec<u128> = Vec::with_capacity(log_size);
    for &element in &basis[..log_size] {
        let mut key = crate::roots::element_key::<F>(element);
        let mut independent = false;
        while let Some(bit) = key.checked_ilog2() {
            if let Some(index) = pivots
                .iter()
                .position(|pivot| pivot.checked_ilog2() == Some(bit))
            {
                key ^= pivots[index];
            } else {
                pivots.push(key);
                independent = true;
                break;
            }
        }
        if !independent {
            return Err(DomainError::NotSubspace {
                size,
                limit: 1 << log_size,
            });
        }
    }
    Ok(())
}

/// The default bit basis `β_j = element with stable value 1 << j`.
fn default_bit_basis<F: FieldKernels>(log_size: u32) -> Vec<F::Elem> {
    (0..log_size)
        .map(|bit| {
            let bytes = (1_u128 << bit).to_le_bytes();
            F::read(&bytes[..F::BYTES])
        })
        .collect()
}

/// The subspace points for `basis` in transform index order.
fn enumerate_subspace<F: FieldKernels>(
    basis: &[F::Elem],
    size: usize,
) -> Result<Vec<F::Elem>, DomainError> {
    use fgf::field::Elem as _;
    let mut points = Vec::new();
    points.try_reserve_exact(size).map_err(|_| {
        DomainError::Config(ConfigError::AllocationFailed {
            context: "subspace points",
            elements: size,
            element_size: core::mem::size_of::<F::Elem>(),
        })
    })?;
    for index in 0..size {
        let mut element = F::Elem::ZERO;
        let mut remaining = index;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            element = element.add(basis[bit]);
            remaining &= remaining - 1;
        }
        points.push(element);
    }
    Ok(points)
}
