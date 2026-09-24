//! Lane-parallel Horner multipoint evaluation over the packed field kernels.
//!
//! One Horner step applied across every point at once turns multipoint
//! evaluation into a sequence of elementwise multiplies and broadcast adds
//! over [`fgf::ops`], so the multiply runs in the vectorized prime-field
//! kernels instead of the scalar per-point loop. The step count decides
//! against the subproduct tree; see `BENCHMARKS.md`.
//!
//! # Layout
//!
//! The points are canonicalized once per call into a packed buffer owned by
//! [`LaneScratch`]; raw prime lanes never reach a packed kernel. One
//! accumulator buffer holds the running values: each Horner step multiplies
//! it by the points in place with [`ops::mul_elementwise_assign`] and folds
//! in the next coefficient with [`ops::add_assign_scalar`], so no ping-pong
//! or broadcast buffer is needed.

use alloc::vec::Vec;

use core::marker::PhantomData;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::ops;

use crate::error::{ConfigError, PolynomialError};

/// Caller-owned reusable storage for the lane-parallel Horner route.
///
/// Holds the accumulator and the canonicalized packed points. A warmed call
/// over the same point count reserves nothing.
#[derive(Debug)]
pub(crate) struct LaneScratch<F: FieldKernels> {
    accumulator: Vec<u8>,
    points: Vec<u8>,
    field: PhantomData<F>,
}

impl<F: FieldKernels> LaneScratch<F> {
    /// Construct empty reusable lane scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            accumulator: Vec::new(),
            points: Vec::new(),
            field: PhantomData,
        }
    }

    /// Reserve every buffer for `count` points.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] through [`PolynomialError`] when the byte
    /// geometry overflows or a reservation fails.
    pub(crate) fn ensure(&mut self, count: usize) -> Result<(), PolynomialError> {
        let bytes = lane_bytes::<F>(count)?;
        ensure_buffer(&mut self.accumulator, bytes)?;
        ensure_buffer(&mut self.points, bytes)?;
        Ok(())
    }

    /// Canonicalize the points and run the Horner steps, leaving the packed
    /// values in the accumulator [`LaneScratch::values`] reports.
    fn horner(
        &mut self,
        coefficients: &[u8],
        coefficient_count: usize,
        points: &[F::Elem],
    ) -> Result<(), PolynomialError> {
        if points.is_empty() {
            return Ok(());
        }
        let coefficient_bytes =
            coefficient_count
                .checked_mul(F::BYTES)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "lane evaluation coefficients",
                })?;
        if coefficients.len() != coefficient_bytes {
            return Err(PolynomialError::Config(ConfigError::BufferLength {
                context: "lane evaluation coefficients",
                expected: coefficient_bytes,
                actual: coefficients.len(),
            }));
        }
        let lane_bytes = lane_bytes::<F>(points.len())?;
        self.ensure(points.len())?;
        for (slot, &point) in self.points[..lane_bytes]
            .chunks_exact_mut(F::BYTES)
            .zip(points)
        {
            F::encode(slot, point.add(F::Elem::ZERO));
        }

        // The accumulator starts at zero, so the first step deposits the
        // leading coefficient; each later step multiplies by every point and
        // adds the next-lower coefficient.
        self.accumulator[..lane_bytes].fill(0);
        let accumulator = &mut self.accumulator[..lane_bytes];
        let packed_points = &self.points[..lane_bytes];
        for index in (0..coefficient_count).rev() {
            let value = F::decode(&coefficients[index * F::BYTES..][..F::BYTES]).add(F::Elem::ZERO);
            ops::mul_elementwise_assign::<F>(accumulator, packed_points);
            ops::add_assign_scalar::<F>(accumulator, value);
        }
        Ok(())
    }

    /// The packed accumulator holding the values after the Horner steps.
    fn values(&self, lane_bytes: usize) -> &[u8] {
        &self.accumulator[..lane_bytes]
    }
}

/// Evaluate the packed coefficients at every point into `values` as field
/// elements, reusing `scratch`.
///
/// `values` receives one value per point, in point order; an empty
/// coefficient list yields zeros and a single coefficient yields that
/// coefficient at every point. The coefficient bytes are low degree first
/// and canonicalized per step, so raw prime lanes are safe.
///
/// # Errors
///
/// Returns [`ConfigError`] through [`PolynomialError`] when a buffer length
/// disagrees with the geometry or a reservation fails.
pub(crate) fn evaluate_lane_into<F: FieldKernels>(
    values: &mut [F::Elem],
    coefficients: &[u8],
    coefficient_count: usize,
    points: &[F::Elem],
    scratch: &mut LaneScratch<F>,
) -> Result<(), PolynomialError> {
    if values.len() != points.len() {
        return Err(PolynomialError::Config(ConfigError::BufferLength {
            context: "lane evaluation values",
            expected: points.len(),
            actual: values.len(),
        }));
    }
    scratch.horner(coefficients, coefficient_count, points)?;
    let lane_bytes = lane_bytes::<F>(points.len())?;
    let accumulator = scratch.values(lane_bytes);
    for (destination, source) in values.iter_mut().zip(accumulator.chunks_exact(F::BYTES)) {
        *destination = F::decode(source);
    }
    Ok(())
}

/// Evaluate the packed coefficients at every point into `values` as packed
/// canonical lanes, reusing `scratch`.
///
/// Same contract as [`evaluate_lane_into`] with the values left packed: one
/// canonical lane per point, in point order.
///
/// # Errors
///
/// Returns [`ConfigError`] through [`PolynomialError`] when a buffer length
/// disagrees with the geometry or a reservation fails.
pub(crate) fn evaluate_lane_packed_into<F: FieldKernels>(
    values: &mut [u8],
    coefficients: &[u8],
    coefficient_count: usize,
    points: &[F::Elem],
    scratch: &mut LaneScratch<F>,
) -> Result<(), PolynomialError> {
    let lane_bytes = lane_bytes::<F>(points.len())?;
    if values.len() != lane_bytes {
        return Err(PolynomialError::Config(ConfigError::BufferLength {
            context: "lane evaluation values",
            expected: lane_bytes,
            actual: values.len(),
        }));
    }
    scratch.horner(coefficients, coefficient_count, points)?;
    let accumulator = scratch.values(lane_bytes);
    values.copy_from_slice(accumulator);
    Ok(())
}

/// The packed byte length of `count` field lanes.
fn lane_bytes<F: FieldKernels>(count: usize) -> Result<usize, PolynomialError> {
    count
        .checked_mul(F::BYTES)
        .ok_or(ConfigError::GeometryOverflow {
            context: "lane evaluation points",
        })
        .map_err(PolynomialError::from)
}

/// Reserve a byte buffer up to `required` without shrinking it.
fn ensure_buffer(buffer: &mut Vec<u8>, required: usize) -> Result<(), PolynomialError> {
    if required > buffer.len() {
        buffer
            .try_reserve_exact(required - buffer.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "lane evaluation buffers",
                elements: required,
                element_size: 1,
            })?;
        buffer.resize(required, 0);
    }
    Ok(())
}
