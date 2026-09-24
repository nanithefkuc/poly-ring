//! Multiplicity-weighted multipoint evaluation.
//!
//! The composition: reduce the input modulo each point's weighted modulus
//! `(X − a_i)^{s_i}` through the ring's remainder tree — one descent for
//! the whole request — then translate each leaf remainder at its own point
//! with a prepared jet. The bridge `f mod (X − a)^s` followed by
//! `X = a + T` yields exactly `f(a + T) mod T^s`, so leaf `i`'s translation
//! is the vector of Hasse derivatives `D^[j]f(a_i)` for `j < s_i`, written
//! straight into the caller's output range. A remainder modulo `X − a`
//! alone would carry only the value, which is why no order above zero is
//! ever produced by a simple-point evaluator here.

use alloc::vec::Vec;

use super::lane::{LaneScratch, evaluate_lane_packed_into};
use super::multipoint::MULTIPOINT_LANE_STEP_CROSSOVER;
use crate::error::HasseError;
use crate::eval::{RemainderScratch, RemainderTree};
use crate::jet::{JetPlan, JetScratch};
use crate::poly::{Polynomial, PolynomialField};
use fgf::field::Elem;

/// A prepared weighted multipoint evaluation request.
///
/// Construction is the expensive phase: the weighted moduli are expanded by
/// exponentiation, the remainder tree and its reciprocal plans are built,
/// and every jet's translation powers are precomputed. Execution is a
/// remainder descent plus one prepared jet per positive-weight point, and a
/// warmed run with prepared scratch allocates nothing.
///
/// Zero multiplicities contribute empty ranges and build no jet. Repeated
/// points stay separate in caller order. There is no `W ≤ field order`
/// restriction: multiplicity is not a count of distinct field elements.
pub struct MultiplicityPlan<F: PolynomialField> {
    points: Vec<F::Elem>,
    multiplicities: Vec<usize>,
    offsets: Vec<usize>,
    max_coefficients: usize,
    total_weight: usize,
    tree: RemainderTree<F>,
    /// One jet per positive-weight point, parallel to `points`.
    jets: Vec<Option<JetPlan<F>>>,
    /// Index into the distinct-multiplicity scratch slots, per point.
    jet_slot: Vec<usize>,
    /// The distinct multiplicities, in first-appearance order.
    distinct_multiplicities: Vec<usize>,
    /// Whether every multiplicity is exactly one — the lane route's shape.
    all_multiplicities_one: bool,
}

/// Reusable evaluation workspace.
///
/// Holds the remainder-tree scratch, the leaf-remainder staging buffer, and
/// one jet scratch per distinct multiplicity (jet scratch is layout-only —
/// never point-dependent — so all jets sharing a multiplicity share one).
pub struct MultiplicityScratch<F: PolynomialField> {
    max_batch: usize,
    max_coefficients: usize,
    total_weight: usize,
    /// The distinct multiplicities, first-appearance order — the jet-slot
    /// layout this scratch serves. Equal weight, capacity, and batch do
    /// not make a scratch reusable across plans with different slot maps.
    distinct_multiplicities: Vec<usize>,
    remainder_scratch: RemainderScratch<F>,
    /// `W` rows of `max_batch` lanes: the tree output, staged leaf by leaf.
    remainders: Vec<u8>,
    /// One per distinct multiplicity.
    jet_scratches: Vec<JetScratch<F>>,
    /// Scalar staging: input coefficients and output values at batch one.
    scalar_input: Vec<u8>,
    scalar_output: Vec<u8>,
    /// Batch input staging: the canonicalized copy odd-characteristic runs
    /// evaluate through, so raw prime lanes never reach a packed kernel.
    batch_input: Vec<u8>,
    /// The lane-parallel Horner buffers for the single-weight route.
    lane: LaneScratch<F>,
}

impl<F: PolynomialField> MultiplicityPlan<F> {
    /// Prepare the request `points[i]` with weight `multiplicities[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`HasseError::LengthMismatch`] when the slices disagree in
    /// length, and the geometry, allocation, and product errors of the
    /// underlying constructions.
    pub fn new(
        points: &[F::Elem],
        multiplicities: &[usize],
        max_coefficients: usize,
    ) -> Result<Self, HasseError> {
        if points.len() != multiplicities.len() {
            return Err(HasseError::LengthMismatch {
                argument: "multiplicities",
                expected: points.len(),
                actual: multiplicities.len(),
            });
        }
        let all_multiplicities_one = multiplicities.iter().all(|&weight| weight == 1);
        let mut offsets = reserved(points.len() + 1, "multiplicity offsets")?;
        let mut total_weight = 0_usize;
        offsets.push(0);
        for &weight in multiplicities {
            total_weight =
                total_weight
                    .checked_add(weight)
                    .ok_or(HasseError::GeometryOverflow {
                        context: "total multiplicity weight",
                    })?;
            offsets.push(total_weight);
        }

        // Canonicalize the points once: a noncanonical prime lane must never
        // reach a packed kernel or a stored modulus.
        let mut canonical = reserved(points.len(), "multiplicity points")?;
        for &point in points {
            canonical.push(point.add(F::Elem::ZERO));
        }

        // Weighted moduli (X − a)^s, by exponentiation through univariate
        // products; weight zero is the constant one and gets an empty range.
        let mut moduli = reserved(points.len(), "weighted moduli")?;
        for (point, &weight) in canonical.iter().zip(multiplicities) {
            moduli.push(Self::weighted_modulus(*point, weight)?);
        }
        let tree = RemainderTree::new(&moduli, max_coefficients).map_err(HasseError::from)?;

        // One jet per positive weight; scratch slots keyed by distinct
        // multiplicity, first appearance order.
        let mut jets = reserved(points.len(), "multiplicity jets")?;
        let mut jet_slot = reserved(points.len(), "multiplicity jet slots")?;
        let mut distinct_multiplicities = Vec::new();
        for (&point, &weight) in canonical.iter().zip(multiplicities) {
            if weight == 0 {
                jets.push(None);
                jet_slot.push(usize::MAX);
                continue;
            }
            let slot =
                if let Some(index) = distinct_multiplicities.iter().position(|&s| s == weight) {
                    index
                } else {
                    distinct_multiplicities.push(weight);
                    distinct_multiplicities.len() - 1
                };
            // The leaf remainder has exactly `weight` coefficients, so the
            // jet never sees more than that.
            jets.push(Some(JetPlan::new(point, weight, weight)?));
            jet_slot.push(slot);
        }
        let mut weights = reserved(multiplicities.len(), "multiplicity weights")?;
        weights.extend_from_slice(multiplicities);
        Ok(Self {
            points: canonical,
            multiplicities: weights,
            offsets,
            max_coefficients,
            total_weight,
            tree,
            jets,
            jet_slot,
            distinct_multiplicities,
            all_multiplicities_one,
        })
    }

    /// Prepare a uniform request: every point at the same multiplicity.
    ///
    /// # Errors
    ///
    /// As [`MultiplicityPlan::new`].
    pub fn uniform(
        points: &[F::Elem],
        multiplicity: usize,
        max_coefficients: usize,
    ) -> Result<Self, HasseError> {
        let mut weights = reserved(points.len(), "multiplicity weights")?;
        weights.resize(points.len(), multiplicity);
        Self::new(points, &weights, max_coefficients)
    }

    /// The prepared points, in caller order.
    #[must_use]
    pub fn points(&self) -> &[F::Elem] {
        &self.points
    }

    /// The prepared multiplicities, in caller order.
    #[must_use]
    pub fn multiplicities(&self) -> &[usize] {
        &self.multiplicities
    }

    /// Output offsets: `offsets[0] = 0`, `offsets[i+1] = offsets[i] +
    /// multiplicities[i]`. Output position `offsets[i] + j` is
    /// `D^[j]f(a_i)`.
    #[must_use]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// The number of jets: points with a positive multiplicity.
    #[must_use]
    pub fn jet_count(&self) -> usize {
        self.jets.iter().flatten().count()
    }
    /// The total output weight `W = Σ multiplicities[i]`.
    #[must_use]
    pub fn total_weight(&self) -> usize {
        self.total_weight
    }

    /// Build scratch for batches of up to `batch_capacity` lanes.
    ///
    /// `scratch(0)` is a valid empty-batch capacity; scalar execution
    /// requires at least one lane.
    ///
    /// # Errors
    ///
    /// Returns [`HasseError::AllocationFailed`] or
    /// [`HasseError::GeometryOverflow`] when a workspace cannot be reserved.
    ///
    /// # Panics
    ///
    /// Panics if a jet scratch cannot be built for a multiplicity this plan
    /// itself just validated — an internal invariant, never a caller input.
    pub fn scratch(&self, batch_capacity: usize) -> Result<MultiplicityScratch<F>, HasseError> {
        let lanes = batch_capacity.max(1);
        let reserve = |bytes: usize, context: &'static str| -> Result<Vec<u8>, HasseError> {
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(bytes)
                .map_err(|_| HasseError::AllocationFailed { context })?;
            buffer.resize(bytes, 0);
            Ok(buffer)
        };
        let remainder_scratch = self
            .tree
            .scratch(batch_capacity)
            .map_err(HasseError::from)?;
        let mut jet_scratches = Vec::with_capacity(self.distinct_multiplicities.len());
        for &weight in &self.distinct_multiplicities {
            let jet = self
                .jets
                .iter()
                .flatten()
                .find(|jet| jet.multiplicity() == weight)
                .expect("every distinct multiplicity has a jet");
            jet_scratches.push(jet.scratch(batch_capacity)?);
        }
        let mut lane = LaneScratch::new();
        lane.ensure(self.points.len()).map_err(HasseError::from)?;
        Ok(MultiplicityScratch {
            max_batch: batch_capacity,
            max_coefficients: self.max_coefficients,
            total_weight: self.total_weight,
            distinct_multiplicities: self.distinct_multiplicities.clone(),
            remainder_scratch,
            remainders: reserve(
                self.total_weight
                    .checked_mul(lanes)
                    .and_then(|rows| rows.checked_mul(F::BYTES))
                    .ok_or(HasseError::GeometryOverflow {
                        context: "leaf remainder staging",
                    })?,
                "leaf remainder staging",
            )?,
            jet_scratches,
            scalar_input: reserve(
                self.max_coefficients.checked_mul(F::BYTES).ok_or(
                    HasseError::GeometryOverflow {
                        context: "scalar input staging",
                    },
                )?,
                "scalar input staging",
            )?,
            scalar_output: reserve(
                self.total_weight
                    .checked_mul(F::BYTES)
                    .ok_or(HasseError::GeometryOverflow {
                        context: "scalar output staging",
                    })?,
                "scalar output staging",
            )?,
            batch_input: reserve(
                self.max_coefficients
                    .checked_mul(lanes)
                    .and_then(|lanes| lanes.checked_mul(F::BYTES))
                    .ok_or(HasseError::GeometryOverflow {
                        context: "batch input staging",
                    })?,
                "batch input staging",
            )?,
            lane,
        })
    }

    /// Evaluate scalar coefficients (low degree first) into `output`.
    ///
    /// # Errors
    ///
    /// As [`MultiplicityPlan::evaluate_batch_into`] at batch one; `output`
    /// must hold exactly `total_weight` elements.
    pub fn evaluate_into(
        &self,
        coefficients: &[F::Elem],
        scratch: &mut MultiplicityScratch<F>,
        output: &mut [F::Elem],
    ) -> Result<(), HasseError> {
        let count = coefficients.len();
        let input_bytes = count
            .checked_mul(F::BYTES)
            .ok_or(HasseError::GeometryOverflow {
                context: "scalar coefficients",
            })?;
        if input_bytes > scratch.scalar_input.len() {
            return Err(HasseError::ScratchMismatch);
        }
        if output.len() != self.total_weight {
            return Err(HasseError::LengthMismatch {
                argument: "multipoint output",
                expected: self.total_weight,
                actual: output.len(),
            });
        }
        let mut input = core::mem::take(&mut scratch.scalar_input);
        let mut staged = core::mem::take(&mut scratch.scalar_output);
        for (destination, value) in input[..input_bytes]
            .chunks_exact_mut(F::BYTES)
            .zip(coefficients)
        {
            F::encode(destination, *value);
        }
        let result =
            self.evaluate_batch_into(&input[..input_bytes], count, 1, scratch, &mut staged[..]);
        if result.is_ok() {
            for (source, destination) in staged[..self.total_weight * F::BYTES]
                .chunks_exact(F::BYTES)
                .zip(output.iter_mut())
            {
                *destination = F::decode(source);
            }
        }
        scratch.scalar_input = input;
        scratch.scalar_output = staged;
        result
    }

    /// Evaluate coefficient-major batched lane rows into `output`.
    ///
    /// `coefficients` holds `coefficient_count` rows of `batch` lanes;
    /// `output` receives `total_weight` rows in the same layout, row
    /// `offsets[i] + j` lane `l` holding `D^[j]f_l(a_i)` for lane `l`'s
    /// polynomial.
    ///
    /// # Errors
    ///
    /// Returns [`HasseError::BatchCapacityExceeded`],
    /// [`HasseError::CoefficientCapacityExceeded`], or
    /// [`HasseError::LengthMismatch`] — all before any mutation of `output`.
    pub fn evaluate_batch_into(
        &self,
        coefficients: &[u8],
        coefficient_count: usize,
        batch: usize,
        scratch: &mut MultiplicityScratch<F>,
        output: &mut [u8],
    ) -> Result<(), HasseError> {
        if batch > scratch.max_batch {
            return Err(HasseError::BatchCapacityExceeded {
                maximum: scratch.max_batch,
                actual: batch,
            });
        }
        if coefficient_count > self.max_coefficients || coefficient_count > scratch.max_coefficients
        {
            return Err(HasseError::CoefficientCapacityExceeded {
                maximum: scratch.max_coefficients,
                actual: coefficient_count,
            });
        }
        let row_bytes = batch * F::BYTES;
        let input_bytes =
            coefficient_count
                .checked_mul(row_bytes)
                .ok_or(HasseError::GeometryOverflow {
                    context: "multipoint coefficients",
                })?;
        if coefficients.len() != input_bytes {
            return Err(HasseError::LengthMismatch {
                argument: "multipoint coefficients",
                expected: input_bytes,
                actual: coefficients.len(),
            });
        }
        let output_bytes = self.total_weight * row_bytes;
        if output.len() != output_bytes {
            return Err(HasseError::LengthMismatch {
                argument: "multipoint output",
                expected: output_bytes,
                actual: output.len(),
            });
        }
        if self.total_weight == 0 || batch == 0 {
            return Ok(());
        }
        if scratch.total_weight != self.total_weight
            || scratch.distinct_multiplicities != self.distinct_multiplicities
            || scratch.max_coefficients != self.max_coefficients
        {
            // Equal weight and capacities do not make a scratch reusable:
            // the jet slots are keyed by the distinct-multiplicity layout,
            // and per-slot jets reject a mismatch only after earlier slots
            // have already written output.
            return Err(HasseError::ScratchMismatch);
        }

        self.run_evaluation(coefficients, coefficient_count, batch, scratch, output)
    }

    /// The validated evaluation: one remainder descent, then one prepared
    /// jet per positive-weight point.
    fn run_evaluation(
        &self,
        coefficients: &[u8],
        coefficient_count: usize,
        batch: usize,
        scratch: &mut MultiplicityScratch<F>,
        output: &mut [u8],
    ) -> Result<(), HasseError> {
        let row_bytes = batch * F::BYTES;
        let MultiplicityScratch {
            remainder_scratch,
            remainders,
            jet_scratches,
            batch_input,
            lane,
            ..
        } = scratch;

        // A single lane over all-unity weights: output row `offsets[i]` is
        // exactly `f(a_i)`, so the lane-parallel Horner route fills the
        // caller's rows directly and the remainder descent is skipped.
        if batch == 1
            && self.all_multiplicities_one
            && self.points.len().saturating_mul(coefficient_count) <= MULTIPOINT_LANE_STEP_CROSSOVER
        {
            evaluate_lane_packed_into(output, coefficients, coefficient_count, &self.points, lane)?;
            return Ok(());
        }

        // One descent carries every lane through every weighted modulus.
        // Odd characteristic runs through a canonicalized copy, so the
        // caller's raw prime lanes never reach a packed kernel.
        let input_bytes_total = coefficient_count * row_bytes;
        if F::CHARACTERISTIC == 2 {
            self.tree
                .remainders_into(
                    coefficients,
                    coefficient_count,
                    batch,
                    remainder_scratch,
                    &mut remainders[..self.total_weight * row_bytes],
                )
                .map_err(HasseError::from)?;
        } else {
            let mut staged = core::mem::take(batch_input);
            if staged.len() < input_bytes_total {
                *batch_input = staged;
                return Err(HasseError::ScratchMismatch);
            }
            staged[..input_bytes_total].copy_from_slice(coefficients);
            for slot in staged[..input_bytes_total].chunks_exact_mut(F::BYTES) {
                let value = F::decode(slot);
                F::encode(slot, value.add(F::Elem::ZERO));
            }
            let result = self
                .tree
                .remainders_into(
                    &staged[..input_bytes_total],
                    coefficient_count,
                    batch,
                    remainder_scratch,
                    &mut remainders[..self.total_weight * row_bytes],
                )
                .map_err(HasseError::from);
            *batch_input = staged;
            result?;
        }

        // Translate each positive-weight leaf into its output range. The
        // leaf remainder is already the jet's input, row for row.
        for (index, jet) in self.jets.iter().enumerate() {
            let Some(jet) = jet else {
                continue;
            };
            let start = self.offsets[index] * row_bytes;
            let length = self.multiplicities[index] * row_bytes;
            let jet_scratch = &mut jet_scratches[self.jet_slot[index]];
            jet.evaluate_batch_into(
                &remainders[start..start + length],
                self.multiplicities[index],
                batch,
                jet_scratch,
                &mut output[start..start + length],
            )?;
        }
        Ok(())
    }

    /// The weighted modulus `(X − a)^weight`, `1` at weight zero.
    fn weighted_modulus(point: F::Elem, weight: usize) -> Result<Polynomial<F>, HasseError> {
        let base = Polynomial::from_coefficients(&[point.neg(), F::Elem::ONE])
            .map_err(HasseError::from)?;
        let mut result = Polynomial::one().map_err(HasseError::from)?;
        let mut remaining = weight;
        let mut power = base;
        while remaining > 0 {
            if remaining & 1 != 0 {
                result = result.multiply(&power).map_err(HasseError::from)?;
            }
            remaining >>= 1;
            if remaining > 0 {
                power = power.square().map_err(HasseError::from)?;
            }
        }
        Ok(result)
    }
}

impl<F: PolynomialField> core::fmt::Debug for MultiplicityPlan<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MultiplicityPlan")
            .field("points", &self.points.len())
            .field("total_weight", &self.total_weight)
            .field("max_coefficients", &self.max_coefficients)
            .finish_non_exhaustive()
    }
}

impl<F: PolynomialField> core::fmt::Debug for MultiplicityScratch<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MultiplicityScratch")
            .field("max_batch", &self.max_batch)
            .field("total_weight", &self.total_weight)
            .field("jet_scratches", &self.jet_scratches.len())
            .finish_non_exhaustive()
    }
}

/// A vector with `capacity` slots reserved fallibly.
fn reserved<T>(capacity: usize, context: &'static str) -> Result<Vec<T>, HasseError> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(capacity)
        .map_err(|_| HasseError::AllocationFailed { context })?;
    Ok(buffer)
}
