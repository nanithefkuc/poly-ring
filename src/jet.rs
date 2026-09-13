//! Prepared local Taylor jets: the truncated translation `f(a + T) mod T^s`.

use alloc::vec::Vec;

use crate::poly::{ConvolutionScratch, PolynomialField, multiply_rows_into};
use fgf::field::Elem;
use fgf::ops::{self, Coeff};

use crate::error::HasseError;

/// Recursion base: at or below this many coefficients the translation runs
/// the division-free Horner-jet kernel. The fixed base bounds the quadratic
/// work of that kernel; tuning it requires measured evidence, not a global
/// fallback to Horner.
const HORNER_BASE: usize = 16;

/// A checked byte length for `count` rows of `batch` lanes.
fn rows_bytes(
    context: &'static str,
    count: usize,
    batch: usize,
    element_bytes: usize,
) -> Result<usize, HasseError> {
    count
        .checked_mul(batch)
        .and_then(|lanes| lanes.checked_mul(element_bytes))
        .ok_or(HasseError::GeometryOverflow { context })
}

/// A prepared one-point truncated Taylor translation.
///
/// `evaluate_into` writes `D^[j]f(a)` for `j < s`: the coefficient of `T^j`
/// in `f(a + T)`, truncated at the multiplicity. The translation is
/// divide-and-conquer — split `f = f0 + X^h·f1`, translate both halves,
/// combine with the precomputed power `(a + T)^h mod T^s` — so it costs
/// quasi-linear multiplication in the field, in every characteristic, with
/// no factorial inversion anywhere. The Horner-jet kernel runs only at the
/// recursion base and in the tests, as the independent oracle.
///
/// A zero point copies the low `s` coefficients and zero-pads: translation
/// by zero is truncation. Multiplicity zero is a legal empty request.
pub struct JetPlan<F: PolynomialField> {
    point: F::Elem,
    multiplicity: usize,
    max_coefficients: usize,
    /// `(a + T)^(2^k) mod T^s`, single-lane rows, for every split the
    /// recursion can take at the prepared capacity.
    powers: Vec<Vec<u8>>,
    prepared_point: Coeff<F>,
}

/// Reusable translation workspace.
///
/// Holds the convolution engine workspace and the per-depth row slots of the
/// recursion, sized to the plan's bounds and the lane capacity. Scratch
/// compatibility is layout only: it holds nothing derived from the point, so
/// two plans with the same geometry can share one scratch.
pub struct JetScratch<F: PolynomialField> {
    max_batch: usize,
    multiplicity: usize,
    convolution: ConvolutionScratch<F>,
    /// Three row buffers per recursion depth: the two child translations and
    /// one product temporary, each `s` rows of `max_batch` lanes.
    slots: Vec<Vec<u8>>,
    /// Single-lane input staging for the scalar entry point.
    scalar_input: Vec<u8>,
    /// Output staging for the scalar entry point.
    scalar_output: Vec<u8>,
    /// Batch input staging: the canonicalized copy odd-characteristic runs
    /// evaluate through, so raw prime lanes never reach a packed kernel.
    batch_input: Vec<u8>,
}

impl<F: PolynomialField> JetPlan<F> {
    /// Prepare the translation at `point` with multiplicity `multiplicity`
    /// for inputs of up to `max_coefficients` coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`HasseError::Product`] when a translation power cannot be
    /// computed, and the geometry/allocation errors of the factor tables.
    pub fn new(
        point: F::Elem,
        multiplicity: usize,
        max_coefficients: usize,
    ) -> Result<Self, HasseError> {
        // Canonicalize once: a noncanonical prime lane must never reach the
        // packed kernels this plan drives.
        let point = point.add(F::Elem::ZERO);
        let prepared_point = Coeff::<F>::new(point);
        let mut powers = Vec::new();
        if multiplicity > 0 && max_coefficients > 1 {
            powers = Self::compute_powers(point, multiplicity, max_coefficients)?;
        }
        Ok(Self {
            point,
            multiplicity,
            max_coefficients,
            powers,
            prepared_point,
        })
    }

    /// The point this plan translates at.
    #[must_use]
    pub fn point(&self) -> F::Elem {
        self.point
    }

    /// The multiplicity (jet length) this plan produces.
    #[must_use]
    pub fn multiplicity(&self) -> usize {
        self.multiplicity
    }

    /// The coefficient capacity this plan was built for.
    #[must_use]
    pub fn max_coefficients(&self) -> usize {
        self.max_coefficients
    }

    /// Build a scratch for batches of up to `batch_capacity` lanes.
    ///
    /// `scratch(0)` is a valid empty-batch capacity; scalar execution
    /// requires at least one lane.
    ///
    /// # Errors
    ///
    /// Returns [`HasseError::AllocationFailed`] or
    /// [`HasseError::GeometryOverflow`] when the workspace cannot be
    /// reserved.
    pub fn scratch(&self, batch_capacity: usize) -> Result<JetScratch<F>, HasseError> {
        let lanes = batch_capacity.max(1);
        let reserve = |bytes: usize, context: &'static str| -> Result<Vec<u8>, HasseError> {
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(bytes)
                .map_err(|_| HasseError::AllocationFailed { context })?;
            buffer.resize(bytes, 0);
            Ok(buffer)
        };
        let s = self.multiplicity;
        // Every product this plan runs is `s × s` rows truncated to `s`.
        let mut convolution =
            ConvolutionScratch::<F>::new(s.max(1), s.max(1), batch_capacity.max(1))
                .map_err(HasseError::from)?;
        let full = s.saturating_add(s).saturating_sub(1);
        if s > 1 {
            convolution.prepare_transform(full, batch_capacity.max(1))?;
        }
        let levels = recursion_depth(self.max_coefficients);
        let slot_bytes = rows_bytes("jet slot bytes", s, batch_capacity.max(1), F::BYTES)?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(4 * levels)
            .map_err(|_| HasseError::AllocationFailed {
                context: "jet recursion slots",
            })?;
        for _ in 0..4 * levels {
            slots.push(reserve(slot_bytes, "jet recursion slot")?);
        }
        Ok(JetScratch {
            max_batch: batch_capacity,
            multiplicity: s,
            convolution,
            slots,
            scalar_input: reserve(
                rows_bytes("jet scalar input", self.max_coefficients, 1, F::BYTES)?,
                "jet scalar input",
            )?,
            scalar_output: reserve(
                rows_bytes("jet scalar output", s, 1, F::BYTES)?,
                "jet scalar output",
            )?,
            batch_input: reserve(
                rows_bytes("jet batch input", self.max_coefficients, lanes, F::BYTES)?,
                "jet batch input",
            )?,
        })
    }

    /// Translate scalar coefficients (low degree first) into `output`.
    ///
    /// # Errors
    ///
    /// Returns the capacity and length errors of
    /// [`JetPlan::evaluate_batch_into`], which this wraps at batch one.
    /// `output` must hold exactly `multiplicity` elements.
    pub fn evaluate_into(
        &self,
        coefficients: &[F::Elem],
        scratch: &mut JetScratch<F>,
        output: &mut [F::Elem],
    ) -> Result<(), HasseError> {
        let count = coefficients.len();
        let input_bytes = rows_bytes("jet coefficients", count, 1, F::BYTES)?;
        if scratch.scalar_input.len() < input_bytes {
            return Err(HasseError::ScratchMismatch);
        }
        if output.len() != self.multiplicity {
            return Err(HasseError::LengthMismatch {
                argument: "jet output",
                expected: self.multiplicity,
                actual: output.len(),
            });
        }
        let mut input = core::mem::take(&mut scratch.scalar_input);
        let mut staged = core::mem::take(&mut scratch.scalar_output);
        for (destination, value) in input[..input_bytes]
            .chunks_exact_mut(F::BYTES)
            .zip(coefficients)
        {
            F::write(destination, *value);
        }
        let result =
            self.evaluate_batch_into(&input[..input_bytes], count, 1, scratch, &mut staged[..]);
        if result.is_ok() {
            for (source, destination) in staged[..self.multiplicity * F::BYTES]
                .chunks_exact(F::BYTES)
                .zip(output.iter_mut())
            {
                *destination = F::read(source);
            }
        }
        scratch.scalar_input = input;
        scratch.scalar_output = staged;
        result
    }

    /// Translate coefficient-major batched lane rows into `output`.
    ///
    /// `coefficients` holds `coefficient_count` rows of `batch` lanes;
    /// `output` receives exactly `multiplicity` rows in the same layout.
    /// Trailing input rows beyond the plan's capacity are rejected, not
    /// truncated.
    ///
    /// # Errors
    ///
    /// Returns [`HasseError::CoefficientCapacityExceeded`],
    /// [`HasseError::BatchCapacityExceeded`], [`HasseError::LengthMismatch`],
    /// or [`HasseError::ScratchMismatch`] — all before any mutation of
    /// `output`.
    pub fn evaluate_batch_into(
        &self,
        coefficients: &[u8],
        coefficient_count: usize,
        batch: usize,
        scratch: &mut JetScratch<F>,
        output: &mut [u8],
    ) -> Result<(), HasseError> {
        // Validation, in full, before any mutation.
        if batch > scratch.max_batch {
            return Err(HasseError::BatchCapacityExceeded {
                maximum: scratch.max_batch,
                actual: batch,
            });
        }
        if scratch.multiplicity != self.multiplicity {
            return Err(HasseError::ScratchMismatch);
        }
        let input_bytes = rows_bytes("jet coefficients", coefficient_count, batch, F::BYTES)?;
        if coefficients.len() != input_bytes {
            return Err(HasseError::LengthMismatch {
                argument: "jet coefficients",
                expected: input_bytes,
                actual: coefficients.len(),
            });
        }
        if coefficient_count > self.max_coefficients {
            return Err(HasseError::CoefficientCapacityExceeded {
                maximum: self.max_coefficients,
                actual: coefficient_count,
            });
        }
        let output_bytes = rows_bytes("jet output", self.multiplicity, batch, F::BYTES)?;
        if output.len() != output_bytes {
            return Err(HasseError::LengthMismatch {
                argument: "jet output",
                expected: output_bytes,
                actual: output.len(),
            });
        }
        if self.multiplicity == 0 || batch == 0 {
            return Ok(());
        }
        let row_bytes = batch * F::BYTES;

        if F::CHARACTERISTIC == 2 {
            return self.run_translation(coefficients, coefficient_count, batch, output, scratch);
        }
        // Odd characteristic: run through a canonicalized copy, so the
        // caller's raw prime lanes never reach a packed kernel.
        let input_bytes_total = coefficient_count * row_bytes;
        let mut staged = core::mem::take(&mut scratch.batch_input);
        if staged.len() < input_bytes_total {
            scratch.batch_input = staged;
            return Err(HasseError::ScratchMismatch);
        }
        staged[..input_bytes_total].copy_from_slice(coefficients);
        for slot in staged[..input_bytes_total].chunks_exact_mut(F::BYTES) {
            let value = F::read(slot);
            F::write(slot, value.add(F::Elem::ZERO));
        }
        let result = self.run_translation(
            &staged[..input_bytes_total],
            coefficient_count,
            batch,
            output,
            scratch,
        );
        scratch.batch_input = staged;
        result
    }

    /// The translation proper, over already-canonical input rows.
    fn run_translation(
        &self,
        coefficients: &[u8],
        coefficient_count: usize,
        batch: usize,
        output: &mut [u8],
        scratch: &mut JetScratch<F>,
    ) -> Result<(), HasseError> {
        let row_bytes = batch * F::BYTES;
        if self.point.is_zero() {
            // Translation by zero is truncation: copy the low s rows and
            // zero-pad — no convolution at all.
            let kept = self.multiplicity.min(coefficient_count) * row_bytes;
            output[..kept].copy_from_slice(&coefficients[..kept]);
            output[kept..].fill(0);
            return Ok(());
        }
        if coefficient_count == 0 {
            // The empty polynomial translates to the empty jet.
            output.fill(0);
            return Ok(());
        }
        if coefficient_count == 1 {
            // f = c0: the jet is c0 followed by zeros.
            output[..row_bytes].copy_from_slice(&coefficients[..row_bytes]);
            output[row_bytes..].fill(0);
            return Ok(());
        }
        let slot_bytes = self.multiplicity * row_bytes;
        let levels = recursion_depth(self.max_coefficients);
        if scratch.slots.len() < 4 * levels || scratch.slots[0].len() < slot_bytes {
            return Err(HasseError::ScratchMismatch);
        }
        // A translation of a `count`-coefficient polynomial has at most
        // `count` nonzero coefficients, so the recursion below carries
        // `min(s, count)` live orders: subtree work is bounded by the
        // subtree's size, never by the full multiplicity.
        let orders = self.multiplicity.min(coefficient_count);
        self.translate(
            coefficients,
            coefficient_count,
            batch,
            output,
            0,
            orders,
            scratch,
        );
        Ok(())
    }

    /// The divide-and-conquer translation, writing `orders` rows into
    /// `target` and zeroing any rows beyond within the target's length.
    ///
    /// `orders` is the live order count: at most the multiplicity and at
    /// most `count`, since a translation preserves length. Every product
    /// below works on `min(orders, operand coefficients)` rows, so the
    /// leaves never pay for the full multiplicity.
    #[allow(clippy::too_many_arguments)]
    fn translate(
        &self,
        coefficients: &[u8],
        count: usize,
        batch: usize,
        target: &mut [u8],
        depth: usize,
        orders: usize,
        scratch: &mut JetScratch<F>,
    ) {
        if count <= HORNER_BASE {
            self.horner_jet(coefficients, count, batch, target, orders);
            return;
        }
        let row_bytes = batch * F::BYTES;
        let orders_bytes = orders * row_bytes;
        // h = the largest power of two strictly below the length.
        let split = count
            .checked_next_power_of_two()
            .expect("count above the base")
            .checked_shr(1)
            .expect("count above the base");
        debug_assert!(split < count);
        let power = &self.powers[split.trailing_zeros() as usize];

        // Four fixed slots at this depth: the two child translations, the
        // broadcast power, and the product. Siblings never alias.
        let mut low = core::mem::take(&mut scratch.slots[4 * depth]);
        let mut high = core::mem::take(&mut scratch.slots[4 * depth + 1]);
        let mut broadcast = core::mem::take(&mut scratch.slots[4 * depth + 2]);
        let mut product = core::mem::take(&mut scratch.slots[4 * depth + 3]);

        // A translation of each half has at most the half's coefficients.
        let low_orders = orders.min(split);
        let high_orders = orders.min(count - split);
        self.translate(
            &coefficients[..split * row_bytes],
            split,
            batch,
            &mut low[..low_orders * row_bytes],
            depth + 1,
            low_orders,
            scratch,
        );
        self.translate(
            &coefficients[split * row_bytes..],
            count - split,
            batch,
            &mut high[..high_orders * row_bytes],
            depth + 1,
            high_orders,
            scratch,
        );

        // target = f0(a+T) + (a+T)^h · f1(a+T) mod T^orders. The power is
        // precomputed mod T^s; its first `orders` rows are exactly the
        // power mod T^orders, and low coefficients of a product depend only
        // on low coefficients of the operands, so the truncated operands
        // give the correct truncated product.
        for degree in 0..orders {
            for lane in 0..batch {
                let source = degree * F::BYTES;
                let destination = degree * row_bytes + lane * F::BYTES;
                broadcast[destination..destination + F::BYTES]
                    .copy_from_slice(&power[source..source + F::BYTES]);
            }
        }
        let product_rows = if high_orders == 0 { 0 } else { orders };
        multiply_rows_into::<F>(
            &broadcast[..orders * row_bytes],
            orders,
            &high[..high_orders * row_bytes],
            high_orders,
            batch,
            orders,
            &mut scratch.convolution,
            &mut product[..product_rows * row_bytes],
        )
        .expect("prepared jet product geometry");

        target[..low_orders * row_bytes].copy_from_slice(&low[..low_orders * row_bytes]);
        target[low_orders * row_bytes..orders_bytes].fill(0);
        ops::add_assign::<F>(
            &mut target[..orders_bytes],
            &product[..product_rows * row_bytes],
        );
        // Rows between the live orders and the caller's target length stay
        // zero (nonempty only at the top level, where the target is the
        // full `s`-row output).
        target[orders_bytes..].fill(0);

        scratch.slots[4 * depth] = low;
        scratch.slots[4 * depth + 1] = high;
        scratch.slots[4 * depth + 2] = broadcast;
        scratch.slots[4 * depth + 3] = product;
    }

    /// The division-free Horner jet kernel, in place over lane rows.
    ///
    /// For descending coefficients and descending orders:
    /// `jet[j] = a·jet[j] + jet[j−1]`, `jet[0] = a·jet[0] + c`. Every row
    /// operation composes `fgf` kernels with the prepared point; no
    /// multiplication instruction is re-implemented here.
    fn horner_jet(
        &self,
        coefficients: &[u8],
        count: usize,
        batch: usize,
        target: &mut [u8],
        orders: usize,
    ) {
        let row_bytes = batch * F::BYTES;
        target[..orders * row_bytes].fill(0);
        for coefficient in (0..count).rev() {
            let source = coefficient * row_bytes;
            for order in (1..orders).rev() {
                let (lower, current) = {
                    let (head, tail) = target.split_at_mut(order * row_bytes);
                    (&head[(order - 1) * row_bytes..], &mut tail[..row_bytes])
                };
                ops::mul_assign_with::<F>(current, &self.prepared_point);
                ops::add_assign::<F>(current, lower);
            }
            ops::mul_assign_with::<F>(&mut target[..row_bytes], &self.prepared_point);
            ops::add_assign::<F>(
                &mut target[..row_bytes],
                &coefficients[source..source + row_bytes],
            );
        }
        // Orders between `orders` and the multiplicity stay zero: a
        // translation of a `count`-coefficient polynomial has at most
        // `count` coefficients and `orders` already covers them.
        target[orders * row_bytes..].fill(0);
    }

    /// `(a + T)^(2^k) mod T^s` for every split the recursion can take.
    fn compute_powers(
        point: F::Elem,
        multiplicity: usize,
        max_coefficients: usize,
    ) -> Result<Vec<Vec<u8>>, HasseError> {
        let depth = recursion_depth(max_coefficients);
        let s_bytes = rows_bytes("jet translation power rows", multiplicity, 1, F::BYTES)?;
        let full = multiplicity.saturating_add(multiplicity).saturating_sub(1);
        let mut convolution = ConvolutionScratch::<F>::new(multiplicity, multiplicity, 1)
            .map_err(HasseError::from)?;
        if multiplicity > 1 {
            convolution.prepare_transform(full, 1)?;
        }
        let mut scratch_product = zeroed(s_bytes, "jet translation power product")?;
        let mut powers = Vec::new();
        powers
            .try_reserve_exact(depth)
            .map_err(|_| HasseError::AllocationFailed {
                context: "jet translation powers",
            })?;
        // p_0 = a + T, truncated: coefficient row 0 is `a`, row 1 is one.
        let mut current = zeroed(s_bytes, "jet translation power rows")?;
        F::write(&mut current[..F::BYTES], point);
        if multiplicity > 1 {
            F::write(&mut current[F::BYTES..2 * F::BYTES], F::Elem::ONE);
        }
        for level in 0..depth {
            if level > 0 {
                multiply_rows_into::<F>(
                    &current,
                    multiplicity,
                    &current,
                    multiplicity,
                    1,
                    multiplicity,
                    &mut convolution,
                    &mut scratch_product,
                )
                .map_err(HasseError::from)?;
                current.copy_from_slice(&scratch_product);
            }
            // The level takes its own copy (one fallible allocation per
            // level); `current` keeps the live value for the next squaring.
            let mut level_rows = zeroed(s_bytes, "jet translation power rows")?;
            level_rows.copy_from_slice(&current);
            powers.push(level_rows);
        }
        Ok(powers)
    }
}

/// A zero-filled buffer reserved fallibly.
fn zeroed(bytes: usize, context: &'static str) -> Result<Vec<u8>, HasseError> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(bytes)
        .map_err(|_| HasseError::AllocationFailed { context })?;
    buffer.resize(bytes, 0);
    Ok(buffer)
}

/// The number of recursion levels a translation of `max_coefficients`
/// coefficients can reach.
fn recursion_depth(max_coefficients: usize) -> usize {
    if max_coefficients <= 1 {
        return 0;
    }
    // Splits reach every power of two strictly below the capacity: one
    // power (and so one recursion level) per bit below the top.
    (max_coefficients - 1).ilog2() as usize + 1
}

impl<F: PolynomialField> core::fmt::Debug for JetPlan<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("JetPlan")
            .field("multiplicity", &self.multiplicity)
            .field("max_coefficients", &self.max_coefficients)
            .field("power_levels", &self.powers.len())
            .finish_non_exhaustive()
    }
}

impl<F: PolynomialField> core::fmt::Debug for JetScratch<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("JetScratch")
            .field("max_batch", &self.max_batch)
            .field("multiplicity", &self.multiplicity)
            .field("slots", &self.slots.len())
            .finish_non_exhaustive()
    }
}
