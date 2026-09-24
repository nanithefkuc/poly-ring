//! Prepared polynomial products over coefficient-major batched rows.
//!
//! One polynomial is batch size one; a batch of `B` independent polynomials is
//! held as `count` coefficient rows of `B` lanes each, lane `l`'s coefficient
//! `d` at `(d * B + l) * F::BYTES`. This layout is an arithmetic contract
//! shared with the remainder tree and the `hasse` crate above, not a wire
//! format.
//!
//! The engine has one stable entry point, [`multiply_rows_into`], and one
//! reusable workspace, [`ConvolutionScratch`]. Routes:
//!
//! - **Schoolbook / Karatsuba** — the corrected packed kernels from
//!   Karatsuba kernels from the poly module, run lane by lane. Always
//!   available, so
//!   `--no-default-features` and transform-disabled fields keep exact
//!   products at `O(n²)` / `O(n^{1.585})` cost.
//! - **Additive transform** (binary fields, `fft`) — the shared AFFT row
//!   pipeline shared with the AFFT batch product.
//! - **Multiplicative transform** (`fft`) — `butterfly-fft`'s NTT over
//!   `Goldilocks` and `QuadMersenne31`, padded to the **full** product size.
//! - **Embedded** (`fft`) — Mersenne31 convolves inside
//!   `QuadMersenne31` by the
//!   exact field embedding `a ↦ (a, 0)`, because its own multiplicative
//!   group admits no useful radix-two transform.
//!
//! `Auto` keeps the measured binary-field AFFT crossovers
//! ([`crate::cost::select_product`]). Goldilocks prepared products select the
//! NTT by shorter operand and batch width. The other prime-field transform
//! routes remain explicit through the forced benchmark entry point.

use alloc::vec::Vec;
use core::marker::PhantomData;

#[cfg(feature = "fft")]
use fgf::field::{Elem as _, Field};
use fgf::kernel::FieldKernels;

use crate::error::{ConfigError, ProductError};
use crate::geometry::checked_product;

use super::karatsuba::{KARATSUBA_CROSSOVER, KaratsubaScratch, karatsuba_into, schoolbook_into};

#[cfg(feature = "fft")]
use {
    super::afft::afft_rows_convolve,
    butterfly_fft::ntt::{NttPlan, NttScratch},
    butterfly_fft::transform::TransformPlan,
    core::any::TypeId,
    fgf::ops,
    fgf::{Goldilocks, QuadMersenne31},
};

/// A field whose polynomials the prepared product engine serves.
///
/// Sealed: the implementor set is exactly `fgf`'s packed fields. Bit-packed
/// GF(2) is deliberately absent — this is a byte-lane engine. Existing
/// generic polynomial APIs keep their `FieldKernels` bounds; only the
// The private supertrait seals the route surface, exactly as in `fgf`.
#[allow(private_bounds)]
pub trait PolynomialField: FieldKernels + private::Sealed + ConvolutionDomain {}
impl<F> PolynomialField for F where F: FieldKernels + private::Sealed + ConvolutionDomain {}

mod private {
    pub trait Sealed {}
}

/// Which transform family serves a field's prepared products.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "fft"), allow(dead_code))]
pub(crate) enum Route {
    /// The additive FFT over a binary field.
    Afft,
    /// A multiplicative NTT over the field itself.
    Ntt,
    /// Convolution inside an exact field extension.
    Embedded,
    /// No transform route; schoolbook and Karatsuba only.
    None,
}

/// Inputs and reusable buffers for one transform-domain convolution.
#[cfg_attr(not(feature = "fft"), allow(dead_code))]
pub(crate) struct TransformRows<'a, P> {
    left: &'a [u8],
    left_count: usize,
    right: &'a [u8],
    right_count: usize,
    batch: usize,
    precision: usize,
    plan: &'a mut P,
    operands: &'a mut [u8],
    products: &'a mut [u8],
    conversion: &'a mut [u8],
    output: &'a mut [u8],
}
/// Per-field convolution workspace and route operations.
///
/// Crate-private supertrait of [`PolynomialField`], mirroring `fgf`'s
/// kernel-dispatch pattern: generic code bounds on the public name and gets
/// the field's route through this trait without being able to name or
/// implement it.
pub(crate) trait ConvolutionDomain: FieldKernels {
    /// A cached, cloneable transform plan (with its execution scratch, where
    /// the route needs one) for a single full-product size.
    type Plan: Clone;

    /// The route this field's prepared products take.
    #[cfg_attr(not(feature = "fft"), allow(dead_code))]
    const ROUTE: Route;

    /// Whether prepared `Auto` may take this field's NTT route.
    #[cfg_attr(not(feature = "fft"), allow(dead_code))]
    const AUTO_NTT: bool = false;

    /// Build the transform plan covering a full product of `full_size`
    /// coefficients over `batch` lanes, or `None` when the size is beyond
    /// this route.
    fn build_plan(full_size: usize, batch: usize) -> Option<Self::Plan>;

    /// Byte sizes of the shared work buffers for one transform size:
    /// `(operands, products, conversion)`.
    fn work_bytes(_: usize, _: usize) -> Result<(usize, usize, usize), ConfigError> {
        Ok((0, 0, 0))
    }

    /// Run the transform product over coefficient-major lane rows.
    ///
    /// All buffer lengths are pre-validated by the caller; `operands`,
    /// `products`, and `conversion` are at least as large as
    /// [`ConvolutionDomain::work_bytes`] reported for the plan's transform
    /// size. The plan is mutable because NTT routes keep their execution
    /// scratch inside it; its geometry is immutable.
    fn transform_rows(_rows: TransformRows<'_, Self::Plan>) -> Result<(), ProductError> {
        unreachable!("this field has no transform route")
    }
}

/// The plan type of a field with no transform route.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NoPlan<F> {
    _field: PhantomData<F>,
}

/// Route selection for the prepared product engine.
///
/// Unstable surface under the `internals` feature: the forced modes exist so
/// benchmark panels can compare routes reproducibly; `Auto` is the only
/// mode callers should rely on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "internals"), expect(dead_code))]
pub enum ProductRoute {
    /// The measured per-field default.
    Auto,
    /// Packed schoolbook convolution, lane by lane.
    Schoolbook,
    /// Karatsuba, lane by lane.
    Karatsuba,
    /// The field's transform route; an error if the size is unsupported.
    Transform,
}

/// An NTT plan bundled with its execution scratch.
///
/// [`NttScratch`] records only the element width and row length, so one
/// scratch built for the largest prepared row length serves every plan of
/// the same field.
#[cfg(feature = "fft")]
#[derive(Clone, Debug)]
pub(crate) struct NttRoute<F: FieldKernels> {
    plan: NttPlan<F>,
    scratch: NttScratch,
}

#[cfg(feature = "fft")]
impl<F: FieldKernels> NttRoute<F> {
    fn new(size: usize, row_len: usize) -> Option<Self> {
        let plan = NttPlan::new(size).ok()?;
        let scratch = plan.scratch(row_len).ok()?;
        Some(Self { plan, scratch })
    }
}

// ---------------------------------------------------------------------------
// Binary fields: the additive transform route.

#[cfg(feature = "fft")]
macro_rules! impl_afft_domain {
    ($($field:ty),+ $(,)?) => {$(
        impl private::Sealed for $field {}

        impl ConvolutionDomain for $field {
            type Plan = TransformPlan<$field>;

            const ROUTE: Route = Route::Afft;

            fn build_plan(full_size: usize, _batch: usize) -> Option<Self::Plan> {
                let size = full_size.checked_next_power_of_two()?;
                TransformPlan::new(size).ok()
            }

            fn work_bytes(
                transform_size: usize,
                batch: usize,
            ) -> Result<(usize, usize, usize), ConfigError> {
                let pair_bytes = checked_product("AFFT lane bytes", batch, Self::BYTES)?;
                let operand_row_bytes = checked_product("AFFT operand row bytes", pair_bytes, 2)?;
                let operands =
                    checked_product("AFFT operand bytes", transform_size, operand_row_bytes)?;
                let products = checked_product("AFFT product bytes", transform_size, pair_bytes)?;
                let conversion = products;
                Ok((operands, products, conversion))
            }

            fn transform_rows(rows: TransformRows<'_, Self::Plan>) -> Result<(), ProductError> {
                let TransformRows {
                    left,
                    left_count,
                    right,
                    right_count,
                    batch,
                    precision,
                    plan,
                    operands,
                    products,
                    conversion,
                    output,
                } = rows;
                let pair_bytes = batch * Self::BYTES;
                operands.fill(0);
                for (rows, count, lane_offset) in
                    [(left, left_count, 0), (right, right_count, pair_bytes)]
                {
                    for degree in 0..count {
                        for lane in 0..batch {
                            let source = (degree * batch + lane) * Self::BYTES;
                            let destination =
                                degree * 2 * pair_bytes + lane_offset + lane * Self::BYTES;
                            operands[destination..destination + Self::BYTES]
                                .copy_from_slice(&rows[source..source + Self::BYTES]);
                        }
                    }
                }
                afft_rows_convolve::<Self>(plan, operands, products, conversion, pair_bytes)?;
                let out_rows = (left_count + right_count - 1).min(precision);
                let kept = out_rows * pair_bytes;
                output[..kept].copy_from_slice(&products[..kept]);
                Ok(())
            }
        }
    )+};
}

#[cfg(feature = "fft")]
impl_afft_domain!(
    fgf::Gf8B,
    fgf::Gf16,
    fgf::Gf32,
    fgf::Gf64,
    fgf::FanPaar8,
    fgf::FanPaar16,
    fgf::FanPaar32,
    fgf::FanPaar64,
);

// ---------------------------------------------------------------------------
// Prime fields: the multiplicative transform route.

#[cfg(feature = "fft")]
macro_rules! impl_ntt_domain {
    ($($field:ty => $auto:expr),+ $(,)?) => {$(
        impl private::Sealed for $field {}

        impl ConvolutionDomain for $field {
            type Plan = NttRoute<$field>;

            const ROUTE: Route = Route::Ntt;

            const AUTO_NTT: bool = $auto;

            fn build_plan(full_size: usize, batch: usize) -> Option<Self::Plan> {
                // Pad to the FULL product size: padding to a truncation
                // length would wrap the high coefficients back into the
                // kept ones.
                let size = full_size.checked_next_power_of_two()?;
                NttRoute::new(size, batch * Self::BYTES)
            }

            fn work_bytes(
                transform_size: usize,
                batch: usize,
            ) -> Result<(usize, usize, usize), ConfigError> {
                let lane_bytes = checked_product("NTT lane bytes", batch, Self::BYTES)?;
                let block = checked_product("NTT block bytes", transform_size, lane_bytes)?;
                let operands = checked_product("NTT operand bytes", block, 2)?;
                Ok((operands, block, 0))
            }

            fn transform_rows(rows: TransformRows<'_, Self::Plan>) -> Result<(), ProductError> {
                let TransformRows {
                    left,
                    left_count,
                    right,
                    right_count,
                    batch,
                    precision,
                    plan,
                    operands,
                    products,
                    output,
                    ..
                } = rows;
                ntt_rows_convolve::<Self>(
                    left,
                    left_count,
                    right,
                    right_count,
                    batch,
                    precision,
                    plan,
                    operands,
                    products,
                    output,
                )
            }
        }
    )+};
}

#[cfg(feature = "fft")]
impl_ntt_domain!(
    Goldilocks => true,
    QuadMersenne31 => false,
);

/// Forward-transform both operand blocks, multiply pointwise, invert.
///
/// The lane rows are already coefficient-major — row `d` is `batch`
/// contiguous lanes — which is exactly the NTT block layout, so packing is a
/// row-by-row copy with zero padding.
#[cfg(feature = "fft")]
#[allow(clippy::too_many_arguments)]
fn ntt_rows_convolve<F: FieldKernels>(
    left: &[u8],
    left_count: usize,
    right: &[u8],
    right_count: usize,
    batch: usize,
    precision: usize,
    route: &mut NttRoute<F>,
    operands: &mut [u8],
    products: &mut [u8],
    output: &mut [u8],
) -> Result<(), ProductError> {
    let row_bytes = batch * F::BYTES;
    let block = route.plan.size() * row_bytes;
    let (left_block, right_block) = operands.split_at_mut(block);
    left_block.fill(0);
    right_block.fill(0);
    for degree in 0..left_count {
        let source = degree * row_bytes;
        left_block[degree * row_bytes..(degree + 1) * row_bytes]
            .copy_from_slice(&left[source..source + row_bytes]);
    }
    for degree in 0..right_count {
        let source = degree * row_bytes;
        right_block[degree * row_bytes..(degree + 1) * row_bytes]
            .copy_from_slice(&right[source..source + row_bytes]);
    }
    let scratch = &mut route.scratch;
    route
        .plan
        .forward_bytes_scratch(left_block, row_bytes, scratch)
        .map_err(ProductError::from)?;
    route
        .plan
        .forward_bytes_scratch(right_block, row_bytes, scratch)
        .map_err(ProductError::from)?;
    for (left_row, right_row, product_row) in left_block
        .chunks_exact(row_bytes)
        .zip(right_block.chunks_exact(row_bytes))
        .zip(products.chunks_exact_mut(row_bytes))
        .map(|((left_row, right_row), product_row)| (left_row, right_row, product_row))
    {
        ops::mul_elementwise::<F>(product_row, left_row, right_row);
    }
    route
        .plan
        .inverse_bytes_scratch(products, row_bytes, scratch)
        .map_err(ProductError::from)?;
    let out_rows = (left_count + right_count - 1).min(precision);
    let kept = out_rows * row_bytes;
    output[..kept].copy_from_slice(&products[..kept]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Mersenne31: exact embedding into QuadMersenne31.
//
// `p − 1 = 2 · (2^30 − 1)` admits no radix-two transform beyond size two, so
// Mersenne31 convolution embeds each canonical base coefficient as `(a, 0)`
// in the quadratic extension (an exact field homomorphism, not an integer
// convolution), convolves there, and projects the real limb.

#[cfg(feature = "fft")]
impl private::Sealed for fgf::Mersenne31 {}

#[cfg(feature = "fft")]
impl ConvolutionDomain for fgf::Mersenne31 {
    type Plan = NttRoute<QuadMersenne31>;

    const ROUTE: Route = Route::Embedded;

    fn build_plan(full_size: usize, batch: usize) -> Option<Self::Plan> {
        let size = full_size.checked_next_power_of_two()?;
        NttRoute::new(size, batch * QuadMersenne31::BYTES)
    }

    fn work_bytes(
        transform_size: usize,
        batch: usize,
    ) -> Result<(usize, usize, usize), ConfigError> {
        let lane_bytes = checked_product("embedded lane bytes", batch, QuadMersenne31::BYTES)?;
        let block = checked_product("embedded block bytes", transform_size, lane_bytes)?;
        let operands = checked_product("embedded operand bytes", block, 2)?;
        Ok((operands, block, 0))
    }

    fn transform_rows(rows: TransformRows<'_, Self::Plan>) -> Result<(), ProductError> {
        let TransformRows {
            left,
            left_count,
            right,
            right_count,
            batch,
            precision,
            plan: route,
            operands,
            products,
            output,
            ..
        } = rows;
        let wide_row = batch * QuadMersenne31::BYTES;
        let block = route.plan.size() * wide_row;
        let (left_block, right_block) = operands.split_at_mut(block);
        left_block.fill(0);
        right_block.fill(0);
        embed_rows::<fgf::Mersenne31>(left, left_count, batch, left_block);
        embed_rows::<fgf::Mersenne31>(right, right_count, batch, right_block);

        let scratch = &mut route.scratch;
        route
            .plan
            .forward_bytes_scratch(left_block, wide_row, scratch)
            .map_err(ProductError::from)?;
        route
            .plan
            .forward_bytes_scratch(right_block, wide_row, scratch)
            .map_err(ProductError::from)?;
        for (left_row, right_row, product_row) in left_block
            .chunks_exact(wide_row)
            .zip(right_block.chunks_exact(wide_row))
            .zip(products.chunks_exact_mut(wide_row))
            .map(|((left_row, right_row), product_row)| (left_row, right_row, product_row))
        {
            ops::mul_elementwise::<QuadMersenne31>(product_row, left_row, right_row);
        }
        route
            .plan
            .inverse_bytes_scratch(products, wide_row, scratch)
            .map_err(ProductError::from)?;

        let full = left_count + right_count - 1;
        let out_rows = full.min(precision);
        for degree in 0..out_rows {
            for lane in 0..batch {
                let source = degree * wide_row + lane * QuadMersenne31::BYTES;
                let re =
                    u32::from_le_bytes(products[source..source + 4].try_into().expect("re limb"));
                let im = u32::from_le_bytes(
                    products[source + 4..source + 8]
                        .try_into()
                        .expect("im limb"),
                );
                debug_assert_eq!(im, 0, "embedded convolution leaves zero imaginary limbs");
                let destination = &mut output[(degree * batch + lane) * fgf::Mersenne31::BYTES..]
                    [..fgf::Mersenne31::BYTES];
                fgf::Mersenne31::encode(destination, fgf::mersenne31::Elem::from_raw(re));
            }
        }
        Ok(())
    }
}

/// Embed base-field lane rows as `(value, 0)` extension lanes.
///
/// Reading through the scalar element canonicalizes each lane, so a
/// noncanonical raw input cannot leak into the extension's packed kernels.
#[cfg(feature = "fft")]
fn embed_rows<Base>(rows: &[u8], count: usize, batch: usize, destination: &mut [u8])
where
    Base: FieldKernels,
{
    let wide_row = batch * QuadMersenne31::BYTES;
    for degree in 0..count {
        for lane in 0..batch {
            let source = (degree * batch + lane) * Base::BYTES;
            let value = Base::decode(&rows[source..source + Base::BYTES]).add(Base::Elem::ZERO);
            let mut encoded = [0_u8; 8];
            Base::encode(&mut encoded[..Base::BYTES], value);
            let target = degree * wide_row + lane * QuadMersenne31::BYTES;
            destination[target..target + Base::BYTES].copy_from_slice(&encoded[..Base::BYTES]);
            // The imaginary limb stays zero from the block fill.
            debug_assert!(
                destination[target + Base::BYTES..target + QuadMersenne31::BYTES]
                    .iter()
                    .all(|byte| *byte == 0)
            );
        }
    }
}

/// Fields whose only prepared route is schoolbook/Karatsuba.
macro_rules! impl_karatsuba_only_domain {
    ($($field:ty),+ $(,)?) => {$(
        impl private::Sealed for $field {}

        impl ConvolutionDomain for $field {
            type Plan = NoPlan<$field>;

            const ROUTE: Route = Route::None;

            fn build_plan(_full_size: usize, _batch: usize) -> Option<Self::Plan> {
                None
            }

        }
    )+};
}

// Gf8D has no additive-transform kernel seat and no usable multiplicative
// group, on every build; the prime fields join it only when the `fft`
// feature is off.
impl_karatsuba_only_domain!(fgf::Gf8D);
#[cfg(not(feature = "fft"))]
impl_karatsuba_only_domain!(
    fgf::Gf8B,
    fgf::Gf16,
    fgf::Gf32,
    fgf::Gf64,
    fgf::FanPaar8,
    fgf::FanPaar16,
    fgf::FanPaar32,
    fgf::FanPaar64,
    fgf::Goldilocks,
    fgf::Mersenne31,
    fgf::QuadMersenne31,
);

// ---------------------------------------------------------------------------
// The prepared engine.

/// Reusable workspace for prepared polynomial products.
///
/// Construct once for a maximum operand geometry, then run
/// [`multiply_rows_into`] any number of times: every execution inside the
/// prepared capacity — fallback or transform route — allocates nothing. The
/// constructor pre-warms the Karatsuba slots and caches a transform plan
/// for every transform size the maximum can request, so even the first
/// transform-routed execution is allocation-free.
/// [`ConvolutionScratch::prepare_transform`] remains for callers that want
/// an explicit confirmation, and is idempotent.
pub struct ConvolutionScratch<F: PolynomialField> {
    max_left: usize,
    max_right: usize,
    max_batch: usize,
    lane_left: Vec<u8>,
    lane_right: Vec<u8>,
    lane_product: Vec<u8>,
    karatsuba: KaratsubaScratch,
    operands: Vec<u8>,
    products: Vec<u8>,
    conversion: Vec<u8>,
    plans: Vec<(usize, F::Plan)>,
}

impl<F: PolynomialField> ConvolutionScratch<F> {
    /// Build scratch for operands of at most `max_left` and `max_right`
    /// coefficients and batches of at most `batch_capacity` lanes.
    ///
    /// # Errors
    ///
    /// Returns [`ProductError::Config`] when a derived byte length overflows
    /// or the buffers cannot be reserved.
    pub fn new(
        max_left: usize,
        max_right: usize,
        batch_capacity: usize,
    ) -> Result<Self, ProductError> {
        let max_operand = max_left.max(max_right);
        let max_full = max_left
            .checked_add(max_right)
            .and_then(|sum| sum.checked_sub(1))
            .unwrap_or(0);
        let lane_bytes = |context: &'static str, count: usize| -> Result<Vec<u8>, ProductError> {
            let bytes = checked_product(context, count, F::BYTES)?;
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(bytes)
                .map_err(|_| ConfigError::AllocationFailed {
                    context,
                    elements: bytes,
                    element_size: 1,
                })?;
            buffer.resize(bytes, 0);
            Ok(buffer)
        };
        let mut prepared = Self {
            max_left,
            max_right,
            max_batch: batch_capacity,
            lane_left: lane_bytes("prepared operand lanes", max_operand)?,
            lane_right: lane_bytes("prepared operand lanes", max_operand)?,
            lane_product: lane_bytes("prepared product lanes", max_full)?,
            karatsuba: KaratsubaScratch::new(),
            operands: Vec::new(),
            products: Vec::new(),
            conversion: Vec::new(),
            plans: Vec::new(),
        };
        // Pre-warm the Karatsuba slot vectors at the maximum geometry: the
        // depth-indexed slots grow lazily otherwise, which would make the
        // first fallback-route execution allocate. One zeroed warm-up
        // product touches every slot this capacity can reach.
        if max_full > 0 && max_operand >= super::karatsuba::KARATSUBA_BASE {
            prepared.lane_left[..max_operand * F::BYTES].fill(0);
            prepared.lane_right[..max_operand * F::BYTES].fill(0);
            prepared.lane_product[..max_full * F::BYTES].fill(0);
            super::karatsuba::karatsuba_into::<F>(
                &mut prepared.lane_product[..max_full * F::BYTES],
                &prepared.lane_left[..max_operand * F::BYTES],
                &prepared.lane_right[..max_operand * F::BYTES],
                0,
                &mut prepared.karatsuba,
            );
        }
        // Cache a transform plan for every transform size the prepared
        // maximum can request: any full product within the caps maps to one
        // of these sizes, so no execution — first or later — allocates.
        // Sizes beyond this field's route cache nothing and stay on the
        // fallback path.
        if max_full > 0 {
            let top = transform_size_of(max_full);
            let mut size = 1_usize;
            while size <= top {
                prepared.prepare_transform_size(size, batch_capacity)?;
                size = size.checked_mul(2).ok_or(ConfigError::GeometryOverflow {
                    context: "prepared transform size ladder",
                })?;
            }
        }
        Ok(prepared)
    }

    /// Cache the transform plan and work buffers for one full-product size.
    ///
    /// Plans are keyed by transform size and built for the scratch's lane
    /// capacity, so a later call for the same size at any batch within the
    /// capacity is a no-op. Safe to call for sizes a particular run never
    /// reaches. If the size is beyond this field's route, nothing is cached
    /// and `Ok(())` is returned — the execution path falls back (or errors,
    /// under a forced route) exactly as it would have.
    ///
    /// # Errors
    ///
    /// Returns [`ProductError::Config`] when the work buffers cannot be
    /// reserved.
    pub fn prepare_transform(
        &mut self,
        full_size: usize,
        batch: usize,
    ) -> Result<(), ProductError> {
        self.prepare_transform_size(transform_size_of(full_size), batch)
    }

    fn prepare_transform_size(
        &mut self,
        transform_size: usize,
        batch: usize,
    ) -> Result<(), ProductError> {
        if transform_size == 0 {
            return Ok(());
        }
        if self.plans.iter().any(|(size, _)| *size == transform_size) {
            return Ok(());
        }
        // Plans and buffers serve any lane count within the capacity: build
        // them once at the largest batch this scratch accepts.
        let lanes = batch.max(self.max_batch);
        let Some(plan) = F::build_plan(transform_size, lanes) else {
            return Ok(());
        };
        let (operands, products, conversion) = F::work_bytes(transform_size, lanes)?;
        let grow = |buffer: &mut Vec<u8>, required: usize, context: &'static str| {
            if buffer.len() < required {
                buffer
                    .try_reserve_exact(required - buffer.len())
                    .map_err(|_| ConfigError::AllocationFailed {
                        context,
                        elements: required,
                        element_size: 1,
                    })?;
                buffer.resize(required, 0);
            }
            Result::<(), ProductError>::Ok(())
        };
        grow(&mut self.operands, operands, "prepared transform operands")?;
        grow(&mut self.products, products, "prepared transform products")?;
        grow(
            &mut self.conversion,
            conversion,
            "prepared transform conversion",
        )?;
        self.plans.push((transform_size, plan));
        Ok(())
    }
}

impl<F: PolynomialField> core::fmt::Debug for ConvolutionScratch<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ConvolutionScratch")
            .field("max_left", &self.max_left)
            .field("max_right", &self.max_right)
            .field("max_batch", &self.max_batch)
            .field("cached_plans", &self.plans.len())
            .finish_non_exhaustive()
    }
}

/// The transform size covering a full product of `full_size` coefficients.
#[cfg(feature = "fft")]
fn transform_size_of(full_size: usize) -> usize {
    full_size.next_power_of_two()
}

#[cfg(not(feature = "fft"))]
fn transform_size_of(_full_size: usize) -> usize {
    0
}

/// The number of output rows a truncated product keeps.
fn output_rows(left_count: usize, right_count: usize, precision: usize) -> Option<usize> {
    if left_count == 0 || right_count == 0 || precision == 0 {
        return Some(0);
    }
    left_count
        .checked_add(right_count)?
        .checked_sub(1)
        .map(|full| full.min(precision))
}

/// Multiply two coefficient-major lane-row operands into `output`.
///
/// Inputs hold `left_count` and `right_count` coefficient rows of `batch`
/// lanes each — lane `l`'s coefficient `d` at `(d * batch + l) * F::BYTES`.
/// `output` receives `min(precision, left_count + right_count − 1)` rows in
/// the same layout, or zero rows when either operand or the precision is
/// empty. Batch zero is a valid empty geometry.
///
/// Route selection is the field's measured default; the forced entry point
/// is [`multiply_rows_route_into`] under `internals`. All lengths, the
/// scratch capacity, and the output size are validated before `output` is
/// touched. The constructor caches every transform size the prepared
/// maximum can request, so any execution inside the capacity — first or
/// later, transform or fallback — allocates nothing.
///
/// # Errors
///
/// Returns [`ProductError`] on a buffer-length or capacity violation, an
/// address-space overflow, or — under a forced transform route — an
/// unsupported transform size.
#[allow(clippy::too_many_arguments)]
pub fn multiply_rows_into<F: PolynomialField>(
    output: &mut [u8],
    left: &[u8],
    left_count: usize,
    right: &[u8],
    right_count: usize,
    batch: usize,
    precision: usize,
    scratch: &mut ConvolutionScratch<F>,
) -> Result<(), ProductError> {
    multiply_rows_route(
        left,
        left_count,
        right,
        right_count,
        batch,
        precision,
        RouteSelection::Auto,
        scratch,
        output,
    )
}

/// Forced-route form of [`multiply_rows_into`].
///
/// Unstable surface under the `internals` feature, for benchmark panels and
/// differential tests. `Transform` on a field (or size) without a transform
/// route, or without a prepared plan, is an error rather than a silent
/// fallback.
///
/// # Errors
///
/// Returns [`ProductError`] exactly as [`multiply_rows_into`], plus
/// [`ConfigError::ScratchTooSmall`] when a forced transform size has no
/// cached plan or workspace.
#[cfg_attr(not(feature = "internals"), expect(dead_code))]
#[allow(clippy::too_many_arguments)]
pub fn multiply_rows_route_into<F: PolynomialField>(
    output: &mut [u8],
    left: &[u8],
    left_count: usize,
    right: &[u8],
    right_count: usize,
    batch: usize,
    precision: usize,
    route: ProductRoute,
    scratch: &mut ConvolutionScratch<F>,
) -> Result<(), ProductError> {
    let selection = match route {
        ProductRoute::Auto => RouteSelection::Auto,
        ProductRoute::Schoolbook => RouteSelection::Schoolbook,
        ProductRoute::Karatsuba => RouteSelection::Karatsuba,
        ProductRoute::Transform => RouteSelection::Transform,
    };
    multiply_rows_route(
        left,
        left_count,
        right,
        right_count,
        batch,
        precision,
        selection,
        scratch,
        output,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "internals"), allow(dead_code))]
enum RouteSelection {
    Auto,
    Schoolbook,
    #[cfg_attr(not(feature = "internals"), allow(dead_code))]
    Karatsuba,
    Transform,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn multiply_rows_route<F: PolynomialField>(
    left: &[u8],
    left_count: usize,
    right: &[u8],
    right_count: usize,
    batch: usize,
    precision: usize,
    selection: RouteSelection,
    scratch: &mut ConvolutionScratch<F>,
    output: &mut [u8],
) -> Result<(), ProductError> {
    // Structural validation, in full, before any mutation.
    let expected = |count: usize, context: &'static str| -> Result<usize, ProductError> {
        checked_product(context, count, batch)
            .and_then(|lanes| checked_product(context, lanes, F::BYTES))
            .map_err(ProductError::Config)
    };
    let left_bytes = expected(left_count, "prepared left operand rows")?;
    let right_bytes = expected(right_count, "prepared right operand rows")?;
    if left.len() != left_bytes {
        return Err(ProductError::Config(ConfigError::BufferLength {
            context: "prepared left operand rows",
            expected: left_bytes,
            actual: left.len(),
        }));
    }
    if right.len() != right_bytes {
        return Err(ProductError::Config(ConfigError::BufferLength {
            context: "prepared right operand rows",
            expected: right_bytes,
            actual: right.len(),
        }));
    }
    let Some(out_rows) = output_rows(left_count, right_count, precision) else {
        return Err(ProductError::Config(ConfigError::GeometryOverflow {
            context: "prepared product rows",
        }));
    };
    let output_bytes = expected(out_rows, "prepared output rows")?;
    if output.len() != output_bytes {
        return Err(ProductError::Config(ConfigError::BufferLength {
            context: "prepared output rows",
            expected: output_bytes,
            actual: output.len(),
        }));
    }
    if left_count > scratch.max_left || right_count > scratch.max_right {
        return Err(ProductError::Config(ConfigError::ScratchTooSmall {
            context: "prepared operand coefficient capacity",
            required: left_count.max(right_count),
            available: scratch.max_left.max(scratch.max_right),
        }));
    }
    if batch > scratch.max_batch {
        return Err(ProductError::Config(ConfigError::ScratchTooSmall {
            context: "prepared lane capacity",
            required: batch,
            available: scratch.max_batch,
        }));
    }
    if out_rows == 0 || batch == 0 {
        return Ok(());
    }
    let full = left_count + right_count - 1;

    // Route decision.
    let use_transform = match selection {
        RouteSelection::Schoolbook | RouteSelection::Karatsuba => false,
        RouteSelection::Transform => true,
        RouteSelection::Auto => auto_uses_transform::<F>(left_count, right_count, batch, full),
    };

    if use_transform {
        let transform_size = transform_size_of(full);
        // A cache miss under Auto prepares the plan once and retries; a
        // forced route without a plan, or a size beyond this field's
        // transform route, is an error rather than a silent fallback.
        let mut plan_slot = scratch
            .plans
            .iter()
            .position(|(size, _)| *size == transform_size);
        if plan_slot.is_none() && selection == RouteSelection::Auto {
            scratch.prepare_transform_size(transform_size, batch)?;
            plan_slot = scratch
                .plans
                .iter()
                .position(|(size, _)| *size == transform_size);
        }
        if let Some(index) = plan_slot {
            let (operand_bytes, product_bytes, conversion_bytes) =
                F::work_bytes(transform_size, batch)?;
            let mut plan = scratch.plans.swap_remove(index).1;
            let mut operands = core::mem::take(&mut scratch.operands);
            let mut products = core::mem::take(&mut scratch.products);
            let mut conversion = core::mem::take(&mut scratch.conversion);
            let result = F::transform_rows(TransformRows {
                left,
                left_count,
                right,
                right_count,
                batch,
                precision,
                plan: &mut plan,
                operands: &mut operands[..operand_bytes],
                products: &mut products[..product_bytes],
                conversion: &mut conversion[..conversion_bytes],
                output,
            });
            scratch.operands = operands;
            scratch.products = products;
            scratch.conversion = conversion;
            scratch.plans.push((transform_size, plan));
            return result;
        }
        if selection == RouteSelection::Transform {
            return Err(ProductError::Config(ConfigError::ScratchTooSmall {
                context: "prepared transform plan",
                required: full,
                available: 0,
            }));
        }
        // Auto with no feasible plan for this size: fall back below.
    }

    // Fallback: schoolbook below the measured crossover, Karatsuba above,
    // lane by lane through the corrected packed kernels.
    let schoolbook = selection == RouteSelection::Schoolbook
        || left_count.min(right_count) < KARATSUBA_CROSSOVER;
    debug_assert!(left_count * F::BYTES <= scratch.lane_left.len());
    debug_assert!(right_count * F::BYTES <= scratch.lane_right.len());
    debug_assert!(full * F::BYTES <= scratch.lane_product.len());
    let full_bytes = full * F::BYTES;
    for lane in 0..batch {
        gather_lane::<F>(left, left_count, batch, lane, &mut scratch.lane_left);
        gather_lane::<F>(right, right_count, batch, lane, &mut scratch.lane_right);
        scratch.lane_product[..full_bytes].fill(0);
        if schoolbook {
            schoolbook_into::<F>(
                &mut scratch.lane_product[..full_bytes],
                &scratch.lane_left[..left_count * F::BYTES],
                &scratch.lane_right[..right_count * F::BYTES],
            );
        } else {
            karatsuba_into::<F>(
                &mut scratch.lane_product[..full_bytes],
                &scratch.lane_left[..left_count * F::BYTES],
                &scratch.lane_right[..right_count * F::BYTES],
                0,
                &mut scratch.karatsuba,
            );
        }
        scatter_lane::<F>(&scratch.lane_product, out_rows, batch, lane, output);
    }
    Ok(())
}

/// The measured Auto rule: binary fields keep their AFFT crossovers;
/// Goldilocks selects the NTT by shorter operand and batch width. Other
/// transform routes stay on Karatsuba.
#[cfg(feature = "fft")]
fn auto_uses_transform<F: PolynomialField>(
    left_count: usize,
    right_count: usize,
    batch: usize,
    full: usize,
) -> bool {
    match F::ROUTE {
        Route::Afft => {
            crate::cost::select_product(crate::cost::ProductCostKey {
                left_coefficients: left_count,
                right_coefficients: right_count,
                output_coefficients: full,
                batch,
                field_order: F::ORDER,
                backend: crate::cost::BackendClass::detect::<F>(),
            }) == crate::cost::ProductBackend::Afft
        }
        Route::Ntt => {
            F::AUTO_NTT
                && crate::cost::select_ntt_product(
                    left_count,
                    right_count,
                    crate::cost::ntt_prepared_product_crossover(batch),
                ) == crate::cost::NttProductBackend::Ntt
        }
        Route::Embedded | Route::None => false,
    }
}

/// Return a one-shot automatic NTT product for the public ring API.
///
/// The result is present when the field is `Goldilocks`, the shorter operand
/// reaches [`crate::cost::NTT_ONESHOT_PRODUCT_CROSSOVER`], a transform plan is
/// available, and the engine accepts the internally generated buffers. Every
/// other case leaves route selection to the caller.
pub(crate) fn oneshot_ntt_product<F: FieldKernels>(
    left: &[u8],
    left_count: usize,
    right: &[u8],
    right_count: usize,
) -> Option<Vec<u8>> {
    #[cfg(feature = "fft")]
    {
        if TypeId::of::<F>() != TypeId::of::<Goldilocks>() || left_count == 0 || right_count == 0 {
            return None;
        }
        if crate::cost::select_ntt_product(
            left_count,
            right_count,
            crate::cost::NTT_ONESHOT_PRODUCT_CROSSOVER,
        ) != crate::cost::NttProductBackend::Ntt
        {
            return None;
        }
        goldilocks_oneshot_convolve(left, left_count, right, right_count)
    }
    #[cfg(not(feature = "fft"))]
    {
        let _ = (left, left_count, right, right_count);
        None
    }
}

/// Run the Goldilocks NTT engine once over freshly allocated work buffers.
#[cfg(feature = "fft")]
fn goldilocks_oneshot_convolve(
    left: &[u8],
    left_count: usize,
    right: &[u8],
    right_count: usize,
) -> Option<Vec<u8>> {
    let full = left_count.checked_add(right_count)?.checked_sub(1)?;
    let size = full.checked_next_power_of_two()?;
    let mut route = NttRoute::<Goldilocks>::new(size, Goldilocks::BYTES)?;
    let (operands_bytes, products_bytes, _) =
        <Goldilocks as ConvolutionDomain>::work_bytes(size, 1).ok()?;
    let reserve = |bytes: usize| -> Option<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(bytes).ok()?;
        buffer.resize(bytes, 0);
        Some(buffer)
    };
    let mut operands = reserve(operands_bytes)?;
    let mut products = reserve(products_bytes)?;
    let output_bytes =
        checked_product("polynomial product coefficients", full, Goldilocks::BYTES).ok()?;
    let mut output = reserve(output_bytes)?;
    ntt_rows_convolve::<Goldilocks>(
        left,
        left_count,
        right,
        right_count,
        1,
        full,
        &mut route,
        &mut operands,
        &mut products,
        &mut output,
    )
    .ok()?;
    Some(output)
}
#[cfg(not(feature = "fft"))]
// The type parameter is unused here but keeps the call sites uniform with
// the `fft` twin above.
#[allow(clippy::extra_unused_type_parameters)]
fn auto_uses_transform<F: PolynomialField>(
    _left_count: usize,
    _right_count: usize,
    _batch: usize,
    _full: usize,
) -> bool {
    false
}

/// Copy lane `lane`'s coefficients out of coefficient-major rows.
fn gather_lane<F: FieldKernels>(
    rows: &[u8],
    count: usize,
    batch: usize,
    lane: usize,
    out: &mut [u8],
) {
    for degree in 0..count {
        let source = (degree * batch + lane) * F::BYTES;
        let destination = degree * F::BYTES;
        out[destination..destination + F::BYTES].copy_from_slice(&rows[source..source + F::BYTES]);
    }
}

/// Copy lane `lane`'s first `count` coefficients into coefficient-major rows.
fn scatter_lane<F: FieldKernels>(
    lanes: &[u8],
    count: usize,
    batch: usize,
    lane: usize,
    out: &mut [u8],
) {
    for degree in 0..count {
        let source = degree * F::BYTES;
        let destination = (degree * batch + lane) * F::BYTES;
        out[destination..destination + F::BYTES].copy_from_slice(&lanes[source..source + F::BYTES]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_free_domain_needs_no_work_buffers() {
        assert_eq!(
            <fgf::Gf8D as ConvolutionDomain>::work_bytes(8, 2),
            Ok((0, 0, 0))
        );
        let mut scratch = ConvolutionScratch::<fgf::Gf8D>::new(0, 0, 1).unwrap();
        scratch.prepare_transform_size(0, 1).unwrap();
    }

    #[test]
    #[should_panic(expected = "this field has no transform route")]
    fn transform_free_domain_rejects_transform_execution() {
        let mut plan = NoPlan::<fgf::Gf8D>::default();
        let mut empty = [];
        <fgf::Gf8D as ConvolutionDomain>::transform_rows(TransformRows {
            left: &[],
            left_count: 0,
            right: &[],
            right_count: 0,
            batch: 0,
            precision: 0,
            plan: &mut plan,
            operands: &mut empty,
            products: &mut [],
            conversion: &mut [],
            output: &mut [],
        })
        .unwrap();
    }
}
