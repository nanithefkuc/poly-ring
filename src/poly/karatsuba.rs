//! The Karatsuba middle tier of the product ladder.
//!
//! Between the schoolbook base and the AFFT batched product, operands are too
//! large for the quadratic convolution but too small to amortize a transform.
//! Karatsuba splits each operand at the half-degree, forms three half-size
//! products, and recombines as
//! `z0 + X^h·(z1 − z0 − z2) + X^{2h}·z2` with
//! `z1 = (a0 + a1)(b0 + b1)`.
//!
//! The signs are carried explicitly through `fgf`'s field addition and
//! subtraction, so the tier is exact in every characteristic. In
//! characteristic two both collapse to XOR and the results are bit-identical
//! to the schoolbook tier, as before.
//!
//! The recursion operates directly on packed coefficient byte slices with
//! depth-indexed scratch slots, so the tier costs field work and no
//! per-level allocation.

use alloc::vec::Vec;

use fgf::field::Elem as _;
use fgf::kernel::FieldKernels;
use fgf::ops;

use crate::error::{ConfigError, PolynomialError};

use super::dense::Polynomial;

/// Operand coefficient count at or above which [`Polynomial::multiply`]
/// dispatches to Karatsuba instead of schoolbook. Measured on GF(2^8)
/// (`BENCHMARKS.md`): schoolbook wins through 1024 coefficients (1.2-2x),
/// Karatsuba wins from 2048 (0.64x at 2048, 0.34x at 4096, 0.56x at 6144,
/// with a marginal 1.09x anomaly at 3072).
pub const KARATSUBA_CROSSOVER: usize = 2048;

/// Recursion base: below this shorter-operand size the recursion falls to
/// the packed schoolbook convolution. Measured; see `BENCHMARKS.md`.
pub(crate) const KARATSUBA_BASE: usize = 48;

/// One depth level of Karatsuba scratch: two split-sum operands and three
/// product slots, all byte buffers.
#[derive(Debug)]
pub(crate) struct KaratsubaScratch {
    sums: Vec<Vec<u8>>,
    products: Vec<Vec<u8>>,
}

impl KaratsubaScratch {
    pub(crate) const fn new() -> Self {
        Self {
            sums: Vec::new(),
            products: Vec::new(),
        }
    }

    /// Take the `index`-th sum slot, sized to at least `bytes` and zeroed
    /// over the active region.
    pub(crate) fn take_sum(&mut self, index: usize, bytes: usize) -> Vec<u8> {
        while self.sums.len() <= index {
            self.sums.push(Vec::new());
        }
        let mut slot = core::mem::take(&mut self.sums[index]);
        if slot.len() < bytes {
            slot.resize(bytes, 0);
        } else {
            slot.truncate(bytes);
        }
        slot.fill(0);
        slot
    }

    /// Take the `index`-th product slot, sized to exactly `bytes` and zeroed.
    pub(crate) fn take_product(&mut self, index: usize, bytes: usize) -> Vec<u8> {
        while self.products.len() <= index {
            self.products.push(Vec::new());
        }
        let mut slot = core::mem::take(&mut self.products[index]);
        if slot.len() < bytes {
            slot.resize(bytes, 0);
        } else {
            slot.truncate(bytes);
        }
        slot.fill(0);
        slot
    }
}

/// Return the Karatsuba product of two polynomials.
///
/// Equal to the schoolbook product coefficient for coefficient; only the
/// cost differs. A zero operand gives the zero polynomial. The recursion
/// bottoms out in the packed schoolbook convolution once the shorter operand
/// falls below its measured base size.
///
/// # Errors
///
/// Returns [`PolynomialError::Config`] when the product buffer cannot be
/// reserved.
///
/// # Panics
///
/// The coefficient-aligned product expectation holds for the byte-exact
/// recombination this function computes itself.
pub fn karatsuba_multiply<F: FieldKernels>(
    a: &Polynomial<F>,
    b: &Polynomial<F>,
) -> Result<Polynomial<F>, PolynomialError> {
    let left = a.coefficient_count();
    let right = b.coefficient_count();
    if left == 0 || right == 0 {
        return Ok(Polynomial::zero());
    }
    let output_count = left
        .checked_add(right)
        .and_then(|sum| sum.checked_sub(1))
        .ok_or(ConfigError::GeometryOverflow {
            context: "polynomial product coefficients",
        })?;
    let mut destination = Vec::new();
    destination
        .try_reserve_exact(output_count * F::BYTES)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "Karatsuba product",
            elements: output_count,
            element_size: F::BYTES,
        })?;
    destination.resize(output_count * F::BYTES, 0);
    let mut scratch = KaratsubaScratch::new();
    karatsuba_into::<F>(
        &mut destination,
        a.as_packed(),
        b.as_packed(),
        0,
        &mut scratch,
    );
    Ok(Polynomial::from_packed(destination).expect("coefficient-aligned product"))
}

/// Accumulate the packed schoolbook convolution `a · b` into zeroed `dst`.
pub(crate) fn schoolbook_into<F: FieldKernels>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len() + F::BYTES, a.len() + b.len());
    for index in 0..b.len() / F::BYTES {
        let scale = F::read(&b[index * F::BYTES..(index + 1) * F::BYTES]);
        if scale.is_zero() {
            continue;
        }
        let offset = index * F::BYTES;
        let target = &mut dst[offset..offset + a.len()];
        ops::mul_add::<F>(target, scale, a);
    }
}

/// Add `source` into `destination` at `offset`, element for element.
///
/// Only the common full-element overlap is touched: the recombination
/// deliberately writes products whose tails fall past the truncated
/// destination, and those coefficients cannot affect an in-range result.
fn add_into<F: FieldKernels>(destination: &mut [u8], offset: usize, source: &[u8]) {
    let Some(window) = destination.get_mut(offset..) else {
        return;
    };
    let overlap = window.len().min(source.len());
    if overlap != 0 {
        ops::add_assign::<F>(&mut window[..overlap], &source[..overlap]);
    }
}

/// Subtract `source` from `destination` at `offset`, element for element.
fn sub_into<F: FieldKernels>(destination: &mut [u8], offset: usize, source: &[u8]) {
    let Some(window) = destination.get_mut(offset..) else {
        return;
    };
    let overlap = window.len().min(source.len());
    if overlap != 0 {
        ops::sub_assign::<F>(&mut window[..overlap], &source[..overlap]);
    }
}

/// Write the Karatsuba product of packed `a` and `b` into zeroed `dst`.
pub(crate) fn karatsuba_into<F: FieldKernels>(
    dst: &mut [u8],
    a: &[u8],
    b: &[u8],
    depth: usize,
    scratch: &mut KaratsubaScratch,
) {
    let a_count = a.len() / F::BYTES;
    let b_count = b.len() / F::BYTES;
    if a_count == 0 || b_count == 0 {
        // The empty product: `dst` is already zeroed by its slot owner.
        return;
    }
    if a_count.min(b_count) < KARATSUBA_BASE {
        schoolbook_into::<F>(dst, a, b);
        return;
    }

    let split = a_count.max(b_count).div_ceil(2);
    let split_bytes = split * F::BYTES;
    let a_split = split_bytes.min(a.len());
    let b_split = split_bytes.min(b.len());
    let (a_low, a_high) = a.split_at(a_split);
    let (b_low, b_high) = b.split_at(b_split);

    // z0 and z2 recurse on the external operand slices. The empty product
    // of a vanishing high part is the zero-length buffer.
    let low_len = (a_low.len() + b_low.len()).saturating_sub(F::BYTES);
    let high_len = (a_high.len() + b_high.len()).saturating_sub(F::BYTES);
    let mut z0 = scratch.take_product(3 * depth, low_len);
    karatsuba_into::<F>(&mut z0, a_low, b_low, depth + 1, scratch);
    let mut z2 = scratch.take_product(3 * depth + 1, high_len);
    karatsuba_into::<F>(&mut z2, a_high, b_high, depth + 1, scratch);

    // Split sums, zero-padded to `split` coefficients each.
    let mut sum_a = scratch.take_sum(2 * depth, split_bytes);
    sum_a[..a_low.len()].copy_from_slice(a_low);
    add_elementwise::<F>(&mut sum_a, a_high);
    let mut sum_b = scratch.take_sum(2 * depth + 1, split_bytes);
    sum_b[..b_low.len()].copy_from_slice(b_low);
    add_elementwise::<F>(&mut sum_b, b_high);

    // middle = (a_low + a_high)(b_low + b_high), or empty when either sum
    // vanished (a pure power-of-two split with an empty high part).
    let middle_len = if sum_a.len() + sum_b.len() >= F::BYTES {
        sum_a.len() + sum_b.len() - F::BYTES
    } else {
        0
    };
    let mut middle = scratch.take_product(3 * depth + 2, middle_len);
    let sums_empty = sum_a.iter().all(|byte| *byte == 0) || sum_b.iter().all(|byte| *byte == 0);
    if !sums_empty {
        karatsuba_into::<F>(&mut middle, &sum_a, &sum_b, depth + 1, scratch);
    }

    // result = z0 + X^s·(middle − z0 − z2) + X^{2s}·z2, every term read from
    // an owned local so no pass aliases `dst`. Out-of-range tails are
    // truncated per element, which never affects in-range results.
    add_into::<F>(dst, 0, &z0);
    add_into::<F>(dst, split_bytes, &middle);
    sub_into::<F>(dst, split_bytes, &z0);
    sub_into::<F>(dst, split_bytes, &z2);
    add_into::<F>(dst, 2 * split_bytes, &z2);

    scratch.products[3 * depth] = z0;
    scratch.products[3 * depth + 1] = z2;
    scratch.products[3 * depth + 2] = middle;
    scratch.sums[2 * depth] = sum_a;
    scratch.sums[2 * depth + 1] = sum_b;
}

/// Add one packed operand into another (in-place field addition), matching
/// element boundaries from the start.
fn add_elementwise<F: FieldKernels>(destination: &mut [u8], source: &[u8]) {
    // Only the common full-element prefix: operands may differ in length,
    // and the tail of the longer one is left untouched.
    let overlap = destination.len().min(source.len());
    if overlap != 0 {
        ops::add_assign::<F>(&mut destination[..overlap], &source[..overlap]);
    }
}
