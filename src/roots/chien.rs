//! Classical Chien search: root-finding by domain scan.
//!
//! The locator `Λ(x) = Σ_j c_j x^j` is evaluated at every element of the
//! field by an incremental update: stepping the evaluation point
//! `γ^i → γ^(i+1)` multiplies the `j`-th running term by `γ^j`, so each
//! successive evaluation costs one packed elementwise multiply plus a byte
//! fold rather than a fresh Horner chain. This is the cheap path for the
//! small locators of bounded-distance decoding; the equal-degree route in
//! [`super::equal_degree`] wins for larger-degree factor extraction, and
//! [`crate::cost::select_base_roots`] picks between them. Both backends
//! produce the same root set in the same canonical order.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::ops;

use crate::error::{ConfigError, RootError};
use crate::poly::Polynomial;

use super::BaseFieldRoots;
use super::equal_degree::element_key;

/// Caller-owned reusable storage for the Chien scan.
///
/// The running term vector, its successor buffer, and the fixed per-lane
/// step factors are drawn from these buffers, so a warmed scan over a
/// changed locator performs no heap allocation.
#[derive(Debug)]
pub struct ChienScratch<F: FieldKernels> {
    state: Vec<u8>,
    state_next: Vec<u8>,
    step: Vec<u8>,
    field: core::marker::PhantomData<F>,
}

impl<F: FieldKernels> ChienScratch<F> {
    /// Construct empty reusable Chien scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: Vec::new(),
            state_next: Vec::new(),
            step: Vec::new(),
            field: core::marker::PhantomData,
        }
    }
}

impl<F: FieldKernels> Default for ChienScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Return every distinct root in the polynomial's coefficient field by the
/// Chien scan.
///
/// # Errors
///
/// Returns [`RootError`] when the field is not a supported binary extension
/// field or storage cannot be reserved.
pub fn chien_roots<F: FieldKernels>(
    polynomial: &Polynomial<F>,
) -> Result<BaseFieldRoots<F::Elem>, RootError> {
    let mut scratch = ChienScratch::new();
    let mut roots = Vec::new();
    if chien_roots_into(polynomial, &mut scratch, &mut roots)? {
        Ok(BaseFieldRoots::All)
    } else {
        Ok(BaseFieldRoots::Finite(roots))
    }
}

/// Write every distinct root of `polynomial` into `roots`, reusing
/// `scratch`. Returns `true` when every field element is a root (the zero
/// polynomial); otherwise `roots` holds the root set in the canonical
/// little-endian element-key order — the same frozen order
/// [`super::equal_degree::base_field_roots_into`] produces, so the two
/// backends are drop-in substitutes.
///
/// The scan visits all `|F|` elements, so it is intended for small fields
/// and short locators; [`crate::cost::select_base_roots`] encodes the
/// measured crossover against the equal-degree backend.
///
/// # Errors
///
/// Returns [`RootError`] when the field is not a supported binary extension
/// field or storage cannot be reserved.
pub fn chien_roots_into<F: FieldKernels>(
    polynomial: &Polynomial<F>,
    scratch: &mut ChienScratch<F>,
    roots: &mut Vec<F::Elem>,
) -> Result<bool, RootError> {
    if !F::ORDER.is_power_of_two() || F::BYTES == 0 || F::BYTES > 16 {
        return Err(RootError::UnsupportedField {
            field_order: F::ORDER,
            element_bytes: F::BYTES,
        });
    }
    roots.clear();
    let coefficient_count = polynomial.coefficient_count();
    if coefficient_count == 0 {
        return Ok(true);
    }
    if coefficient_count == 1 {
        return Ok(false);
    }

    let degree = coefficient_count - 1;
    let capacity = degree.min(usize::try_from(F::ORDER).unwrap_or(usize::MAX));
    if roots.capacity() < capacity {
        roots
            .try_reserve(capacity - roots.capacity())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "Chien roots",
                elements: capacity,
                element_size: core::mem::size_of::<F::Elem>(),
            })?;
    }

    // Point zero is a root exactly when the constant term vanishes.
    if polynomial.coefficient(0).is_zero() {
        roots.push(F::Elem::ZERO);
    }

    let byte_len = coefficient_count * F::BYTES;
    ensure_len(&mut scratch.state, byte_len, "Chien state")?;
    ensure_len(&mut scratch.state_next, byte_len, "Chien state")?;
    ensure_len(&mut scratch.step, byte_len, "Chien steps")?;
    scratch.state[..byte_len].copy_from_slice(&polynomial.as_packed()[..byte_len]);
    scratch.state_next[..byte_len].copy_from_slice(&polynomial.as_packed()[..byte_len]);
    for lane in 0..coefficient_count {
        let factor = F::GENERATOR.pow(lane as u64);
        let start = lane * F::BYTES;
        F::write(&mut scratch.step[start..start + F::BYTES], factor);
    }

    // `state_j = c_j * point^j` for point = 1 = γ^0 initially; each
    // iteration evaluates the current point, then multiplies lane j by γ^j.
    let mut point = F::Elem::ONE;
    let nonzero_points = F::ORDER - 1;
    for _ in 0..nonzero_points {
        if packed_xor_sum_is_zero(&scratch.state[..byte_len], F::BYTES) {
            roots.push(point);
        }
        point = point.mul(F::GENERATOR);
        let (state, state_next) = (
            &mut scratch.state[..byte_len],
            &mut scratch.state_next[..byte_len],
        );
        ops::mul_elementwise::<F>(state_next, state, &scratch.step[..byte_len]);
        scratch.state[..byte_len].copy_from_slice(&scratch.state_next[..byte_len]);
    }

    roots.sort_by_key(|root| element_key::<F>(*root));
    roots.dedup();
    if roots.len() > degree {
        return Err(RootError::FactorizationInvariant {
            reason: "the Chien scan found more roots than the polynomial degree",
        });
    }
    Ok(false)
}

/// Whether the XOR sum of the packed field elements is zero.
///
/// Field addition is bytewise XOR, so the evaluation at the current point —
/// the sum of the running terms — is the bytewise fold of the buffer; the
/// point is a root exactly when that fold is zero.
fn packed_xor_sum_is_zero(bytes: &[u8], element_bytes: usize) -> bool {
    let mut fold = [0_u8; 16];
    for chunk in bytes.chunks_exact(element_bytes) {
        for (accumulator, byte) in fold.iter_mut().zip(chunk) {
            *accumulator ^= byte;
        }
    }
    fold.iter().all(|byte| *byte == 0)
}

fn ensure_len(
    values: &mut Vec<u8>,
    required: usize,
    context: &'static str,
) -> Result<(), RootError> {
    if required > values.len() {
        values
            .try_reserve_exact(required - values.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context,
                elements: required,
                element_size: 1,
            })?;
        values.resize(required, 0);
    }
    Ok(())
}
