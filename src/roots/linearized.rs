//! Standalone solver for 2-linearized and affine polynomials.
//!
//! A 2-linearized polynomial `L(X) = Σ_k a_k X^(2^k)` is GF(2)-linear: its
//! root set is an affine GF(2)-subspace of the field, found directly by
//! solving the associated linear system instead of scanning or splitting.
//! The solver evaluates `L` on the power basis `1, γ, …, γ^(m-1)`, reduces
//! the images to GF(2) coordinate vectors, and eliminates over `u128`
//! bitmask rows — no matrix type is constructed; the bitmasks are the
//! solver's private scratch.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use crate::error::{ConfigError, RootError};
use crate::poly::Polynomial;

use super::equal_degree::element_key;

/// Return every root of the affine 2-linearized polynomial
/// `L(X) + affine`, where `L` is given in ordinary polynomial form with
/// nonzero coefficients only at power-of-two degrees and a zero constant
/// term.
///
/// The roots are returned in the canonical little-endian element-key order —
/// the same frozen order the Chien and equal-degree backends produce, and
/// the solver agrees with [`super::chien::chien_roots`] on every affine
/// input.
///
/// The root set is an affine GF(2)-subspace of dimension at most
/// `log2(|F|)`. When `L` is the zero polynomial and `affine` is zero, every
/// field element is a root and the full finite list is returned — the
/// solver is intended for the small fields and short linearized degrees the
/// decoders use, so callers over wide fields keep the linearized degree
/// small.
///
/// # Errors
///
/// Returns [`RootError::NotLinearized`] when a coefficient sits at degree
/// zero or at a non-power-of-two degree, and [`RootError`] when the field is
/// unsupported or storage cannot be reserved.
///
/// # Panics
///
/// The internal span expectations hold for a validated binary field basis,
/// which every `fgf` field provides.
pub fn linearized_roots<F: FieldKernels>(
    linearized: &Polynomial<F>,
    affine: F::Elem,
) -> Result<Vec<F::Elem>, RootError> {
    if !F::ORDER.is_power_of_two() || F::BYTES == 0 || F::BYTES > 16 {
        return Err(RootError::UnsupportedField {
            field_order: F::ORDER,
            element_bytes: F::BYTES,
        });
    }
    let extension_degree = F::ORDER.trailing_zeros() as usize;
    for degree in 0..linearized.coefficient_count() {
        if !linearized.coefficient(degree).is_zero() && !degree.is_power_of_two() {
            return Err(RootError::NotLinearized { degree });
        }
    }
    if linearized.is_zero() {
        if affine.is_zero() {
            return enumerate_field::<F>();
        }
        return Ok(Vec::new());
    }

    // Power basis of GF(2^m) over GF(2): 1, γ, ..., γ^(m-1).
    let basis: Vec<F::Elem> = (0..extension_degree)
        .map(|exponent| F::GENERATOR.pow(exponent as u64))
        .collect();

    // Reduced table for coordinate extraction: pairs (element key,
    // combination mask over the power basis) with distinct pivot bits.
    let mut reduction: Vec<(u128, u128)> = Vec::new();
    for (index, &basis_element) in basis.iter().enumerate() {
        reduce_element(
            &mut reduction,
            element_key::<F>(basis_element),
            1_u128 << index,
        );
    }

    // Column j of the GF(2) system is the coordinate vector of L(basis[j]);
    // row i masks the solution coefficients b_0..b_{m-1} whose image has bit
    // i set, with the affine constant as right-hand side.
    let affine_coordinates = coordinates_of(&reduction, element_key::<F>(affine))
        .expect("every field element lies in the power-basis span");
    let mut rows: Vec<(u128, u128)> = Vec::with_capacity(extension_degree);
    for row_index in 0..extension_degree {
        let mut mask = 0_u128;
        for (column, &basis_element) in basis.iter().enumerate() {
            let image = evaluate_linearized(linearized, basis_element);
            let coordinates = coordinates_of(&reduction, element_key::<F>(image))
                .expect("every field element lies in the power-basis span");
            if coordinates >> row_index & 1 != 0 {
                mask |= 1_u128 << column;
            }
        }
        let rhs = affine_coordinates >> row_index & 1;
        rows.push((mask, rhs));
    }

    eliminate(&mut rows);
    // An inconsistent system means the affine constant lies outside the
    // image of L: the equation has no solution and the root set is empty.
    let Some(particular) = back_substitute(&rows) else {
        return Ok(Vec::new());
    };

    // Enumerate the affine root space particular ⊕ ker(L).
    let kernel = kernel_basis(&rows, extension_degree);
    let combination_count = 1_usize << kernel.len();
    let mut roots = Vec::new();
    roots.try_reserve_exact(combination_count).map_err(|_| {
        RootError::from(ConfigError::AllocationFailed {
            context: "linearized roots",
            elements: combination_count,
            element_size: core::mem::size_of::<F::Elem>(),
        })
    })?;
    for ordinal in 0..combination_count {
        let mut solution = particular;
        let mut digits = ordinal;
        for &basis_vector in &kernel {
            if digits & 1 != 0 {
                solution ^= basis_vector;
            }
            digits >>= 1;
        }
        roots.push(element_from_coordinates::<F>(&basis, solution));
    }

    roots.sort_by_key(|root| element_key::<F>(*root));
    roots.dedup();
    Ok(roots)
}

/// Every field element in canonical key order.
fn enumerate_field<F: FieldKernels>() -> Result<Vec<F::Elem>, RootError> {
    let order = usize::try_from(F::ORDER).map_err(|_| {
        RootError::from(ConfigError::GeometryOverflow {
            context: "linearized roots of the zero polynomial",
        })
    })?;
    let mut all = Vec::new();
    all.try_reserve_exact(order).map_err(|_| {
        RootError::from(ConfigError::AllocationFailed {
            context: "linearized roots of the zero polynomial",
            elements: order,
            element_size: core::mem::size_of::<F::Elem>(),
        })
    })?;
    for key in 0..F::ORDER {
        all.push(element_from_key::<F>(key));
    }
    Ok(all)
}

/// Evaluate `L(point)` for a 2-linearized polynomial.
fn evaluate_linearized<F: FieldKernels>(linearized: &Polynomial<F>, point: F::Elem) -> F::Elem {
    let mut value = F::Elem::ZERO;
    let mut power = point;
    let mut degree = 1_usize;
    while degree < linearized.coefficient_count() {
        let coefficient = linearized.coefficient(degree);
        if !coefficient.is_zero() {
            value = value.add(coefficient.mul(power));
        }
        power = power.square();
        degree *= 2;
    }
    value
}

/// Reduce `key` against the reduced table, accumulating the combination
/// mask; pushes a new pivot row when the key does not reduce to zero.
fn reduce_element(reduction: &mut Vec<(u128, u128)>, key: u128, mask: u128) {
    let mut key = key;
    let mut mask = mask;
    while let Some(pivot) = key.checked_ilog2() {
        let Some(&(row_key, row_mask)) = reduction
            .iter()
            .find(|&&(row_key, _)| row_key.checked_ilog2() == Some(pivot))
        else {
            reduction.push((key, mask));
            return;
        };
        key ^= row_key;
        mask ^= row_mask;
    }
}

/// The combination mask expressing the element with key `key` in the power
/// basis, or `None` when the key does not reduce to zero.
fn coordinates_of(reduction: &[(u128, u128)], key: u128) -> Option<u128> {
    let mut key = key;
    let mut mask = 0_u128;
    while let Some(pivot) = key.checked_ilog2() {
        let &(row_key, row_mask) = reduction
            .iter()
            .find(|&&(row_key, _)| row_key.checked_ilog2() == Some(pivot))?;
        key ^= row_key;
        mask ^= row_mask;
    }
    Some(mask)
}

/// In-place GF(2) Gaussian elimination, pivoting from the high bit down.
fn eliminate(rows: &mut [(u128, u128)]) {
    let mut pivot_row = 0_usize;
    for pivot_bit in (0..u128::BITS).rev() {
        let Some(offset) = rows[pivot_row..]
            .iter()
            .position(|&(mask, _)| mask >> pivot_bit & 1 != 0)
        else {
            continue;
        };
        let index = pivot_row + offset;
        rows.swap(pivot_row, index);
        for other in (pivot_row + 1)..rows.len() {
            if rows[other].0 >> pivot_bit & 1 != 0 {
                rows[other].0 ^= rows[pivot_row].0;
                rows[other].1 ^= rows[pivot_row].1;
            }
        }
        pivot_row += 1;
        if pivot_row == rows.len() {
            break;
        }
    }
}

/// A particular solution of the eliminated system with free variables set
/// to zero, or `None` when inconsistent (a zero row with a set right-hand
/// side).
fn back_substitute(rows: &[(u128, u128)]) -> Option<u128> {
    let mut solution = 0_u128;
    for &(mask, rhs) in rows.iter().rev() {
        if mask == 0 {
            if rhs != 0 {
                return None;
            }
            continue;
        }
        let pivot = u128::BITS - 1 - mask.leading_zeros();
        let others = mask ^ (1_u128 << pivot);
        // In GF(2), subtraction is XOR: the pivot bit is the parity of the
        // right-hand side and the already-solved contributions.
        let bit = (rhs as u32 ^ (solution & others).count_ones()) & 1;
        if bit != 0 {
            solution |= 1_u128 << pivot;
        }
    }
    Some(solution)
}

/// Basis masks of the null space of the eliminated system.
///
/// Each free column yields one kernel vector: the free bit set, then every
/// pivot bit resolved by back substitution against the already-determined
/// lower bits (the same reverse-order accumulation as the particular
/// solution, with zero right-hand side).
fn kernel_basis(rows: &[(u128, u128)], columns: usize) -> Vec<u128> {
    let pivots: Vec<usize> = rows
        .iter()
        .filter(|&&(mask, _)| mask != 0)
        .map(|&(mask, _)| (u128::BITS - 1 - mask.leading_zeros()) as usize)
        .collect();
    let mut kernel = Vec::new();
    for column in 0..columns {
        if pivots.contains(&column) {
            continue;
        }
        let mut vector = 1_u128 << column;
        for &(mask, _) in rows.iter().rev() {
            if mask == 0 {
                continue;
            }
            let pivot = u128::BITS - 1 - mask.leading_zeros();
            let others = mask ^ (1_u128 << pivot);
            if (vector & others).count_ones() & 1 != 0 {
                vector |= 1_u128 << pivot;
            }
        }
        kernel.push(vector);
    }
    kernel
}

/// Decode a field element from its power-basis coordinate mask.
fn element_from_coordinates<F: FieldKernels>(basis: &[F::Elem], mask: u128) -> F::Elem {
    let mut element = F::Elem::ZERO;
    for (index, &basis_element) in basis.iter().enumerate() {
        if mask >> index & 1 != 0 {
            element = element.add(basis_element);
        }
    }
    element
}

/// Decode a field element from its canonical little-endian key.
fn element_from_key<F: FieldKernels>(key: u128) -> F::Elem {
    let bytes = key.to_le_bytes();
    F::read(&bytes[..F::BYTES])
}
