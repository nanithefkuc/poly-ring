// `as_chunks::<F::BYTES>()` needs a const generic depending on a
// type parameter, which Rust rejects in const argument position.
#![allow(clippy::chunks_exact_to_as_chunks)]

//! Prepared Hasse derivatives: characteristic-two rows, accessors, and the
//! documented capacity errors.
//!
//! The existing jet fixtures exercise the odd-characteristic scalar path;
//! the parity-mask path, the batched rows, and every capacity rejection
//! live here, each against hand-expanded binomial values.

use fgf::field::Field;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{DerivativePlan, HasseError};

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn gf8b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// `D^[1](1 + X + X² + X³) = [1, 0, 1]` over Gf8B: `C(1,1) = C(3,1) = 1`
/// while `C(2,1) = 0` — the parity mask, not the integer binomial.
#[test]
fn binary_scalar_derivative_applies_the_parity_mask() {
    let plan = DerivativePlan::<Gf8B>::new(1, 8).expect("plan");
    assert_eq!(plan.order(), 1);
    assert_eq!(plan.max_coefficients(), 8);
    assert_eq!(plan.output_coefficients(4).expect("output"), 3);
    let input = [gf8b(1), gf8b(1), gf8b(1), gf8b(1)];
    let mut output = [gf8b(0); 3];
    plan.apply_into(&input, &mut output).expect("apply");
    assert_eq!(output, [gf8b(1), gf8b(0), gf8b(1)]);
}

/// An order at the capacity is legal and yields empty outputs, while an
/// input beyond the capacity names the bound and the actual count.
#[test]
fn capacity_contracts_hold() {
    let plan = DerivativePlan::<Mersenne31>::new(8, 8).expect("plan");
    assert_eq!(plan.output_coefficients(8).expect("output"), 0);
    assert_eq!(plan.output_coefficients(3).expect("output"), 0);
    assert_eq!(
        plan.output_coefficients(9).map(|_| ()),
        Err(HasseError::CoefficientCapacityExceeded {
            maximum: 8,
            actual: 9
        })
    );

    let input = [m31(1); 9];
    let mut output = [m31(0); 1];
    assert_eq!(
        plan.apply_into(&input, &mut output).map(|_| ()),
        Err(HasseError::CoefficientCapacityExceeded {
            maximum: 8,
            actual: 9
        })
    );

    // A short output is rejected before any write happens.
    let plan = DerivativePlan::<Mersenne31>::new(2, 8).expect("plan");
    let input = [m31(1), m31(2), m31(3), m31(4)];
    let mut output = [m31(0); 1];
    assert_eq!(
        plan.apply_into(&input, &mut output).map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "derivative output",
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(output, [m31(0)]);
}

/// Batched rows over Gf8B: lane 0 holds `[1,1,1,1]`, lane 1 holds
/// `[1,0,1,0]`; the order-1 outputs are `[1,0,1]` and `[0,0,0]`.
#[test]
fn binary_batch_rows_match_the_scalar_parity_rule() {
    let plan = DerivativePlan::<Gf8B>::new(1, 4).expect("plan");
    let lanes: [[u8; 4]; 2] = [[1, 1, 1, 1], [1, 0, 1, 0]];
    let mut packed = [0_u8; 8];
    for (degree, slot) in packed.chunks_exact_mut(2).enumerate() {
        for (lane, row) in lanes.iter().enumerate() {
            <Gf8B as Field>::encode(&mut slot[lane..lane + 1], gf8b(row[degree]));
        }
    }
    let mut output = [0_u8; 6];
    plan.apply_batch_into(&packed, 4, 2, &mut output)
        .expect("batch");
    let expected: [[u8; 3]; 2] = [[1, 0, 1], [0, 0, 0]];
    for (degree, slot) in output.as_chunks::<2>().0.iter().enumerate() {
        for (lane, row) in expected.iter().enumerate() {
            assert_eq!(
                <Gf8B as Field>::decode(&slot[lane..lane + 1]),
                gf8b(row[degree])
            );
        }
    }
}

/// Batched length and capacity rejections name the offending buffer.
#[test]
fn batch_geometry_errors_name_the_buffer() {
    let plan = DerivativePlan::<Gf8B>::new(1, 4).expect("plan");
    let mut output = [0_u8; 6];

    // Three bytes cannot hold four one-lane rows.
    assert_eq!(
        plan.apply_batch_into(&[0_u8; 3], 4, 1, &mut output)
            .map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "derivative coefficients",
            expected: 4,
            actual: 3
        })
    );
    // Five rows exceed the prepared capacity of four.
    assert_eq!(
        plan.apply_batch_into(&[0_u8; 5], 5, 1, &mut output)
            .map(|_| ()),
        Err(HasseError::CoefficientCapacityExceeded {
            maximum: 4,
            actual: 5
        })
    );
    // Two output bytes cannot hold three one-lane rows.
    assert_eq!(
        plan.apply_batch_into(&[0_u8; 4], 4, 1, &mut output[..2])
            .map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "derivative output",
            expected: 3,
            actual: 2
        })
    );
    assert_eq!(output, [0_u8; 6]);
}
