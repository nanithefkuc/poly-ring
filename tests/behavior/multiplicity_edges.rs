//! Weighted multipoint evaluation: the batch error contracts and
//! cross-plan scratch rejection.
//!
//! The happy paths and the scalar/oracle agreement live in
//! `multiplicity.rs`; every rejection here asserts the documented variant
//! with its payload, and every failing call leaves the output untouched.

use fgf::{Gf8B, gf8b};
use poly_ring::{HasseError, MultiplicityPlan};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn two_point_plan() -> MultiplicityPlan<Gf8B> {
    MultiplicityPlan::<Gf8B>::new(&[b(0), b(1)], &[1, 2], 8).expect("plan")
}

/// Batch validation order: lane capacity, coefficient capacity, input
/// length, output length.
#[test]
fn batch_validation_names_each_violation() {
    let plan = two_point_plan();
    let mut scratch = plan.scratch(1).expect("scratch");
    // Two coefficient rows of one lane, three output rows of one lane.
    let input = [0_u8; 2];
    let mut output = [0_u8; 3];

    assert_eq!(
        plan.evaluate_batch_into(&input, 2, 2, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::BatchCapacityExceeded {
            maximum: 1,
            actual: 2
        })
    );

    // Nine rows exceed the prepared capacity of eight.
    let big = [0_u8; 9];
    assert_eq!(
        plan.evaluate_batch_into(&big, 9, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::CoefficientCapacityExceeded {
            maximum: 8,
            actual: 9
        })
    );

    // One byte cannot hold two one-lane rows.
    assert_eq!(
        plan.evaluate_batch_into(&[0_u8; 1], 2, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "multipoint coefficients",
            expected: 2,
            actual: 1
        })
    );

    // Two output bytes cannot hold three one-lane rows.
    assert_eq!(
        plan.evaluate_batch_into(&input, 2, 1, &mut scratch, &mut output[..2])
            .map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "multipoint output",
            expected: 3,
            actual: 2
        })
    );
    assert_eq!(output, [0_u8; 3]);
}

/// Equal weight and capacity do not make a scratch reusable: the jet slots
/// are keyed by the distinct-multiplicity layout, so a scratch staged for
/// three unit jets cannot serve a `[1, 2]` request.
#[test]
fn cross_layout_scratch_is_rejected() {
    let plan = two_point_plan();
    let other = MultiplicityPlan::<Gf8B>::new(&[b(0), b(1), b(2)], &[1, 1, 1], 8).expect("plan");
    assert_eq!(other.total_weight(), plan.total_weight());
    let mut scratch = other.scratch(1).expect("scratch");
    let input = [0_u8; 2];
    let mut output = [0_u8; 3];
    assert_eq!(
        plan.evaluate_batch_into(&input, 2, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::ScratchMismatch)
    );
    assert_eq!(output, [0_u8; 3]);
}

/// The scalar entry rejects an input beyond the staging before reading the
/// output, and a short output before evaluating.
#[test]
fn scalar_entry_rejects_geometry_before_evaluating() {
    let plan = two_point_plan();
    let narrow = MultiplicityPlan::<Gf8B>::new(&[b(0), b(1)], &[1, 2], 2).expect("plan");
    let mut scratch = narrow.scratch(1).expect("scratch");
    let coefficients = [b(1); 4];
    let mut output = [b(0); 3];
    assert_eq!(
        plan.evaluate_into(&coefficients, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::ScratchMismatch)
    );

    let mut scratch = plan.scratch(1).expect("scratch");
    let mut short = [b(0); 2];
    assert_eq!(
        plan.evaluate_into(&coefficients, &mut scratch, &mut short)
            .map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "multipoint output",
            expected: 3,
            actual: 2
        })
    );
    assert_eq!(short, [b(0); 2]);
}
