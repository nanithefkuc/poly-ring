// `as_chunks::<F::BYTES>()` needs a const generic depending on a
// type parameter, which Rust rejects in const argument position.
#![allow(clippy::chunks_exact_to_as_chunks)]

//! Taylor jets: accessors, the scalar/batch error contracts, and
//! cross-geometry scratch rejection.
//!
//! Every rejection asserts the documented variant with its payload, and
//! every failing call leaves the caller's output untouched.

use fgf::field::Field;
use fgf::{Gf8B, gf8b};
use poly_ring::{HasseError, JetPlan};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn make_plan(point: u8, multiplicity: usize, max: usize) -> JetPlan<Gf8B> {
    JetPlan::<Gf8B>::new(gf8b::Elem::from_raw(point), multiplicity, max).expect("plan")
}

/// Accessors round-trip the construction arguments.
#[test]
fn accessors_report_the_prepared_geometry() {
    let plan = make_plan(3, 5, 16);
    assert_eq!(plan.point(), gf8b::Elem::from_raw(3));
    assert_eq!(plan.multiplicity(), 5);
    assert_eq!(plan.max_coefficients(), 16);
}

/// The scalar entry rejects a short output before touching it.
#[test]
fn scalar_output_length_is_checked_first() {
    let plan = make_plan(1, 3, 8);
    let mut scratch = plan.scratch(1).expect("scratch");
    let coefficients = [gf8b::Elem::from_raw(1); 4];
    let mut output = [gf8b::Elem::from_raw(9); 2];
    assert_eq!(
        plan.evaluate_into(&coefficients, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "jet output",
            expected: 3,
            actual: 2
        })
    );
    assert_eq!(output, [gf8b::Elem::from_raw(9); 2]);
}

/// Scalar staging is sized to the plan: four coefficients do not fit a
/// scratch built for two.
#[test]
fn scalar_staging_rejects_an_oversized_input() {
    let plan = make_plan(1, 3, 8);
    let narrow = make_plan(2, 3, 2);
    let mut scratch = narrow.scratch(1).expect("scratch");
    let coefficients = [gf8b::Elem::from_raw(1); 4];
    let mut output = [gf8b::Elem::from_raw(0); 3];
    assert_eq!(
        plan.evaluate_into(&coefficients, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::ScratchMismatch)
    );
}

/// A scratch multiplicity mismatch fails the whole scalar call without
/// writing a single decoded lane.
#[test]
fn scalar_call_writes_nothing_on_a_scratch_mismatch() {
    let plan = make_plan(1, 3, 8);
    let other = make_plan(1, 4, 8);
    let mut scratch = other.scratch(1).expect("scratch");
    let coefficients = [gf8b::Elem::from_raw(1); 2];
    let mut output = [gf8b::Elem::from_raw(7); 3];
    assert_eq!(
        plan.evaluate_into(&coefficients, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::ScratchMismatch)
    );
    assert_eq!(output, [gf8b::Elem::from_raw(7); 3]);
}

/// Batch validation order: lane capacity, scratch geometry, input length,
/// coefficient capacity, output length.
#[test]
fn batch_validation_names_each_violation() {
    let plan = make_plan(1, 3, 8);
    let mut scratch = plan.scratch(1).expect("scratch");
    let input = [0_u8; 4];
    let mut output = [0_u8; 3];

    assert_eq!(
        plan.evaluate_batch_into(&input, 4, 2, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::BatchCapacityExceeded {
            maximum: 1,
            actual: 2
        })
    );

    // Sixteen rows exceed the prepared capacity of eight.
    let big = [0_u8; 16];
    assert_eq!(
        plan.evaluate_batch_into(&big, 16, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::CoefficientCapacityExceeded {
            maximum: 8,
            actual: 16
        })
    );

    // Three bytes cannot hold four one-lane rows.
    assert_eq!(
        plan.evaluate_batch_into(&[0_u8; 3], 4, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "jet coefficients",
            expected: 4,
            actual: 3
        })
    );

    // Two output bytes cannot hold three one-lane rows.
    assert_eq!(
        plan.evaluate_batch_into(&input, 4, 1, &mut scratch, &mut output[..2])
            .map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "jet output",
            expected: 3,
            actual: 2
        })
    );
    assert_eq!(output, [0_u8; 3]);
}

/// Batch zero with empty buffers is the valid empty geometry.
#[test]
fn empty_batch_geometry_succeeds() {
    let plan = make_plan(1, 3, 8);
    let mut scratch = plan.scratch(1).expect("scratch");
    plan.evaluate_batch_into(&[], 0, 0, &mut scratch, &mut [])
        .expect("empty geometry");
}

/// Equal multiplicity and capacity do not make a scratch reusable: a
/// sixteen-row input does not fit staging built for eight, and a
/// four-row input overruns recursion slots built for the smaller plan.
#[test]
fn cross_plan_scratch_is_rejected_by_layout() {
    let plan = make_plan(1, 3, 64);
    let smaller = make_plan(2, 3, 8);
    let mut scratch = smaller.scratch(1).expect("scratch");

    // Sixteen rows exceed the eight-row staging: caught at the copy.
    let wide = [0_u8; 16];
    let mut output = [0_u8; 3];
    assert_eq!(
        plan.evaluate_batch_into(&wide, 16, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::ScratchMismatch)
    );

    // Four rows fit the staging but need six recursion levels while the
    // scratch holds three: caught at the slot check, output untouched.
    let narrow = [0_u8; 4];
    assert_eq!(
        plan.evaluate_batch_into(&narrow, 4, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::ScratchMismatch)
    );
    assert_eq!(output, [0_u8; 3]);
}

/// A scratch multiplicity mismatch is caught before any length is read.
#[test]
fn batch_scratch_multiplicity_mismatch_is_caught_first() {
    let plan = make_plan(1, 3, 8);
    let other = make_plan(1, 4, 8);
    let mut scratch = other.scratch(1).expect("scratch");
    let input = [0_u8; 2];
    let mut output = [0_u8; 3];
    assert_eq!(
        plan.evaluate_batch_into(&input, 2, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::ScratchMismatch)
    );
}

/// An unrepresentable staging size is an allocation failure, never a
/// wrapped buffer: the scalar staging for `2^63` coefficients cannot
/// exist on a 64-bit host.
#[test]
fn unrepresentable_staging_is_an_allocation_failure() {
    let plan = JetPlan::<Gf8B>::new(gf8b::Elem::from_raw(1), 3, 1 << 63).expect("plan");
    assert_eq!(
        plan.scratch(1).map(|_| ()).map_err(|error| match error {
            HasseError::AllocationFailed { context } => context,
            other => panic!("unexpected error: {other:?}"),
        }),
        Err("jet scalar input")
    );
}

/// Batched lanes agree with the scalar translation row for row.
#[test]
fn batch_lanes_agree_with_the_scalar_translation() {
    // f = 1 + X² over Gf8B at a = 1 with s = 3: f(1+T) = T².
    let plan = make_plan(1, 3, 8);
    let coefficients = [b(1), b(0), b(1)];
    let mut scalar = [b(0); 3];
    let mut scratch = plan.scratch(2).expect("scratch");
    plan.evaluate_into(&coefficients, &mut scratch, &mut scalar)
        .expect("scalar");
    assert_eq!(scalar, [b(0), b(0), b(1)]);

    // Two lanes: lane 0 holds f, lane 1 holds 1 + X + X². At a = 1 the
    // second is 1 + (1+T) + (1+T)² = 1 + T + T² in characteristic two.
    let mut packed = [0_u8; 6];
    let lane_polys = [[1_u8, 0, 1], [1, 1, 1]];
    for (degree, slot) in packed.chunks_exact_mut(2).enumerate() {
        for (lane, poly) in lane_polys.iter().enumerate() {
            <Gf8B as Field>::encode(&mut slot[lane..lane + 1], b(poly[degree]));
        }
    }
    let mut output = [0_u8; 6];
    plan.evaluate_batch_into(&packed, 3, 2, &mut scratch, &mut output)
        .expect("batch");
    let expected = [[0_u8, 0, 1], [1, 1, 1]];
    for (degree, slot) in output.as_chunks::<2>().0.iter().enumerate() {
        for (lane, row) in expected.iter().enumerate() {
            assert_eq!(
                <Gf8B as Field>::decode(&slot[lane..lane + 1]),
                b(row[degree])
            );
        }
    }
}
