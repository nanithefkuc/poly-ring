//! Translation scratch behavior and error boundaries.

use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{HasseError, JetPlan};

use crate::oracles;
use oracles::noise;

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

/// Deep translations (past the Horner base) match the Horner-jet oracle on
/// both characteristics.
#[test]
fn deep_translations_match_the_horner_oracle() {
    fn horner_jet<F: poly_ring::PolynomialField>(
        coefficients: &[F::Elem],
        point: F::Elem,
        multiplicity: usize,
    ) -> Vec<F::Elem> {
        use fgf::field::Elem;
        let mut jet = vec![F::Elem::ZERO; multiplicity];
        if multiplicity == 0 {
            return jet;
        }
        for &coefficient in coefficients.iter().rev() {
            for order in (1..multiplicity).rev() {
                jet[order] = point.mul(jet[order]).add(jet[order - 1]);
            }
            jet[0] = point.mul(jet[0]).add(coefficient);
        }
        jet
    }

    // 40 coefficients exceeds the Horner base (16): the recursion runs.
    let coefficients = noise::<Gf8B>(40, 0xF401);
    let plan = JetPlan::<Gf8B>::new(b(7), 9, 40).expect("plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = vec![b(0); 9];
    plan.evaluate_into(&coefficients, &mut scratch, &mut output)
        .expect("evaluate");
    assert_eq!(output, horner_jet::<Gf8B>(&coefficients, b(7), 9));

    let coefficients = noise::<Mersenne31>(40, 0xF402);
    let plan = JetPlan::<Mersenne31>::new(m31(11), 9, 40).expect("plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = vec![m31(0); 9];
    plan.evaluate_into(&coefficients, &mut scratch, &mut output)
        .expect("evaluate");
    assert_eq!(output, horner_jet::<Mersenne31>(&coefficients, m31(11), 9));
}

/// Batch-2 deep translations match the scalar jets lane by lane.
#[test]
fn batch_deep_translations_match_scalar_lanes() {
    let plan = JetPlan::<Gf8B>::new(b(3), 6, 32).expect("plan");
    let mut scratch = plan.scratch(2).expect("scratch");
    let lane0 = noise::<Gf8B>(32, 0xF403);
    let lane1 = noise::<Gf8B>(32, 0xF404);
    let mut packed = vec![0_u8; 32 * 2];
    for (degree, (a, c)) in lane0.iter().zip(&lane1).enumerate() {
        packed[degree * 2] = a.to_bytes()[0];
        packed[degree * 2 + 1] = c.to_bytes()[0];
    }
    let mut output = vec![0_u8; 6 * 2];
    plan.evaluate_batch_into(&packed, 32, 2, &mut scratch, &mut output)
        .expect("batch");
    let mut scalar_scratch = plan.scratch(1).expect("scratch");
    for (lane, coefficients) in [&lane0, &lane1].iter().enumerate() {
        let mut expected = vec![b(0); 6];
        plan.evaluate_into(coefficients, &mut scalar_scratch, &mut expected)
            .expect("scalar");
        for (order, want) in expected.iter().enumerate() {
            assert_eq!(
                output[order * 2 + lane],
                want.to_bytes()[0],
                "lane {lane} order {order} diverged"
            );
        }
    }
}

/// A foreign scratch (same capacities, different multiplicity) is rejected
/// before any mutation.
#[test]
fn foreign_scratch_is_rejected_before_mutation() {
    let plan = JetPlan::<Gf8B>::new(b(2), 3, 8).expect("plan");
    let other = JetPlan::<Gf8B>::new(b(2), 5, 8).expect("other");
    let mut foreign = other.scratch(1).expect("scratch");
    let input = [0_u8; 6];
    let mut output = [9_u8; 3];
    // Coefficient count 6 with multiplicity 3: input length matches, but
    // the scratch multiplicity differs.
    assert_eq!(
        plan.evaluate_batch_into(&input, 6, 1, &mut foreign, &mut output)
            .map(|_| ()),
        Err(HasseError::ScratchMismatch)
    );
    assert_eq!(output, [9_u8; 3]);
}

/// Odd-characteristic undersized staging is a scratch mismatch, and the
/// output is untouched.
#[test]
fn odd_characteristic_undersized_staging_is_a_mismatch() {
    let plan = JetPlan::<Mersenne31>::new(m31(5), 4, 10).expect("plan");
    let narrow = JetPlan::<Mersenne31>::new(m31(5), 4, 3).expect("narrow");
    let mut foreign = narrow.scratch(1).expect("scratch");
    let input = vec![0_u8; 8 * 4];
    let mut output = vec![7_u8; 4 * 4];
    assert_eq!(
        plan.evaluate_batch_into(&input, 8, 1, &mut foreign, &mut output)
            .map(|_| ()),
        Err(HasseError::ScratchMismatch)
    );
    assert_eq!(output, vec![7_u8; 4 * 4]);
}

/// Scratch construction validates its geometry: extreme batches fail as
/// allocation failures, and multiplicity-zero plans build empty staging.
#[test]
fn scratch_construction_validates_geometry() {
    let plan = JetPlan::<Gf8B>::new(b(1), 3, 8).expect("plan");
    assert!(plan.scratch(1 << 60).is_err());
    let empty = JetPlan::<Gf8B>::new(b(1), 0, 8).expect("plan");
    let mut scratch = empty.scratch(1).expect("scratch");
    let mut output: [gf8b::Elem; 0] = [];
    empty
        .evaluate_into(&[b(1), b(2)], &mut scratch, &mut output)
        .expect("empty");
}

/// Jet scratch for huge maxima fails as an allocation failure (the plan
/// itself builds lazily; its scratch cannot be reserved).
#[test]
fn huge_jet_maxima_fail_scratch_construction() {
    let plan = JetPlan::<Gf8B>::new(b(1), 3, 1 << 63).expect("plan");
    assert!(plan.scratch(1).is_err());
}
