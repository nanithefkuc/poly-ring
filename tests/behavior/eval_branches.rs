//! Evaluation branches: Newton duplicate detection, multipoint descent,
//! domain backends, remainder-tree edges, and Hermite CRT failures.
//!
//! The agreement suites run the happy paths; every test here takes a branch
//! they never reach — duplicate supports, crossover dispatch on both sides,
//! scratch/geometry violations, empty and constant moduli, and the Hermite
//! weight conflicts — each against exact values or error payloads.

use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, Mersenne31, gf8b};
use poly_ring::{
    ConfigError, DomainError, EvalError, EvaluationDomain, HermiteError, HermitePlan,
    MultiplicityPlan, NewtonBasis, Polynomial,
};

use crate::oracles;
use oracles::{naive_evaluate, noise, noise_poly};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn distinct_points<F: FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
    let mut points = Vec::new();
    let mut state = seed;
    while points.len() < len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let candidate = F::decode(&state.to_le_bytes()[..F::BYTES]);
        if !points.contains(&candidate) {
            points.push(candidate);
        }
    }
    points
}

/// A repeated point is a duplicate, never a zero denominator: construction
/// names both indices.
#[test]
fn newton_basis_names_duplicate_points() {
    let points = [b(1), b(2), b(1)];
    assert_eq!(
        NewtonBasis::<Gf8B>::new(&points).map(|_| ()),
        Err(EvalError::Domain(DomainError::DuplicatePoint {
            first: 0,
            second: 2
        }))
    );
    // Empty support and over-capacity support are domain errors too.
    assert!(matches!(
        NewtonBasis::<Gf8B>::new(&[]),
        Err(EvalError::Domain(DomainError::Config(
            ConfigError::ZeroParameter { .. }
        )))
    ));
}

/// Newton interpolation through duplicate points fails before any basis
/// arithmetic runs.
#[test]
fn newton_interpolation_rejects_duplicate_supports() {
    let points = [b(4), b(4)];
    let values = [b(1), b(2)];
    assert_eq!(
        poly_ring::interpolate_newton::<Gf8B>(&points, &values).map(|_| ()),
        Err(EvalError::Domain(DomainError::DuplicatePoint {
            first: 0,
            second: 1
        }))
    );
}

/// The Newton interpolant through `n` points matches per-point Horner
/// evaluation at every support point, on both sides of the crossover.
#[test]
fn newton_interpolant_matches_horner_at_support() {
    for len in [3_usize, 12] {
        let points = distinct_points::<Gf16>(len, 0xC301 + len as u64);
        let values = noise::<Gf16>(len, 0xC302 + len as u64);
        let interpolant = poly_ring::interpolate_newton(&points, &values).expect("interpolant");
        for (point, value) in points.iter().zip(&values) {
            assert_eq!(interpolant.evaluate(*point), *value);
        }
        // The `into` form writes the same polynomial into reused storage.
        let mut reused = noise_poly::<Gf16>(20, 0xC303);
        poly_ring::interpolate_newton_into(&mut reused, &points, &values).expect("into");
        assert_eq!(reused, interpolant);
    }
}

/// A reused multipoint scratch recycles its tree and remainder slots:
/// evaluating over 40, then 20, then 40 points reuses every pool without
/// changing the values.
#[test]
fn multipoint_scratch_recycles_tree_and_slots() {
    use poly_ring::{MultipointScratch, evaluate_multipoint_into};

    let polynomial = noise_poly::<Gf16>(9, 0xC312);
    let mut scratch = MultipointScratch::new();
    let mut values = Vec::new();
    for (seed, len) in [(0xC313_u64, 40_usize), (0xC314, 20), (0xC315, 40)] {
        let points = distinct_points::<Gf16>(len, seed);
        evaluate_multipoint_into(&mut values, &polynomial, &points, &mut scratch).expect("eval");
        assert_eq!(values.len(), len);
        for (point, value) in points.iter().zip(&values) {
            assert_eq!(*value, naive_evaluate(&polynomial, *point));
        }
    }
    let _ = MultipointScratch::<Gf16>::default();
}

/// Multipoint evaluation below the crossover is Horner; above it the
/// subproduct tree — both agree with the naive scan.
#[test]
fn multipoint_evaluation_matches_naive_on_both_sides() {
    let polynomial = noise_poly::<Gf16>(9, 0xC304);
    for len in [3_usize, 40] {
        let points = distinct_points::<Gf16>(len, 0xC305 + len as u64);
        let values = poly_ring::evaluate_multipoint(&polynomial, &points).expect("eval");
        assert_eq!(values.len(), len);
        for (point, value) in points.iter().zip(&values) {
            assert_eq!(*value, naive_evaluate(&polynomial, *point));
        }
    }
}

/// Lagrange interpolation rejects mismatched lengths and duplicate points
/// with the domain payloads.
#[test]
fn lagrange_rejects_length_and_duplicate_inputs() {
    let points = distinct_points::<Gf16>(4, 0xC306);
    let short = noise::<Gf16>(3, 0xC307);
    assert_eq!(
        poly_ring::interpolate_lagrange::<Gf16>(&points, &short).map(|_| ()),
        Err(EvalError::Domain(DomainError::LengthMismatch {
            expected: 4,
            found: 3
        }))
    );
    let mut duplicated = points.clone();
    duplicated[3] = duplicated[0];
    let values = noise::<Gf16>(4, 0xC308);
    assert!(matches!(
        poly_ring::interpolate_lagrange::<Gf16>(&duplicated, &values),
        Err(EvalError::Domain(DomainError::DuplicatePoint { .. }))
    ));
}

/// Lagrange interpolation through `n > crossover` points inverts
/// multipoint evaluation exactly.
#[test]
fn lagrange_inverts_multipoint_above_crossover() {
    let points = distinct_points::<Gf16>(20, 0xC309);
    let polynomial = noise_poly::<Gf16>(7, 0xC30A);
    let values = poly_ring::evaluate_multipoint(&polynomial, &points).expect("eval");
    let recovered = poly_ring::interpolate_lagrange(&points, &values).expect("lagrange");
    assert_eq!(recovered, polynomial);
}

/// Arbitrary domains validate their supports: empty, over-capacity, and
/// duplicate points are domain errors with payloads.
#[test]
fn arbitrary_domains_validate_supports() {
    assert!(matches!(
        EvaluationDomain::<Gf8B>::arbitrary(vec![]),
        Err(DomainError::Config(ConfigError::ZeroParameter { .. }))
    ));
    let dup = [b(1), b(2), b(1)];
    assert_eq!(
        EvaluationDomain::<Gf8B>::arbitrary(dup.to_vec()).map(|_| ()),
        Err(DomainError::DuplicatePoint {
            first: 0,
            second: 2
        })
    );
    // Over capacity: 257 points exceed the 256-element field.
    let mut too_many = Vec::new();
    for key in 0..=256_u32 {
        too_many.push(Gf8B::decode(&(key as u128).to_le_bytes()[..Gf8B::BYTES]));
    }
    assert!(matches!(
        EvaluationDomain::<Gf8B>::arbitrary(too_many),
        Err(DomainError::Config(
            ConfigError::FieldCapacityExceeded { .. }
        ))
    ));
    // Backend selection: small supports are Horner, large ones the tree.
    let small =
        EvaluationDomain::<Gf8B>::arbitrary(distinct_points::<Gf8B>(4, 0xC30B)).expect("domain");
    assert_eq!(small.backend(), poly_ring::EvaluationBackend::Horner);
    let large =
        EvaluationDomain::<Gf8B>::arbitrary(distinct_points::<Gf8B>(40, 0xC30C)).expect("domain");
    assert_eq!(
        large.backend(),
        poly_ring::EvaluationBackend::SubproductTree
    );
    assert!(!small.is_empty());
    assert_eq!(small.len(), 4);
}

/// Subspace sizes must be powers of two within the field; an explicit
/// basis must be long enough and independent.
#[cfg(feature = "fft")]
#[test]
fn subspace_construction_validates_sizes_and_bases() {
    use fgf::Gf16;

    assert!(matches!(
        EvaluationDomain::<Gf16>::additive_subspace(7),
        Err(DomainError::NotSubspace { .. })
    ));
    assert!(matches!(
        EvaluationDomain::<Gf16>::additive_subspace(1 << 17),
        Err(DomainError::Config(
            ConfigError::FieldCapacityExceeded { .. }
        ))
    ));
    // A short basis prefix cannot span the requested size.
    assert!(matches!(
        EvaluationDomain::<Gf16>::additive_subspace_with_basis(8, &[Gf16::decode(&[1_u8, 0])]),
        Err(DomainError::NotSubspace { .. })
    ));
    // A dependent basis (repeated element) is rejected.
    let element = Gf16::decode(&[9_u8, 0]);
    assert!(matches!(
        EvaluationDomain::<Gf16>::additive_subspace_with_basis(8, &[element, element]),
        Err(DomainError::NotSubspace { .. })
    ));
}

/// Remainder trees reject a zero modulus and accept constant moduli as
/// empty ranges; the descent validates every length and capacity before
/// touching the output.
#[test]
fn remainder_trees_validate_geometry_before_descending() {
    use poly_ring::{ProductError, RemainderTree};

    let zero = Polynomial::<Gf8B>::zero();
    assert_eq!(
        RemainderTree::<Gf8B>::new(&[zero], 4).map(|_| ()),
        Err(ProductError::Polynomial(
            poly_ring::PolynomialError::DivisionByZero
        ))
    );
    // All-constant moduli: no tree nodes, zero total rows, empty output.
    let constants = [
        Polynomial::<Gf8B>::from_coefficients(&[b(3)]).expect("const"),
        Polynomial::<Gf8B>::from_coefficients(&[b(5)]).expect("const"),
    ];
    let tree = RemainderTree::<Gf8B>::new(&constants, 4).expect("tree");
    assert_eq!(tree.leaf_offsets(), &[0, 0, 0]);
    let mut scratch = tree.scratch(1).expect("scratch");
    // total_rows == 0 returns Ok without touching anything.
    tree.remainders_into(&[], 0, 1, &mut scratch, &mut [])
        .expect("empty descent");
    // Batch zero is a valid empty geometry on a nontrivial tree.
    let moduli = [noise_poly::<Gf8B>(3, 0xC30D)];
    let live = RemainderTree::<Gf8B>::new(&moduli, 4).expect("live");
    let mut live_scratch = live.scratch(1).expect("scratch");
    live.remainders_into(&[], 0, 0, &mut live_scratch, &mut [])
        .expect("batch zero");
    // Wrong input length names both byte counts.
    assert!(matches!(
        live.remainders_into(&[0u8; 3], 1, 1, &mut live_scratch, &mut [0u8; 2]),
        Err(ProductError::Config(ConfigError::BufferLength { .. }))
    ));
    // Over-capacity coefficients and over-capacity lanes name their bounds.
    let coefficients = vec![0u8; 8 * Gf8B::BYTES];
    let mut too_short = vec![0u8; 2 * Gf8B::BYTES];
    assert!(matches!(
        live.remainders_into(&coefficients, 8, 1, &mut live_scratch, &mut too_short),
        Err(ProductError::Config(ConfigError::ScratchTooSmall { .. }))
    ));
    assert!(matches!(
        live.remainders_into(&[], 0, 2, &mut live_scratch, &mut []),
        Err(ProductError::Config(ConfigError::ScratchTooSmall { .. }))
    ));
    // A scratch from a different tree is rejected before any mutation.
    let other = RemainderTree::<Gf8B>::new(&[noise_poly::<Gf8B>(5, 0xC311)], 6).expect("other");
    let mut foreign = other.scratch(1).expect("scratch");
    let total = live.leaf_offsets().last().copied().unwrap_or(0);
    let mut output = vec![0xAA_u8; total * Gf8B::BYTES];
    let before = output.clone();
    assert!(matches!(
        live.remainders_into(&[0u8; 2 * Gf8B::BYTES], 2, 1, &mut foreign, &mut output),
        Err(ProductError::Config(ConfigError::ScratchTooSmall { .. }))
    ));
    assert_eq!(output, before);
}

/// Multiplicity evaluation over a zero-weight plan is the empty vector,
/// and mismatched scratch geometry is rejected.
#[test]
fn multiplicity_zero_weight_and_scratch_mismatch() {
    let points = [b(1), b(2)];
    let plan = MultiplicityPlan::<Gf8B>::new(&points, &[0, 0], 4).expect("plan");
    assert_eq!(plan.total_weight(), 0);
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = Vec::new();
    plan.evaluate_into(&[], &mut scratch, &mut output)
        .expect("empty");
    assert!(output.is_empty());
    // A scratch built for a different plan is a mismatch, not garbage.
    let other = MultiplicityPlan::<Gf8B>::new(&points, &[1, 1], 4).expect("other");
    let mut other_scratch = other.scratch(1).expect("scratch");
    assert!(
        plan.evaluate_into(&[], &mut other_scratch, &mut Vec::new())
            .is_err()
    );
    let _ = scratch;
}

/// Multiplicity batch evaluation on the prime field runs through the
/// canonicalized copy and agrees with the scalar path.
#[test]
fn prime_multiplicity_batch_matches_scalar() {
    use fgf::mersenne31;

    let points = [
        mersenne31::Elem::from_raw(3),
        mersenne31::Elem::from_raw(11),
    ];
    let plan = MultiplicityPlan::<Mersenne31>::new(&points, &[2, 1], 8).expect("plan");
    let coefficients = noise::<Mersenne31>(6, 0xC30E);
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut scalar = vec![mersenne31::Elem::from_raw(0); plan.total_weight()];
    plan.evaluate_into(&coefficients, &mut scratch, &mut scalar)
        .expect("scalar");
    // Batch form at batch one over the same coefficients.
    let mut batch_scratch = plan.scratch(1).expect("scratch");
    let row_bytes = Mersenne31::BYTES;
    let mut input = vec![0_u8; coefficients.len() * row_bytes];
    for (degree, coefficient) in coefficients.iter().enumerate() {
        Mersenne31::encode(
            &mut input[degree * row_bytes..(degree + 1) * row_bytes],
            *coefficient,
        );
    }
    let mut output = vec![0_u8; plan.total_weight() * row_bytes];
    plan.evaluate_batch_into(
        &input,
        coefficients.len(),
        1,
        &mut batch_scratch,
        &mut output,
    )
    .expect("batch");
    for (slot, expected) in output.chunks_exact(row_bytes).zip(&scalar) {
        assert_eq!(Mersenne31::decode(slot), *expected);
    }
}

/// Hermite construction rejects length mismatches and conflicting
/// duplicates; interpolation rejects short value vectors.
#[test]
fn hermite_rejects_mismatches_and_conflicts() {
    let points = [b(1), b(2)];
    let multiplicities = [1_usize];
    assert_eq!(
        HermitePlan::<Gf8B>::new(&points, &multiplicities).map(|_| ()),
        Err(HermiteError::LengthMismatch {
            expected: 2,
            actual: 1
        })
    );
    // Two positive weights on the same point conflict; a zero weight does not.
    assert_eq!(
        HermitePlan::<Gf8B>::new(&[b(1), b(1)], &[1, 1]).map(|_| ()),
        Err(HermiteError::DuplicatePoint {
            first: 0,
            second: 1
        })
    );
    let peaceful =
        HermitePlan::<Gf8B>::new(&[b(1), b(1)], &[1, 0]).expect("zero weight never conflicts");
    assert_eq!(peaceful.total_weight(), 1);
    // Interpolation with a short value vector names both lengths.
    let plan = HermitePlan::<Gf8B>::new(&[b(1)], &[2]).expect("plan");
    assert_eq!(
        plan.interpolate(&[b(1)]).map(|_| ()),
        Err(HermiteError::LengthMismatch {
            expected: 2,
            actual: 1
        })
    );
}

/// Hermite round-trip through multiplicity evaluation: interpolating the
/// evaluated jets recovers the original polynomial below the weight.
#[test]
fn hermite_round_trip_recovers_low_degree() {
    let points = [b(0x11), b(0x23)];
    let multiplicities = [2_usize, 1];
    let plan = HermitePlan::<Gf8B>::new(&points, &multiplicities).expect("plan");
    let multi = MultiplicityPlan::<Gf8B>::new(&points, &multiplicities, 8).expect("multi");
    let polynomial = noise_poly::<Gf8B>(3, 0xC30F);
    let coefficients: Vec<gf8b::Elem> = polynomial.coefficients().collect();
    let mut scratch = multi.scratch(1).expect("scratch");
    let mut jets = vec![gf8b::Elem::from_raw(0); plan.total_weight()];
    multi
        .evaluate_into(&coefficients, &mut scratch, &mut jets)
        .expect("jets");
    let recovered = plan.interpolate(&jets).expect("interpolate");
    assert_eq!(recovered, polynomial);
    assert_eq!(plan.offsets(), &[0, 2, 3]);
    assert_eq!(plan.points(), &points);
    assert_eq!(plan.multiplicities(), &multiplicities);
}

/// Non-default fields run the same paths: Goldilocks multiplicity plans
/// evaluate jets that Hermite re-interpolates in principle; here the plan
/// construction and weight bookkeeping hold.
#[test]
fn goldilocks_multiplicity_bookkeeping_holds() {
    use fgf::{Goldilocks, goldilocks};

    let points = [goldilocks::Elem::from_raw(7), goldilocks::Elem::from_raw(9)];
    let plan = MultiplicityPlan::<Goldilocks>::new(&points, &[1, 2], 6).expect("plan");
    assert_eq!(plan.total_weight(), 3);
    assert_eq!(plan.offsets(), &[0, 1, 3]);
    let uniform = MultiplicityPlan::<Goldilocks>::uniform(&points, 2, 6).expect("uniform");
    assert_eq!(uniform.total_weight(), 4);
    let _ = noise_poly::<Mersenne31>(3, 0xC310);
}
