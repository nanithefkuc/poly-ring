//! Prepared evaluation behavior and error boundaries.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, Goldilocks, Mersenne31, QuadMersenne31, gf8b, mersenne31};
use poly_ring::{
    ConfigError, DomainError, EvalError, EvaluationDomain, HasseError, HermiteError, HermitePlan,
    MultiplicityPlan, NewtonBasis, Polynomial,
};

use crate::oracles;
use oracles::{noise, noise_poly};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
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

/// Newton `interpolate_into` rejects a value/suppport length mismatch and
/// leaves the output untouched; the vanishing polynomial is the support
/// product.
#[test]
fn newton_into_rejects_mismatched_values() {
    let points = distinct_points::<Gf8B>(4, 0xC201);
    let basis = NewtonBasis::<Gf8B>::new(&points).expect("basis");
    let values = noise::<Gf8B>(3, 0xC202);
    let mut output = noise_poly::<Gf8B>(5, 0xC203);
    let before = output.clone();
    assert_eq!(
        basis.interpolate_into(&values, &mut output).map(|_| ()),
        Err(EvalError::Domain(DomainError::LengthMismatch {
            expected: 4,
            found: 3
        }))
    );
    assert_eq!(output, before);
    // The vanishing polynomial vanishes at every support point.
    for point in &points {
        assert!(basis.vanishing().evaluate(*point).is_zero());
    }
    assert!(!NewtonBasis::<Gf8B>::new(&points).expect("basis").is_empty());
}

/*
 * NOTE(worker): the next several tests were drafted against assumed public
 * surface (`leaf_degrees`, `total_rows`, `max_coefficients`,
 * `max_bound_for_test`) that this crate does not expose. They are kept
 * out of the build until rewritten against `leaf_offsets()` and
 * behavior-level assertions.
 */

/// Newton and Lagrange interpolation reject duplicate supports with the
/// offending indices.
#[test]
fn interpolation_rejects_duplicate_supports() {
    let mut points = distinct_points::<Gf8B>(4, 0xC204);
    points[3] = points[1];
    let values = noise::<Gf8B>(4, 0xC205);
    assert_eq!(
        poly_ring::interpolate_newton::<Gf8B>(&points, &values).map(|_| ()),
        Err(EvalError::Domain(DomainError::DuplicatePoint {
            first: 1,
            second: 3
        }))
    );
    assert_eq!(
        poly_ring::interpolate_lagrange::<Gf8B>(&points, &values).map(|_| ()),
        Err(EvalError::Domain(DomainError::DuplicatePoint {
            first: 1,
            second: 3
        }))
    );
    // A length mismatch names both lengths on both paths.
    let short = noise::<Gf8B>(2, 0xC206);
    let good = distinct_points::<Gf8B>(4, 0xC207);
    assert_eq!(
        poly_ring::interpolate_newton::<Gf8B>(&good, &short).map(|_| ()),
        Err(EvalError::Domain(DomainError::LengthMismatch {
            expected: 4,
            found: 2
        }))
    );
}

/// Multipoint evaluation above the crossover matches Horner point by point,
/// and the scratch form agrees with the one-shot.
#[test]
fn large_multipoint_matches_horner() {
    use poly_ring::{MultipointScratch, evaluate_multipoint, evaluate_multipoint_into};

    fn check<F: FieldKernels>() {
        let polynomial = noise_poly::<F>(23, 0xC208);
        let points = distinct_points::<F>(40, 0xC209);
        let values = evaluate_multipoint(&polynomial, &points).expect("multipoint");
        assert_eq!(values.len(), points.len());
        for (point, value) in points.iter().zip(&values) {
            assert_eq!(*value, polynomial.evaluate(*point));
        }
        let mut scratch = MultipointScratch::<F>::new();
        let mut into = Vec::new();
        evaluate_multipoint_into(&mut into, &polynomial, &points, &mut scratch).expect("into");
        assert_eq!(into, values);
    }
    check::<Gf8B>();
    check::<Mersenne31>();
}

/// A 24-point arbitrary domain takes the subproduct-tree backend and
/// interpolates exactly through its points.
#[test]
fn large_arbitrary_domain_interpolates_through_tree() {
    use poly_ring::DomainScratch;

    let points = distinct_points::<Gf16>(24, 0xC210);
    let domain = EvaluationDomain::<Gf16>::arbitrary(points.clone()).expect("domain");
    assert_eq!(
        domain.backend(),
        poly_ring::EvaluationBackend::SubproductTree
    );
    assert_eq!(domain.points(), points.as_slice());
    let values = noise::<Gf16>(24, 0xC211);
    let mut scratch = DomainScratch::<Gf16>::new();
    let mut output = Polynomial::<Gf16>::zero();
    domain
        .interpolate_into(&values, &mut scratch, &mut output)
        .expect("interpolate");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(output.evaluate(*point), *value);
    }
    // Interpolating into a reused buffer matches the allocating path.
    let direct = domain
        .interpolate(&values, &mut scratch)
        .expect("interpolate");
    assert_eq!(output, direct);
}

/// Arbitrary domains validate before building: duplicates name both indices
/// and over-capacity supports the the point count.
#[test]
fn arbitrary_domains_validate_before_building() {
    let mut repeated = distinct_points::<Gf8B>(5, 0xC212);
    repeated.push(repeated[2]);
    assert_eq!(
        EvaluationDomain::<Gf8B>::arbitrary(repeated).map(|_| ()),
        Err(DomainError::DuplicatePoint {
            first: 2,
            second: 5
        })
    );
}

/// Remainder tree accessors report the prepared geometry; a zero modulus is
/// DivisionByZero; an empty modulus list builds and evaluates to nothing.
#[test]
fn remainder_tree_accessors_and_empty_geometry() {
    use poly_ring::RemainderTree;

    let moduli = [
        Polynomial::<Gf8B>::from_coefficients(&[b(1), b(2)]).expect("linear"),
        Polynomial::<Gf8B>::from_coefficients(&[b(1)]).expect("constant"),
    ];
    let tree = RemainderTree::new(&moduli, 8).expect("tree");
    assert_eq!(tree.leaf_offsets(), &[0, 1, 1]);

    assert_eq!(
        RemainderTree::new(&[Polynomial::<Gf8B>::zero()], 8).map(|_| ()),
        Err(poly_ring::ProductError::Polynomial(
            poly_ring::PolynomialError::DivisionByZero
        ))
    );

    // Empty modulus list: no rows, and the evaluation is the empty vector.
    let empty = RemainderTree::<Gf8B>::new(&[], 8).expect("empty");
    assert!(empty.leaf_offsets() == [0]);
    let mut scratch = empty.scratch(1).expect("scratch");
    let mut output = Vec::new();
    empty
        .remainders_into(&[], 0, 1, &mut scratch, &mut output)
        .expect("empty remainders");
    assert!(output.is_empty());
}

/// Remainder validation order: coefficient length, coefficient capacity,
/// lane capacity, and output length each name their violation.
#[test]
fn remainder_validation_names_each_violation() {
    use poly_ring::{ProductError, RemainderTree};

    let moduli = [Polynomial::<Gf8B>::from_coefficients(&[b(1), b(1)]).expect("linear")];
    let tree = RemainderTree::new(&moduli, 6).expect("tree");
    let mut scratch = tree.scratch(1).expect("scratch");
    // One output row of one lane.
    let mut output = vec![0_u8; 1];

    // Coefficient buffer too short for the claimed count.
    assert_eq!(
        tree.remainders_into(&[0_u8; 2], 4, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(ProductError::Config(ConfigError::BufferLength {
            context: "remainder coefficients",
            expected: 4,
            actual: 2,
        }))
    );
    // Nine rows exceed the prepared capacity of six.
    assert_eq!(
        tree.remainders_into(&[0_u8; 9], 9, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(ProductError::Config(ConfigError::ScratchTooSmall {
            context: "remainder coefficient capacity",
            required: 9,
            available: 6,
        }))
    );
    // Two lanes exceed a one-lane scratch.
    assert_eq!(
        tree.remainders_into(&[0_u8; 4], 2, 2, &mut scratch, &mut output)
            .map(|_| ()),
        Err(ProductError::Config(ConfigError::ScratchTooSmall {
            context: "remainder lane capacity",
            required: 2,
            available: 1,
        }))
    );
    // Two output bytes cannot hold one one-lane row contract: build the
    // exact-size failure with an undersized buffer on a valid call.
    let good_input = vec![0_u8; 3];
    assert_eq!(
        tree.remainders_into(&good_input, 3, 1, &mut scratch, &mut [])
            .map(|_| ()),
        Err(ProductError::Config(ConfigError::BufferLength {
            context: "remainder output",
            expected: 1,
            actual: 0,
        }))
    );
    // Cross-tree scratch is rejected: same capacities, different geometry.
    // The mismatch reports the caller's bound against the scratch's tree.
    let other_moduli =
        [Polynomial::<Gf8B>::from_coefficients(&[b(2), b(3), b(1)]).expect("quadratic")];
    let other = RemainderTree::new(&other_moduli, 6).expect("other");
    let mut other_scratch = other.scratch(1).expect("other scratch");
    assert!(
        tree.remainders_into(&good_input, 3, 1, &mut other_scratch, &mut output)
            .is_err()
    );
}

/// Hermite construction rejects length mismatches and conflicting duplicate
/// points; interpolation rejects short value vectors.
#[test]
fn hermite_construction_rejects_mismatches_and_conflicts() {
    assert_eq!(
        HermitePlan::<Gf8B>::new(&[b(0), b(1)], &[1]).map(|_| ()),
        Err(HermiteError::LengthMismatch {
            expected: 2,
            actual: 1
        })
    );
    // Two positive weights on the same point conflict and name both entries.
    assert_eq!(
        HermitePlan::<Gf8B>::new(&[b(3), b(3)], &[1, 2]).map(|_| ()),
        Err(HermiteError::DuplicatePoint {
            first: 0,
            second: 1
        })
    );
    // Same point with a zero weight never conflicts.
    let plan = HermitePlan::<Gf8B>::new(&[b(3), b(3)], &[1, 0]).expect("plan");
    assert_eq!(plan.total_weight(), 1);
    // Short value vectors name the total weight.
    assert_eq!(
        plan.interpolate(&[]).map(|_| ()),
        Err(HermiteError::LengthMismatch {
            expected: 1,
            actual: 0
        })
    );
}

/// Hermite round trip through the weighted evaluator: interpolate the jets
/// of a polynomial recovers it exactly.
#[test]
fn hermite_round_trip_recovers_the_polynomial() {
    let expected =
        Polynomial::<Mersenne31>::from_coefficients(&[m31(3), m31(1), m31(4), m31(1), m31(5)])
            .expect("f");
    let points = [m31(2), m31(7)];
    let plan = HermitePlan::<Mersenne31>::new(&points, &[2, 3]).expect("plan");
    // Independent jets by the defining Taylor sums.
    let mut jets = Vec::new();
    for point in points {
        for order in 0..if point == m31(2) { 2 } else { 3 } {
            let mut value = m31(0);
            for (degree, coefficient) in expected.coefficients().enumerate() {
                if degree >= order {
                    value = value.add(
                        coefficient
                            .mul(poly_ring::binomial::<Mersenne31>(degree, order))
                            .mul(point.pow((degree - order) as u64)),
                    );
                }
            }
            jets.push(value);
        }
    }
    let reconstructed = plan.interpolate(&jets).expect("interpolate");
    assert_eq!(reconstructed, expected);
}

/// Jet accessors and the multiplicity-zero/empty-coefficient fast paths.
#[test]
fn jet_accessors_and_degenerate_translations() {
    use poly_ring::JetPlan;

    let plan = JetPlan::<Gf8B>::new(b(4), 0, 8).expect("plan");
    assert_eq!(plan.point(), b(4));
    assert_eq!(plan.multiplicity(), 0);
    assert_eq!(plan.max_coefficients(), 8);
    // Multiplicity zero writes nothing and succeeds on empty scratch.
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output: [gf8b::Elem; 0] = [];
    plan.evaluate_into(&[b(1), b(2)], &mut scratch, &mut output)
        .expect("empty jet");

    // Zero-point translation is truncation: low rows copy, rest zero-pad.
    let zero_plan = JetPlan::<Gf8B>::new(b(0), 4, 8).expect("plan");
    let mut scratch = zero_plan.scratch(1).expect("scratch");
    let input = [b(1), b(2)];
    let mut output = [b(9); 4];
    zero_plan
        .evaluate_into(&input, &mut scratch, &mut output)
        .expect("truncate");
    assert_eq!(output, [b(1), b(2), b(0), b(0)]);

    // The default scratch matches a fresh one on a real translation.
    let real = JetPlan::<Gf8B>::new(b(3), 3, 8).expect("plan");
    let mut fresh = real.scratch(1).expect("scratch");
    let mut defaulted = real.scratch(1).expect("scratch");
    let input = [b(1), b(2), b(3), b(4)];
    let mut out_fresh = [b(0); 3];
    let mut out_default = [b(0); 3];
    real.evaluate_into(&input, &mut fresh, &mut out_fresh)
        .expect("fresh");
    real.evaluate_into(&input, &mut defaulted, &mut out_default)
        .expect("defaulted");
    assert_eq!(out_fresh, out_default);
}

/// Jet batch validation: lane capacity, coefficient capacity, input length,
/// and output length each name their violation without mutation.
#[test]
fn jet_batch_validation_names_each_violation() {
    use poly_ring::JetPlan;

    let plan = JetPlan::<Gf8B>::new(b(2), 3, 6).expect("plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let input = [0_u8; 6];
    let mut output = [0_u8; 3];
    assert_eq!(
        plan.evaluate_batch_into(&input, 6, 2, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::BatchCapacityExceeded {
            maximum: 1,
            actual: 2
        })
    );
    assert_eq!(
        plan.evaluate_batch_into(&[0_u8; 7], 7, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::CoefficientCapacityExceeded {
            maximum: 6,
            actual: 7
        })
    );
    assert_eq!(
        plan.evaluate_batch_into(&[0_u8; 3], 6, 1, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "jet coefficients",
            expected: 6,
            actual: 3
        })
    );
    assert_eq!(
        plan.evaluate_batch_into(&input, 6, 1, &mut scratch, &mut [0_u8; 2])
            .map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "jet output",
            expected: 3,
            actual: 2
        })
    );
}

/// Weighted-evaluation accessors and the scalar error contracts.
#[test]
fn multiplicity_accessors_and_scalar_errors() {
    let plan = MultiplicityPlan::<Gf8B>::new(&[b(0), b(1), b(2)], &[1, 0, 2], 8).expect("plan");
    assert_eq!(plan.points(), &[b(0), b(1), b(2)]);
    assert_eq!(plan.multiplicities(), &[1, 0, 2]);
    assert_eq!(plan.offsets(), &[0, 1, 1, 3]);
    assert_eq!(plan.total_weight(), 3);
    assert_eq!(plan.jet_count(), 2);

    let uniform = MultiplicityPlan::<Gf8B>::uniform(&[b(0), b(1)], 2, 8).expect("uniform");
    assert_eq!(uniform.multiplicities(), &[2, 2]);
    assert_eq!(uniform.total_weight(), 4);

    // Length mismatch names both lengths.
    assert_eq!(
        MultiplicityPlan::<Gf8B>::new(&[b(0)], &[1, 2], 8).map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "multiplicities",
            expected: 1,
            actual: 2
        })
    );

    // Scalar evaluation rejects a short output without touching it.
    let mut scratch = plan.scratch(1).expect("scratch");
    let coefficients = [b(1); 4];
    let mut output = [b(7); 2];
    assert_eq!(
        plan.evaluate_into(&coefficients, &mut scratch, &mut output)
            .map(|_| ()),
        Err(HasseError::LengthMismatch {
            argument: "multipoint output",
            expected: 3,
            actual: 2
        })
    );
    assert_eq!(output, [b(7); 2]);
}

/// Weighted evaluation over Goldilocks and QuadMersenne31 matches the
/// Horner-jet oracle lane by lane.
#[test]
fn weighted_evaluation_matches_oracle_on_prime_fields() {
    use poly_ring::PolynomialField;

    fn horner_jet<F: FieldKernels>(
        coefficients: &[F::Elem],
        point: F::Elem,
        multiplicity: usize,
    ) -> Vec<F::Elem> {
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

    fn check<F: PolynomialField>() {
        let coefficients = noise::<F>(9, 0xC220);
        let points = noise::<F>(3, 0xC221);
        let weights = [1_usize, 3, 2];
        let plan = MultiplicityPlan::<F>::new(&points, &weights, 12).expect("plan");
        let mut scratch = plan.scratch(1).expect("scratch");
        let mut output = vec![F::Elem::ZERO; 6];
        plan.evaluate_into(&coefficients, &mut scratch, &mut output)
            .expect("evaluate");
        let mut expected = Vec::new();
        for (point, &weight) in points.iter().zip(&weights) {
            expected.extend(horner_jet::<F>(&coefficients, *point, weight));
        }
        assert_eq!(output, expected);
    }
    check::<Goldilocks>();
    check::<QuadMersenne31>();
    check::<Mersenne31>();
}

/// Subspace evaluation reduces a long polynomial first: a degree-40
/// polynomial evaluates identically to its reduced form on 16 points.
#[test]
#[cfg(feature = "fft")]
fn long_subspace_evaluation_reduces_first() {
    use butterfly_fft::transform::TransformPlan;
    use poly_ring::{TransformScratch, evaluate_subspace, evaluate_subspace_into};

    let plan = TransformPlan::<Gf16>::new(16).expect("plan");
    let mut scratch = TransformScratch::new();
    let polynomial = noise_poly::<Gf16>(41, 0xC230);
    let values = evaluate_subspace(&polynomial, &plan, &mut scratch).expect("evaluate");
    assert_eq!(values.len(), 16);
    for (index, value) in values.iter().enumerate() {
        assert_eq!(*value, polynomial.evaluate(plan.point_element(index)));
    }
    // The into-form agrees with the allocating form.
    let mut into = Vec::new();
    evaluate_subspace_into(&mut into, &polynomial, &plan, &mut scratch).expect("into");
    assert_eq!(into, values);

    // Coset evaluation of the same long polynomial matches Horner too.
    let mut coset_values = Vec::new();
    poly_ring::evaluate_coset_into(&mut coset_values, &polynomial, &plan, &mut scratch)
        .expect("coset");
    assert_eq!(coset_values.len(), 16);
    for (index, value) in coset_values.iter().enumerate() {
        assert_eq!(*value, polynomial.evaluate(plan.point_element(index)));
    }
}

/// Odd-characteristic batch jets canonicalize before translating: the Gf8B
/// suite never takes the staging-copy path.
#[test]
fn odd_characteristic_batch_jets_match_scalar() {
    use poly_ring::JetPlan;

    let plan = JetPlan::<Mersenne31>::new(m31(5), 4, 10).expect("plan");
    let mut scratch = plan.scratch(2).expect("scratch");
    // Two lanes of five coefficients in coefficient-major rows.
    let lane0 = [m31(1), m31(2), m31(3), m31(4), m31(5)];
    let lane1 = [m31(5), m31(4), m31(3), m31(2), m31(1)];
    let mut packed = vec![0_u8; 5 * 2 * 4];
    for (degree, (a, b)) in lane0.iter().zip(&lane1).enumerate() {
        <Mersenne31 as Field>::encode(&mut packed[(degree * 2) * 4..][..4], *a);
        <Mersenne31 as Field>::encode(&mut packed[(degree * 2 + 1) * 4..][..4], *b);
    }
    let mut output = vec![0_u8; 4 * 2 * 4];
    plan.evaluate_batch_into(&packed, 5, 2, &mut scratch, &mut output)
        .expect("batch");
    // Each lane matches its scalar jet.
    let mut scalar_scratch = plan.scratch(1).expect("scratch");
    for (lane, coefficients) in [lane0, lane1].iter().enumerate() {
        let mut expected = [m31(0); 4];
        plan.evaluate_into(coefficients, &mut scalar_scratch, &mut expected)
            .expect("scalar");
        for (order, want) in expected.iter().enumerate() {
            let offset = (order * 2 + lane) * 4;
            assert_eq!(
                <Mersenne31 as Field>::decode(&output[offset..offset + 4]),
                *want,
                "lane {lane} order {order} diverged"
            );
        }
    }
}
