//! Weighted multipoint evaluation against the division-free Horner-jet
//! oracle: structurally unrelated to the production path (scalar element
//! loops only, no binomials, no translation powers, no remainder tree).

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::mersenne31;
use fgf::{Gf8B, Gf16, Goldilocks, Mersenne31, QuadMersenne31};
use poly_ring::MultiplicityPlan;
use proptest::prelude::*;
use proptest::test_runner::RngSeed;

/// The independent oracle: Taylor translation by the Horner recurrence on
/// jets, one scalar element operation at a time.
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

/// The full request oracle: every point's jet, concatenated in caller order.
fn horner_weighted<F: FieldKernels>(
    coefficients: &[F::Elem],
    points: &[F::Elem],
    multiplicities: &[usize],
) -> Vec<F::Elem> {
    let mut expected = Vec::new();
    for (point, &multiplicity) in points.iter().zip(multiplicities) {
        expected.extend(horner_jet::<F>(coefficients, *point, multiplicity));
    }
    expected
}

fn noise<F: FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bytes = state.to_le_bytes();
            F::read(&bytes[..F::BYTES])
        })
        .collect()
}

fn check_request<F: poly_ring::PolynomialField>(
    coefficients: &[F::Elem],
    points: &[F::Elem],
    multiplicities: &[usize],
) {
    let plan =
        MultiplicityPlan::<F>::new(points, multiplicities, coefficients.len() + 8).expect("plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = vec![F::Elem::ZERO; plan.total_weight()];
    plan.evaluate_into(coefficients, &mut scratch, &mut output)
        .expect("evaluate");
    assert_eq!(
        output,
        horner_weighted::<F>(coefficients, points, multiplicities)
    );
}

/// One point's jet through the production `MultiplicityPlan`: the same
/// request the properties make, so generated inputs cross the real
/// remainder-tree and translation paths, not only the oracle.
fn production_jet<F: poly_ring::PolynomialField>(
    coefficients: &[F::Elem],
    point: F::Elem,
    multiplicity: usize,
) -> Vec<F::Elem> {
    let plan =
        MultiplicityPlan::<F>::new(&[point], &[multiplicity], coefficients.len()).expect("plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = vec![F::Elem::ZERO; plan.total_weight()];
    plan.evaluate_into(coefficients, &mut scratch, &mut output)
        .expect("evaluate");
    output
}

fn small_points<F: FieldKernels>(count: usize) -> Vec<F::Elem> {
    // Small canonical representatives as distinct points, by field value.
    (0..count)
        .map(|index| {
            let mut current = F::Elem::ONE;
            for _ in 0..index {
                current = current.add(F::Elem::ONE);
            }
            current
        })
        .collect()
}

#[test]
fn weighted_tree_matches_the_oracle_across_fields() {
    fn check<F: poly_ring::PolynomialField>() {
        for seed in 0..4 {
            let coefficients = noise::<F>(17 + 9 * seed, 0x5EED_0000 + seed as u64);
            let points = small_points::<F>(5);
            check_request::<F>(&coefficients, &points, &[1, 2, 3, 5, 8]);
            check_request::<F>(&coefficients, &points[..3], &[4, 1, 2]);
        }
    }
    check::<Gf8B>();
    check::<Gf16>();
    check::<Mersenne31>();
    check::<Goldilocks>();
    check::<QuadMersenne31>();
}

#[test]
fn offsets_and_ordering_follow_the_caller() {
    let points = small_points::<Mersenne31>(4);
    let multiplicities = [2, 0, 3, 1];
    let coefficients = noise::<Mersenne31>(9, 0x5EED_0100);
    let plan = MultiplicityPlan::<Mersenne31>::new(&points, &multiplicities, 16).expect("plan");
    assert_eq!(plan.points(), points.as_slice());
    assert_eq!(plan.multiplicities(), multiplicities.as_slice());
    let offsets = plan.offsets().to_vec();
    assert_eq!(offsets, &[0, 2, 2, 5, 6]);
    assert_eq!(plan.total_weight(), 6);
    assert_eq!(plan.jet_count(), 3);
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = vec![mersenne_zero(); 6];
    plan.evaluate_into(&coefficients, &mut scratch, &mut output)
        .expect("evaluate");
    let expected = horner_weighted::<Mersenne31>(&coefficients, &points, &multiplicities);
    assert_eq!(output, expected);
    // Output position offsets[i] + j is D^[j]f(a_i) for the ORIGINAL point
    // order, zero weights included as empty ranges.
    for (index, value) in output.iter().enumerate() {
        let point = points
            .iter()
            .enumerate()
            .rev()
            .find(|(candidate, _)| offsets[*candidate] <= index)
            .map(|(candidate, _)| candidate)
            .unwrap_or(0);
        let j = index - offsets[point];
        assert_eq!(
            *value,
            horner_weighted::<Mersenne31>(&coefficients, &points, &multiplicities)[index],
            "position {index} is point {point} order {j}"
        );
    }
}

fn mersenne_zero() -> mersenne31::Elem {
    <Mersenne31 as Field>::Elem::ZERO
}

#[test]
fn degenerate_requests_hold_their_contracts() {
    let points = small_points::<Gf8B>(3);
    let coefficients = noise::<Gf8B>(6, 0x5EED_0200);

    // All-zero weights: empty output, valid request.
    check_request::<Gf8B>(&coefficients, &points, &[0, 0, 0]);
    // Zero polynomial.
    check_request::<Gf8B>(&[], &points, &[2, 1, 4]);
    // Degree far below the total weight.
    check_request::<Gf8B>(&coefficients[..2], &points, &[4, 4, 4]);
    // Multiplicity far above the degree.
    check_request::<Gf8B>(&coefficients[..3], &points[..1], &[11]);
    // Multiplicity above the characteristic, over a prime field: 11 > 2 is
    // nothing special, and 11-th Hasse derivatives exist at every point.
    let prime_points = small_points::<Mersenne31>(2);
    let prime_coefficients = noise::<Mersenne31>(4, 0x5EED_0201);
    check_request::<Mersenne31>(&prime_coefficients, &prime_points, &[7, 11]);
    // Repeated points stay separate, in caller order.
    let repeated = [prime_points[0], prime_points[1], prime_points[0]];
    check_request::<Mersenne31>(&prime_coefficients, &repeated, &[1, 2, 3]);
    // Skewed weights: one heavy leaf beside light ones.
    check_request::<Mersenne31>(&prime_coefficients, &prime_points, &[1, 9]);
    // Empty request.
    check_request::<Mersenne31>(&prime_coefficients, &[], &[]);
    // Uniform construction takes the same validated path.
    let plan = MultiplicityPlan::<Mersenne31>::uniform(&prime_points, 3, 12).expect("uniform");
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = vec![mersenne_zero(); 6];
    plan.evaluate_into(&prime_coefficients, &mut scratch, &mut output)
        .expect("evaluate");
    assert_eq!(
        output,
        horner_weighted::<Mersenne31>(&prime_coefficients, &prime_points, &[3, 3])
    );
}

#[test]
fn scalar_and_batch_lanes_agree_for_distinct_polynomials() {
    fn check<F: poly_ring::PolynomialField>() {
        let batch = 4_usize;
        let count = 13_usize;
        let lanes: Vec<Vec<F::Elem>> = (0..batch)
            .map(|lane| noise::<F>(count, 0x5EED_0300 + lane as u64))
            .collect();
        let points = small_points::<F>(3);
        let multiplicities = [1, 3, 2];
        let plan = MultiplicityPlan::<F>::new(&points, &multiplicities, count).expect("plan");
        let mut scratch = plan.scratch(batch).expect("scratch");
        let weight = plan.total_weight();

        let mut packed = vec![0_u8; count * batch * F::BYTES];
        for (lane, coefficients) in lanes.iter().enumerate() {
            for (degree, value) in coefficients.iter().enumerate() {
                let offset = (degree * batch + lane) * F::BYTES;
                F::write(&mut packed[offset..offset + F::BYTES], *value);
            }
        }
        let mut batched = vec![0_u8; weight * batch * F::BYTES];
        plan.evaluate_batch_into(&packed, count, batch, &mut scratch, &mut batched)
            .expect("batch evaluate");

        // Each lane must equal its own scalar evaluation, and adjacent lanes
        // hold genuinely different polynomials so a lane-stride bug cannot
        // pass by accident.
        for (lane, coefficients) in lanes.iter().enumerate() {
            let mut scalar = vec![F::Elem::ZERO; weight];
            plan.evaluate_into(coefficients, &mut scratch, &mut scalar)
                .expect("scalar evaluate");
            for (row, expected) in scalar.iter().enumerate() {
                let offset = (row * batch + lane) * F::BYTES;
                assert_eq!(F::read(&batched[offset..offset + F::BYTES]), *expected);
            }
        }
        assert_ne!(lanes[0], lanes[1]);
    }
    check::<Gf8B>();
    check::<Mersenne31>();
    check::<Goldilocks>();
}

#[test]
fn noncanonical_prime_lanes_evaluate_their_field_values() {
    const MODULUS: u32 = 0x7FFF_FFFF;
    // Lane values p (zero) and p + 5 (five), by raw lanes.
    let raw = [
        mersenne31::Elem::from_raw(MODULUS),
        mersenne31::Elem::from_raw(MODULUS + 5),
    ];
    let canonical = [mersenne31::Elem::from_raw(0), mersenne31::Elem::from_raw(5)];
    let points = small_points::<Mersenne31>(2);
    for coefficients in [raw, canonical] {
        let _ = coefficients;
    }
    check_request::<Mersenne31>(&raw, &points, &[2, 3]);
    check_request::<Mersenne31>(&canonical, &points, &[2, 3]);
    // The packed batch path canonicalizes on the way in as well: a raw `p`
    // lane is the zero coefficient.
    let plan = MultiplicityPlan::<Mersenne31>::new(&points[..1], &[2], 2).expect("plan");
    let mut scratch = plan.scratch(2).expect("scratch");
    let mut packed = vec![0_u8; 2 * 2 * Mersenne31::BYTES];
    for (degree, value) in raw.iter().enumerate() {
        for lane in 0..2 {
            let offset = (degree * 2 + lane) * Mersenne31::BYTES;
            Mersenne31::write(&mut packed[offset..offset + Mersenne31::BYTES], *value);
        }
    }
    let mut output = vec![0_u8; 2 * 2 * Mersenne31::BYTES];
    plan.evaluate_batch_into(&packed, 2, 2, &mut scratch, &mut output)
        .expect("evaluate");
    let scalar = [raw[0], raw[1]];
    let mut scalar_output = vec![mersenne_zero(); 2];
    plan.evaluate_into(&scalar, &mut scratch, &mut scalar_output)
        .expect("scalar");
    for (row, expected) in scalar_output.iter().enumerate() {
        for lane in 0..2 {
            let offset = (row * 2 + lane) * Mersenne31::BYTES;
            assert_eq!(
                <Mersenne31 as Field>::read(&output[offset..offset + Mersenne31::BYTES]),
                *expected
            );
        }
    }
}

#[test]
fn errors_reject_before_mutating_and_scratch_survives() {
    let points = small_points::<Gf8B>(2);
    let coefficients = noise::<Gf8B>(5, 0x5EED_0400);
    let plan = MultiplicityPlan::<Gf8B>::new(&points, &[2, 2], 8).expect("plan");
    let mut scratch = plan.scratch(2).expect("scratch");

    // Mismatched request lengths.
    assert!(matches!(
        MultiplicityPlan::<Gf8B>::new(&points, &[2], 8),
        Err(poly_ring::HasseError::LengthMismatch { .. })
    ));
    // Capacity violations.
    let small_plan = MultiplicityPlan::<Gf8B>::new(&points, &[2, 2], 4).unwrap();
    let mut four = [<Gf8B as Field>::Elem::ZERO; 4];
    assert!(matches!(
        small_plan.evaluate_into(&coefficients, &mut scratch, &mut four),
        Err(poly_ring::HasseError::CoefficientCapacityExceeded { .. })
    ));
    let mut short = [<Gf8B as Field>::Elem::ZERO; 3];
    assert!(matches!(
        plan.evaluate_into(&coefficients, &mut scratch, &mut short),
        Err(poly_ring::HasseError::LengthMismatch { .. })
    ));
    // Batch beyond capacity.
    let packed = vec![0_u8; 5 * 4 * Gf8B::BYTES];
    let mut batched = vec![0_u8; 4 * 4 * Gf8B::BYTES];
    assert!(matches!(
        plan.evaluate_batch_into(&packed, 5, 4, &mut scratch, &mut batched),
        Err(poly_ring::HasseError::BatchCapacityExceeded { .. })
    ));
    // Destination sentinels survive a wrong-length output.
    let mut sentinel = vec![0xAB_u8; 5 * 2 * Gf8B::BYTES];
    assert!(matches!(
        plan.evaluate_batch_into(&packed[..5 * 2], 5, 2, &mut scratch, &mut sentinel),
        Err(poly_ring::HasseError::LengthMismatch { .. })
    ));
    assert!(sentinel.iter().all(|byte| *byte == 0xAB));

    // A valid run immediately after the errors proves the scratch is intact.
    let mut output = vec![<Gf8B as Field>::Elem::ZERO; 4];
    plan.evaluate_into(&coefficients, &mut scratch, &mut output)
        .expect("evaluate");
    assert_eq!(
        output,
        horner_weighted::<Gf8B>(&coefficients, &points, &[2, 2])
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        rng_seed: RngSeed::Fixed(0x5EED_C0FF_EE01_0001),
        ..ProptestConfig::default()
    })]

    #[test]
    fn product_rule_holds_over_mersenne31(seed in any::<u64>()) {
        // D^[r](f·g) = Σ_j D^[j]f · D^[r−j]g — the Leibniz rule for Hasse
        // derivatives. Every jet comes from a production MultiplicityPlan
        // and is first checked against the Horner-jet oracle, so the rule
        // is proven of what the crate returns, not only of the oracle.
        let f = noise::<Mersenne31>(9, seed);
        let g = noise::<Mersenne31>(7, seed ^ 0x5EED);
        let point = noise::<Mersenne31>(1, seed ^ 0xBEEF)[0];
        let product: Vec<mersenne31::Elem> = {
            let mut coefficients = vec![mersenne_zero(); f.len() + g.len()];
            for (i, a) in f.iter().enumerate() {
                for (j, b) in g.iter().enumerate() {
                    coefficients[i + j] = coefficients[i + j].add(a.mul(*b));
                }
            }
            coefficients
        };
        for r in [0_usize, 1, 2, 5, 9] {
            let jet_f = production_jet::<Mersenne31>(&f, point, r + 1);
            let jet_g = production_jet::<Mersenne31>(&g, point, r + 1);
            let jet_product = production_jet::<Mersenne31>(&product, point, r + 1);
            prop_assert_eq!(jet_f.as_slice(), horner_jet::<Mersenne31>(&f, point, r + 1));
            prop_assert_eq!(jet_g.as_slice(), horner_jet::<Mersenne31>(&g, point, r + 1));
            prop_assert_eq!(
                jet_product.as_slice(),
                horner_jet::<Mersenne31>(&product, point, r + 1)
            );
            let mut total = mersenne_zero();
            for j in 0..=r {
                total = total.add(jet_f[j].mul(jet_g[r - j]));
            }
            prop_assert_eq!(jet_product[r], total);
        }
    }

    #[test]
    fn taylor_identity_holds_over_gf16(seed in any::<u64>()) {
        // Σ_j D^[j]f(a)·h^j equals f(a + h): the jets and the shifted value
        // come from one production plan over both points, the jets are
        // checked against the Horner-jet oracle, and the sum against the
        // oracle's scalar Horner evaluation of f at a + h.
        let f = noise::<Gf16>(11, seed);
        let a = noise::<Gf16>(1, seed ^ 0x1111)[0];
        let h = noise::<Gf16>(1, seed ^ 0x2222)[0];
        let shifted = a.add(h);
        let plan = MultiplicityPlan::<Gf16>::new(&[a, shifted], &[f.len(), 1], f.len())
            .expect("plan");
        let mut scratch = plan.scratch(1).expect("scratch");
        let mut output = vec![fgf::gf16::Elem::ZERO; plan.total_weight()];
        plan.evaluate_into(&f, &mut scratch, &mut output)
            .expect("evaluate");
        let jet = &output[..f.len()];
        let value_at_shifted = output[f.len()];
        let oracle_jet = horner_jet::<Gf16>(&f, a, f.len());
        prop_assert_eq!(jet, oracle_jet.as_slice());
        // f(a + h) = Σ_j D^[j]f(a) · h^j.
        let mut value = fgf::gf16::Elem::ZERO;
        for (j, derivative) in jet.iter().enumerate() {
            value = value.add(derivative.mul(h.pow(j as u64)));
        }
        prop_assert_eq!(value, value_at_shifted);
        let mut direct = fgf::gf16::Elem::ZERO;
        for coefficient in f.iter().rev() {
            direct = direct.mul(shifted).add(*coefficient);
        }
        prop_assert_eq!(value, direct);
    }

    #[test]
    fn generated_requests_match_the_oracle_above_the_jet_horner_base(
        seed in any::<u64>(),
        coefficient_count in 1_usize..=48,
        multiplicity in 17_usize..=40,
    ) {
        // Multiplicity above the jet's recursion base of 16 drives the
        // divide-and-conquer translation path; the polynomial, the points,
        // and the neighbouring weights are all generated, and the whole
        // request goes through check_request — production plan against the
        // Horner-jet oracle — over a binary and a prime field.
        fn check<F: poly_ring::PolynomialField>(seed: u64, count: usize, multiplicity: usize) {
            let coefficients = noise::<F>(count, seed);
            let points = noise::<F>(3, seed ^ 0x600D);
            check_request::<F>(&coefficients, &points, &[multiplicity, 1, multiplicity - 16]);
        }
        check::<Gf16>(seed, coefficient_count, multiplicity);
        check::<Mersenne31>(seed, coefficient_count, multiplicity);
    }
}
