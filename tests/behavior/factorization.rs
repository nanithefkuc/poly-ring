//! Finite-field factorization contracts and independent reconstruction checks.

use fgf::field::Elem;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{BaseFieldRoots, FactorizationError, Polynomial, base_field_roots};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn power<F: fgf::kernel::FieldKernels>(factor: &Polynomial<F>, exponent: usize) -> Polynomial<F> {
    let mut result = Polynomial::one().expect("one");
    for _ in 0..exponent {
        result = result.multiply(factor).expect("product");
    }
    result
}

#[test]
fn dense_subtraction_negation_and_identity_hold_in_odd_characteristic() {
    let left = Polynomial::<Mersenne31>::from_coefficients(&[m31(9), m31(4)]).expect("left");
    let right = Polynomial::<Mersenne31>::from_coefficients(&[m31(2), m31(7)]).expect("right");
    let difference = left.sub(&right).expect("difference");
    assert_eq!(difference.add(&right).expect("reconstruct"), left);
    assert!(left.add(&left.negated()).expect("inverse").is_zero());

    let mut assigned = left.clone();
    assigned.sub_assign(&right).expect("subtract in place");
    assert_eq!(assigned, difference);
    assigned.negate_assign();
    assert_eq!(assigned, difference.negated());
    assert!(Polynomial::<Mersenne31>::one().expect("one").is_one());
    assert!(!Polynomial::<Mersenne31>::zero().is_one());
}

#[test]
fn square_free_decomposition_recovers_binary_multiplicities() {
    let first = Polynomial::<Gf8B>::from_coefficients(&[b(1), b(1)]).expect("first");
    let second = Polynomial::<Gf8B>::from_coefficients(&[b(2), b(1)]).expect("second");
    let input = power(&first, 2)
        .multiply(&power(&second, 3))
        .expect("input");

    let factors = input
        .square_free_factorization()
        .expect("square-free decomposition");
    assert_eq!(factors.len(), 2);
    assert_eq!(factors[0].multiplicity, 2);
    assert_eq!(factors[0].factor, first);
    assert_eq!(factors[1].multiplicity, 3);
    assert_eq!(factors[1].factor, second);

    let mut reconstructed = Polynomial::<Gf8B>::one().expect("one");
    for entry in factors {
        reconstructed = reconstructed
            .multiply(&power(&entry.factor, entry.multiplicity))
            .expect("reconstruct");
    }
    assert_eq!(reconstructed, input.monic());
}

#[test]
fn square_free_decomposition_extracts_pth_roots() {
    let first = Polynomial::<Gf8B>::from_coefficients(&[b(3), b(1)]).expect("first");
    let second = Polynomial::<Gf8B>::from_coefficients(&[b(5), b(1)]).expect("second");
    let square_free = first.multiply(&second).expect("square-free product");
    let input = square_free.square().expect("Frobenius square");
    assert!(input.formal_derivative().expect("derivative").is_zero());

    let factors = input.square_free_factorization().expect("decomposition");
    assert_eq!(factors.len(), 1);
    assert_eq!(factors[0].factor, square_free);
    assert_eq!(factors[0].multiplicity, 2);
}

#[test]
fn distinct_and_equal_degree_factorization_cover_odd_fields() {
    // M31 is congruent to seven modulo eight. Thus -1 and -2 are quadratic
    // nonresidues, making X² + 1 and X² + 2 irreducible.
    let linear = Polynomial::<Mersenne31>::from_coefficients(&[m31(3), m31(1)]).expect("linear");
    let first = Polynomial::<Mersenne31>::from_coefficients(&[m31(1), m31(0), m31(1)])
        .expect("first quadratic");
    let second = Polynomial::<Mersenne31>::from_coefficients(&[m31(2), m31(0), m31(1)])
        .expect("second quadratic");
    let quadratics = first.multiply(&second).expect("quadratic product");
    let input = linear.multiply(&quadratics).expect("mixed product");

    let groups = input
        .distinct_degree_factorization()
        .expect("distinct degrees");
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].factor_degree, 1);
    assert_eq!(groups[0].factor, linear);
    assert_eq!(groups[1].factor_degree, 2);
    assert_eq!(groups[1].factor, quadratics);

    let split = quadratics
        .equal_degree_factorization(2)
        .expect("equal degrees");
    assert_eq!(split.len(), 2);
    assert!(split.contains(&first));
    assert!(split.contains(&second));
    assert_eq!(
        split[0].multiply(&split[1]).expect("reconstruct"),
        quadratics
    );
}

#[test]
fn equal_degree_factorization_covers_binary_extensions() {
    let mut irreducibles = Vec::new();
    'outer: for linear in 0_u16..=u8::MAX.into() {
        for constant in 1_u16..=u8::MAX.into() {
            let linear = b(linear as u8);
            let constant = b(constant as u8);
            let has_root = (0_u16..=u8::MAX.into()).any(|raw| {
                let root = b(raw as u8);
                root.mul(root).add(linear.mul(root)).add(constant).is_zero()
            });
            if !has_root {
                irreducibles.push(
                    Polynomial::<Gf8B>::from_coefficients(&[constant, linear, b(1)])
                        .expect("quadratic"),
                );
                if irreducibles.len() == 2 {
                    break 'outer;
                }
            }
        }
    }
    assert_eq!(irreducibles.len(), 2);
    let input = irreducibles[0].multiply(&irreducibles[1]).expect("input");
    let split = input.equal_degree_factorization(2).expect("equal degrees");
    assert_eq!(split.len(), 2);
    assert!(split.contains(&irreducibles[0]));
    assert!(split.contains(&irreducibles[1]));
}

#[test]
fn complete_factorization_preserves_irreducibles_and_multiplicities() {
    let linear = Polynomial::<Mersenne31>::from_coefficients(&[m31(3), m31(1)]).expect("linear");
    let first = Polynomial::<Mersenne31>::from_coefficients(&[m31(1), m31(0), m31(1)])
        .expect("first quadratic");
    let second = Polynomial::<Mersenne31>::from_coefficients(&[m31(2), m31(0), m31(1)])
        .expect("second quadratic");
    let input = power(&linear, 2)
        .multiply(&first)
        .unwrap()
        .multiply(&second)
        .unwrap();

    let factors = input.factor().expect("complete factorization");
    assert_eq!(factors.len(), 3);
    assert!(
        factors
            .iter()
            .any(|entry| entry.factor == linear && entry.multiplicity == 2)
    );
    assert!(
        factors
            .iter()
            .any(|entry| entry.factor == first && entry.multiplicity == 1)
    );
    assert!(
        factors
            .iter()
            .any(|entry| entry.factor == second && entry.multiplicity == 1)
    );
    assert!(first.is_irreducible().expect("irreducible"));
    assert!(second.is_irreducible().expect("irreducible"));
    assert!(linear.is_irreducible().expect("linear"));
    assert!(!input.is_irreducible().expect("reducible"));
    assert!(
        !Polynomial::<Mersenne31>::one()
            .unwrap()
            .is_irreducible()
            .unwrap()
    );
}

#[test]
fn allocating_root_extraction_supports_prime_fields() {
    let root_a = m31(3);
    let root_b = m31(11);
    let linear_a =
        Polynomial::<Mersenne31>::from_coefficients(&[root_a.neg(), m31(1)]).expect("linear");
    let linear_b =
        Polynomial::<Mersenne31>::from_coefficients(&[root_b.neg(), m31(1)]).expect("linear");
    let rootless = Polynomial::<Mersenne31>::from_coefficients(&[m31(1), m31(0), m31(1)])
        .expect("irreducible quadratic");
    let input = power(&linear_a, 2)
        .multiply(&linear_b)
        .unwrap()
        .multiply(&rootless)
        .unwrap();

    let mut expected = vec![root_a, root_b];
    expected.sort_by_key(|root| poly_ring::internals::element_key::<Mersenne31>(*root));
    assert_eq!(
        base_field_roots(&input).expect("prime roots"),
        BaseFieldRoots::Finite(expected)
    );
    assert_eq!(
        base_field_roots(&Polynomial::<Mersenne31>::zero()).unwrap(),
        BaseFieldRoots::All
    );
}

#[test]
fn factorization_rejects_undefined_stage_inputs() {
    assert_eq!(
        Polynomial::<Gf8B>::zero().square_free_factorization(),
        Err(FactorizationError::ZeroPolynomial)
    );
    let repeated = power(
        &Polynomial::<Gf8B>::from_coefficients(&[b(1), b(1)]).expect("linear"),
        2,
    );
    assert_eq!(
        repeated.distinct_degree_factorization(),
        Err(FactorizationError::NotSquareFree)
    );
    assert_eq!(
        repeated.equal_degree_factorization(0),
        Err(FactorizationError::ZeroFactorDegree)
    );
}
