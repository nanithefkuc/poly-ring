//! Prepared polynomial quotient-ring behavior.

use fgf::{Mersenne31, mersenne31};
use poly_ring::{ModulusPlan, ModulusScratch, Polynomial, PolynomialError};

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn polynomial(coefficients: &[u32]) -> Polynomial<Mersenne31> {
    Polynomial::from_coefficients(&coefficients.iter().copied().map(m31).collect::<Vec<_>>())
        .expect("polynomial")
}

#[test]
fn prepared_arithmetic_matches_direct_reduction() {
    let modulus = polynomial(&[1, 0, 0, 1]);
    let plan = ModulusPlan::new(&modulus).expect("plan");
    let left = polynomial(&[3, 4, 5, 6]);
    let right = polynomial(&[7, 8, 9]);

    assert_eq!(plan.modulus(), &modulus);
    assert_eq!(
        plan.reduce(&left).expect("reduce"),
        left.remainder(&modulus).unwrap()
    );
    assert_eq!(
        plan.multiply(&left, &right).expect("multiply"),
        left.multiply(&right).unwrap().remainder(&modulus).unwrap()
    );
    assert_eq!(
        plan.square(&left).expect("square"),
        left.square().unwrap().remainder(&modulus).unwrap()
    );

    let mut expected = Polynomial::<Mersenne31>::one().expect("one");
    for _ in 0..9 {
        expected = expected
            .multiply(&left)
            .unwrap()
            .remainder(&modulus)
            .unwrap();
    }
    assert_eq!(plan.pow(&left, 9).expect("power"), expected);
    assert!(plan.pow(&left, 0).expect("unit power").is_one());
}

#[test]
fn quotient_inverse_obeys_the_multiplicative_identity() {
    let modulus = polynomial(&[1, 0, 1]);
    let plan = ModulusPlan::new(&modulus).expect("plan");
    let value = polynomial(&[0, 1]);
    let inverse = plan.inverse(&value).expect("inverse");
    assert!(plan.multiply(&value, &inverse).expect("identity").is_one());

    let factor = polynomial(&[1, 1]);
    let reducible_modulus = factor.multiply(&polynomial(&[2, 1])).unwrap();
    let reducible_plan = ModulusPlan::new(&reducible_modulus).expect("plan");
    assert_eq!(
        reducible_plan.inverse(&factor),
        Err(PolynomialError::NotInvertibleModulo)
    );
}

#[test]
fn prepared_modular_composition_matches_full_composition() {
    let modulus = polynomial(&[2, 0, 0, 1]);
    let outer = polynomial(&[1, 2, 3, 4]);
    let inner = polynomial(&[5, 1, 1]);
    let plan = ModulusPlan::new(&modulus).expect("plan");
    let expected = outer.compose(&inner).unwrap().remainder(&modulus).unwrap();

    assert_eq!(plan.compose(&outer, &inner).expect("compose"), expected);
    assert_eq!(
        outer.compose_mod(&inner, &modulus).expect("compose mod"),
        expected
    );

    let mut scratch = ModulusScratch::new();
    let mut output = Polynomial::zero();
    plan.compose_into(&outer, &inner, &mut scratch, &mut output)
        .expect("compose into");
    assert_eq!(output, expected);
    plan.compose_into(&outer, &inner, &mut scratch, &mut output)
        .expect("repeat compose");
    assert_eq!(output, expected);
}

#[test]
fn quotient_plan_rejects_degenerate_moduli() {
    assert_eq!(
        ModulusPlan::<Mersenne31>::new(&Polynomial::zero()),
        Err(PolynomialError::DivisionByZero)
    );
    assert_eq!(
        ModulusPlan::new(&polynomial(&[7])),
        Err(PolynomialError::ConstantModulus)
    );
}

#[test]
fn general_composition_agrees_after_evaluation() {
    let outer = polynomial(&[3, 1, 4, 1]);
    let inner = polynomial(&[5, 9, 2]);
    let composed = outer.compose(&inner).expect("compose");
    let point = m31(6);
    assert_eq!(
        composed.evaluate(point),
        outer.evaluate(inner.evaluate(point))
    );
}
