//! Property tests: randomized inputs, fixed seed, across representative
//! fields.

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16};
use poly_ring::Polynomial;
use proptest::prelude::*;

fn noise<F: FieldKernels>(len: usize, seed: u64) -> Polynomial<F> {
    let mut state = seed;
    let coefficients: Vec<F::Elem> = (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bytes = state.to_le_bytes();
            F::read(&bytes[..F::BYTES])
        })
        .collect();
    Polynomial::from_coefficients(&coefficients).expect("noise polynomial")
}

fn coefficients_strategy(max_len: usize) -> impl Strategy<Value = Vec<u8>> {
    (1..max_len).prop_flat_map(|len| proptest::collection::vec(any::<u8>(), len))
}

fn poly_from_bytes<F: FieldKernels>(bytes: &[u8], width: usize) -> Polynomial<F> {
    let coefficients: Vec<F::Elem> = bytes
        .chunks(width)
        .map(|chunk| {
            let mut buffer = [0_u8; 16];
            buffer[..chunk.len()].copy_from_slice(chunk);
            F::read(&buffer[..width])
        })
        .collect();
    Polynomial::from_coefficients(&coefficients).expect("polynomial")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn multiplication_is_commutative_and_distributive_gf8(a in coefficients_strategy(24), b in coefficients_strategy(24), c in coefficients_strategy(24)) {
        let a = poly_from_bytes::<Gf8B>(&a, 1);
        let b = poly_from_bytes::<Gf8B>(&b, 1);
        let c = poly_from_bytes::<Gf8B>(&c, 1);
        prop_assert_eq!(a.multiply(&b).unwrap(), b.multiply(&a).unwrap());
        // a·(b + c) == a·b + a·c.
        let left = a.multiply(&b.add(&c).unwrap()).unwrap();
        let right = a
            .multiply(&b)
            .unwrap()
            .add(&a.multiply(&c).unwrap())
            .unwrap();
        prop_assert_eq!(left, right);
    }

    #[test]
    fn division_identity_holds_gf8(a in coefficients_strategy(28), b in coefficients_strategy(12)) {
        let a = poly_from_bytes::<Gf8B>(&a, 1);
        let b = poly_from_bytes::<Gf8B>(&b, 1);
        proptest::prop_assume!(!b.is_zero());
        let (q, r) = a.div_rem(&b).unwrap();
        prop_assert!(r.degree().is_none_or(|degree| degree < b.degree().unwrap()));
        prop_assert_eq!(q.multiply(&b).unwrap().add(&r).unwrap(), a);
    }

    #[test]
    fn bezout_identity_holds_gf16(a in coefficients_strategy(24), b in coefficients_strategy(18)) {
        let a = poly_from_bytes::<Gf16>(&a, 2);
        let b = poly_from_bytes::<Gf16>(&b, 2);
        proptest::prop_assume!(!a.is_zero() && !b.is_zero());
        let relation = a.gcd_ext(&b).unwrap();
        let identity = relation
            .a_cofactor
            .multiply(&a)
            .unwrap()
            .add(&relation.b_cofactor.multiply(&b).unwrap())
            .unwrap();
        let gcd = relation.gcd.clone();
        prop_assert_eq!(&identity, &gcd);
        prop_assert!(a.div_rem(&gcd).unwrap().1.is_zero());
        prop_assert!(b.div_rem(&gcd).unwrap().1.is_zero());
    }

    #[test]
    fn karatsuba_is_byte_identical_to_schoolbook_gf8(a in coefficients_strategy(260), b in coefficients_strategy(260)) {
        let a = poly_from_bytes::<Gf8B>(&a, 1);
        let b = poly_from_bytes::<Gf8B>(&b, 1);
        proptest::prop_assume!(!a.is_zero() && !b.is_zero());
        prop_assert_eq!(
            poly_ring::karatsuba_multiply(&a, &b).unwrap(),
            a.multiply_truncated(&b, a.coefficient_count() + b.coefficient_count()).unwrap()
        );
    }

    #[test]
    fn chien_and_equal_degree_root_sets_agree_gf8(a in coefficients_strategy(10)) {
        let a = poly_from_bytes::<Gf8B>(&a, 1);
        proptest::prop_assume!(!a.is_zero());
        let chien = poly_ring::chien_roots(&a).unwrap();
        let split = poly_ring::base_field_roots(&a).unwrap();
        prop_assert_eq!(chien, split);
    }

    #[test]
    fn newton_and_lagrange_interpolation_agree_gf16(points_seed in any::<u64>()) {
        // Distinct points drawn from generator powers.
        let points: Vec<<Gf16 as fgf::field::Field>::Elem> = (1..=9)
            .map(|exponent| <Gf16 as fgf::field::Field>::GENERATOR.pow(exponent + (points_seed % 7)))
            .collect();
        let values = noise::<Gf16>(9, points_seed).coefficients().take(9).collect::<Vec<_>>();
        let newton = poly_ring::interpolate_newton::<Gf16>(&points, &values).unwrap();
        let lagrange = poly_ring::interpolate_lagrange::<Gf16>(&points, &values).unwrap();
        prop_assert_eq!(&newton, &lagrange);
        for (point, value) in points.iter().zip(&values) {
            prop_assert_eq!(newton.evaluate(*point), *value);
        }
        drop(lagrange);
    }

    #[test]
    fn series_inverse_is_self_inverse_gf8(a in coefficients_strategy(40), t in 1usize..40) {
        let a = poly_from_bytes::<Gf8B>(&a, 1);
        proptest::prop_assume!(!a.coefficient(0).is_zero());
        let inverse = a.inverse_mod_x_power(t).unwrap();
        let product = a.multiply_truncated(&inverse, t).unwrap();
        // The product below x^t is exactly the constant one.
        prop_assert!(product.coefficient_count() <= 1 && product.coefficient(0).is_one());
    }
}
