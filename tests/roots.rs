//! Root-finding: Chien, equal-degree, linearized, and power-series lifting,
//! each cross-checked against an independent oracle.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16};
use poly_ring::roots::element_key;
use poly_ring::{
    BaseFieldRoots, Polynomial, RootError, chien_roots, linearized_roots, roth_ruckenstein_roots,
};

mod oracles;
use oracles::{naive_evaluate, noise, noise_poly};

/// A random polynomial with a planted set of distinct roots.
/// A polynomial with exactly `root_count` planted distinct roots (constant
/// filler, so the root count is exact).
fn with_roots<F: FieldKernels>(seed: u64, root_count: usize) -> Polynomial<F> {
    let mut polynomial = Polynomial::one().expect("one");
    let mut roots = Vec::new();
    for index in 0..root_count {
        let mut root = noise::<F>(1, seed + 100 * index as u64 + 1)[0];
        while roots.contains(&root) || root.is_zero() {
            root = root.mul(F::GENERATOR).add(F::Elem::ONE);
        }
        roots.push(root);
        polynomial = polynomial.multiply_x_plus(root).expect("linear factor");
    }
    polynomial
        .multiply(&noise_poly::<F>(1, seed + 7))
        .expect("constant filler")
}

fn assert_backend_agreement<F: FieldKernels>() {
    for (seed, count) in [(0x11, 1), (0x22, 2), (0x33, 3), (0x44, 5), (0x55, 8)] {
        let polynomial = with_roots::<F>(seed, count);
        let chien = chien_roots(&polynomial).expect("chien");
        let equal_degree = poly_ring::base_field_roots(&polynomial).expect("equal degree");

        // Both backends return the same finite set in the same frozen order,
        // or both report the zero polynomial.
        match (&chien, &equal_degree) {
            (BaseFieldRoots::Finite(a), BaseFieldRoots::Finite(b)) => assert_eq!(a, b),
            (BaseFieldRoots::All, BaseFieldRoots::All) => {}
            mismatch => panic!("backends disagree: {mismatch:?}"),
        }

        let roots = chien.into_finite().expect("finite roots");
        assert_eq!(
            roots.len(),
            count,
            "root count must match the planted factors"
        );
        // Independent Horner verification of every root.
        for root in &roots {
            assert!(naive_evaluate(&polynomial, *root).is_zero());
        }
        // Root order is the frozen canonical element-key order.
        let keys: Vec<u128> = roots.iter().map(|root| element_key::<F>(*root)).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted);
        assert!(roots.windows(2).all(|pair| pair[0] != pair[1]));
    }

    // The zero polynomial vanishes everywhere; a nonzero constant nowhere.
    assert_eq!(
        chien_roots(&Polynomial::<F>::zero()).expect("chien"),
        BaseFieldRoots::All
    );
    assert_eq!(
        poly_ring::base_field_roots(&Polynomial::<F>::zero()).expect("equal degree"),
        BaseFieldRoots::All
    );
    assert_eq!(
        chien_roots(&noise_poly::<F>(1, 0x77)).expect("chien"),
        BaseFieldRoots::Finite(Vec::new())
    );
    assert_eq!(
        poly_ring::base_field_roots(&noise_poly::<F>(1, 0x77)).expect("equal degree"),
        BaseFieldRoots::Finite(Vec::new())
    );
    // X itself has the single root 0, exercising the constant-term path.
    let x = Polynomial::<F>::from_coefficients(&[F::Elem::ZERO, F::Elem::ONE]).expect("X");
    let expected = BaseFieldRoots::Finite(vec![F::Elem::ZERO]);
    assert_eq!(chien_roots(&x).expect("chien"), expected);
    assert_eq!(
        poly_ring::base_field_roots(&x).expect("equal degree"),
        expected
    );
}

#[test]
fn chien_and_equal_degree_agree_across_fields() {
    assert_backend_agreement::<Gf8B>();
    assert_backend_agreement::<Gf8B>();
    assert_backend_agreement::<Gf16>();
}

#[test]
fn chien_matches_naive_full_scan() {
    for seed in 0..8 {
        let polynomial = noise_poly::<Gf8B>(7, 0xA000 + seed);
        let chien = chien_roots(&polynomial)
            .expect("chien")
            .into_finite()
            .expect("finite");
        let naive = oracles::naive_roots_small_field(&polynomial);
        assert_eq!(chien, naive);
    }
}

fn assert_linearized_agreement<F: FieldKernels>() {
    // L(X) = a·X + b·X² over the field: roots form a GF(2)-affine space and
    // must match both the Chien scan and independent Horner checks.
    let a = noise::<F>(1, 0xB001)[0];
    let b = noise::<F>(1, 0xB002)[0];
    if a.is_zero() && b.is_zero() {
        return;
    }
    let mut coefficients = vec![F::Elem::ZERO; 3];
    coefficients[1] = a;
    coefficients[2] = b;
    let linearized = Polynomial::<F>::from_coefficients(&coefficients).expect("linearized");
    for affine_seed in 0..4 {
        let affine = noise::<F>(1, 0xB100 + affine_seed)[0];
        let roots = linearized_roots(&linearized, affine).expect("linearized roots");
        let ordinary = {
            let mut shifted = linearized.clone();
            shifted.set_coefficient(0, affine).expect("constant");
            shifted
        };
        // The ordinary polynomial view has the same root set as the
        // linearized solver, and the Chien scan agrees (frozen order).
        let chien = chien_roots(&ordinary)
            .expect("chien")
            .into_finite()
            .expect("finite");
        assert_eq!(roots, chien);
        for root in &roots {
            // Independent Horner over the ordinary representation.
            assert!(naive_evaluate(&ordinary, *root).is_zero());
        }
        assert!(!roots.is_empty() || !ordinary.is_zero());
    }

    // A coefficient at a non-power-of-two degree is rejected with the
    // offending degree.
    let bad = Polynomial::<F>::from_coefficients(&[
        F::Elem::ZERO,
        noise::<F>(1, 0xB200)[0],
        F::Elem::ZERO,
        F::Elem::ONE,
    ])
    .expect("degree 3");
    assert_eq!(
        linearized_roots(&bad, F::Elem::ZERO),
        Err(RootError::NotLinearized { degree: 3 })
    );
}

#[test]
fn linearized_solver_agrees_with_chien() {
    assert_linearized_agreement::<Gf8B>();
    assert_linearized_agreement::<Gf8B>();
    assert_linearized_agreement::<Gf16>();
}

#[test]
fn linearized_zero_polynomial_covers_the_field() {
    let all = linearized_roots(&Polynomial::<Gf8B>::zero(), <Gf8B as Field>::Elem::ZERO)
        .expect("all roots");
    assert_eq!(all.len(), 256);
    let none = linearized_roots(&Polynomial::<Gf8B>::zero(), <Gf8B as Field>::Elem::ONE)
        .expect("no roots");
    assert!(none.is_empty());
}

/// Build `Q(X, Y) = Q_0(X) + Q_1(X)·Y + Q_2(X)·Y²` with planted polynomial
/// roots: for each planted root `f`, multiply `(Y + f(X))` into the product.
fn bivariate_with_roots<F: FieldKernels>(seed: u64, roots: &[&[F::Elem]]) -> Vec<Polynomial<F>> {
    let mut rows = vec![
        Polynomial::one().expect("one"),
        Polynomial::zero(),
        Polynomial::zero(),
    ];
    for coefficients in roots {
        let root = Polynomial::from_coefficients(coefficients).expect("planted root");
        // a(Y)·(f + Y): new_j = f·a_j + a_{j-1}.
        let mut next = vec![Polynomial::zero(); 3];
        for (j, row) in rows.iter().enumerate() {
            let scaled = row.multiply(&root).expect("f·a");
            next[j] = next[j].add(&scaled).expect("add");
        }
        for j in 1..rows.len() {
            next[j] = next[j].add(&rows[j - 1]).expect("add");
        }
        rows = next;
    }
    let _ = seed;
    rows
}

#[test]
fn roth_ruckenstein_finds_planted_roots() {
    let root_a: Vec<<Gf8B as Field>::Elem> = oracles::noise::<Gf8B>(3, 0xC001);
    let root_b: Vec<<Gf8B as Field>::Elem> = oracles::noise::<Gf8B>(2, 0xC002);
    let rows = bivariate_with_roots::<Gf8B>(0xC000, &[&root_a, &root_b]);
    let found = roth_ruckenstein_roots(&rows, 4, poly_ring::RothRuckensteinLimits::new(10_000, 64))
        .expect("roots");
    let to_poly = |coefficients: &[<Gf8B as Field>::Elem]| {
        Polynomial::<Gf8B>::from_coefficients(coefficients).expect("planted")
    };
    assert!(found.contains(&to_poly(&root_a)));
    assert!(found.contains(&to_poly(&root_b)));
    // Every returned candidate is a true root of Q(X, f(X)) == 0.
    for candidate in &found {
        let mut composition = Polynomial::<Gf8B>::zero();
        for row in rows.iter().rev() {
            composition = composition.multiply(candidate).expect("multiply");
            composition = composition.add(row).expect("add");
        }
        assert!(composition.is_zero());
    }
    // The empty-rows input is the zero bivariate.
    assert_eq!(
        roth_ruckenstein_roots::<Gf8B>(&[], 2, poly_ring::RothRuckensteinLimits::new(10_000, 64)),
        Err(RootError::ZeroBivariatePolynomial)
    );
}

#[cfg(feature = "fft")]
#[test]
fn alekhnovich_and_roth_ruckenstein_return_the_same_roots() {
    use poly_ring::{AlekhnovichLimits, AlekhnovichScratch, alekhnovich_roots};

    let root_a: Vec<<Gf16 as Field>::Elem> = oracles::noise::<Gf16>(3, 0xD001);
    let root_b: Vec<<Gf16 as Field>::Elem> = oracles::noise::<Gf16>(2, 0xD002);
    let rows = bivariate_with_roots::<Gf16>(0xD000, &[&root_a, &root_b]);
    let limits = AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 22, 1 << 26, 256);
    let mut scratch = AlekhnovichScratch::new();
    let lifted = alekhnovich_roots(&rows, 4, limits, &mut scratch).expect("alekhnovich roots");
    let prefixed = roth_ruckenstein_roots(
        &rows,
        4,
        poly_ring::RothRuckensteinLimits::new(1_000_000, 256),
    )
    .expect("roth-ruckenstein roots");
    assert_eq!(lifted, prefixed);

    // The forced D&C path (crossover 0) also agrees with the prefix lift.
    let divide_and_conquer = alekhnovich_roots(
        &rows,
        4,
        limits.with_roth_ruckenstein_crossover(0),
        &mut scratch,
    )
    .expect("divide-and-conquer roots");
    assert_eq!(divide_and_conquer, prefixed);
}

#[test]
fn root_order_is_stable_across_runs() {
    let polynomial = with_roots::<Gf16>(0x91, 6);
    let mut first = None;
    for _ in 0..3 {
        let roots = chien_roots(&polynomial)
            .expect("chien")
            .into_finite()
            .expect("finite");
        if let Some(previous) = &first {
            assert_eq!(&roots, previous);
        } else {
            first = Some(roots);
        }
        let split = poly_ring::base_field_roots(&polynomial)
            .expect("equal degree")
            .into_finite()
            .expect("finite");
        assert_eq!(Some(&split), first.as_ref());
    }
}
