//! Lifting reuse behavior and error boundaries.

use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, gf8b};
use poly_ring::{
    Polynomial, RootError, RothRuckensteinLimits, RothRuckensteinScratch, roth_ruckenstein_roots,
    roth_ruckenstein_roots_into,
};

use crate::oracles;
use oracles::{noise, noise_poly};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// Build `Q(X,Y) = prod_k (Y + f_k(X))` in row form for planted roots.
fn bivariate_with_roots<F: FieldKernels>(roots: &[&[F::Elem]]) -> Vec<Polynomial<F>> {
    let width = roots.len() + 1;
    let mut rows = vec![Polynomial::one().expect("one")];
    rows.resize_with(width, Polynomial::zero);
    for coefficients in roots {
        let root = Polynomial::from_coefficients(coefficients).expect("planted root");
        let mut next = vec![Polynomial::zero(); width];
        for (j, row) in rows.iter().enumerate() {
            next[j] = next[j].add(&row.multiply(&root).expect("f")).expect("add");
        }
        for j in 1..width {
            next[j] = next[j].add(&rows[j - 1]).expect("add");
        }
        rows = next;
    }
    rows
}

fn check_composition<F: FieldKernels>(rows: &[Polynomial<F>], candidate: &Polynomial<F>) -> bool {
    let mut composition = Polynomial::<F>::zero();
    for row in rows.iter().rev() {
        composition = composition.multiply(candidate).expect("multiply");
        composition = composition.add(row).expect("add");
    }
    composition.is_zero()
}

/// Three planted roots exercise the stack traversal past depth one, with
/// output sorting and dedup over the verified set.
#[test]
fn three_planted_roots_come_back_sorted_and_verified() {
    let root_a = noise::<Gf16>(3, 0xE201);
    let root_b = noise::<Gf16>(2, 0xE202);
    let root_c = noise::<Gf16>(4, 0xE203);
    let rows = bivariate_with_roots::<Gf16>(&[&root_a, &root_b, &root_c]);
    let limits = RothRuckensteinLimits::new(100_000, 64);
    let found = roth_ruckenstein_roots(&rows, 5, limits).expect("roots");
    assert_eq!(found.len(), 3);
    let to_poly = |coefficients: &[<Gf16 as Field>::Elem]| {
        Polynomial::<Gf16>::from_coefficients(coefficients).expect("planted")
    };
    assert!(found.contains(&to_poly(&root_a)));
    assert!(found.contains(&to_poly(&root_b)));
    assert!(found.contains(&to_poly(&root_c)));
    // Deduplicated: no candidate appears twice.
    for (i, a) in found.iter().enumerate() {
        for other in &found[i + 1..] {
            assert_ne!(a, other);
        }
    }
    for candidate in &found {
        assert!(check_composition(&rows, candidate));
    }
}

/// The `into` form recycles a pre-filled output and a warmed scratch: the
/// second run over new rows of the same shape reuses every pool.
#[test]
fn into_form_recycles_output_and_scratch_pools() {
    let rows = bivariate_with_roots::<Gf8B>(&[&noise::<Gf8B>(2, 0xE204)]);
    let limits = RothRuckensteinLimits::new(10_000, 64);
    let mut scratch = RothRuckensteinScratch::<Gf8B>::new();
    let mut output = vec![
        noise_poly::<Gf8B>(30, 0xE205),
        noise_poly::<Gf8B>(30, 0xE206),
    ];
    roth_ruckenstein_roots_into(&mut output, &rows, 2, limits, &mut scratch).expect("first");
    assert_eq!(output.len(), 1);
    assert!(check_composition(&rows, &output[0]));
    let capacity = scratch.capacity();
    // Second run over different rows of the same shape: pools are reused.
    let rows2 = bivariate_with_roots::<Gf8B>(&[&noise::<Gf8B>(2, 0xE207)]);
    let mut output2 = std::mem::take(&mut output);
    roth_ruckenstein_roots_into(&mut output2, &rows2, 2, limits, &mut scratch).expect("second");
    assert_eq!(output2.len(), 1);
    assert!(check_composition(&rows2, &output2[0]));
    assert!(scratch.capacity() >= capacity);
}

/// A tight output limit fails once the second verified root arrives, naming
/// the resource and both counts.
#[test]
fn tight_output_limit_names_resource_and_counts() {
    let root_a = noise::<Gf8B>(2, 0xE208);
    let root_b = noise::<Gf8B>(2, 0xE209);
    let rows = bivariate_with_roots::<Gf8B>(&[&root_a, &root_b]);
    assert_eq!(
        roth_ruckenstein_roots(&rows, 3, RothRuckensteinLimits::new(100_000, 1)).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein output roots",
            required: 2,
            limit: 1,
        })
    );
}

/// A tight work-item limit fails on the first child expansion.
#[test]
fn tight_work_limit_fails_the_first_expansion() {
    let root_a = noise::<Gf8B>(2, 0xE210);
    let root_b = noise::<Gf8B>(2, 0xE211);
    let rows = bivariate_with_roots::<Gf8B>(&[&root_a, &root_b]);
    assert_eq!(
        roth_ruckenstein_roots(&rows, 3, RothRuckensteinLimits::new(1, 64)).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein work items",
            required: 2,
            limit: 1,
        })
    );
}

/// A bivariate with no base-field root at depth zero returns the empty set
/// without error (the initial frame has no roots to expand).
#[test]
fn rootless_constant_term_returns_empty() {
    // Q(X,Y) = (X + 1) + Y over Gf8B with max_degree 0: the constant-X row
    // is (1, 1), whose only base root expands to a nonroot at full check...
    // simpler: Q(X,Y) = c + Y with c nonzero constant and max_degree 0
    // forces the depth-zero frame to hold exactly one root; instead use a
    // Q whose constant term has no base root at all.
    // Q(X,Y) = (X^2 + X + 1) + X·Y: constant-X row is (1, 0) -> polynomial
    // "1", which has no roots, so extraction is empty.
    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(1)]).expect("one"),
        Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1)]).expect("x"),
    ];
    let found =
        roth_ruckenstein_roots(&rows, 4, RothRuckensteinLimits::new(10_000, 64)).expect("empty");
    assert!(found.is_empty());
}

/// Max degree zero still finds constant roots: the full-depth check runs on
/// the first frame's roots directly.
#[test]
fn degree_zero_finds_constant_roots() {
    // Q(X,Y) = Y^2 + (f0+f1)Y + f0·f1 with distinct constants f0, f1:
    // both constants are exact roots at max_degree 0.
    let f0 = b(3);
    let f1 = b(7);
    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[f0.mul(f1)]).expect("q0"),
        Polynomial::<Gf8B>::from_coefficients(&[f0.add(f1)]).expect("q1"),
        Polynomial::<Gf8B>::one().expect("q2"),
    ];
    let found =
        roth_ruckenstein_roots(&rows, 0, RothRuckensteinLimits::new(10_000, 64)).expect("roots");
    assert_eq!(found.len(), 2);
    for candidate in &found {
        assert_eq!(candidate.coefficient_count(), 1);
        assert!(check_composition(&rows, candidate));
    }
}

/// An all-zero row set is the zero bivariate polynomial, rejected up front.
#[test]
fn all_zero_rows_are_rejected() {
    let rows = vec![Polynomial::<Gf8B>::zero(), Polynomial::<Gf8B>::zero()];
    assert_eq!(
        roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(10_000, 64)).map(|_| ()),
        Err(RootError::ZeroBivariatePolynomial)
    );
}

/// A factor with an X valuation lifts through the initial divide: Q has a
/// common X power and still recovers the planted root.
#[test]
fn valuation_lifts_through_the_common_x_power() {
    // Q(X,Y) = X·(Y + f(X)): rows [X·f, X]; valuation 1 divides out first.
    let f = Polynomial::<Gf8B>::from_coefficients(&noise::<Gf8B>(2, 0xE212)).expect("f");
    let x = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1)]).expect("x");
    let rows = vec![x.multiply(&f).expect("xf"), x.clone()];
    let found =
        roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(10_000, 64)).expect("roots");
    assert!(found.contains(&f));
    for candidate in &found {
        assert!(check_composition(&rows, candidate));
    }
}
