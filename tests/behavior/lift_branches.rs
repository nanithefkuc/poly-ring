//! Roth–Ruckenstein traversal branches: limits, empty frames, and reuse.
//!
//! The agreement suites lift typical planted roots; every test here takes a
//! branch they never reach — the empty input, the `Y`-free input, the
//! work-item and output limits with exact payloads, the rootless constant
//! term, the common-`X` valuation, and scratch/output recycling — each
//! against exact root sets or error payloads.

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

/// Empty rows are the zero bivariate polynomial, rejected up front; a
/// `Y`-free row set has no bounded roots and returns empty.
#[test]
fn empty_and_y_free_inputs_take_early_exits() {
    assert_eq!(
        roth_ruckenstein_roots::<Gf8B>(&[], 2, RothRuckensteinLimits::new(10_000, 64)).map(|_| ()),
        Err(RootError::ZeroBivariatePolynomial)
    );
    let y_free = [noise_poly::<Gf8B>(4, 0xE501)];
    assert!(
        roth_ruckenstein_roots(&y_free, 4, RothRuckensteinLimits::new(10_000, 64))
            .expect("y-free")
            .is_empty()
    );
    // The limits accessors report the configured bounds.
    let limits = RothRuckensteinLimits::new(7, 9);
    assert_eq!(limits.max_work_items(), 7);
    assert_eq!(limits.max_output_roots(), 9);
}

/// A zero work-item budget fails the very first frame push with the exact
/// resource name and counts.
#[test]
fn zero_work_budget_fails_the_first_push() {
    let rows = bivariate_with_roots::<Gf8B>(&[&noise::<Gf8B>(2, 0xE502)]);
    assert_eq!(
        roth_ruckenstein_roots(&rows, 3, RothRuckensteinLimits::new(0, 64)).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein work items",
            required: 1,
            limit: 0,
        })
    );
}

/// A zero output budget fails on the first verified root, naming the
/// output resource.
#[test]
fn zero_output_budget_fails_the_first_root() {
    let rows = bivariate_with_roots::<Gf8B>(&[&noise::<Gf8B>(2, 0xE503)]);
    assert_eq!(
        roth_ruckenstein_roots(&rows, 3, RothRuckensteinLimits::new(100_000, 0)).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein output roots",
            required: 1,
            limit: 0,
        })
    );
}

/// A `max_degree` of `usize::MAX` overflows the coefficient count instead
/// of reserving the address space.
#[test]
fn max_degree_overflow_is_a_geometry_error() {
    let rows = bivariate_with_roots::<Gf8B>(&[&noise::<Gf8B>(2, 0xE504)]);
    assert!(matches!(
        roth_ruckenstein_roots(&rows, usize::MAX, RothRuckensteinLimits::new(10_000, 64)),
        Err(RootError::Polynomial(poly_ring::PolynomialError::Config(
            poly_ring::ConfigError::GeometryOverflow { .. }
        )))
    ));
}

/// A common `X` power divides out before lifting: `Q = X·(Y + f)` still
/// recovers the planted root, verified by composition.
#[test]
fn common_x_power_divides_out_first() {
    let f = Polynomial::<Gf8B>::from_coefficients(&noise::<Gf8B>(2, 0xE505)).expect("f");
    let x = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1)]).expect("x");
    let rows = vec![x.multiply(&f).expect("xf"), x.clone()];
    let found =
        roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(10_000, 64)).expect("roots");
    assert!(found.contains(&f));
    for candidate in &found {
        assert!(check_composition(&rows, candidate));
    }
}

/// Duplicate planted factors deduplicate: `Q = (Y + f)²` reports `f` once.
#[test]
fn duplicate_factors_report_each_root_once() {
    let coefficients = noise::<Gf16>(3, 0xE506);
    let rows = bivariate_with_roots::<Gf16>(&[&coefficients, &coefficients]);
    let found =
        roth_ruckenstein_roots(&rows, 4, RothRuckensteinLimits::new(100_000, 64)).expect("roots");
    assert_eq!(found.len(), 1);
    assert!(check_composition(&rows, &found[0]));
}

/// The `into` form recycles a pre-filled output and a warmed scratch: the
/// second run over new rows of the same shape reuses every pool, and the
/// retained capacity never shrinks.
#[test]
fn into_form_recycles_output_and_pools() {
    let rows = bivariate_with_roots::<Gf8B>(&[&noise::<Gf8B>(2, 0xE507)]);
    let limits = RothRuckensteinLimits::new(10_000, 64);
    let mut scratch = RothRuckensteinScratch::<Gf8B>::new();
    let mut output = vec![noise_poly::<Gf8B>(30, 0xE508)];
    roth_ruckenstein_roots_into(&mut output, &rows, 2, limits, &mut scratch).expect("first");
    assert_eq!(output.len(), 1);
    let capacity = scratch.capacity();
    let rows2 = bivariate_with_roots::<Gf8B>(&[&noise::<Gf8B>(2, 0xE509)]);
    let mut output2 = std::mem::take(&mut output);
    roth_ruckenstein_roots_into(&mut output2, &rows2, 2, limits, &mut scratch).expect("second");
    assert_eq!(output2.len(), 1);
    assert!(check_composition(&rows2, &output2[0]));
    assert!(scratch.capacity() >= capacity);
    let _ = RothRuckensteinScratch::<Gf8B>::default();
}

/// A deep three-root lift sorts the verified set deterministically: two
/// runs return the identical order.
#[test]
fn deep_lift_returns_deterministic_order() {
    let rows = bivariate_with_roots::<Gf16>(&[
        &noise::<Gf16>(3, 0xE50A),
        &noise::<Gf16>(2, 0xE50B),
        &noise::<Gf16>(4, 0xE50C),
    ]);
    let limits = RothRuckensteinLimits::new(100_000, 64);
    let first = roth_ruckenstein_roots(&rows, 5, limits).expect("first");
    let second = roth_ruckenstein_roots(&rows, 5, limits).expect("second");
    assert_eq!(first, second);
    assert_eq!(first.len(), 3);
    for candidate in &first {
        assert!(check_composition(&rows, candidate));
    }
}

/// A branch that dies mid-lift recycles its child frame: `Q = Y^2 + (X+1)`
/// has one depth-zero root, but the substituted child has a nonzero
/// constant term with no roots, so extraction returns empty.
#[test]
fn dead_branch_recycles_its_child_frame() {
    // Q(X,Y) = (X + 1) + Y^2 over Gf8B.
    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(1), b(1)]).expect("x+1"),
        Polynomial::<Gf8B>::zero(),
        Polynomial::<Gf8B>::one().expect("one"),
    ];
    let found = roth_ruckenstein_roots(&rows, 3, RothRuckensteinLimits::new(10_000, 64))
        .expect("extraction");
    assert!(found.is_empty());
    for candidate in &found {
        assert!(check_composition(&rows, candidate));
    }
}

/// Trailing zero rows in the caller's input are normalized away through
/// the pooled rows: `Q = (Y + f) + 0·Y^2` still recovers `f`.
#[test]
fn trailing_zero_rows_normalize_through_the_pool() {
    let coefficients = noise::<Gf8B>(2, 0xE50D);
    let f = Polynomial::<Gf8B>::from_coefficients(&coefficients).expect("f");
    let mut rows = bivariate_with_roots::<Gf8B>(&[&coefficients]);
    rows.push(Polynomial::zero());
    rows.push(Polynomial::zero());
    let found =
        roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(10_000, 64)).expect("roots");
    assert!(found.contains(&f));
}

/// Reusing a scratch after a limit error recycles the abandoned frames:
/// the second extraction over fresh rows succeeds and reuses capacity.
#[test]
fn scratch_reuse_after_limit_error_recycles_frames() {
    let root_a = noise::<Gf8B>(2, 0xE50E);
    let root_b = noise::<Gf8B>(2, 0xE50F);
    let rows = bivariate_with_roots::<Gf8B>(&[&root_a, &root_b]);
    let mut scratch = RothRuckensteinScratch::<Gf8B>::new();
    let mut output = Vec::new();
    // Limit 1 admits the initial frame but fails the first child
    // expansion, stranding the initial frame on the stack.
    assert_eq!(
        roth_ruckenstein_roots_into(
            &mut output,
            &rows,
            3,
            RothRuckensteinLimits::new(1, 64),
            &mut scratch
        )
        .map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein work items",
            required: 2,
            limit: 1,
        })
    );
    // The abandoned frame is still on the stack; the next call recycles
    // it instead of leaking or double-running.
    roth_ruckenstein_roots_into(
        &mut output,
        &rows,
        2,
        RothRuckensteinLimits::new(10_000, 64),
        &mut scratch,
    )
    .expect("second");
    assert_eq!(output.len(), 2);
    for candidate in &output {
        assert!(check_composition(&rows, candidate));
    }
}

/// Over a prime field the base-field factorizer is unsupported: lifting
/// surfaces that error instead of inventing roots.
#[test]
fn prime_field_lift_reports_unsupported_base_factorization() {
    use fgf::Mersenne31;

    let coefficients = oracles::noise::<Mersenne31>(2, 0xE510);
    let f = Polynomial::<Mersenne31>::from_coefficients(&coefficients).expect("f");
    let one = Polynomial::<Mersenne31>::one().expect("one");
    let rows = vec![f, one];
    assert_eq!(
        roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(10_000, 64)).map(|_| ()),
        Err(RootError::UnsupportedField {
            field_order: <Mersenne31 as fgf::field::Field>::ORDER,
            element_bytes: Mersenne31::BYTES,
        })
    );
}

/// Candidates sharing a constant prefix sort by length: a constant root
/// and a linear root with the same constant term exercise the
/// count-then-compare tail of the candidate ordering.
#[test]
fn shared_prefix_candidates_sort_by_length() {
    use fgf::gf8b;

    // Q(X,Y) = (Y + a)(Y + a + b·X) with nonzero a, b: roots [a] and
    // [a, b] share the degree-zero coefficient.
    let a = gf8b::Elem::from_raw(0x29);
    let b_coef = gf8b::Elem::from_raw(0x53);
    let f1 = Polynomial::<Gf8B>::from_coefficients(&[a]).expect("f1");
    let f2 = Polynomial::<Gf8B>::from_coefficients(&[a, b_coef]).expect("f2");
    let rows = bivariate_with_roots::<Gf8B>(&[&[a], &[a, b_coef]]);
    let found =
        roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(10_000, 64)).expect("roots");
    assert_eq!(found.len(), 2);
    assert!(found.contains(&f1));
    assert!(found.contains(&f2));
    // Deterministic order: two runs agree exactly.
    let again =
        roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(10_000, 64)).expect("roots");
    assert_eq!(found, again);
}

/// An internal zero `Y` row is skipped by the substitution without
/// affecting the lift: `Q = X + Y^2` still extracts its roots.
#[test]
fn internal_zero_row_skips_substitution_terms() {
    // Q(X,Y) = X + Y^2: rows [X, 0, 1].
    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1)]).expect("x"),
        Polynomial::<Gf8B>::zero(),
        Polynomial::<Gf8B>::one().expect("one"),
    ];
    let found =
        roth_ruckenstein_roots(&rows, 3, RothRuckensteinLimits::new(10_000, 64)).expect("roots");
    for candidate in &found {
        assert!(check_composition(&rows, candidate));
    }
}
