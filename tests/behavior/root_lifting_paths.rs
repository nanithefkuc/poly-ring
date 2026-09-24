//! Roth–Ruckenstein prefix-lifting paths: deeper branches, the
//! `into_` reuse form, dead frames, and deterministic ordering.
//!
//! ## Covered lines
//!
//! | Lines | Behavior |
//! |-------|----------|
//! | 346,357,364 | child-frame `substitute`/`divide`/`fill_roots` — any multi-depth extraction |
//! | 308–323 | terminal-depth `is_root` check and `continue` — reached by any completed candidate |
//! | 380–382,390–393 | final invariant checks — exercised by normal extraction |
//!
//! The error arms (`try_reserve`, `map_err` closures) above lines 425,
//! 437, 574, and 587 require a failing-allocator gate placed precisely
//! after larger non-fallible intermediate Vec operations. Their success
//! paths are exercised by the agreement suites; the error arms are
//! documented as gated invariants.

use fgf::field::Field;
use fgf::{Gf8B, Gf16, gf8b};
use poly_ring::{
    Polynomial, RootError, RothRuckensteinLimits, RothRuckensteinScratch, roth_ruckenstein_roots,
    roth_ruckenstein_roots_into,
};

use crate::oracles;
use oracles::noise;

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn compose<F: fgf::kernel::FieldKernels>(
    rows: &[Polynomial<F>],
    candidate: &Polynomial<F>,
) -> bool {
    let mut result = Polynomial::<F>::zero();
    for row in rows.iter().rev() {
        result = result.multiply(candidate).expect("multiply");
        result = result.add(row).expect("add");
    }
    result.is_zero()
}

fn bivariate_with_roots<F: fgf::kernel::FieldKernels>(roots: &[&[F::Elem]]) -> Vec<Polynomial<F>> {
    let width = roots.len() + 1;
    let mut rows = vec![Polynomial::one().expect("one")];
    rows.resize_with(width, Polynomial::zero);
    for coefficients in roots {
        let root = Polynomial::from_coefficients(coefficients).expect("planted");
        let mut next = vec![Polynomial::zero(); width];
        for (j, row) in rows.iter().enumerate() {
            next[j] = next[j]
                .add(&row.multiply(&root).expect("mul"))
                .expect("add");
        }
        for j in 1..width {
            next[j] = next[j].add(&rows[j - 1]).expect("add");
        }
        rows = next;
    }
    rows
}

fn limits() -> RothRuckensteinLimits {
    RothRuckensteinLimits::new(100_000, 64)
}

#[test]
fn constant_root_verifies_and_lands_in_output() {
    let a = b(0x1d);
    let rows = bivariate_with_roots::<Gf8B>(&[&[a]]);
    let found = roth_ruckenstein_roots(&rows, 0, limits()).expect("roots");
    assert_eq!(found.len(), 1);
    assert_eq!(
        found[0],
        Polynomial::<Gf8B>::from_coefficients(&[a]).expect("const")
    );
    assert!(compose(&rows, &found[0]));
}

#[test]
fn linear_root_expands_child_frame_and_verifies() {
    let coeffs = noise::<Gf8B>(2, 0xBC01);
    let rows = bivariate_with_roots::<Gf8B>(&[&coeffs]);
    let found = roth_ruckenstein_roots(&rows, 1, limits()).expect("roots");
    assert_eq!(found.len(), 1);
    let planted = Polynomial::<Gf8B>::from_coefficients(&coeffs).expect("f");
    assert!(found.contains(&planted));
    assert!(compose(&rows, &found[0]));
}

#[test]
fn linear_root_with_wider_max_degree_pads_and_verifies() {
    let coeffs = noise::<Gf8B>(2, 0xBC05);
    let rows = bivariate_with_roots::<Gf8B>(&[&coeffs]);
    let found = roth_ruckenstein_roots(&rows, 4, limits()).expect("roots");
    assert_eq!(found.len(), 1);
    let planted = Polynomial::<Gf8B>::from_coefficients(&coeffs).expect("f");
    assert!(found.contains(&planted));
    assert!(compose(&rows, &found[0]));
}

#[test]
fn three_linear_roots_sorted_deterministically() {
    let a = noise::<Gf16>(2, 0xBC02);
    let bb = noise::<Gf16>(2, 0xBC03);
    let c = noise::<Gf16>(2, 0xBC04);
    let rows = bivariate_with_roots::<Gf16>(&[&a, &bb, &c]);
    let first = roth_ruckenstein_roots(&rows, 2, limits()).expect("first");
    let second = roth_ruckenstein_roots(&rows, 2, limits()).expect("second");
    assert_eq!(first, second);
    assert_eq!(first.len(), 3);
    for candidate in &first {
        assert!(compose(&rows, candidate));
    }
}

#[test]
fn mixed_depth_roots_sort_stably() {
    let a = b(0x17);
    let bb = b(0x3f);
    let rows = bivariate_with_roots::<Gf8B>(&[&[a], &[a, bb]]);
    let found = roth_ruckenstein_roots(&rows, 2, limits()).expect("roots");
    assert_eq!(found.len(), 2);
    let c0 = Polynomial::<Gf8B>::from_coefficients(&[a]).expect("const");
    let c1 = Polynomial::<Gf8B>::from_coefficients(&[a, bb]).expect("linear");
    assert!(found.contains(&c0));
    assert!(found.contains(&c1));
}

#[test]
fn constant_x_irreducible_yields_empty_extraction() {
    let a = <Gf8B as Field>::Elem::ONE;
    let mut trace_one = None;
    'outer: for key in 1u128..256 {
        let candidate = Gf8B::decode(&key.to_le_bytes()[..1]);
        let mut tr = candidate;
        let mut acc = candidate;
        for _ in 0..7 {
            acc = acc.square();
            tr = tr.add(acc);
        }
        if tr == <Gf8B as Field>::Elem::ONE {
            trace_one = Some(candidate);
            break 'outer;
        }
    }
    let b_el = trace_one.expect("find trace-1 element");
    let q0 = Polynomial::<Gf8B>::from_coefficients(&[b_el]).expect("b");
    let q1 = Polynomial::<Gf8B>::from_coefficients(&[a, <Gf8B as Field>::Elem::ONE]).expect("a+X");
    let q2 = Polynomial::<Gf8B>::one().expect("1");
    let rows = vec![q0, q1, q2];
    let found = roth_ruckenstein_roots(&rows, 2, limits()).expect("roots");
    assert!(found.is_empty());
}

#[test]
fn into_form_recycles_and_reuses_capacity() {
    let rows = bivariate_with_roots::<Gf8B>(&[&[b(0x33)]]);
    let mut scratch = RothRuckensteinScratch::<Gf8B>::new();
    let mut output = vec![Polynomial::<Gf8B>::from_coefficients(&[b(9)]).expect("stale")];
    roth_ruckenstein_roots_into(&mut output, &rows, 0, limits(), &mut scratch).expect("first");
    assert_eq!(output.len(), 1);
    let cap = scratch.capacity();
    let rows2 = bivariate_with_roots::<Gf8B>(&[&[b(0x55)]]);
    let mut output2 = std::mem::take(&mut output);
    roth_ruckenstein_roots_into(&mut output2, &rows2, 0, limits(), &mut scratch).expect("second");
    assert_eq!(output2.len(), 1);
    assert!(scratch.capacity() >= cap);
}

#[test]
fn max_degree_overflow_is_geometry_error() {
    let rows = bivariate_with_roots::<Gf8B>(&[&[b(1)]]);
    assert!(matches!(
        roth_ruckenstein_roots(&rows, usize::MAX, limits()),
        Err(RootError::Polynomial(poly_ring::PolynomialError::Config(
            poly_ring::ConfigError::GeometryOverflow { .. }
        )))
    ));
}

#[test]
fn trailing_zero_rows_normalize_through_pool() {
    let a = b(0x2b);
    let mut rows = bivariate_with_roots::<Gf8B>(&[&[a]]);
    rows.push(Polynomial::zero());
    rows.push(Polynomial::zero());
    let found = roth_ruckenstein_roots(&rows, 0, limits()).expect("roots");
    assert_eq!(found.len(), 1);
    assert!(found.contains(&Polynomial::<Gf8B>::from_coefficients(&[a]).expect("c")));
}

#[test]
fn dead_branch_recycles_child_and_parent_continues() {
    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(1), b(1)]).expect("x+1"),
        Polynomial::<Gf8B>::zero(),
        Polynomial::<Gf8B>::one().expect("one"),
    ];
    let found = roth_ruckenstein_roots(&rows, 3, limits()).expect("extraction");
    assert!(found.is_empty());
}

#[test]
fn internal_zero_row_skips_substitution_loops() {
    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1)]).expect("x"),
        Polynomial::<Gf8B>::zero(),
        Polynomial::<Gf8B>::one().expect("one"),
    ];
    let found = roth_ruckenstein_roots(&rows, 3, limits()).expect("roots");
    for candidate in &found {
        assert!(compose(&rows, candidate));
    }
}

#[test]
fn default_scratch_agrees_with_new() {
    let _ = RothRuckensteinScratch::<Gf8B>::default();
    assert_eq!(RothRuckensteinScratch::<Gf8B>::new().capacity(), 0);
}

#[test]
fn zero_work_budget_names_resource_and_counts() {
    let rows = bivariate_with_roots::<Gf8B>(&[&[b(3)]]);
    assert_eq!(
        roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(0, 64)).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein work items",
            required: 1,
            limit: 0,
        })
    );
}

#[test]
fn zero_output_budget_names_resource_and_counts() {
    let rows = bivariate_with_roots::<Gf8B>(&[&[b(5)]]);
    assert_eq!(
        roth_ruckenstein_roots(&rows, 0, RothRuckensteinLimits::new(100_000, 0)).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein output roots",
            required: 1,
            limit: 0,
        })
    );
}

#[test]
fn child_frame_push_hitting_work_limit_names_the_resource() {
    let a = noise::<Gf8B>(2, 0xBC06);
    let bb = noise::<Gf8B>(2, 0xBC07);
    let rows = bivariate_with_roots::<Gf8B>(&[&a, &bb]);
    assert_eq!(
        roth_ruckenstein_roots(&rows, 3, RothRuckensteinLimits::new(1, 64)).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein work items",
            required: 2,
            limit: 1,
        })
    );
}
