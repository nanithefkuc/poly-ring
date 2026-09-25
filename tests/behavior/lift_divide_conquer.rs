//! Alekhnovich divide-and-conquer branches: materialization limits and
//! family filtering.
//!
//! The agreement suites run the D&C path on typical planted roots; every
//! test here takes a branch they never reach — the output-root and
//! intermediate-family limits with exact payloads, and family prefixes
//! that overshoot the requested degree — each against exact root sets or
//! error payloads.

#[cfg(feature = "fft")]
use fgf::{Gf8B, gf8b};
#[cfg(feature = "fft")]
use poly_ring::{AlekhnovichLimits, AlekhnovichScratch, Polynomial, RootError, alekhnovich_roots};

#[cfg(feature = "fft")]
fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

/// Build `Q(X,Y) = Y + f(X)` in row form for one planted root.
#[cfg(feature = "fft")]
fn single_root_rows() -> Vec<Polynomial<Gf8B>> {
    let f = Polynomial::<Gf8B>::from_coefficients(&[b(3), b(5)]).expect("f");
    let one = Polynomial::<Gf8B>::one().expect("one");
    vec![f, one]
}

#[cfg(feature = "fft")]
fn forced_limits(max_output_roots: usize) -> AlekhnovichLimits {
    AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 24, 1 << 28, max_output_roots)
        .with_roth_ruckenstein_crossover(0)
}

/// A zero output-root budget fails during candidate materialization once
/// the first family completes, naming the resource and both counts.
#[cfg(feature = "fft")]
#[test]
fn zero_output_budget_fails_materialization() {
    let rows = single_root_rows();
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    assert_eq!(
        alekhnovich_roots(&rows, 10, forced_limits(0), &mut scratch).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich output roots",
            required: 1,
            limit: 0,
        })
    );
}

/// A zero intermediate-family budget fails on the first scalar family.
#[cfg(feature = "fft")]
#[test]
fn zero_family_budget_fails_scalar_materialization() {
    let rows = single_root_rows();
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let limits = AlekhnovichLimits::new(1_000_000, 0, 1 << 24, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    assert_eq!(
        alekhnovich_roots(&rows, 4, limits, &mut scratch).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich intermediate families",
            required: 1,
            limit: 0,
        })
    );
}

/// A `max_degree` below every root degree filters all families: the
/// degree-1 planted root overshoots `max_degree` zero and extraction is
/// empty without error.
#[cfg(feature = "fft")]
#[test]
fn over_degree_families_filter_to_empty() {
    let rows = single_root_rows();
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let found = alekhnovich_roots(&rows, 0, forced_limits(256), &mut scratch).expect("empty");
    assert!(found.is_empty());
    // The same input at a sufficient degree recovers the planted root.
    let found = alekhnovich_roots(&rows, 4, forced_limits(256), &mut scratch).expect("roots");
    assert_eq!(found.len(), 1);
    let expected = Polynomial::<Gf8B>::from_coefficients(&[b(3), b(5)]).expect("f");
    assert_eq!(found[0], expected);
}

/// Limit accessors report the configured bounds, and the default scratch
/// agrees with a fresh one.
#[cfg(feature = "fft")]
#[test]
fn limit_accessors_report_configured_bounds() {
    let limits = AlekhnovichLimits::new(11, 22, 33, 44, 55);
    assert_eq!(limits.max_work_items(), 11);
    assert_eq!(limits.max_intermediate_families(), 22);
    assert_eq!(limits.max_coefficients(), 33);
    assert_eq!(limits.max_scratch_bytes(), 44);
    assert_eq!(limits.max_output_roots(), 55);
    assert_eq!(limits.roth_ruckenstein_crossover(), 20_000);
    assert_eq!(
        limits
            .with_roth_ruckenstein_crossover(7)
            .roth_ruckenstein_crossover(),
        7
    );
    let _ = AlekhnovichScratch::<Gf8B>::default();
}

/// An internal zero `Y` row skips the weighted-degree scan without
/// affecting extraction: `Q = X + Y^2` runs the full divide-and-conquer
/// machine and every returned candidate verifies by composition.
#[cfg(feature = "fft")]
#[test]
fn zero_rows_skip_weighted_degree_and_verify() {
    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1)]).expect("x"),
        Polynomial::<Gf8B>::zero(),
        Polynomial::<Gf8B>::one().expect("one"),
    ];
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let found = alekhnovich_roots(&rows, 3, forced_limits(256), &mut scratch).expect("roots");
    for candidate in &found {
        let mut composition = Polynomial::<Gf8B>::zero();
        for row in rows.iter().rev() {
            composition = composition.multiply(candidate).expect("multiply");
            composition = composition.add(row).expect("add");
        }
        assert!(composition.is_zero());
    }
}

/// Empty rows are the zero bivariate polynomial, rejected up front; a
/// `Y`-free row set clears the output and returns empty.
#[cfg(feature = "fft")]
#[test]
fn empty_and_y_free_inputs_take_early_exits() {
    use poly_ring::RootError;

    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    assert_eq!(
        alekhnovich_roots(&[], 2, forced_limits(256), &mut scratch).map(|_| ()),
        Err(RootError::ZeroBivariatePolynomial)
    );
    // A Y-free row set clears a pre-filled output and succeeds empty.
    let rows = vec![Polynomial::<Gf8B>::from_coefficients(&[b(1), b(2)]).expect("row")];
    let mut output = vec![Polynomial::<Gf8B>::from_coefficients(&[b(9)]).expect("stale")];
    poly_ring::alekhnovich_roots_into(&mut output, &rows, 2, forced_limits(256), &mut scratch)
        .expect("y-free");
    assert!(output.is_empty());
}

/// A unit work budget admits the initial frame but fails the first coarse
/// split, naming the work resource and both counts.
#[cfg(feature = "fft")]
#[test]
fn unit_work_budget_fails_the_first_coarse_split() {
    use poly_ring::RootError;

    let rows = single_root_rows();
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let limits = AlekhnovichLimits::new(1, 1_000_000, 1 << 24, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    assert_eq!(
        alekhnovich_roots(&rows, 4, limits, &mut scratch).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich work items",
            required: 2,
            limit: 1,
        })
    );
}

/// An exact coefficient budget fails the first coarse charge: the weighted
/// input fits exactly, but the coarse frame does not.
#[cfg(feature = "fft")]
#[test]
fn exact_coefficient_budget_fails_the_coarse_charge() {
    use poly_ring::RootError;

    // Single-root rows at max_degree 4: weighted size 10 fits exactly.
    let rows = single_root_rows();
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let limits = AlekhnovichLimits::new(1_000_000, 1_000_000, 10, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    assert_eq!(
        alekhnovich_roots(&rows, 4, limits, &mut scratch).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich coefficients",
            required: 16,
            limit: 10,
        })
    );
}

/// A narrow free tail enumerates and filters: at `max_degree` one the
/// single linear root verifies through completion enumeration.
#[cfg(feature = "fft")]
#[test]
fn narrow_tail_enumeration_verifies_the_root() {
    let rows = single_root_rows();
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let found = alekhnovich_roots(&rows, 1, forced_limits(256), &mut scratch).expect("roots");
    assert_eq!(found.len(), 1);
    let expected = Polynomial::<Gf8B>::from_coefficients(&[b(3), b(5)]).expect("f");
    assert_eq!(found[0], expected);
}

/// Constant planted roots complete without tail work: the substituted
/// families vanish exactly, so refinement keeps them and materialization
/// enumerates the free tail.
#[cfg(feature = "fft")]
#[test]
fn constant_roots_complete_without_tail_work() {
    let q0 = Polynomial::<Gf8B>::from_coefficients(&[b(3).mul(b(7))]).expect("q0");
    let q1 = Polynomial::<Gf8B>::from_coefficients(&[b(3).add(b(7))]).expect("q1");
    let q2 = Polynomial::<Gf8B>::one().expect("q2");
    let rows = vec![q0, q1, q2];
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let found = alekhnovich_roots(&rows, 2, forced_limits(256), &mut scratch).expect("roots");
    assert_eq!(found.len(), 2);
    assert!(found.contains(&Polynomial::<Gf8B>::from_coefficients(&[b(3)]).expect("c0")));
    assert!(found.contains(&Polynomial::<Gf8B>::from_coefficients(&[b(7)]).expect("c1")));
}
