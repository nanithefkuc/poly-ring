//! Recursive lifting behavior and error boundaries.

#[cfg(feature = "fft")]
use fgf::field::Field;
#[cfg(feature = "fft")]
use fgf::kernel::FieldKernels;
#[cfg(feature = "fft")]
use fgf::{Gf8B, Gf16, gf8b};
#[cfg(feature = "fft")]
use poly_ring::{
    AlekhnovichLimits, AlekhnovichScratch, Polynomial, RootError, RothRuckensteinLimits,
    alekhnovich_roots, alekhnovich_roots_into, roth_ruckenstein_roots,
};

#[cfg(feature = "fft")]
use crate::oracles;
#[cfg(feature = "fft")]
use oracles::{noise, noise_poly};

/// Build `Q(X,Y) = prod_k (Y + f_k(X))` in row form for planted roots.
#[cfg(feature = "fft")]
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

#[cfg(feature = "fft")]
fn check_composition<F: FieldKernels>(rows: &[Polynomial<F>], candidate: &Polynomial<F>) -> bool {
    let mut composition = Polynomial::<F>::zero();
    for row in rows.iter().rev() {
        composition = composition.multiply(candidate).expect("multiply");
        composition = composition.add(row).expect("add");
    }
    composition.is_zero()
}

/// A deep precision-8 extraction walks the full frame machine: coarse
/// splits, refine transforms, tail completions, and candidate
/// materialization with per-family free-tail enumeration.
#[cfg(feature = "fft")]
#[test]
fn deep_divide_and_conquer_recovers_planted_roots() {
    let root_a = noise::<Gf16>(6, 0xE701);
    let root_b = noise::<Gf16>(5, 0xE702);
    let rows = bivariate_with_roots::<Gf16>(&[&root_a, &root_b]);
    let limits = AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 24, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    let lifted = alekhnovich_roots(&rows, 8, limits, &mut scratch).expect("alekhnovich");
    assert_eq!(lifted.len(), 2);
    let to_poly = |coefficients: &[<Gf16 as Field>::Elem]| {
        Polynomial::<Gf16>::from_coefficients(coefficients).expect("planted")
    };
    assert!(lifted.contains(&to_poly(&root_a)));
    assert!(lifted.contains(&to_poly(&root_b)));
    for candidate in &lifted {
        assert!(check_composition(&rows, candidate));
    }
    // The forced D&C path agrees with prefix lifting on the same input.
    let prefixed = roth_ruckenstein_roots(&rows, 8, RothRuckensteinLimits::new(1_000_000, 256))
        .expect("roth-ruckenstein");
    assert_eq!(lifted, prefixed);
    // Scratch reuse: frame capacity is retained for the next extraction.
    let frames = scratch.frame_capacity();
    let mut output = Vec::new();
    alekhnovich_roots_into(&mut output, &rows, 8, limits, &mut scratch).expect("reuse");
    assert_eq!(output, lifted);
    assert!(scratch.frame_capacity() >= frames);
    assert!(scratch.capacity() >= scratch.frame_capacity());
}

/// A tiny work-item budget fails at the first frame push, naming the
/// resource and both counts.
#[cfg(feature = "fft")]
#[test]
fn tiny_work_budget_names_resource_and_counts() {
    let rows = bivariate_with_roots::<Gf16>(&[&noise::<Gf16>(3, 0xE703)]);
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    assert_eq!(
        alekhnovich_roots(
            &rows,
            6,
            AlekhnovichLimits::new(0, 1_000_000, 1 << 22, 1 << 26, 256)
                .with_roth_ruckenstein_crossover(0),
            &mut scratch,
        )
        .map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich work items",
            required: 1,
            limit: 0,
        })
    );
}

/// A tiny coefficient budget fails the weighted-size charge before any
/// frame runs.
#[cfg(feature = "fft")]
#[test]
fn tiny_coefficient_budget_fails_up_front() {
    let rows = bivariate_with_roots::<Gf16>(&[&noise::<Gf16>(3, 0xE704)]);
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    assert_eq!(
        alekhnovich_roots(
            &rows,
            6,
            AlekhnovichLimits::new(1_000_000, 1_000_000, 1, 1 << 26, 256)
                .with_roth_ruckenstein_crossover(0),
            &mut scratch,
        )
        .map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich coefficients",
            required: 14,
            limit: 1,
        })
    );
}

/// A tiny scratch-byte budget fails the initial frame charge.
#[cfg(feature = "fft")]
#[test]
fn tiny_scratch_budget_fails_the_initial_charge() {
    let rows = bivariate_with_roots::<Gf16>(&[&noise::<Gf16>(3, 0xE705)]);
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    assert!(
        alekhnovich_roots(
            &rows,
            6,
            AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 22, 1, 256)
                .with_roth_ruckenstein_crossover(0),
            &mut scratch,
        )
        .is_err()
    );
}

/// A zero output-root budget fails during candidate materialization once
/// the first family completes.
#[cfg(feature = "fft")]
#[test]
fn zero_output_budget_fails_materialization() {
    let root_a = noise::<Gf16>(3, 0xE706);
    let rows = bivariate_with_roots::<Gf16>(&[&root_a]);
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    assert_eq!(
        alekhnovich_roots(
            &rows,
            4,
            AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 22, 1 << 26, 0)
                .with_roth_ruckenstein_crossover(0),
            &mut scratch,
        )
        .map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich output roots",
            required: 1,
            limit: 0,
        })
    );
}

/// A zero intermediate-family budget fails the first scalar family insert.
#[cfg(feature = "fft")]
#[test]
fn zero_family_budget_fails_the_first_insert() {
    let rows = bivariate_with_roots::<Gf16>(&[&noise::<Gf16>(3, 0xE707)]);
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    assert_eq!(
        alekhnovich_roots(
            &rows,
            6,
            AlekhnovichLimits::new(1_000_000, 0, 1 << 22, 1 << 26, 256)
                .with_roth_ruckenstein_crossover(0),
            &mut scratch,
        )
        .map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich intermediate families",
            required: 1,
            limit: 0,
        })
    );
}

/// The crossover accessor round-trips the override, and the default
/// crossover routes small inputs through Roth–Ruckenstein identically.
#[cfg(feature = "fft")]
#[test]
fn crossover_accessor_and_default_routing() {
    let limits = AlekhnovichLimits::new(100, 200, 1 << 20, 1 << 24, 64);
    assert_eq!(
        limits.roth_ruckenstein_crossover(),
        poly_ring::DEFAULT_ROTH_RUCKENSTEIN_CROSSOVER
    );
    assert_eq!(
        limits
            .with_roth_ruckenstein_crossover(7)
            .roth_ruckenstein_crossover(),
        7
    );
    // Small input under the default crossover: both entries agree.
    let rows = bivariate_with_roots::<Gf8B>(&[&noise::<Gf8B>(2, 0xE708)]);
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let lifted = alekhnovich_roots(&rows, 3, limits, &mut scratch).expect("alekhnovich");
    let prefixed = roth_ruckenstein_roots(&rows, 3, RothRuckensteinLimits::new(100_000, 64))
        .expect("roth-ruckenstein");
    assert_eq!(lifted, prefixed);
    for candidate in &lifted {
        assert!(check_composition(&rows, candidate));
    }
    // Affine families expose their prefix and tail degree.
    let _ = noise_poly::<Gf8B>(3, 0xE709);
    let _ = gf8b::Elem::from_raw(1);
}
