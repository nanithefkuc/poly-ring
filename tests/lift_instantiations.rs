//! Root-lifter instantiation sweep: the traversal shapes over fields and
//! entry contexts the existing suites leave cold.
//!
//! Roth–Ruckenstein and Alekhnovich are generic, and line coverage counts
//! per monomorphized copy, so a shape exercised only over one field still
//! misses inside every other instantiation. This file replays the load-
//! bearing shapes — deep planted roots on cold scratch, rootless constant
//! branches, dead child frames, vanishing refinements, shared-prefix
//! families, budget binds — over GF(16), the second GF(8) polynomial,
//! GF(32), the fan-paär GF(8) tower, and the odd-characteristic base-field
//! wrapper, driving both the forced divide-and-conquer path and the
//! default-crossover delegation to prefix lifting.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{FanPaar8, Gf8B, Gf8D, Gf16, Gf32, Goldilocks, QuadMersenne31};
use poly_ring::{
    BaseFieldRoots, BinaryRootScratch, Polynomial, RootError, RothRuckensteinLimits,
    RothRuckensteinScratch, base_field_roots, binary_field_roots_into, roth_ruckenstein_roots,
    roth_ruckenstein_roots_into,
};

#[cfg(feature = "fft")]
use poly_ring::{AlekhnovichLimits, AlekhnovichScratch, alekhnovich_roots, alekhnovich_roots_into};

/// The generator raised to `exponent`, the one constructor every field here
/// shares.
fn element<F: Field>(exponent: u64) -> F::Elem {
    F::GENERATOR.pow(exponent)
}

fn one<F: Field>() -> F::Elem {
    F::Elem::ONE
}

fn zero<F: Field>() -> F::Elem {
    F::Elem::ZERO
}

fn poly<F: FieldKernels>(coefficients: &[F::Elem]) -> Polynomial<F> {
    Polynomial::from_coefficients(coefficients).expect("coefficients build a polynomial")
}

/// Rows of `prod_k (Y + f_k(X))` over the `Y`-coefficient rows.
fn rows_with_roots<F: FieldKernels>(roots: &[&[F::Elem]]) -> Vec<Polynomial<F>> {
    let width = roots.len() + 1;
    let mut rows = vec![poly::<F>(&[one::<F>()])];
    rows.resize_with(width, Polynomial::zero);
    for coefficients in roots {
        let root = poly(coefficients);
        let mut next = vec![Polynomial::<F>::zero(); width];
        for (j, row) in rows.iter().enumerate() {
            next[j] = next[j]
                .add(&row.multiply(&root).expect("row"))
                .expect("add");
        }
        for j in 1..width {
            next[j] = next[j].add(&rows[j - 1]).expect("add");
        }
        rows = next;
    }
    rows
}

fn compose_is_zero<F: FieldKernels>(rows: &[Polynomial<F>], candidate: &Polynomial<F>) -> bool {
    let mut composition = Polynomial::<F>::zero();
    for row in rows.iter().rev() {
        composition = composition.multiply(candidate).expect("multiply");
        composition = composition.add(row).expect("add");
    }
    composition.is_zero()
}

fn roth_limits() -> RothRuckensteinLimits {
    RothRuckensteinLimits::new(100_000, 64)
}

#[cfg(feature = "fft")]
fn forced_limits() -> AlekhnovichLimits {
    AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 24, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0)
}

#[cfg(feature = "fft")]
fn default_limits() -> AlekhnovichLimits {
    AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 24, 1 << 28, 256)
}

/// Every element of a small field: zero plus the powers of the generator.
fn field_elements<F: Field>() -> Vec<F::Elem> {
    let order = usize::try_from(F::ORDER).expect("small field order");
    let mut elements = vec![F::Elem::ZERO];
    let mut power = F::Elem::ONE;
    for _ in 1..order {
        elements.push(power);
        power = power.mul(F::GENERATOR);
    }
    elements
}

/// Coefficients `(c1, c2)` such that `c1 + c2·Y + Y^2` has no root in the
/// field, found by a fixed-order scan over the generator basis.
fn rootless_y_quadratic<F: FieldKernels>() -> (F::Elem, F::Elem) {
    let elements = field_elements::<F>();
    let nonzero: Vec<F::Elem> = elements[1..].to_vec();
    for &c1 in &nonzero {
        for &c2 in &nonzero {
            let rooted = elements.iter().any(|&y| {
                let y_squared = y.mul(y);
                c1.add(c2.mul(y)).add(y_squared).is_zero()
            });
            if !rooted {
                return (c1, c2);
            }
        }
    }
    panic!("an irreducible Y-quadratic exists over every field scanned here");
}

/// The planted roots as a frozen-order list, sorted by the canonical
/// little-endian element key the whole crate shares.
fn planted_list<F: FieldKernels>(roots: &[F::Elem]) -> Vec<F::Elem> {
    let mut list = roots.to_vec();
    list.sort_by_key(|root| poly_ring::internals::element_key::<F>(*root));
    list.dedup();
    list
}

// ── Roth–Ruckenstein over GF(16), GF(8) under 0x11D, GF(32) ─────────────────

/// A three-root extraction over GF(16) from a cold scratch with a stale
/// output: the child-frame machinery runs to depth three, every planted
/// root comes back verified, and the allocating form agrees exactly.
#[test]
fn roth_gf16_deep_traversal_on_cold_scratch_with_stale_output() {
    let constant = [element::<Gf16>(0)];
    let linear = [element::<Gf16>(3), element::<Gf16>(7)];
    let quadratic = [element::<Gf16>(1), element::<Gf16>(2), element::<Gf16>(5)];
    let rows = rows_with_roots::<Gf16>(&[&constant, &linear, &quadratic]);

    let mut scratch = RothRuckensteinScratch::<Gf16>::new();
    let mut output = vec![poly::<Gf16>(&[element::<Gf16>(9), element::<Gf16>(11)])];
    roth_ruckenstein_roots_into(&mut output, &rows, 3, roth_limits(), &mut scratch)
        .expect("deep extraction");
    assert_eq!(output.len(), 3);
    assert!(output.contains(&poly(&constant)));
    assert!(output.contains(&poly(&linear)));
    assert!(output.contains(&poly(&quadratic)));
    for candidate in &output {
        assert!(compose_is_zero(&rows, candidate));
    }
    assert_eq!(
        roth_ruckenstein_roots(&rows, 3, roth_limits()).expect("allocating form"),
        output
    );

    // The warmed scratch recycles its pools over a changed single-root input.
    let capacity = scratch.capacity();
    let rows2 = rows_with_roots::<Gf16>(&[&linear]);
    let mut output2 = output;
    roth_ruckenstein_roots_into(&mut output2, &rows2, 2, roth_limits(), &mut scratch)
        .expect("changed input");
    assert_eq!(output2, vec![poly(&linear)]);
    assert!(scratch.capacity() >= capacity);
}

/// The same planted-root shapes over GF(8) under `0x11D` and over GF(32):
/// every root verifies and the `into` and allocating forms agree.
#[test]
fn roth_gf8d_and_gf32_recover_planted_roots() {
    let rows = rows_with_roots::<Gf8D>(&[
        &[element::<Gf8D>(0)],
        &[element::<Gf8D>(5), element::<Gf8D>(9)],
        &[element::<Gf8D>(3), element::<Gf8D>(1), element::<Gf8D>(7)],
    ]);
    let found = roth_ruckenstein_roots(&rows, 3, roth_limits()).expect("gf8d extraction");
    assert_eq!(found.len(), 3);
    for candidate in &found {
        assert!(compose_is_zero(&rows, candidate));
    }
    let mut scratch = RothRuckensteinScratch::<Gf8D>::new();
    let mut output = Vec::new();
    roth_ruckenstein_roots_into(&mut output, &rows, 3, roth_limits(), &mut scratch)
        .expect("into form");
    assert_eq!(output, found);

    let rows = rows_with_roots::<Gf32>(&[
        &[element::<Gf32>(0)],
        &[element::<Gf32>(1), element::<Gf32>(3)],
        &[element::<Gf32>(7), element::<Gf32>(2)],
    ]);
    let found = roth_ruckenstein_roots(&rows, 2, roth_limits()).expect("gf32 extraction");
    assert_eq!(found.len(), 3);
    for candidate in &found {
        assert!(compose_is_zero(&rows, candidate));
    }
}

/// Rejections and degenerate geometries over the freshly instantiated
/// fields: empty rows error, a `Y`-free row set clears a stale output, an
/// overflowing degree bound is a geometry error, and trailing zero rows
/// normalize away without changing the roots.
#[test]
fn roth_rejects_empty_and_y_free_rows_over_new_fields() {
    assert_eq!(
        roth_ruckenstein_roots::<Gf16>(&[], 2, roth_limits()).map(|_| ()),
        Err(RootError::ZeroBivariatePolynomial)
    );
    assert_eq!(
        roth_ruckenstein_roots::<Gf8D>(&[], 2, roth_limits()).map(|_| ()),
        Err(RootError::ZeroBivariatePolynomial)
    );
    assert_eq!(
        roth_ruckenstein_roots::<Gf32>(&[], 2, roth_limits()).map(|_| ()),
        Err(RootError::ZeroBivariatePolynomial)
    );

    // A single row has Y-degree zero: nothing to lift, stale output cleared.
    let mut scratch = RothRuckensteinScratch::<Gf16>::new();
    let mut output = vec![poly::<Gf16>(&[element::<Gf16>(13)])];
    roth_ruckenstein_roots_into(
        &mut output,
        &[poly::<Gf16>(&[element::<Gf16>(1), element::<Gf16>(6)])],
        2,
        roth_limits(),
        &mut scratch,
    )
    .expect("y-free rows");
    assert!(output.is_empty());
    assert!(
        roth_ruckenstein_roots::<Gf8D>(&[poly(&[element::<Gf8D>(3)])], 2, roth_limits())
            .expect("y-free allocating form")
            .is_empty()
    );

    // max_degree + 1 overflows before any traversal starts.
    let rows = rows_with_roots::<Gf16>(&[&[element::<Gf16>(4)]]);
    assert!(matches!(
        roth_ruckenstein_roots::<Gf16>(&rows, usize::MAX, roth_limits()),
        Err(RootError::Polynomial(poly_ring::PolynomialError::Config(
            poly_ring::ConfigError::GeometryOverflow { .. }
        )))
    ));

    // Trailing zero rows are normalized away without changing the roots.
    let mut padded = rows;
    padded.push(Polynomial::zero());
    padded.push(Polynomial::zero());
    assert_eq!(
        roth_ruckenstein_roots(&padded, 0, roth_limits()).expect("padded rows"),
        vec![poly::<Gf16>(&[element::<Gf16>(4)])]
    );
}

/// A constant-Y quadratic with no root in the coefficient field stops the
/// traversal before any frame is pushed; the same shape over GF(8) under
/// `0x11D` and the fan-paär GF(8) tower instantiates the scan there too.
#[test]
fn roth_rootless_constant_y_exits_before_the_first_frame() {
    let (c1, c2) = rootless_y_quadratic::<Gf16>();
    let rows = vec![
        poly::<Gf16>(&[c1]),
        poly::<Gf16>(&[c2]),
        poly(&[one::<Gf16>()]),
    ];
    assert!(
        roth_ruckenstein_roots(&rows, 2, roth_limits())
            .expect("rootless constant")
            .is_empty()
    );
    let mut scratch = RothRuckensteinScratch::<Gf16>::new();
    let mut output = vec![poly::<Gf16>(&[element::<Gf16>(8)])];
    roth_ruckenstein_roots_into(&mut output, &rows, 2, roth_limits(), &mut scratch)
        .expect("into form agrees");
    assert!(output.is_empty());

    let (d1, d2) = rootless_y_quadratic::<Gf8D>();
    let rows = vec![
        poly::<Gf8D>(&[d1]),
        poly::<Gf8D>(&[d2]),
        poly(&[one::<Gf8D>()]),
    ];
    assert!(
        roth_ruckenstein_roots(&rows, 2, roth_limits())
            .expect("rootless over gf8d")
            .is_empty()
    );

    let (p1, p2) = rootless_y_quadratic::<FanPaar8>();
    let rows = vec![
        poly::<FanPaar8>(&[p1]),
        poly::<FanPaar8>(&[p2]),
        poly(&[one::<FanPaar8>()]),
    ];
    assert!(
        roth_ruckenstein_roots(&rows, 2, roth_limits())
            .expect("rootless over the fan-paär tower")
            .is_empty()
    );
}

/// `Q = X + 1 + Y^2` walks one branch to a child whose constant row is
/// rootless: the dead child is recycled and the parent finishes empty; an
/// internal zero row leaves the lift unchanged.
#[test]
fn roth_dead_child_frame_is_recycled_over_new_fields() {
    let rows = vec![
        poly::<Gf16>(&[one::<Gf16>(), one::<Gf16>()]),
        Polynomial::zero(),
        poly(&[one::<Gf16>()]),
    ];
    assert!(
        roth_ruckenstein_roots(&rows, 3, roth_limits())
            .expect("gf16 dead branch")
            .is_empty()
    );
    let rows = vec![
        poly::<Gf8D>(&[one::<Gf8D>(), one::<Gf8D>()]),
        Polynomial::zero(),
        poly(&[one::<Gf8D>()]),
    ];
    assert!(
        roth_ruckenstein_roots(&rows, 3, roth_limits())
            .expect("gf8d dead branch")
            .is_empty()
    );

    let rows = vec![
        poly::<Gf16>(&[zero::<Gf16>(), one::<Gf16>()]),
        Polynomial::zero(),
        poly(&[one::<Gf16>()]),
    ];
    for candidate in roth_ruckenstein_roots(&rows, 3, roth_limits()).expect("zero row skipped") {
        assert!(compose_is_zero(&rows, &candidate));
    }
}

/// Caller budgets bind over GF(16) exactly as over GF(8): the work budget
/// fails up front and again at the first child push, the output budget
/// fails at the first verified root — each naming the resource and both
/// counts — and the same scratch recovers the root afterwards.
#[test]
fn roth_budgets_bind_over_gf16() {
    let rows = rows_with_roots::<Gf16>(&[&[element::<Gf16>(4)]]);
    assert_eq!(
        roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(0, 64)).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein work items",
            required: 1,
            limit: 0,
        })
    );
    assert_eq!(
        roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(1, 64)).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein work items",
            required: 2,
            limit: 1,
        })
    );
    assert_eq!(
        roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(100_000, 0)).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein output roots",
            required: 1,
            limit: 0,
        })
    );
    let mut scratch = RothRuckensteinScratch::<Gf16>::new();
    let mut output = Vec::new();
    roth_ruckenstein_roots_into(
        &mut output,
        &rows,
        2,
        RothRuckensteinLimits::new(100_000, 64),
        &mut scratch,
    )
    .expect("recovered");
    assert_eq!(output.len(), 1);
}

// ── Alekhnovich over the same instantiations ────────────────────────────────

/// The vanishing-square input over GF(16): the half-precision transform of
/// the surviving family is identically zero, so the traversal keeps the
/// family as-is and finishes its tail from a residual node.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_gf16_keeps_vanishing_families_and_agrees_with_prefix_lifting() {
    let f = poly::<Gf16>(&[element::<Gf16>(1), element::<Gf16>(3)]);
    let squared = f.multiply(&f).expect("square");
    let rows = vec![squared, Polynomial::zero(), poly(&[one::<Gf16>()])];
    let expected = roth_ruckenstein_roots(&rows, 2, roth_limits()).expect("prefix lift");
    assert_eq!(expected, vec![f]);

    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    let lifted = alekhnovich_roots(&rows, 2, forced_limits(), &mut scratch).expect("forced");
    assert_eq!(lifted, expected);
    let frames = scratch.frame_capacity();
    let mut output = Vec::new();
    alekhnovich_roots_into(&mut output, &rows, 2, forced_limits(), &mut scratch).expect("reuse");
    assert_eq!(output, expected);
    assert!(scratch.frame_capacity() >= frames);

    // Below the default crossover the entry point delegates to prefix
    // lifting and returns the same root without forcing.
    let routed = alekhnovich_roots(&rows, 2, default_limits(), &mut scratch).expect("routed");
    assert_eq!(routed, expected);
}

/// A cubic in `Y` with three planted roots over GF(16) walks the full
/// divide-and-conquer machine — coarse split, refine transform, tail
/// completion, candidate materialization — and both limit regimes agree
/// with prefix lifting.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_gf16_cubic_walks_the_full_refinement_machine() {
    let rows = rows_with_roots::<Gf16>(&[
        &[element::<Gf16>(2)],
        &[element::<Gf16>(5), element::<Gf16>(9)],
        &[element::<Gf16>(11), zero::<Gf16>(), element::<Gf16>(3)],
    ]);
    let expected = roth_ruckenstein_roots(&rows, 3, roth_limits()).expect("prefix lift");
    assert_eq!(expected.len(), 3);

    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    let lifted = alekhnovich_roots(&rows, 3, forced_limits(), &mut scratch).expect("forced");
    assert_eq!(lifted, expected);
    for candidate in &lifted {
        assert!(compose_is_zero(&rows, candidate));
    }
    let mut output = vec![poly::<Gf16>(&[element::<Gf16>(14)])];
    alekhnovich_roots_into(&mut output, &rows, 3, forced_limits(), &mut scratch).expect("into");
    assert_eq!(output, expected);

    // Default crossover: the same rows route through prefix lifting.
    let routed = alekhnovich_roots(&rows, 3, default_limits(), &mut scratch).expect("routed");
    assert_eq!(routed, expected);
}

/// Two roots sharing their degree-zero coefficient — a constant and a
/// quadratic with the same constant term — complete to nested affine
/// families, so the family set deduplicates instead of carrying a family
/// already contained in another.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_shared_prefix_roots_deduplicate_families() {
    let shared = element::<Gf16>(7);
    let tail = element::<Gf16>(4);
    let rows = rows_with_roots::<Gf16>(&[&[shared], &[shared, zero::<Gf16>(), tail]]);
    let expected = roth_ruckenstein_roots(&rows, 3, roth_limits()).expect("prefix lift");
    assert_eq!(expected.len(), 2);

    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    let lifted = alekhnovich_roots(&rows, 3, forced_limits(), &mut scratch).expect("forced");
    assert_eq!(lifted, expected);
    assert!(lifted.contains(&poly::<Gf16>(&[shared])));
    assert!(lifted.contains(&poly::<Gf16>(&[shared, zero::<Gf16>(), tail])));

    // The same shape over GF(8) under 0x11B instantiates the dedup there.
    let shared = element::<Gf8B>(9);
    let tail = element::<Gf8B>(5);
    let rows = rows_with_roots::<Gf8B>(&[&[shared], &[shared, zero::<Gf8B>(), tail]]);
    let expected = roth_ruckenstein_roots(&rows, 3, roth_limits()).expect("prefix lift");
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    assert_eq!(
        alekhnovich_roots(&rows, 3, forced_limits(), &mut scratch).expect("forced"),
        expected
    );
}

/// The forced divide-and-conquer path over GF(8) under `0x11D`, GF(32), and
/// the fan-paär GF(8) tower: a precision-one leaf solves a linear constant
/// row directly, planted roots come back verified, and the default
/// crossover delegates to prefix lifting identically.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_new_fields_recover_planted_roots_and_leaf_nodes() {
    // Leaf node: Q = a + b·Y solves at precision one.
    let rows = vec![
        poly::<Gf8D>(&[element::<Gf8D>(3)]),
        poly::<Gf8D>(&[element::<Gf8D>(5)]),
    ];
    let mut scratch = AlekhnovichScratch::<Gf8D>::new();
    let lifted = alekhnovich_roots(&rows, 0, forced_limits(), &mut scratch).expect("leaf");
    let expected = roth_ruckenstein_roots(&rows, 0, roth_limits()).expect("prefix lift");
    assert_eq!(lifted, expected);
    assert_eq!(lifted.len(), 1);
    assert!(compose_is_zero(&rows, &lifted[0]));

    // Two planted roots through the full machine, then via default routing.
    let rows = rows_with_roots::<Gf8D>(&[
        &[element::<Gf8D>(1)],
        &[element::<Gf8D>(4), element::<Gf8D>(9)],
    ]);
    let expected = roth_ruckenstein_roots(&rows, 2, roth_limits()).expect("prefix lift");
    assert_eq!(expected.len(), 2);
    assert_eq!(
        alekhnovich_roots(&rows, 2, forced_limits(), &mut scratch).expect("forced"),
        expected
    );
    assert_eq!(
        alekhnovich_roots(&rows, 2, default_limits(), &mut scratch).expect("routed"),
        expected
    );

    // GF(32) and the fan-paär GF(8) tower instantiate the same machinery.
    let rows = rows_with_roots::<Gf32>(&[
        &[element::<Gf32>(2)],
        &[element::<Gf32>(1), element::<Gf32>(6)],
    ]);
    let expected = roth_ruckenstein_roots(&rows, 2, roth_limits()).expect("prefix lift");
    let mut scratch = AlekhnovichScratch::<Gf32>::new();
    assert_eq!(
        alekhnovich_roots(&rows, 2, forced_limits(), &mut scratch).expect("forced"),
        expected
    );
    for candidate in &expected {
        assert!(compose_is_zero(&rows, candidate));
    }

    let rows = rows_with_roots::<FanPaar8>(&[
        &[element::<FanPaar8>(3)],
        &[element::<FanPaar8>(1), element::<FanPaar8>(2)],
    ]);
    let expected = roth_ruckenstein_roots(&rows, 2, roth_limits()).expect("prefix lift");
    let mut scratch = AlekhnovichScratch::<FanPaar8>::new();
    assert_eq!(
        alekhnovich_roots(&rows, 2, forced_limits(), &mut scratch).expect("forced"),
        expected
    );
}

/// `Q = X^5 + Y` has the single power-series root `X^5`: below the bound it
/// is dropped during materialization, at the bound it is returned exactly.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_gf16_drops_the_family_above_the_degree_bound() {
    let x5 = [
        zero::<Gf16>(),
        zero::<Gf16>(),
        zero::<Gf16>(),
        zero::<Gf16>(),
        zero::<Gf16>(),
        one::<Gf16>(),
    ];
    let rows = vec![poly::<Gf16>(&x5), poly(&[one::<Gf16>()])];
    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    let lifted = alekhnovich_roots(&rows, 3, forced_limits(), &mut scratch).expect("bounded");
    assert!(lifted.is_empty());
    assert_eq!(
        roth_ruckenstein_roots(&rows, 3, roth_limits()).expect("prefix lift"),
        lifted
    );
    let lifted = alekhnovich_roots(&rows, 5, forced_limits(), &mut scratch).expect("admitted");
    assert_eq!(lifted, vec![poly::<Gf16>(&x5)]);
}

/// The Alekhnovich entry points over the new fields reject empty rows and
/// clear a stale output on `Y`-free rows, matching the prefix lifter's
/// degenerate-geometry contract.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_new_fields_reject_empty_and_clear_y_free_outputs() {
    assert_eq!(
        alekhnovich_roots::<Gf8D>(&[], 2, forced_limits(), &mut AlekhnovichScratch::new())
            .map(|_| ()),
        Err(RootError::ZeroBivariatePolynomial)
    );
    assert_eq!(
        alekhnovich_roots::<Gf32>(&[], 2, forced_limits(), &mut AlekhnovichScratch::new())
            .map(|_| ()),
        Err(RootError::ZeroBivariatePolynomial)
    );

    let y_free = vec![poly::<FanPaar8>(&[
        element::<FanPaar8>(1),
        element::<FanPaar8>(5),
    ])];
    let mut scratch = AlekhnovichScratch::<FanPaar8>::new();
    let mut output = vec![poly::<FanPaar8>(&[element::<FanPaar8>(9)])];
    alekhnovich_roots_into(&mut output, &y_free, 2, forced_limits(), &mut scratch).expect("y-free");
    assert!(output.is_empty());
}

// ── Base-field roots: the allocating wrapper and new field bodies ──────────

/// The allocating wrapper over GF(16): the zero polynomial reports the whole
/// field, a nonzero constant reports nothing, and a four-root product splits
/// in the frozen order shared with the scratch form.
#[test]
fn base_field_roots_gf16_wrapper_shapes() {
    assert_eq!(
        base_field_roots::<Gf16>(&Polynomial::zero()).expect("zero polynomial"),
        BaseFieldRoots::All
    );
    assert_eq!(
        base_field_roots(&poly::<Gf16>(&[element::<Gf16>(6)])).expect("constant"),
        BaseFieldRoots::Finite(Vec::new())
    );

    let roots = [
        element::<Gf16>(0),
        element::<Gf16>(1),
        element::<Gf16>(2),
        element::<Gf16>(9),
    ];
    let mut polynomial = poly::<Gf16>(&[one::<Gf16>()]);
    for root in roots {
        polynomial = polynomial.multiply_x_plus(root).expect("X + root");
    }
    let expected = planted_list::<Gf16>(&roots);
    assert_eq!(
        base_field_roots(&polynomial).expect("four roots"),
        BaseFieldRoots::Finite(expected.clone())
    );

    // Repeated roots collapse to the distinct set through the gcd.
    let squared = polynomial.multiply(&polynomial).expect("square");
    assert_eq!(
        base_field_roots(&squared).expect("repeated roots"),
        BaseFieldRoots::Finite(expected)
    );
}

/// Deep multi-root splits over GF(8) under `0x11D`, GF(32), and the fan-paär
/// GF(8) tower: the scratch form returns the planted set in the frozen
/// order, a warmed scratch handles a changed input, and the allocating
/// wrapper agrees.
#[test]
fn base_field_roots_new_fields_split_deep_products() {
    fn deep_split<F: FieldKernels>(exponents: &[u64], changed_exponent: u64) {
        let elements: Vec<F::Elem> = exponents.iter().map(|&e| element::<F>(e)).collect();
        let mut polynomial = poly::<F>(&[one::<F>()]);
        for root in &elements {
            polynomial = polynomial.multiply_x_plus(*root).expect("X + root");
        }
        let expected = planted_list::<F>(&elements);

        let mut scratch = BinaryRootScratch::<F>::new();
        let mut found = Vec::new();
        assert!(!binary_field_roots_into(&mut found, &polynomial, &mut scratch).expect("split"));
        assert_eq!(found, expected);
        for root in &found {
            assert!(polynomial.evaluate(*root).is_zero());
        }
        assert_eq!(
            base_field_roots(&polynomial).expect("allocating form"),
            BaseFieldRoots::Finite(expected.clone())
        );

        // Warmed scratch over a changed input returns the new root set.
        let mut all = elements.clone();
        all.push(element::<F>(changed_exponent));
        let changed = polynomial
            .multiply_x_plus(all[all.len() - 1])
            .expect("one more root");
        let mut found = Vec::new();
        assert!(
            !binary_field_roots_into(&mut found, &changed, &mut scratch).expect("changed input")
        );
        assert_eq!(found, planted_list::<F>(&all));
    }

    deep_split::<Gf8D>(&[0, 3, 7, 11, 13], 20);
    deep_split::<Gf32>(&[0, 1, 5, 9], 12);
    deep_split::<FanPaar8>(&[0, 2, 6, 10, 14, 18], 22);
}

/// The odd-characteristic wrapper keeps only linear factors with negated
/// roots: planted linear factors return their roots in the frozen order,
/// repeated factors deduplicate, a constant reports nothing, and the zero
/// polynomial reports the whole field.
#[test]
fn base_field_roots_prime_fields_keep_linear_factors() {
    fn prime_shapes<F: FieldKernels>() {
        let first = element::<F>(1);
        let second = element::<F>(3);
        let linear = |root: F::Elem| poly::<F>(&[root.neg(), one::<F>()]);
        let product = linear(first)
            .multiply(&linear(second))
            .expect("linear product");
        let expected = planted_list::<F>(&[first, second]);
        assert_eq!(
            base_field_roots(&product).expect("two linear factors"),
            BaseFieldRoots::Finite(expected.clone())
        );
        for root in &expected {
            assert!(product.evaluate(*root).is_zero());
        }

        let squared = linear(first).multiply(&linear(first)).expect("square");
        assert_eq!(
            base_field_roots(&squared).expect("repeated factor"),
            BaseFieldRoots::Finite(vec![first])
        );
        assert_eq!(
            base_field_roots(&poly::<F>(&[first])).expect("constant"),
            BaseFieldRoots::Finite(Vec::new())
        );
        assert_eq!(
            base_field_roots::<F>(&Polynomial::zero()).expect("zero polynomial"),
            BaseFieldRoots::All
        );
    }

    prime_shapes::<Goldilocks>();
    prime_shapes::<QuadMersenne31>();
}
