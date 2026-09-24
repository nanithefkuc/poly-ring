//! Root-lifting edges: deep Roth–Ruckstein branches, the Alekhnovich
//! divide-and-conquer traversal, caller-provided limit failures, and
//! allocation-failure resistance for both lifters.
//!
//! Every test drives the public surface: `roth_ruckenstein_roots`,
//! `roth_ruckenstein_roots_into`, `alekhnovich_roots`, and
//! `alekhnovich_roots_into`, with the two lifters cross-checked against
//! each other on shared inputs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use fgf::{Gf8B, gf8b};
use poly_ring::{
    Polynomial, RootError, RothRuckensteinLimits, RothRuckensteinScratch, roth_ruckenstein_roots,
    roth_ruckenstein_roots_into,
};

#[cfg(feature = "fft")]
use poly_ring::{AlekhnovichLimits, AlekhnovichScratch, alekhnovich_roots, alekhnovich_roots_into};

struct FailingAllocator;

// The gates are scoped to the calling thread: sibling tests running in
// parallel must not be charged to them.
thread_local! {
    static THRESHOLD: Cell<usize> = const { Cell::new(usize::MAX) };
    static MATCHING_ALLOCATION: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // A failing assertion allocates its message while the gate is still
        // armed; unwinding threads always get their memory.
        let refused = THRESHOLD
            .try_with(|threshold| layout.size() > threshold.get())
            .unwrap_or(false)
            && !std::thread::panicking();
        let refused = refused
            || MATCHING_ALLOCATION
                .try_with(|gate| {
                    let Some((size, remaining)) = gate.get() else {
                        return false;
                    };
                    if layout.size() != size {
                        return false;
                    }
                    if remaining == 0 {
                        gate.set(None);
                        true
                    } else {
                        gate.set(Some((size, remaining - 1)));
                        false
                    }
                })
                .unwrap_or(false);
        if refused {
            std::ptr::null_mut()
        } else {
            unsafe { System.alloc(layout) }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: FailingAllocator = FailingAllocator;

/// Arms the size threshold for one operation and disarms it however the
/// operation leaves, so a panicking case cannot leak the gate.
fn with_threshold<R>(bytes: usize, operation: impl FnOnce() -> R) -> R {
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            THRESHOLD.with(|threshold| threshold.set(usize::MAX));
        }
    }
    THRESHOLD.with(|threshold| threshold.set(bytes));
    let _disarm = Disarm;
    operation()
}

/// Refuses the `skip + 1`-th allocation of exactly `size` bytes for one
/// operation, letting every other allocation through.
fn reject_matching<R>(size: usize, skip: usize, operation: impl FnOnce() -> R) -> R {
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            MATCHING_ALLOCATION.with(|gate| gate.set(None));
        }
    }
    MATCHING_ALLOCATION.with(|gate| gate.set(Some((size, skip))));
    let _disarm = Disarm;
    operation()
}

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn poly(coefficients: &[gf8b::Elem]) -> Polynomial<Gf8B> {
    Polynomial::from_coefficients(coefficients).expect("polynomial")
}

/// Rows of `prod_j (Y + root_j)` by synthetic multiplication over the
/// `Y`-coefficient rows.
fn rows_with_roots(roots: &[&[gf8b::Elem]]) -> Vec<Polynomial<Gf8B>> {
    let width = roots.len() + 1;
    let mut rows = vec![Polynomial::one().expect("one")];
    rows.resize_with(width, Polynomial::zero);
    for coefficients in roots {
        let root = poly(coefficients);
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

/// `Q(X, Y) = Y^2 + f(X)^2` with `f` nonconstant: the unique bounded root
/// is `f` itself, and the half-precision transforms vanish identically, so
/// the traversal keeps affine families without recursing on them.
fn square_rows() -> Vec<Polynomial<Gf8B>> {
    let f = poly(&[b(0x5), b(0x3)]);
    let squared = f.multiply(&f).expect("square");
    vec![squared, Polynomial::zero(), Polynomial::one().expect("one")]
}

/// Two planted roots: one constant, one linear.
fn two_root_rows() -> Vec<Polynomial<Gf8B>> {
    rows_with_roots(&[&[b(0x1d)], &[b(0x53), b(0xc7)]])
}

/// Three planted roots for a cubic in `Y`: constant, linear, linear.
fn three_root_rows() -> Vec<Polynomial<Gf8B>> {
    rows_with_roots(&[&[b(0x1d)], &[b(0x53), b(0xc7)], &[b(0x91), b(0x2a)]])
}

fn compose_zero(rows: &[Polynomial<Gf8B>], candidate: &Polynomial<Gf8B>) -> bool {
    let mut result = Polynomial::<Gf8B>::zero();
    for row in rows.iter().rev() {
        result = result.multiply(candidate).expect("multiply");
        result = result.add(row).expect("add");
    }
    result.is_zero()
}

fn limits() -> RothRuckensteinLimits {
    RothRuckensteinLimits::new(100_000, 64)
}

#[cfg(feature = "fft")]
fn forced_limits() -> AlekhnovichLimits {
    AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 24, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0)
}

/// Planted roots survive a deep traversal: both bounded roots of a
/// factored bivariate come back, every returned candidate composes to
/// zero, and a rootless prefix input yields the empty list.
#[test]
fn roth_finds_deep_planted_roots_and_rejects_rootless() {
    let rows = two_root_rows();
    let found = roth_ruckenstein_roots(&rows, 2, limits()).expect("roots");
    assert_eq!(found.len(), 2);
    assert!(found.contains(&poly(&[b(0x1d)])));
    assert!(found.contains(&poly(&[b(0x53), b(0xc7)])));
    for candidate in &found {
        assert!(compose_zero(&rows, candidate));
    }

    // A deeper bound than any root: the planted pair is still exactly it.
    let found = roth_ruckenstein_roots(&rows, 4, limits()).expect("deep");
    assert_eq!(found.len(), 2);
    for candidate in &found {
        assert!(compose_zero(&rows, candidate));
    }

    // No bounded root: every full-depth candidate fails verification.
    let q0 = poly(&[b(0x7), b(0x1)]);
    let q1 = poly(&[b(0x21), b(0x5)]);
    let rootless = vec![q0, q1];
    let found = roth_ruckenstein_roots(&rootless, 2, limits()).expect("rootless");
    assert!(found.is_empty());

    // Three roots through a cubic in Y.
    let rows = three_root_rows();
    let found = roth_ruckenstein_roots(&rows, 2, limits()).expect("cubic");
    assert_eq!(found.len(), 3);
    for candidate in &found {
        assert!(compose_zero(&rows, candidate));
    }
}

/// The `into` form recycles a stale output, keeps its pools warm across a
/// changed input, and agrees with the allocating form.
#[test]
fn roth_into_form_recycles_output_and_pools() {
    let rows = two_root_rows();
    let mut scratch = RothRuckensteinScratch::<Gf8B>::new();
    let mut output = vec![poly(&[b(0xff), b(0xff), b(0xff)])];
    roth_ruckenstein_roots_into(&mut output, &rows, 2, limits(), &mut scratch).expect("first");
    assert_eq!(output.len(), 2);
    let expected = output.clone();
    let capacity = scratch.capacity();

    let rows2 = rows_with_roots(&[&[b(0x7), b(0x11)]]);
    let mut output2 = std::mem::take(&mut output);
    roth_ruckenstein_roots_into(&mut output2, &rows2, 2, limits(), &mut scratch).expect("second");
    assert_eq!(output2.len(), 1);
    assert!(compose_zero(&rows2, &output2[0]));
    assert!(scratch.capacity() >= capacity);

    assert_eq!(
        roth_ruckenstein_roots(&rows, 2, limits()).expect("allocating form"),
        expected
    );
}

#[cfg(feature = "fft")]
fn expected_roots(rows: &[Polynomial<Gf8B>], max_degree: usize) -> Vec<Polynomial<Gf8B>> {
    roth_ruckenstein_roots(rows, max_degree, limits()).expect("prefix lift")
}

/// The vanishing-square input drives the half-precision refinement where a
/// family's transform is identically zero: the family is kept as-is and
/// the traversal finishes its tail from a residual node.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_keeps_vanishing_affine_families() {
    let rows = square_rows();
    let expected = expected_roots(&rows, 2);
    assert_eq!(expected.len(), 1);
    assert_eq!(expected[0], poly(&[b(0x5), b(0x3)]));

    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let lifted = alekhnovich_roots(&rows, 2, forced_limits(), &mut scratch).expect("alekhnovich");
    assert_eq!(lifted, expected);
    for candidate in &lifted {
        assert!(compose_zero(&rows, candidate));
    }

    // Reuse over the same geometry: identical output, retained frames.
    let frames = scratch.frame_capacity();
    let mut output = Vec::new();
    alekhnovich_roots_into(&mut output, &rows, 2, forced_limits(), &mut scratch).expect("reuse");
    assert_eq!(output, expected);
    assert!(scratch.frame_capacity() >= frames);
    assert!(scratch.capacity() >= scratch.frame_capacity());
}

/// A precision-one node solves the constant-`X` polynomial directly, and a
/// `Y`-free row set clears a pre-filled output.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_leaf_node_and_y_free_rows() {
    let rows = vec![poly(&[b(0x9)]), poly(&[b(0x4)])];
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let lifted = alekhnovich_roots(&rows, 0, forced_limits(), &mut scratch).expect("leaf");
    // 9 + 4Y has the single root Y = 9/4.
    let expected = roth_ruckenstein_roots(&rows, 0, limits()).expect("prefix lift");
    assert_eq!(lifted, expected);
    assert_eq!(lifted.len(), 1);
    assert!(compose_zero(&rows, &lifted[0]));

    // Y-free rows: every bounded polynomial would be a root of nothing to
    // check against, so the output is cleared without error.
    let y_free = vec![poly(&[b(0x5), b(0x2)])];
    let mut output = vec![poly(&[b(0x77)])];
    alekhnovich_roots_into(&mut output, &y_free, 2, forced_limits(), &mut scratch).expect("y-free");
    assert!(output.is_empty());
}

/// A cubic in `Y` with three planted bounded roots agrees between the two
/// lifters and between the allocating and `into` forms.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_cubic_agrees_with_prefix_lifting() {
    let rows = three_root_rows();
    let expected = expected_roots(&rows, 2);
    assert_eq!(expected.len(), 3);

    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let lifted = alekhnovich_roots(&rows, 2, forced_limits(), &mut scratch).expect("alekhnovich");
    assert_eq!(lifted, expected);

    let mut output = vec![poly(&[b(0x42)])];
    alekhnovich_roots_into(&mut output, &rows, 2, forced_limits(), &mut scratch).expect("into");
    assert_eq!(output, expected);
    for candidate in &output {
        assert!(compose_zero(&rows, candidate));
    }

    // Below the default crossover the entry point routes to Roth–Ruckenstein
    // and returns the same roots without forcing.
    let default_limits = AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 24, 1 << 28, 256);
    let routed = alekhnovich_roots(&rows, 2, default_limits, &mut scratch).expect("routed");
    assert_eq!(routed, expected);
}

/// A power-series root whose degree exceeds the caller's bound is filtered
/// during candidate materialization: the affine family is determined, but
/// its prefix already outranks `max_degree`, so the output is empty.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_drops_family_above_the_degree_bound() {
    // Q = X^5 + Y has the single power-series root X^5.
    let rows = vec![
        poly(&[b(0), b(0), b(0), b(0), b(0), b(0x1)]),
        poly(&[b(0x1)]),
    ];
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let lifted = alekhnovich_roots(&rows, 3, forced_limits(), &mut scratch).expect("bounded");
    assert!(lifted.is_empty());
    assert_eq!(
        roth_ruckenstein_roots(&rows, 3, limits()).expect("prefix lift"),
        lifted
    );

    // At a bound that admits it, the same rows return exactly X^5.
    let lifted = alekhnovich_roots(&rows, 5, forced_limits(), &mut scratch).expect("admitted");
    assert_eq!(lifted, vec![poly(&[b(0), b(0), b(0), b(0), b(0), b(0x1)])]);
}

/// The scalar-splitter charge at a precision-one node is bounded by the
/// caller's coefficient budget: one below the requirement fails, at it
/// succeeds.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_coefficient_budget_binds_at_leaf_charge() {
    let rows = vec![poly(&[b(0x9)]), poly(&[b(0x4)])];
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let strict = AlekhnovichLimits::new(1_000_000, 1_000_000, 17, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    assert_eq!(
        alekhnovich_roots(&rows, 0, strict, &mut scratch).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich coefficients",
            required: 18,
            limit: 17,
        })
    );
    let exact = AlekhnovichLimits::new(1_000_000, 1_000_000, 18, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    let lifted = alekhnovich_roots(&rows, 0, exact, &mut scratch).expect("within budget");
    assert_eq!(lifted.len(), 1);
}

/// The work-item budget binds at the residual push of a refinement: one
/// below the traversal's push count fails there, at it succeeds.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_work_item_budget_binds_during_traversal() {
    let rows = square_rows();
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let strict = AlekhnovichLimits::new(5, 1_000_000, 1 << 24, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    assert_eq!(
        alekhnovich_roots(&rows, 2, strict, &mut scratch).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich work items",
            required: 6,
            limit: 5,
        })
    );
    let exact = AlekhnovichLimits::new(6, 1_000_000, 1 << 24, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    let lifted = alekhnovich_roots(&rows, 2, exact, &mut scratch).expect("within budget");
    assert_eq!(lifted.len(), 1);
}

/// The intermediate-family budget binds at a tail insertion.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_family_budget_binds_at_tail_insertion() {
    let rows = square_rows();
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let strict = AlekhnovichLimits::new(1_000_000, 5, 1 << 24, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    assert_eq!(
        alekhnovich_roots(&rows, 2, strict, &mut scratch).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich intermediate families",
            required: 6,
            limit: 5,
        })
    );
    let exact = AlekhnovichLimits::new(1_000_000, 6, 1 << 24, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    let lifted = alekhnovich_roots(&rows, 2, exact, &mut scratch).expect("within budget");
    assert_eq!(lifted.len(), 1);
}

/// The scratch-byte budget binds during traversal, after the initial
/// charge passes: the minimal succeeding limit minus one fails with the
/// cumulative requirement, and the minimal limit itself succeeds.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_scratch_byte_budget_binds_during_traversal() {
    let rows = square_rows();
    let base = AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 24, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    let expected = alekhnovich_roots(&rows, 2, base, &mut scratch).expect("unbounded");
    assert_eq!(expected.len(), 1);

    // Find the minimal scratch-byte limit that lets the traversal finish.
    let mut minimal = 1_usize;
    while minimal < (1 << 20) {
        let limits = AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 24, minimal, 256)
            .with_roth_ruckenstein_crossover(0);
        let mut probe = AlekhnovichScratch::<Gf8B>::new();
        if alekhnovich_roots(&rows, 2, limits, &mut probe).is_ok() {
            break;
        }
        minimal += 1;
    }
    assert!(
        minimal < (1 << 20),
        "some scratch-byte limit below 2^20 must succeed"
    );

    let strict = AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 24, minimal - 1, 256)
        .with_roth_ruckenstein_crossover(0);
    let mut probe = AlekhnovichScratch::<Gf8B>::new();
    assert_eq!(
        alekhnovich_roots(&rows, 2, strict, &mut probe).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Alekhnovich scratch bytes",
            required: minimal,
            limit: minimal - 1,
        })
    );
    let exact = AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 24, minimal, 256)
        .with_roth_ruckenstein_crossover(0);
    let mut probe = AlekhnovichScratch::<Gf8B>::new();
    assert_eq!(
        alekhnovich_roots(&rows, 2, exact, &mut probe).expect("within budget"),
        expected
    );
}

fn is_reservation_failure(error: &RootError) -> bool {
    match error {
        RootError::Polynomial(poly_ring::PolynomialError::Config(
            poly_ring::ConfigError::AllocationFailed { .. },
        )) => true,
        #[cfg(feature = "fft")]
        RootError::Product(poly_ring::ProductError::Config(
            poly_ring::ConfigError::AllocationFailed { .. },
        )) => true,
        _ => false,
    }
}

/// Allocation refusal anywhere in the prefix lift reports a reservation
/// failure, and the same scratch recovers the planted roots afterwards.
#[test]
fn roth_allocation_refusal_reports_and_recovers() {
    let rows = two_root_rows();
    let expected = roth_ruckenstein_roots(&rows, 2, limits()).expect("oracle");
    assert_eq!(expected.len(), 2);

    let check = |scratch: &mut RothRuckensteinScratch<Gf8B>| {
        let mut output = Vec::new();
        roth_ruckenstein_roots_into(&mut output, &rows, 2, limits(), scratch).expect("recovered");
        assert_eq!(output, expected);
    };

    for threshold in [8_usize, 32, 96, 168] {
        let mut scratch = RothRuckensteinScratch::<Gf8B>::new();
        let mut output = Vec::new();
        let gated = with_threshold(threshold, || {
            roth_ruckenstein_roots_into(&mut output, &rows, 2, limits(), &mut scratch).map(|_| ())
        });
        assert!(
            is_reservation_failure(&gated.unwrap_err()),
            "threshold {threshold}"
        );
        check(&mut scratch);
    }

    for (size, skip) in [
        (2_usize, 0_usize),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 10),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5),
        (4, 0),
        (8, 0),
        (8, 4),
        (8, 11),
        (96, 0),
        (96, 1),
        (96, 2),
        (96, 3),
        (96, 4),
    ] {
        let mut scratch = RothRuckensteinScratch::<Gf8B>::new();
        let mut output = Vec::new();
        let gated = reject_matching(size, skip, || {
            roth_ruckenstein_roots_into(&mut output, &rows, 2, limits(), &mut scratch).map(|_| ())
        });
        match gated {
            Ok(()) => assert_eq!(output, expected, "size {size} skip {skip}"),
            Err(error) => {
                assert!(
                    is_reservation_failure(&error),
                    "size {size} skip {skip}: {error:?}"
                )
            }
        }
        check(&mut scratch);
    }
}

/// Allocation refusal inside the divide-and-conquer traversal — including
/// inside the affine-substitution product — reports a reservation failure
/// on warmed and on cold scratch, and the extraction recovers afterwards.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_allocation_refusal_reports_and_recovers() {
    let rows = square_rows();
    let expected = expected_roots(&rows, 2);
    assert_eq!(expected.len(), 1);

    let check = |scratch: &mut AlekhnovichScratch<Gf8B>| {
        let mut output = Vec::new();
        alekhnovich_roots_into(&mut output, &rows, 2, forced_limits(), scratch).expect("recovered");
        assert_eq!(output, expected);
    };

    // Warmed scratch: backend tables and pools are already in place, so
    // each gate lands on a traversal-time reservation.
    let mut warm = AlekhnovichScratch::<Gf8B>::new();
    let _ = alekhnovich_roots_into(&mut Vec::new(), &rows, 2, forced_limits(), &mut warm);
    {
        let mut scratch = AlekhnovichScratch::<Gf8B>::new();
        let _ = alekhnovich_roots_into(&mut Vec::new(), &rows, 2, forced_limits(), &mut scratch);
        let mut output = Vec::new();
        let gated = with_threshold(80, || {
            alekhnovich_roots_into(&mut output, &rows, 2, forced_limits(), &mut scratch).map(|_| ())
        });
        assert!(
            is_reservation_failure(&gated.unwrap_err()),
            "warmed threshold"
        );
        check(&mut scratch);
    }
    for (size, skip) in [
        (2_usize, 0_usize),
        (2, 1),
        (2, 5),
        (3, 3),
        (3, 4),
        (3, 5),
        (32, 0),
        (32, 1),
        (32, 2),
        (96, 0),
        (96, 1),
        (128, 0),
        (128, 1),
        (128, 2),
    ] {
        let mut scratch = AlekhnovichScratch::<Gf8B>::new();
        let _ = alekhnovich_roots_into(&mut Vec::new(), &rows, 2, forced_limits(), &mut scratch);
        let mut output = Vec::new();
        let gated = reject_matching(size, skip, || {
            alekhnovich_roots_into(&mut output, &rows, 2, forced_limits(), &mut scratch).map(|_| ())
        });
        match gated {
            Ok(()) => assert_eq!(output, expected, "size {size} skip {skip}"),
            Err(error) => {
                assert!(
                    is_reservation_failure(&error),
                    "size {size} skip {skip}: {error:?}"
                )
            }
        }
        check(&mut scratch);
    }

    // Cold scratch: the explicit work stack and the substitution product
    // allocate for the first time under the gate.
    for (size, skip) in [
        (2_usize, 0_usize),
        (2, 1),
        (2, 2),
        (32, 0),
        (48, 0),
        (96, 0),
        (96, 1),
        (96, 2),
        (128, 0),
        (128, 1),
        (128, 2),
        (480, 0),
    ] {
        let mut scratch = AlekhnovichScratch::<Gf8B>::new();
        let mut output = Vec::new();
        let gated = reject_matching(size, skip, || {
            alekhnovich_roots_into(&mut output, &rows, 2, forced_limits(), &mut scratch).map(|_| ())
        });
        match gated {
            Ok(()) => assert_eq!(output, expected, "cold size {size} skip {skip}"),
            Err(error) => {
                assert!(
                    is_reservation_failure(&error),
                    "cold size {size} skip {skip}: {error:?}"
                )
            }
        }
        check(&mut scratch);
    }
}

/// The precision-one leaf path has its own reservation chain; refusal at
/// any of its links reports, and a fresh extraction recovers.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_leaf_allocation_refusal_reports_and_recovers() {
    let rows = vec![poly(&[b(0x9)]), poly(&[b(0x4)])];
    let expected = expected_roots(&rows, 0);
    assert_eq!(expected.len(), 1);

    let check = |scratch: &mut AlekhnovichScratch<Gf8B>| {
        let mut output = Vec::new();
        alekhnovich_roots_into(&mut output, &rows, 0, forced_limits(), scratch).expect("recovered");
        assert_eq!(output, expected);
    };

    {
        let mut scratch = AlekhnovichScratch::<Gf8B>::new();
        let mut output = Vec::new();
        let gated = with_threshold(64, || {
            alekhnovich_roots_into(&mut output, &rows, 0, forced_limits(), &mut scratch).map(|_| ())
        });
        assert!(is_reservation_failure(&gated.unwrap_err()), "threshold");
        check(&mut scratch);
    }
    for (size, skip) in [
        (1_usize, 2_usize),
        (2, 0),
        (2, 1),
        (32, 0),
        (48, 0),
        (96, 0),
        (96, 1),
        (96, 2),
        (480, 0),
    ] {
        let mut scratch = AlekhnovichScratch::<Gf8B>::new();
        let mut output = Vec::new();
        let gated = reject_matching(size, skip, || {
            alekhnovich_roots_into(&mut output, &rows, 0, forced_limits(), &mut scratch).map(|_| ())
        });
        match gated {
            Ok(()) => assert_eq!(output, expected, "size {size} skip {skip}"),
            Err(error) => {
                assert!(
                    is_reservation_failure(&error),
                    "size {size} skip {skip}: {error:?}"
                )
            }
        }
        check(&mut scratch);
    }
}
