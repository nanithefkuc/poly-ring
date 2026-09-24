//! Equal-degree splitting edges: deep multi-root factorizations, repeated
//! roots through the squarefree gcd, and allocation-failure resistance.
//!
//! Every test drives the public surface only: `binary_field_roots_into`
//! (scratch form), `base_field_roots` (allocating form), and the Chien scan
//! as an independent backend sharing the frozen enumeration order.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, gf8b, gf16};
use poly_ring::{
    BaseFieldRoots, BinaryRootScratch, Polynomial, RootError, base_field_roots,
    binary_field_roots_into, chien_roots,
};

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

fn g16(value: u16) -> gf16::Elem {
    gf16::Elem::from_raw(value)
}

/// The product of `X + root` over every planted root.
fn with_planted_roots<F: FieldKernels>(roots: &[F::Elem]) -> Polynomial<F> {
    let mut polynomial = Polynomial::one().expect("one");
    for root in roots {
        polynomial = polynomial.multiply_x_plus(*root).expect("X + root");
    }
    polynomial
}

/// The frozen-order root list of a small-field polynomial, taken from the
/// independent Chien backend.
fn chien_list<F: FieldKernels>(polynomial: &Polynomial<F>) -> Vec<F::Elem> {
    chien_roots(polynomial)
        .expect("chien")
        .into_finite()
        .expect("finite")
}

/// Every root of a polynomial whose roots all lie in the coefficient field:
/// the planted set itself.
fn planted_list<F: FieldKernels>(roots: &[F::Elem]) -> Vec<F::Elem> {
    let mut list = roots.to_vec();
    list.sort_by_key(|root| poly_ring::internals::element_key::<F>(*root));
    list.dedup();
    list
}

/// A deep split: seven distinct planted roots over GF(16) force several
/// trace-split rounds, and the enumeration order agrees with the Chien scan.
#[test]
fn seven_planted_roots_split_in_frozen_order() {
    let roots: Vec<gf16::Elem> = (1_u16..=7).map(g16).collect();
    let polynomial = with_planted_roots::<Gf16>(&roots);
    let expected = planted_list::<Gf16>(&roots);
    assert_eq!(chien_list(&polynomial), expected);

    let mut scratch = BinaryRootScratch::<Gf16>::new();
    let mut found = Vec::new();
    let all = binary_field_roots_into(&mut found, &polynomial, &mut scratch).expect("split");
    assert!(!all);
    assert_eq!(found, expected);
    assert_eq!(
        base_field_roots(&polynomial).expect("allocating form"),
        BaseFieldRoots::Finite(expected)
    );
    for root in &found {
        assert!(polynomial.evaluate(*root).is_zero());
    }
}

/// The full-field polynomial `X^|F| + X` splits into every field element,
/// driving the factor stack to its deepest shape.
#[test]
fn full_field_polynomial_yields_every_element() {
    // X^16 + X over GF(2^4) vanishes on all sixteen elements.
    let mut coefficients = vec![<Gf16 as Field>::Elem::ZERO; 17];
    coefficients[1] = <Gf16 as Field>::Elem::ONE;
    coefficients[16] = <Gf16 as Field>::Elem::ONE;
    let polynomial = Polynomial::<Gf16>::from_coefficients(&coefficients).expect("X^16 + X");
    let expected = chien_list(&polynomial);
    assert_eq!(expected.len(), 16);

    let mut scratch = BinaryRootScratch::<Gf16>::new();
    let mut found = Vec::new();
    assert!(!binary_field_roots_into(&mut found, &polynomial, &mut scratch).expect("split"));
    assert_eq!(found, expected);
    // The scratch form preserves the allocating form's result exactly.
    assert_eq!(
        base_field_roots(&polynomial).expect("allocating form"),
        BaseFieldRoots::Finite(expected)
    );
}

/// Repeated roots collapse through `gcd(p, X^|F| + X)`: the square factor
/// contributes its root once, matching the distinct-root Chien list.
#[test]
fn repeated_roots_collapse_to_distinct_set() {
    let a = g16(0x3);
    let double = g16(0x5);
    let c = g16(0x9);
    // (X + double)^2 keeps the root distinct-set unchanged.
    let squared = with_planted_roots::<Gf16>(&[double])
        .multiply(&with_planted_roots::<Gf16>(&[double]))
        .expect("square");
    let mut polynomial = squared;
    for root in [a, c] {
        polynomial = polynomial.multiply_x_plus(root).expect("X + root");
    }
    let expected = planted_list::<Gf16>(&[a, double, c]);
    assert_eq!(chien_list(&polynomial), expected);

    let mut scratch = BinaryRootScratch::<Gf16>::new();
    let mut found = Vec::new();
    assert!(!binary_field_roots_into(&mut found, &polynomial, &mut scratch).expect("split"));
    assert_eq!(found, expected);
}

/// Boundary shapes: a linear factor extracts its root directly, an
/// irreducible quadratic yields nothing, a nonzero constant has no roots,
/// and the zero polynomial reports the whole field.
#[test]
fn boundary_shapes_report_directly() {
    // X + a: the single root is a itself in characteristic two.
    let linear = with_planted_roots::<Gf8B>(&[b(0x5)]);
    let mut scratch = BinaryRootScratch::<Gf8B>::new();
    let mut found = Vec::new();
    assert!(!binary_field_roots_into(&mut found, &linear, &mut scratch).expect("linear"));
    assert_eq!(found, vec![b(0x5)]);

    // An irreducible quadratic by field scan: some `X^2 + X + c` has no
    // root at all, and the split must agree with the Chien scan on it.
    let one = <Gf8B as Field>::Elem::ONE;
    let mut chosen = None;
    for raw in 1_u16..256 {
        let quad =
            Polynomial::<Gf8B>::from_coefficients(&[gf8b::Elem::from_raw(raw as u8), one, one])
                .expect("quad");
        let rooted =
            (0_u16..256).any(|point| quad.evaluate(gf8b::Elem::from_raw(point as u8)).is_zero());
        if !rooted {
            chosen = Some(quad);
            break;
        }
    }
    let irreducible = chosen.expect("an irreducible X^2 + X + c exists over GF(2^8)");
    assert_eq!(chien_list(&irreducible), Vec::new());
    let mut found = Vec::new();
    assert!(!binary_field_roots_into(&mut found, &irreducible, &mut scratch).expect("irreducible"));
    assert!(found.is_empty());

    // A nonzero constant: degree zero, no roots.
    let constant = Polynomial::<Gf8B>::from_coefficients(&[b(0x7)]).expect("constant");
    let mut found = Vec::new();
    assert!(!binary_field_roots_into(&mut found, &constant, &mut scratch).expect("constant"));
    assert!(found.is_empty());
    assert_eq!(
        base_field_roots(&constant).expect("constant"),
        BaseFieldRoots::Finite(Vec::new())
    );

    // The zero polynomial: every element is a root.
    let mut found = Vec::new();
    assert!(
        binary_field_roots_into(&mut found, &Polynomial::<Gf8B>::zero(), &mut scratch)
            .expect("zero")
    );
    assert_eq!(
        base_field_roots(&Polynomial::<Gf8B>::zero()).expect("zero"),
        BaseFieldRoots::All
    );

    // X^3: root zero with multiplicity three.
    let cube = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(0), b(0), b(1)]).expect("X^3");
    let mut found = Vec::new();
    assert!(!binary_field_roots_into(&mut found, &cube, &mut scratch).expect("cube"));
    assert_eq!(found, vec![b(0)]);
}

/// A warmed scratch keeps its capacity across a changed input and returns
/// the changed polynomial's roots, not the previous run's.
#[test]
fn scratch_reuse_handles_changed_inputs() {
    let first = with_planted_roots::<Gf16>(&[g16(1), g16(2), g16(4), g16(8)]);
    let second = with_planted_roots::<Gf16>(&[g16(3), g16(5), g16(6), g16(7), g16(10)]);
    let mut scratch = BinaryRootScratch::<Gf16>::new();

    let mut found = Vec::new();
    binary_field_roots_into(&mut found, &first, &mut scratch).expect("first");
    assert_eq!(
        found,
        planted_list::<Gf16>(&[g16(1), g16(2), g16(4), g16(8)])
    );
    let capacity = scratch.capacity();

    let mut found = Vec::new();
    binary_field_roots_into(&mut found, &second, &mut scratch).expect("second");
    assert_eq!(
        found,
        planted_list::<Gf16>(&[g16(3), g16(5), g16(6), g16(7), g16(10)])
    );
    assert!(scratch.capacity() >= capacity);
    assert!(scratch.capacity() > 0);
}

fn is_allocation_failure(error: &RootError) -> bool {
    matches!(
        error,
        RootError::Polynomial(poly_ring::PolynomialError::Config(
            poly_ring::ConfigError::AllocationFailed { .. }
        ))
    )
}

/// Allocation refusal anywhere inside a split is reported as a reservation
/// failure, and the same scratch recovers the full root set afterwards —
/// including the runs whose failure left factors on the stack, which the
/// next call must recycle rather than leak.
#[test]
fn allocation_refusal_reports_and_recovers() {
    let roots: Vec<gf16::Elem> = (1_u16..=8).map(g16).collect();
    let polynomial = with_planted_roots::<Gf16>(&roots);
    let expected = chien_list(&polynomial);
    assert_eq!(expected.len(), 8);

    let check = |scratch: &mut BinaryRootScratch<Gf16>| {
        let mut found = Vec::new();
        binary_field_roots_into(&mut found, &polynomial, scratch).expect("recovered");
        assert_eq!(found, expected);
    };

    // Size thresholds: each refuses a progressively later reservation.
    // The list steps around the buffer sizes that supporting arithmetic
    // grows by plain cloning; refusing one of those aborts the process
    // instead of reporting, so every threshold here lands on a checked
    // reservation instead.
    for threshold in [
        0_usize, 1, 2, 3, 8, 9, 16, 18, 20, 24, 28, 36, 40, 48, 64, 96, 120, 168,
    ] {
        let mut scratch = BinaryRootScratch::<Gf16>::new();
        let mut found = Vec::new();
        let gated = with_threshold(threshold, || {
            binary_field_roots_into(&mut found, &polynomial, &mut scratch).map(|_| ())
        });
        match gated {
            Ok(()) => assert_eq!(found, expected),
            Err(error) => assert!(
                is_allocation_failure(&error),
                "threshold {threshold}: {error:?}"
            ),
        }
        check(&mut scratch);
    }

    // Matching gates: refusing a specific allocation of one exact size
    // lands the failure at a different supporting operation each time.
    for (size, skip) in [
        (4_usize, 0_usize),
        (4, 1),
        (4, 2),
        (10, 0),
        (10, 1),
        (16, 1),
        (16, 3),
        (18, 0),
        (30, 0),
    ] {
        let mut scratch = BinaryRootScratch::<Gf16>::new();
        let mut found = Vec::new();
        let gated = reject_matching(size, skip, || {
            binary_field_roots_into(&mut found, &polynomial, &mut scratch).map(|_| ())
        });
        match gated {
            Ok(()) => assert_eq!(found, expected),
            Err(error) => {
                assert!(
                    is_allocation_failure(&error),
                    "size {size} skip {skip}: {error:?}"
                )
            }
        }
        check(&mut scratch);
    }

    // The same discipline over GF(2^8), where the two-byte coefficient
    // reservations of the trace splitter are matched exactly: a failure
    // there leaves factors on the stack, and the recovery call must
    // recycle them rather than leak or double-run.
    let roots8: Vec<gf8b::Elem> = (1_u8..=10).map(b).collect();
    let polynomial8 = with_planted_roots::<Gf8B>(&roots8);
    let expected8 = chien_list(&polynomial8);
    assert_eq!(expected8.len(), 10);
    for (size, skip) in [
        (2_usize, 0_usize),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (4, 0),
        (5, 0),
        (5, 1),
        (5, 2),
        (7, 0),
        (9, 0),
        (9, 1),
        (10, 0),
        (17, 0),
        (240, 0),
        (240, 1),
    ] {
        let mut scratch = BinaryRootScratch::<Gf8B>::new();
        let mut found = Vec::new();
        let gated = reject_matching(size, skip, || {
            binary_field_roots_into(&mut found, &polynomial8, &mut scratch).map(|_| ())
        });
        match gated {
            Ok(()) => assert_eq!(found, expected8),
            Err(error) => {
                assert!(
                    is_allocation_failure(&error),
                    "size {size} skip {skip}: {error:?}"
                )
            }
        }
        let mut recovered = Vec::new();
        binary_field_roots_into(&mut recovered, &polynomial8, &mut scratch).expect("recovered");
        assert_eq!(recovered, expected8);
        // A reused root buffer needs no new reservation at all.
        binary_field_roots_into(&mut recovered, &polynomial8, &mut scratch).expect("rewarmed");
        assert_eq!(recovered, expected8);
    }
    for threshold in [100_usize, 200] {
        let mut scratch = BinaryRootScratch::<Gf8B>::new();
        let mut found = Vec::new();
        let gated = with_threshold(threshold, || {
            binary_field_roots_into(&mut found, &polynomial8, &mut scratch).map(|_| ())
        });
        assert!(
            is_allocation_failure(&gated.unwrap_err()),
            "threshold {threshold}"
        );
        let mut recovered = Vec::new();
        binary_field_roots_into(&mut recovered, &polynomial8, &mut scratch).expect("recovered");
        assert_eq!(recovered, expected8);
    }
}
