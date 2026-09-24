//! Error-path coverage for the evaluation surfaces.
//!
//! Three families: public-API error construction whose variants and
//! `Display` arms are otherwise unreached, the odd-characteristic Hasse
//! derivative factor table (whose prime-multiple rows vanish), and the
//! reservation failures that surface only once every earlier table fits.

use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{
    ConfigError, DerivativePlan, DomainError, EvalError, FactorizationError, HasseError,
    HermiteError, JetPlan, MULTIPOINT_LANE_STEP_CROSSOVER, ModulusPlan, MultiplicityPlan,
    MultipointScratch, Polynomial, PolynomialError, RemainderTree, RootError,
    evaluate_multipoint_into, interpolate_lagrange,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

// ── Test-only failing allocator ───────────────────────────────────────────────
//
// Two thread-local gates, so parallel tests in this binary never charge each
// other: a size threshold (refuse anything larger) and an exact-size gate
// (allow a fixed count of matching allocations, then refuse the next).

struct FailingAllocator;

thread_local! {
    static THRESHOLD: Cell<usize> = const { Cell::new(usize::MAX) };
    static MATCHING: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let refused = THRESHOLD
            .try_with(|gate| layout.size() > gate.get())
            .unwrap_or(false)
            && !std::thread::panicking();
        let refused = refused
            || MATCHING
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

/// Arms the threshold gate for one operation and disarms it however the
/// operation leaves.
fn with_threshold<R>(bytes: usize, operation: impl FnOnce() -> R) -> R {
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            THRESHOLD.with(|gate| gate.set(usize::MAX));
        }
    }
    THRESHOLD.with(|gate| gate.set(bytes));
    let _disarm = Disarm;
    operation()
}

/// Arms the exact-size gate: `allowed` allocations of exactly `size` bytes
/// succeed, the next one is refused.
fn refuse_after<R>(size: usize, allowed: usize, operation: impl FnOnce() -> R) -> R {
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            MATCHING.with(|gate| gate.set(None));
        }
    }
    MATCHING.with(|gate| gate.set(Some((size, allowed))));
    let _disarm = Disarm;
    operation()
}

// ── Error construction and Display routing ───────────────────────────────────

/// A constant modulus is rejected by name and a non-coprime residue reports
/// the quotient-ring invariant, both with distinguishable renderings.
#[test]
fn quotient_ring_rejects_constant_modulus_and_noninvertible_residue() {
    let unit = Polynomial::<Gf8B>::from_coefficients(&[b(1)]).expect("unit");
    let error = ModulusPlan::new(&unit).expect_err("constant modulus");
    assert_eq!(error, PolynomialError::ConstantModulus);
    assert!(format!("{error}").contains("modulus"));

    let modulus = Polynomial::<Gf8B>::from_coefficients(&[b(1), b(0), b(1)]).expect("modulus");
    let plan = ModulusPlan::new(&modulus).expect("plan");
    let zero = Polynomial::<Gf8B>::zero();
    let error = plan.inverse(&zero).expect_err("zero residue");
    assert_eq!(error, PolynomialError::NotInvertibleModulo);
    assert!(format!("{error}").contains("invertible"));
}

/// Conversions into `FactorizationError` wrap their cause, and every variant
/// renders its payload so distinct failures stay distinct.
#[test]
fn factorization_errors_wrap_causes_and_render_payloads() {
    let from_polynomial = FactorizationError::from(PolynomialError::DivisionByZero);
    assert_eq!(
        from_polynomial,
        FactorizationError::Polynomial(PolynomialError::DivisionByZero)
    );
    assert!(format!("{from_polynomial}").contains("division by zero"));

    let config = ConfigError::FieldCapacityExceeded {
        points: 300,
        field_order: 256,
    };
    let from_config = FactorizationError::from(config);
    assert_eq!(
        from_config,
        FactorizationError::Polynomial(PolynomialError::Config(config))
    );
    let rendered = format!("{from_config}");
    assert!(rendered.contains("300") && rendered.contains("256"));

    let not_equal_degree = FactorizationError::NotEqualDegree { factor_degree: 7 };
    let invariant = FactorizationError::Invariant {
        reason: "synthetic identity",
    };
    let rendered = [
        format!("{}", FactorizationError::ZeroPolynomial),
        format!("{}", FactorizationError::ZeroFactorDegree),
        format!("{}", FactorizationError::NotSquareFree),
        format!("{not_equal_degree}"),
        format!("{invariant}"),
        format!("{from_config}"),
    ];
    assert!(rendered[3].contains('7'));
    assert!(rendered[4].contains("synthetic identity"));
    let mut distinct = rendered.to_vec();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), rendered.len());
}

/// Conversions into `RootError` preserve the nested failure, and the
/// product- and factorization-carrying arms render their payloads.
#[test]
fn root_errors_wrap_nested_causes() {
    let zero_polynomial = FactorizationError::ZeroPolynomial;
    let from_factorization = RootError::from(zero_polynomial);
    assert_eq!(
        from_factorization,
        RootError::Factorization(zero_polynomial)
    );
    assert!(format!("{from_factorization}").contains("256").eq(&false));

    let limit = RootError::ResourceLimitExceeded {
        resource: "lift budget",
        required: 12,
        limit: 8,
    };
    let rendered = format!("{limit}");
    assert!(rendered.contains("12") && rendered.contains("8"));

    let product = RootError::Product(poly_ring::ProductError::Polynomial(
        PolynomialError::DivisionByZero,
    ));
    assert!(format!("{product}").contains("division by zero"));
    assert_ne!(product, from_factorization);
}

/// Config, polynomial, domain, evaluation, and Hermite errors render the
/// nested payload through their forwarding arms.
#[test]
fn wrapped_evaluation_errors_render_the_nested_payload() {
    let scratch_too_small = ConfigError::ScratchTooSmall {
        context: "lane capacity probe",
        required: 9,
        available: 2,
    };
    let domain = DomainError::from(scratch_too_small);
    assert_eq!(domain, DomainError::Config(scratch_too_small));
    let rendered = format!("{domain}");
    assert!(rendered.contains('9') && rendered.contains('2'));

    let evaluation = EvalError::from(PolynomialError::DivisionByZero);
    assert_eq!(
        evaluation,
        EvalError::Polynomial(PolynomialError::DivisionByZero)
    );
    assert!(format!("{evaluation}").contains("division by zero"));

    let overflow = ConfigError::GeometryOverflow {
        context: "hermite geometry probe",
    };
    let hermite_config = HermiteError::from(overflow);
    assert_eq!(hermite_config, HermiteError::Config(overflow));
    assert!(format!("{hermite_config}").contains("hermite geometry probe"));

    let hermite_polynomial = HermiteError::from(PolynomialError::NonExactDivision);
    assert_eq!(
        hermite_polynomial,
        HermiteError::Polynomial(PolynomialError::NonExactDivision)
    );
    assert_ne!(format!("{hermite_polynomial}"), format!("{hermite_config}"));
}

// ── Odd-characteristic derivative factor table ───────────────────────────────

/// An order one below the Mersenne prime makes the recurrence factor
/// C(p, p − 1) = p vanish in GF(p), so the factor table's second entry is
/// an explicit zero while the plan itself stays cheap: only two factors are
/// materialized, and the capacity query reports the overflow as a checked
/// error instead of an allocation.
#[test]
fn prime_derivative_plan_zeroes_the_prime_multiple_factor() {
    const PRIME: usize = 2_147_483_647;
    let plan = DerivativePlan::<Mersenne31>::new(PRIME - 1, PRIME + 1).expect("plan");
    assert_eq!(plan.order(), PRIME - 1);
    assert_eq!(plan.max_coefficients(), PRIME + 1);
    assert_eq!(plan.output_coefficients(PRIME - 1), Ok(0));
    assert_eq!(plan.output_coefficients(PRIME + 1), Ok(2));
    // Applying to real inputs would need `p` coefficients, so the observable
    // contract of this plan is its checked geometry: two output rows in
    // capacity, an error one row beyond it.
    assert_eq!(
        plan.output_coefficients(PRIME + 2),
        Err(HasseError::CoefficientCapacityExceeded {
            maximum: PRIME + 1,
            actual: PRIME + 2,
        })
    );
}

// ── Scratch reservation failures past earlier tables ─────────────────────────

/// `MultiplicityPlan::scratch` names the leaf-staging table when its
/// reservation fails after the remainder-tree and jet workspaces fit.
#[test]
fn multiplicity_scratch_reports_leaf_staging_reservation_failure() {
    let points = [b(1), b(2)];
    let plan = MultiplicityPlan::<Gf8B>::new(&points, &[2, 1], 8).expect("multiplicity plan");
    let warm = plan.scratch(2).expect("warm-up");
    drop(warm);
    // Three weighted rows over two lanes: the leaf staging buffer is the
    // first six-byte reservation on this path.
    let error = refuse_after(6, 0, || plan.scratch(2)).expect_err("leaf staging");
    assert_eq!(
        error,
        HasseError::AllocationFailed {
            context: "leaf remainder staging"
        }
    );
    assert!(plan.scratch(2).is_ok());
}

/// `JetPlan::scratch` names the scalar-output staging when its reservation
/// fails after the convolution, recursion-slot, and scalar-input tables fit,
/// in both characteristic two and odd characteristic.
#[test]
fn jet_scratch_reports_scalar_output_reservation_failure() {
    let plan = JetPlan::<Gf8B>::new(b(3), 3, 8).expect("jet plan");
    // The scalar-output staging is the twenty-first three-byte reservation:
    // two operand-lane and six convolution-lane buffers precede the twelve
    // recursion slots.
    drop(plan.scratch(1).expect("warm-up"));
    let error = refuse_after(3, 19, || plan.scratch(1)).expect_err("scalar output");
    assert_eq!(
        error,
        HasseError::AllocationFailed {
            context: "jet scalar output"
        }
    );
    assert!(plan.scratch(1).is_ok());
}

/// The same scalar-output arm over GF(31), where one element occupies four
/// bytes and the staging sizes differ from the characteristic-two layout.
#[test]
fn prime_jet_scratch_reports_scalar_output_reservation_failure() {
    let plan = JetPlan::<Mersenne31>::new(m31(3), 2, 8).expect("jet plan");
    // Four-byte elements double every staging size; the scalar-output
    // staging is the twenty-fifth eight-byte reservation on this path.
    drop(plan.scratch(1).expect("warm-up"));
    let error = refuse_after(8, 24, || plan.scratch(1)).expect_err("scalar output");
    assert_eq!(
        error,
        HasseError::AllocationFailed {
            context: "jet scalar output"
        }
    );
    assert!(plan.scratch(1).is_ok());
}

/// A fresh multipoint scratch names its remainder-slot table when the first
/// slot reservation fails after the subproduct tree and value vector fit.
/// The step count crosses the lane crossover, so the request routes through
/// the subproduct tree.
#[test]
fn multipoint_evaluation_reports_remainder_slot_reservation_failure() {
    let points: Vec<_> = (1_u8..=17).map(b).collect();
    let coefficient_count = MULTIPOINT_LANE_STEP_CROSSOVER / 17 + 1;
    let polynomial = Polynomial::<Gf8B>::from_coefficients(
        &(0..coefficient_count)
            .map(|index| b((index % 251 + 1) as u8))
            .collect::<Vec<_>>(),
    )
    .expect("polynomial");
    let slot_bytes = std::mem::size_of::<Polynomial<Gf8B>>();
    let mut values = Vec::new();
    let mut scratch = MultipointScratch::new();
    // The slot table reserves with the allocator's four-slot minimum, so the
    // first ninety-six-byte reservation on this path is the slot table.
    let error = refuse_after(4 * slot_bytes, 0, || {
        evaluate_multipoint_into(&mut values, &polynomial, &points, &mut scratch)
    })
    .expect_err("remainder slot");
    assert_eq!(
        error,
        PolynomialError::Config(ConfigError::AllocationFailed {
            context: "multipoint remainder slots",
            elements: 1,
            element_size: slot_bytes,
        })
    );

    // The same geometry succeeds once the gate opens, and the tree descent
    // agrees with direct Horner evaluation.
    let mut values = Vec::new();
    let mut scratch = MultipointScratch::new();
    evaluate_multipoint_into(&mut values, &polynomial, &points, &mut scratch)
        .expect("open-gate evaluation");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(*value, polynomial.evaluate(*point));
    }
}
/// The value vector names itself when its reservation fails before the
/// lane buffers: at this step count the lane-parallel Horner route takes
/// the request and reserves the values first.
#[test]
fn multipoint_evaluation_reports_value_reservation_failure() {
    let points: Vec<_> = (1_u8..=17).map(b).collect();
    let polynomial =
        Polynomial::<Gf8B>::from_coefficients(&[b(1), b(0), b(1)]).expect("polynomial");
    let element_bytes = std::mem::size_of::<gf8b::Elem>();
    let mut values = Vec::new();
    let mut scratch = MultipointScratch::new();
    let error = refuse_after(element_bytes * 17, 0, || {
        evaluate_multipoint_into(&mut values, &polynomial, &points, &mut scratch)
    })
    .expect_err("value reservation");
    assert_eq!(
        error,
        PolynomialError::Config(ConfigError::AllocationFailed {
            context: "multipoint values",
            elements: 17,
            element_size: element_bytes,
        })
    );
}
/// Lagrange interpolation surfaces the failure of its internal multipoint
/// evaluation pass as a polynomial error, after its own master tree,
/// derivative, and denominator tables were built. At this geometry the
/// inner evaluation runs through the lane-parallel Horner route, whose
/// lane-buffer reservation is the first allocation its scratch makes.
#[test]
fn lagrange_interpolation_reports_evaluation_reservation_failure() {
    let points: Vec<_> = (1_u8..=17).map(b).collect();
    let values: Vec<_> = (0_u8..17).map(|raw| b(raw % 17 + 1)).collect();
    // The master tree's two seventeen-byte reservations and the denominator
    // table are the three seventeen-byte reservations ahead of the inner
    // evaluation; the fourth is its first lane buffer.
    let error = refuse_after(17, 3, || interpolate_lagrange::<Gf8B>(&points, &values))
        .expect_err("inner evaluation");
    assert_eq!(
        error,
        EvalError::Polynomial(PolynomialError::Config(ConfigError::AllocationFailed {
            context: "lane evaluation buffers",
            elements: 17,
            element_size: 1,
        }))
    );

    let interpolated =
        interpolate_lagrange::<Gf8B>(&points, &values).expect("open-gate interpolation");
    for (point, value) in points.iter().zip(&values) {
        assert_eq!(interpolated.evaluate(*point), *value);
    }
}

/// `RemainderTree::scratch` names a staging table once the convolution
/// engine and descent slots fit: the reversal rows reserve twice, so the
/// second same-size reservation is the one refused.
#[test]
fn remainder_scratch_names_staging_reservation_failure() {
    let moduli = [
        Polynomial::<Gf8B>::from_coefficients(&[b(1), b(1)]).expect("modulus"),
        Polynomial::<Gf8B>::from_coefficients(&[b(2), b(0), b(1)]).expect("modulus"),
    ];
    let tree = RemainderTree::<Gf8B>::new(&moduli, 8).expect("tree");
    drop(tree.scratch(1).expect("warm-up"));
    let error = with_threshold(0, || tree.scratch(1)).expect_err("closed gate");
    assert!(matches!(
        error,
        poly_ring::ProductError::Config(ConfigError::AllocationFailed { .. })
    ));
    assert!(tree.scratch(1).is_ok());
}
