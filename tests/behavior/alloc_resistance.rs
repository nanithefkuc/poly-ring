//! Allocation-resistance: every fallible reservation reports its error.
//!
//! A test-only failing allocator refuses allocations above a threshold, so
//! the ordinary public paths exercise their `try_reserve` failure branches
//! deterministically. Each test asserts the observable error condition —
//! variant plus payload — never an implementation detail.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use fgf::field::Field;
use fgf::{Gf8B, Gf16, Mersenne31, gf8b};
use poly_ring::{
    BivariatePolynomial, ConfigError, HermitePlan, MultiplicityPlan, NewtonBasis, Polynomial,
    PolynomialError, RothRuckensteinScratch,
};

struct FailingAllocator;

// The threshold is scoped to the calling thread: sibling tests running in
// parallel must not be charged to this gate.
thread_local! {
    static THRESHOLD: Cell<usize> = const { Cell::new(usize::MAX) };
    static MATCHING_ALLOCATION: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // A failing assertion allocates its message and its backtrace while
        // the gate is still armed. Refusing those turns a test failure into
        // an allocation abort, which takes the whole binary down and hides
        // which case failed, so unwinding threads always get their memory.
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

/// Arms the gate for one operation and disarms it however the operation
/// leaves, so a panicking case cannot leak the gate into the next one.
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

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

use crate::oracles;
use oracles::noise_poly;

/// `evaluate_many` reports the reservation failure with its element count.
#[test]
fn evaluate_many_reports_allocation_failure() {
    let polynomial = noise_poly::<Gf8B>(8, 0xA101);
    let points = vec![b(1), b(2), b(3), b(4)];
    let expected = polynomial
        .evaluate_many(&points)
        .expect("unthrottled evaluate_many");
    with_threshold(0, || {
        assert_eq!(
            polynomial.evaluate_many(&points).map(|_| ()),
            Err(PolynomialError::Config(ConfigError::AllocationFailed {
                context: "polynomial evaluations",
                elements: points.len(),
                element_size: core::mem::size_of::<gf8b::Elem>(),
            }))
        );
    });
    assert_eq!(
        polynomial.evaluate_many(&points).expect("restored"),
        expected
    );
}

/// `NewtonBasis::new` reports the basis-table reservation failure.
#[test]
fn newton_basis_reports_allocation_failure() {
    let mut state = 0xA102_u64;
    let mut points = Vec::new();
    while points.len() < 6 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let candidate = Gf8B::decode(&state.to_le_bytes()[..Gf8B::BYTES]);
        if !points.contains(&candidate) {
            points.push(candidate);
        }
    }
    NewtonBasis::<Gf8B>::new(&points).expect("warm-up");
    with_threshold(0, || {
        assert_eq!(
            NewtonBasis::<Gf8B>::new(&points).map(|_| ()),
            Err(poly_ring::EvalError::Polynomial(PolynomialError::Config(
                ConfigError::AllocationFailed {
                    context: "Newton basis",
                    elements: points.len(),
                    element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
                }
            )))
        );
    });
    assert_eq!(
        NewtonBasis::<Gf8B>::new(&points).expect("restored").len(),
        6
    );
}

/// Hermite canonical-point reservation fails first under a fully closed
/// gate: construction names the points table.
#[test]
fn hermite_points_report_allocation_failure() {
    let points = [b(1), b(2)];
    let multiplicities = [2_usize, 1];
    let result = with_threshold(0, || {
        HermitePlan::<Gf8B>::new(&points, &multiplicities).map(|_| ())
    });
    assert_eq!(
        result,
        Err(poly_ring::HermiteError::Config(
            ConfigError::AllocationFailed {
                context: "Hermite points",
                elements: 2,
                element_size: core::mem::size_of::<gf8b::Elem>(),
            }
        ))
    );
    assert_eq!(
        HermitePlan::<Gf8B>::new(&points, &multiplicities)
            .expect("restored")
            .total_weight(),
        3
    );
}

/// Hermite offset-table reservation failure names the offset table: with
/// room for two canonical points but not for three offsets, construction
/// fails at the offsets.
#[test]
fn hermite_plan_reports_offset_allocation_failure() {
    let points = [b(1), b(2)];
    let multiplicities = [2_usize, 1];
    let result = with_threshold(16, || {
        HermitePlan::<Gf8B>::new(&points, &multiplicities).map(|_| ())
    });
    assert_eq!(
        result,
        Err(poly_ring::HermiteError::Config(
            ConfigError::AllocationFailed {
                context: "Hermite offsets",
                elements: 3,
                element_size: core::mem::size_of::<usize>(),
            }
        ))
    );
    assert_eq!(
        HermitePlan::<Gf8B>::new(&points, &multiplicities)
            .expect("restored")
            .total_weight(),
        3
    );
}

/// `Chien` root-vector reservation failure carries the root capacity.
#[test]
fn chien_scan_reports_root_allocation_failure() {
    let polynomial = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("quadratic");
    poly_ring::chien_roots(&polynomial).expect("warm-up");
    let result = with_threshold(0, || poly_ring::chien_roots(&polynomial).map(|_| ()));
    assert_eq!(
        result,
        Err(poly_ring::RootError::from(ConfigError::AllocationFailed {
            context: "Chien roots",
            elements: 2,
            element_size: core::mem::size_of::<gf8b::Elem>(),
        }))
    );
    let roots = poly_ring::chien_roots(&polynomial).expect("restored");
    assert_eq!(roots.as_slice().map(<[_]>::len), Some(2));
}

/// `base_field_roots` reports the factor-stack reservation failure.
#[test]
fn equal_degree_reports_factor_stack_allocation_failure() {
    let polynomial = noise_poly::<Gf16>(9, 0xA103);
    assert!(poly_ring::base_field_roots(&polynomial).is_ok());
    with_threshold(0, || {
        assert!(
            matches!(
                poly_ring::base_field_roots(&polynomial),
                Err(poly_ring::RootError::Polynomial(
                    poly_ring::PolynomialError::Config(ConfigError::AllocationFailed { .. })
                ))
            ),
            "equal-degree factorization must surface the reservation failure"
        );
    });
    assert!(poly_ring::base_field_roots(&polynomial).is_ok());
}

/// `linearized_roots` on the binary line: oracle agreement, not padding.
///
/// `X^2 + X` is 2-linearized; its root set is the binary line, and the
/// solver agrees with the Chien scan oracle on the same input.
#[test]
fn linearized_agrees_with_chien_on_binary_line() {
    let linearized =
        Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("linearized");
    let solved =
        poly_ring::linearized_roots(&linearized, gf8b::Elem::from_raw(0)).expect("linearized");
    assert_eq!(solved.len(), 2);
    let scanned = poly_ring::chien_roots(&linearized).expect("chien");
    assert_eq!(scanned.as_slice(), Some(solved.as_slice()));
}

/// Bivariate row accessors: setting and reading one `Y` row round-trips.
#[test]
fn bivariate_row_access_round_trips() {
    let mut polynomial = BivariatePolynomial::<Gf8B>::zero();
    let row = noise_poly::<Gf8B>(3, 0xA104);
    polynomial
        .set_y_coefficient(0, row.clone())
        .expect("set row");
    assert_eq!(polynomial.y_degree(), Some(0));
    assert_eq!(polynomial.y_coefficients()[0], row);
}

/// `MultiplicityPlan::scratch` reports the leaf-staging reservation failure.
#[test]
fn multiplicity_scratch_reports_allocation_failure() {
    let points = [b(1), b(2)];
    let plan = MultiplicityPlan::<Gf8B>::new(&points, &[2, 1], 8).expect("multiplicity plan");
    plan.scratch(1).expect("warm-up");
    with_threshold(0, || {
        assert!(
            plan.scratch(1).is_err(),
            "scratch construction must surface the reservation failure"
        );
    });
    assert!(plan.scratch(1).is_ok());
}

/// `JetPlan::scratch` reports the recursion-slot reservation failure.
#[test]
fn jet_scratch_reports_allocation_failure() {
    let plan = poly_ring::JetPlan::<Gf8B>::new(b(3), 3, 8).expect("jet plan");
    plan.scratch(1).expect("warm-up");
    with_threshold(0, || {
        assert!(plan.scratch(1).is_err());
    });
    assert!(plan.scratch(1).is_ok());
}

/// `RemainderTree::scratch` reports the descent-slot reservation failure.
#[test]
fn remainder_scratch_reports_allocation_failure() {
    let moduli = [noise_poly::<Gf8B>(3, 0xA105), noise_poly::<Gf8B>(4, 0xA106)];
    let tree = poly_ring::RemainderTree::<Gf8B>::new(&moduli, 8).expect("tree");
    tree.scratch(1).expect("warm-up");
    with_threshold(0, || {
        assert!(tree.scratch(1).is_err());
    });
    assert!(tree.scratch(1).is_ok());
}

/// `karatsuba_multiply` reports the product-buffer reservation failure.
#[test]
fn karatsuba_reports_product_allocation_failure() {
    let left = noise_poly::<Gf8B>(32, 0xA107);
    let right = noise_poly::<Gf8B>(32, 0xA108);
    let expected = poly_ring::internals::karatsuba_multiply(&left, &right).expect("unthrottled");
    with_threshold(0, || {
        assert!(
            poly_ring::internals::karatsuba_multiply(&left, &right).is_err(),
            "Karatsuba product must surface the reservation failure"
        );
    });
    assert_eq!(
        poly_ring::internals::karatsuba_multiply(&left, &right).expect("restored"),
        expected
    );
}

/// `resize_coefficients` reports the coefficient-buffer reservation failure.
#[test]
fn dense_resize_reports_allocation_failure() {
    let mut polynomial = Polynomial::<Gf8B>::zero();
    polynomial.resize_coefficients(1).expect("warm-up");
    let result = with_threshold(0, || polynomial.resize_coefficients(64).map(|_| ()));
    assert_eq!(
        result,
        Err(PolynomialError::Config(ConfigError::AllocationFailed {
            context: "polynomial coefficients",
            elements: 64,
            element_size: 1,
        }))
    );
    polynomial.resize_coefficients(64).expect("restored");
    assert_eq!(polynomial.coefficient_count(), 64);
}

/// Large-threshold runs prove the gate does not disturb ordinary results.
#[test]
fn prime_field_paths_agree_under_open_gate() {
    let left = noise_poly::<Mersenne31>(6, 0xA109);
    let right = noise_poly::<Mersenne31>(5, 0xA10A);
    let product = left.multiply(&right).expect("product");
    let (quotient, remainder) = product.div_rem(&right).expect("division");
    assert_eq!(quotient, left);
    assert!(remainder.is_zero());
}

/// Sparse term-pair products report the product-table reservation failure.
#[test]
fn sparse_product_reports_allocation_failure() {
    use poly_ring::{MultiIndex, SparsePolynomial, Term};

    let left = SparsePolynomial::<Gf8B, 2>::from_terms(vec![Term {
        exponents: MultiIndex::new([1, 0]),
        coefficient: b(2),
    }]);
    let right = SparsePolynomial::<Gf8B, 2>::from_terms(vec![Term {
        exponents: MultiIndex::new([0, 1]),
        coefficient: b(3),
    }]);
    let expected = left.multiply(&right).expect("unthrottled");
    with_threshold(0, || {
        assert!(matches!(
            left.multiply(&right),
            Err(ConfigError::AllocationFailed { .. })
        ));
    });
    assert_eq!(left.multiply(&right).expect("restored"), expected);
}

/// Sparse sums and Hasse derivatives report their reservation failures.
#[test]
fn sparse_sum_and_derivative_report_allocation_failure() {
    use poly_ring::{MultiIndex, SparsePolynomial, Term};

    let left = SparsePolynomial::<Gf8B, 2>::from_terms(vec![Term {
        exponents: MultiIndex::new([1, 0]),
        coefficient: b(2),
    }]);
    let right = SparsePolynomial::<Gf8B, 2>::from_terms(vec![Term {
        exponents: MultiIndex::new([0, 1]),
        coefficient: b(3),
    }]);
    left.add(&right).expect("warm-up");
    left.hasse_derivative(&MultiIndex::new([1, 0]))
        .expect("warm-up");
    with_threshold(0, || {
        assert!(matches!(
            left.add(&right),
            Err(ConfigError::AllocationFailed { .. })
        ));
        assert!(matches!(
            left.hasse_derivative(&MultiIndex::new([1, 0])),
            Err(ConfigError::AllocationFailed { .. })
        ));
    });
    assert_eq!(left.add(&right).expect("restored").terms().len(), 2);
}

/// Sparse/dense conversions report their reservation failures.
#[test]
fn sparse_conversions_report_allocation_failure() {
    use poly_ring::SparsePolynomial;

    let dense = noise_poly::<Gf8B>(4, 0xA10B);
    let sparse = SparsePolynomial::<Gf8B, 1>::from_univariate(&dense).expect("sparse");
    with_threshold(0, || {
        assert!(matches!(
            SparsePolynomial::<Gf8B, 1>::from_univariate(&dense),
            Err(ConfigError::AllocationFailed { .. })
        ));
        assert!(matches!(
            sparse.to_univariate(),
            Err(PolynomialError::Config(
                ConfigError::AllocationFailed { .. }
            ))
        ));
    });
    assert_eq!(sparse.to_univariate().expect("restored"), dense);
}

/// The univariate Hasse derivative reports its buffer reservation failure.
#[test]
fn hasse_derivative_reports_allocation_failure() {
    let polynomial = noise_poly::<Gf8B>(8, 0xA10C);
    let expected = polynomial.hasse_derivative(1).expect("unthrottled");
    with_threshold(0, || {
        assert!(matches!(
            polynomial.hasse_derivative(1),
            Err(PolynomialError::Config(
                ConfigError::AllocationFailed { .. }
            ))
        ));
    });
    assert_eq!(polynomial.hasse_derivative(1).expect("restored"), expected);
}

/// The zero linearized polynomial enumerating the full field reports the
/// output-table reservation failure (no prior allocation on that path).
#[test]
fn linearized_zero_polynomial_reports_enumeration_failure() {
    poly_ring::linearized_roots(&Polynomial::<Gf8B>::zero(), gf8b::Elem::from_raw(0))
        .expect("warm-up");
    with_threshold(0, || {
        assert!(matches!(
            poly_ring::linearized_roots(&Polynomial::<Gf8B>::zero(), gf8b::Elem::from_raw(0)),
            Err(poly_ring::RootError::Polynomial(PolynomialError::Config(
                ConfigError::AllocationFailed { .. }
            )))
        ));
    });
    assert_eq!(
        poly_ring::linearized_roots(&Polynomial::<Gf8B>::zero(), gf8b::Elem::from_raw(0))
            .expect("restored")
            .len(),
        Gf8B::ORDER as usize
    );
}

/// A cold Chien scratch reserves its state and reports the failure; a warmed
/// one with a pre-reserved output scans the same polynomial under the same
/// closed gate, because the steady state asks for no memory.
#[test]
fn chien_state_reports_allocation_failure() {
    use poly_ring::ChienScratch;

    let polynomial = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).expect("quadratic");
    let mut cold = ChienScratch::<Gf8B>::new();
    let mut roots = Vec::with_capacity(2);
    let cold_result = with_threshold(0, || {
        poly_ring::chien_roots_into(&mut roots, &polynomial, &mut cold)
    });
    assert!(matches!(
        cold_result,
        Err(poly_ring::RootError::Polynomial(PolynomialError::Config(
            ConfigError::AllocationFailed { .. }
        )))
    ));

    let mut warm = ChienScratch::<Gf8B>::new();
    let mut roots = Vec::with_capacity(2);
    poly_ring::chien_roots_into(&mut roots, &polynomial, &mut warm).expect("warm-up");
    roots.clear();
    let warm_result = with_threshold(0, || {
        poly_ring::chien_roots_into(&mut roots, &polynomial, &mut warm)
    });
    assert!(!warm_result.expect("a warmed scan reserves nothing"));
    assert_eq!(roots.len(), 2);
}

/// Bivariate row-vector growth reports the row-table failure with the
/// requested count.
#[test]
fn bivariate_row_growth_reports_allocation_failure() {
    let mut polynomial = BivariatePolynomial::<Gf8B>::zero();
    let row = noise_poly::<Gf8B>(3, 0xA201);
    let probe = row.clone();
    let result = with_threshold(0, || polynomial.set_y_coefficient(4, probe).map(|_| ()));
    assert_eq!(
        result,
        Err(ConfigError::AllocationFailed {
            context: "bivariate Y coefficients",
            elements: 5,
            element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
        })
    );
    polynomial
        .set_y_coefficient(4, row.clone())
        .expect("restored");
    assert_eq!(polynomial.y_degree(), Some(4));
}

/// Bivariate products report the output-row reservation failure.
#[test]
fn bivariate_product_reports_allocation_failure() {
    let left = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        noise_poly::<Gf8B>(3, 0xA202),
        noise_poly::<Gf8B>(2, 0xA203),
    ]);
    let right =
        BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![noise_poly::<Gf8B>(2, 0xA204)]);
    let expected = left.multiply(&right).expect("unthrottled");
    let result = with_threshold(0, || left.multiply(&right).map(|_| ()));
    assert_eq!(
        result,
        Err(PolynomialError::Config(ConfigError::AllocationFailed {
            context: "bivariate product rows",
            elements: 2,
            element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
        }))
    );
    assert_eq!(left.multiply(&right).expect("restored"), expected);
}

/// Bivariate Hasse derivatives report the derivative-row failure.
#[test]
fn bivariate_derivative_reports_allocation_failure() {
    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        noise_poly::<Gf8B>(3, 0xA205),
        noise_poly::<Gf8B>(2, 0xA206),
    ]);
    polynomial.hasse_derivative(0, 0).expect("warm-up");
    with_threshold(0, || {
        assert!(matches!(
            polynomial.hasse_derivative(0, 0),
            Err(PolynomialError::Config(
                ConfigError::AllocationFailed { .. }
            ))
        ));
    });
    assert_eq!(
        polynomial
            .hasse_derivative(0, 0)
            .expect("restored")
            .y_coefficient_count(),
        2
    );
}

/// Bivariate linear row products report the row-table failure.
#[test]
fn bivariate_linear_product_reports_allocation_failure() {
    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        noise_poly::<Gf8B>(3, 0xA207),
        noise_poly::<Gf8B>(2, 0xA208),
    ]);
    polynomial.multiply_x_plus(b(1)).expect("warm-up");
    with_threshold(0, || {
        assert!(matches!(
            polynomial.multiply_x_plus(b(1)),
            Err(PolynomialError::Config(
                ConfigError::AllocationFailed { .. }
            ))
        ));
    });
    assert_eq!(
        polynomial
            .multiply_x_plus(b(1))
            .expect("restored")
            .y_coefficient_count(),
        2
    );
}

/// Bivariate `Y` substitutions report the substitution-row failure.
#[test]
fn bivariate_substitution_reports_allocation_failure() {
    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        noise_poly::<Gf8B>(3, 0xA209),
        noise_poly::<Gf8B>(2, 0xA20A),
    ]);
    let prefix = noise_poly::<Gf8B>(2, 0xA20B);
    polynomial.substitute_y_linear(b(1)).expect("warm-up");
    polynomial
        .substitute_y_affine_truncated(&prefix, 1, 8)
        .expect("warm-up");
    with_threshold(0, || {
        assert!(matches!(
            polynomial.substitute_y_linear(b(1)),
            Err(PolynomialError::Config(
                ConfigError::AllocationFailed { .. }
            ))
        ));
        assert!(matches!(
            polynomial.substitute_y_affine_truncated(&prefix, 1, 8),
            Err(PolynomialError::Config(
                ConfigError::AllocationFailed { .. }
            ))
        ));
    });
    assert_eq!(
        polynomial
            .substitute_y_linear(b(1))
            .expect("restored")
            .y_coefficient_count(),
        2
    );
}

/// Bivariate `X`-power quotients report the quotient-row failure.
#[test]
fn bivariate_quotient_reports_allocation_failure() {
    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(0), b(4)]).expect("row"),
        Polynomial::<Gf8B>::from_coefficients(&[b(0), b(0), b(6)]).expect("row"),
    ]);
    polynomial.divide_by_x_power(1).expect("warm-up");
    with_threshold(0, || {
        assert!(matches!(
            polynomial.divide_by_x_power(1),
            Err(PolynomialError::Config(
                ConfigError::AllocationFailed { .. }
            ))
        ));
    });
    assert_eq!(
        polynomial
            .divide_by_x_power(1)
            .expect("restored")
            .y_coefficient_count(),
        2
    );
}

/// Sparse bivariate-term enumeration reports the term-table failure.
#[test]
fn sparse_bivariate_terms_report_allocation_failure() {
    use poly_ring::SparsePolynomial;

    let dense = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        noise_poly::<Gf8B>(3, 0xA20C),
        noise_poly::<Gf8B>(2, 0xA20D),
    ]);
    let expected = SparsePolynomial::from_bivariate(&dense).expect("unthrottled");
    with_threshold(0, || {
        assert!(matches!(
            SparsePolynomial::from_bivariate(&dense),
            Err(ConfigError::AllocationFailed { .. })
        ));
    });
    assert_eq!(
        SparsePolynomial::from_bivariate(&dense).expect("restored"),
        expected
    );
}

/// Sparse-to-dense-bivariate materialization reports the row-table failure.
#[test]
fn sparse_to_dense_rows_report_allocation_failure() {
    use poly_ring::{MultiIndex, SparsePolynomial, Term};

    let sparse = SparsePolynomial::<Gf8B, 2>::from_terms(vec![
        Term {
            exponents: MultiIndex::new([2, 1]),
            coefficient: b(4),
        },
        Term {
            exponents: MultiIndex::new([0, 0]),
            coefficient: b(5),
        },
    ]);
    let expected = sparse.to_bivariate().expect("unthrottled");
    with_threshold(0, || {
        assert!(matches!(
            sparse.to_bivariate(),
            Err(PolynomialError::Config(
                ConfigError::AllocationFailed { .. }
            ))
        ));
    });
    assert_eq!(sparse.to_bivariate().expect("restored"), expected);
}

/// Hermite moduli-table reservation fails once the earlier tables fit:
/// with room for offsets and canonical points but not for two row
/// polynomials, construction names the moduli table.
#[test]
fn hermite_moduli_report_allocation_failure() {
    let points = [b(1), b(2)];
    let multiplicities = [2_usize, 1];
    let expected = HermitePlan::<Gf8B>::new(&points, &multiplicities).expect("unthrottled");
    // Offsets need 3*8 = 24 bytes, canonical points 2 bytes; the moduli
    // table needs 2 row polynomials. A 40-byte gate admits the first two.
    let result = with_threshold(40, || {
        HermitePlan::<Gf8B>::new(&points, &multiplicities).map(|_| ())
    });
    assert_eq!(
        result,
        Err(poly_ring::HermiteError::Config(
            ConfigError::AllocationFailed {
                context: "Hermite moduli",
                elements: 2,
                element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
            }
        ))
    );
    assert_eq!(
        HermitePlan::<Gf8B>::new(&points, &multiplicities)
            .expect("restored")
            .total_weight(),
        expected.total_weight()
    );
}

/// Hermite translation-table reservation fails once moduli fit: with room
/// for two row polynomials but not for two translation plans,
/// construction names the translations table.
#[test]
fn hermite_translations_report_allocation_failure() {
    let points = [b(1), b(2)];
    let multiplicities = [2_usize, 1];
    // Moduli pushes grow to four row polynomials; the translations table
    // needs two translation plans. A 100-byte gate admits moduli but not
    // translations.
    let result = with_threshold(100, || {
        HermitePlan::<Gf8B>::new(&points, &multiplicities).map(|_| ())
    });
    assert_eq!(
        result,
        Err(poly_ring::HermiteError::Config(
            ConfigError::AllocationFailed {
                context: "Hermite jet translations",
                elements: 2,
                element_size: core::mem::size_of::<poly_ring::JetPlan<Gf8B>>(),
            }
        ))
    );
    HermitePlan::<Gf8B>::new(&points, &multiplicities).expect("restored");
}

/// The fast batched substitution reports each staging-table failure in
/// order: powers, then pairs, then metadata — probed at gates admitting
/// every earlier table.
#[cfg(feature = "fft")]
#[test]
fn fast_substitution_staging_reports_allocation_failures() {
    use poly_ring::PolynomialProductScratch;

    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(1), b(2)]).expect("row"),
        Polynomial::<Gf8B>::from_coefficients(&[b(3)]).expect("row"),
        Polynomial::<Gf8B>::from_coefficients(&[b(4)]).expect("row"),
    ]);
    let prefix = Polynomial::<Gf8B>::from_coefficients(&[b(5), b(6)]).expect("prefix");
    for (gate, context, elements, element_size) in [
        (
            0_usize,
            "fast affine substitution powers",
            3_usize,
            24_usize,
        ),
        (
            100_usize,
            "fast affine substitution products",
            9_usize,
            16_usize,
        ),
        (
            200_usize,
            "fast affine substitution metadata",
            9_usize,
            24_usize,
        ),
    ] {
        let mut scratch = PolynomialProductScratch::<Gf8B>::new();
        let result = with_threshold(gate, || {
            polynomial
                .substitute_y_affine_truncated_fast(&prefix, 1, 8, &mut scratch)
                .map(|_| ())
        });
        assert_eq!(
            result,
            Err(poly_ring::ProductError::Config(
                ConfigError::AllocationFailed {
                    context,
                    elements,
                    element_size,
                }
            ))
        );
    }
    let mut scratch = PolynomialProductScratch::<Gf8B>::new();
    polynomial
        .substitute_y_affine_truncated_fast(&prefix, 1, 8, &mut scratch)
        .expect("restored");
}

/// Roth–Ruckenstein prefix reservation fails first under a closed gate,
/// naming the coefficient table and its length.
#[test]
fn lift_prefix_reports_allocation_failure() {
    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(3), b(5)]).expect("f"),
        Polynomial::<Gf8B>::one().expect("one"),
    ];
    let mut scratch = RothRuckensteinScratch::<Gf8B>::new();
    let mut output = Vec::new();
    let result = with_threshold(0, || {
        poly_ring::roth_ruckenstein_roots_into(
            &mut output,
            &rows,
            2,
            poly_ring::RothRuckensteinLimits::new(10_000, 64),
            &mut scratch,
        )
        .map(|_| ())
    });
    assert_eq!(
        result,
        Err(poly_ring::RootError::Polynomial(PolynomialError::Config(
            ConfigError::AllocationFailed {
                context: "Roth–Ruckenstein coefficient prefix",
                elements: 3,
                element_size: 1,
            }
        )))
    );
    poly_ring::roth_ruckenstein_roots_into(
        &mut output,
        &rows,
        2,
        poly_ring::RothRuckensteinLimits::new(10_000, 64),
        &mut scratch,
    )
    .expect("restored");
    assert_eq!(output.len(), 1);
}

/// Roth–Ruckenstein pooled-row reservation fails once the prefix fits: a
/// 64-byte gate admits three prefix elements but not two pooled rows.
#[test]
fn lift_pooled_rows_report_allocation_failure() {
    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(3), b(5)]).expect("f"),
        Polynomial::<Gf8B>::one().expect("one"),
    ];
    let mut scratch = RothRuckensteinScratch::<Gf8B>::new();
    let mut output = Vec::new();
    let result = with_threshold(64, || {
        poly_ring::roth_ruckenstein_roots_into(
            &mut output,
            &rows,
            2,
            poly_ring::RothRuckensteinLimits::new(10_000, 64),
            &mut scratch,
        )
        .map(|_| ())
    });
    assert_eq!(
        result,
        Err(poly_ring::RootError::Polynomial(PolynomialError::Config(
            ConfigError::AllocationFailed {
                context: "lifting rows",
                elements: 2,
                element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
            }
        )))
    );
    poly_ring::roth_ruckenstein_roots_into(
        &mut output,
        &rows,
        2,
        poly_ring::RothRuckensteinLimits::new(10_000, 64),
        &mut scratch,
    )
    .expect("restored");
    assert_eq!(output.len(), 1);
}

/// Roth–Ruckenstein frame-stack reservation fails once pooled rows fit: a
/// 100-byte gate admits them but not three frames.
#[test]
fn lift_frame_stack_reports_allocation_failure() {
    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(3), b(5)]).expect("f"),
        Polynomial::<Gf8B>::one().expect("one"),
    ];
    let mut scratch = RothRuckensteinScratch::<Gf8B>::new();
    let mut output = Vec::new();
    let result = with_threshold(100, || {
        poly_ring::roth_ruckenstein_roots_into(
            &mut output,
            &rows,
            2,
            poly_ring::RothRuckensteinLimits::new(10_000, 64),
            &mut scratch,
        )
        .map(|_| ())
    });
    assert!(matches!(
        result,
        Err(poly_ring::RootError::Polynomial(PolynomialError::Config(
            ConfigError::AllocationFailed { .. }
        )))
    ));
    poly_ring::roth_ruckenstein_roots_into(
        &mut output,
        &rows,
        2,
        poly_ring::RothRuckensteinLimits::new(10_000, 64),
        &mut scratch,
    )
    .expect("restored");
    assert_eq!(output.len(), 1);
}

/// Alekhnovich row-quotient staging fails first under a closed gate,
/// naming the quotient table and its row count.
#[cfg(feature = "fft")]
#[test]
fn alekhnovich_quotient_reports_allocation_failure() {
    use poly_ring::{AlekhnovichLimits, AlekhnovichScratch, alekhnovich_roots};

    let rows = vec![
        Polynomial::<Gf8B>::from_coefficients(&[b(3), b(5)]).expect("f"),
        Polynomial::<Gf8B>::one().expect("one"),
    ];
    let limits = AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 24, 1 << 28, 256)
        .with_roth_ruckenstein_crossover(0);
    let mut scratch = AlekhnovichScratch::<Gf8B>::new();
    // Warm up outside the gate first: backend selection and lazy tables
    // must not initialize while allocations are refused.
    assert_eq!(
        alekhnovich_roots(&rows, 4, limits, &mut scratch)
            .expect("warm-up")
            .len(),
        1
    );
    let result = with_threshold(0, || {
        alekhnovich_roots(&rows, 4, limits, &mut scratch).map(|_| ())
    });
    assert_eq!(
        result,
        Err(poly_ring::RootError::Polynomial(PolynomialError::Config(
            ConfigError::AllocationFailed {
                context: "bivariate X-power quotient",
                elements: 2,
                element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
            }
        )))
    );
    assert_eq!(
        alekhnovich_roots(&rows, 4, limits, &mut scratch)
            .expect("restored")
            .len(),
        1
    );
}

/// Equal-degree factor-stack reservation fails once supporting arithmetic
/// fits: a 48-byte gate admits the powering buffers but not the factor
/// stack entry.
#[test]
fn equal_degree_factor_stack_reports_allocation_failure() {
    use poly_ring::{BinaryRootScratch, binary_field_roots_into};

    let polynomial = noise_poly::<Gf16>(7, 0xD404);
    let expected_roots = {
        let mut scratch = BinaryRootScratch::<Gf16>::new();
        let mut roots = Vec::new();
        poly_ring::binary_field_roots_into(&mut roots, &polynomial, &mut scratch)
            .expect("unthrottled");
        roots
    };
    let mut scratch = BinaryRootScratch::<Gf16>::new();
    let mut roots = Vec::new();
    let result = with_threshold(48, || {
        binary_field_roots_into(&mut roots, &polynomial, &mut scratch).map(|_| ())
    });
    assert!(matches!(
        result,
        Err(poly_ring::RootError::Polynomial(PolynomialError::Config(
            ConfigError::AllocationFailed { .. }
        )))
    ));
    let mut scratch = BinaryRootScratch::<Gf16>::new();
    let mut roots = Vec::new();
    poly_ring::binary_field_roots_into(&mut roots, &polynomial, &mut scratch).expect("restored");
    assert_eq!(roots, expected_roots);
}

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

#[test]
fn affine_output_reservation_failure_preserves_the_input() {
    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        Polynomial::constant(b(3)).unwrap(),
        Polynomial::one().unwrap(),
    ]);
    let prefix = Polynomial::constant(b(5)).unwrap();
    let expected = BivariatePolynomial::from_y_coefficients(vec![
        Polynomial::constant(b(6)).unwrap(),
        Polynomial::from_coefficients(&[b(0), b(1)]).unwrap(),
    ]);
    let result = reject_matching(2 * core::mem::size_of::<Polynomial<Gf8B>>(), 1, || {
        polynomial.substitute_y_affine_truncated(&prefix, 1, 3)
    });
    assert!(matches!(
        result,
        Err(PolynomialError::Config(
            ConfigError::AllocationFailed { .. }
        ))
    ));
    assert_eq!(
        polynomial
            .substitute_y_affine_truncated(&prefix, 1, 3)
            .unwrap(),
        expected
    );
}

#[cfg(feature = "fft")]
#[test]
fn fast_affine_output_reservation_failure_is_recoverable() {
    let polynomial = BivariatePolynomial::<Gf8B>::from_y_coefficients(vec![
        Polynomial::constant(b(3)).unwrap(),
        Polynomial::one().unwrap(),
    ]);
    let prefix = Polynomial::constant(b(5)).unwrap();
    let expected = polynomial
        .substitute_y_affine_truncated(&prefix, 1, 3)
        .unwrap();
    let mut scratch = poly_ring::PolynomialProductScratch::new();
    let result = reject_matching(2 * core::mem::size_of::<Polynomial<Gf8B>>(), 1, || {
        polynomial.substitute_y_affine_truncated_fast(&prefix, 1, 3, &mut scratch)
    });
    assert!(matches!(
        result,
        Err(poly_ring::ProductError::Config(
            ConfigError::AllocationFailed { .. }
        ))
    ));
    assert_eq!(
        polynomial
            .substitute_y_affine_truncated_fast(&prefix, 1, 3, &mut scratch)
            .unwrap(),
        expected
    );
}

#[cfg(feature = "fft")]
#[test]
fn transform_reservations_recover_without_changing_values() {
    use butterfly_fft::{basis::conversion_scratch_elements, transform::TransformPlan};
    use poly_ring::{
        TransformScratch, evaluate_coset_into, evaluate_subspace_into, interpolate_subspace_into,
    };

    let plan = TransformPlan::<Gf16>::new(16).unwrap();
    let polynomial = noise_poly::<Gf16>(7, 0xA301);
    let expected: Vec<_> = (0..16)
        .map(|i| polynomial.evaluate(plan.point_element(i)))
        .collect();
    for evaluate in [evaluate_subspace_into::<Gf16>, evaluate_coset_into::<Gf16>] {
        for failed_size in [32, conversion_scratch_elements(16) * Gf16::BYTES] {
            let mut scratch = TransformScratch::new();
            let mut values = Vec::with_capacity(16);
            let result = reject_matching(failed_size, 0, || {
                evaluate(&mut values, &polynomial, &plan, &mut scratch)
            });
            assert!(matches!(
                result,
                Err(PolynomialError::Config(
                    ConfigError::AllocationFailed { .. }
                ))
            ));
            evaluate(&mut values, &polynomial, &plan, &mut scratch).unwrap();
            assert_eq!(values, expected);
        }
        let mut scratch = TransformScratch::new();
        let mut values = Vec::new();
        let result = with_threshold(0, || {
            evaluate(&mut values, &polynomial, &plan, &mut scratch)
        });
        assert!(matches!(
            result,
            Err(PolynomialError::Config(
                ConfigError::AllocationFailed { .. }
            ))
        ));
        evaluate(&mut values, &polynomial, &plan, &mut scratch).unwrap();
        assert_eq!(values, expected);
    }
    for failed_size in [32, conversion_scratch_elements(16) * Gf16::BYTES] {
        let mut scratch = TransformScratch::new();
        let mut output = Polynomial::zero();
        let result = reject_matching(failed_size, 0, || {
            interpolate_subspace_into(&mut output, &plan, &expected, &mut scratch)
        });
        assert!(matches!(
            result,
            Err(PolynomialError::Config(
                ConfigError::AllocationFailed { .. }
            ))
        ));
        interpolate_subspace_into(&mut output, &plan, &expected, &mut scratch).unwrap();
        assert_eq!(output, polynomial);
    }
}

#[test]
fn hermite_retained_weights_and_crt_reservations_are_fallible() {
    for (points, weights, size) in [
        (vec![b(1), b(2)], vec![0, 0], 2 * size_of::<usize>()),
        (
            vec![b(1), b(2)],
            vec![1, 1],
            2 * size_of::<Polynomial<Gf8B>>(),
        ),
    ] {
        let result = reject_matching(size, 0, || HermitePlan::<Gf8B>::new(&points, &weights));
        assert!(matches!(
            result,
            Err(poly_ring::HermiteError::Config(
                ConfigError::AllocationFailed { .. }
            ))
        ));
        let plan = HermitePlan::<Gf8B>::new(&points, &weights).unwrap();
        let values = vec![b(7); weights.iter().sum()];
        let recovered = plan.interpolate(&values).unwrap();
        if values.is_empty() {
            assert_eq!(recovered, Polynomial::zero());
        } else {
            assert_eq!(recovered, Polynomial::constant(b(7)).unwrap());
        }
    }
}

#[test]
fn linearized_root_output_reservation_is_fallible() {
    let polynomial = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1), b(1)]).unwrap();
    let result = reject_matching(2 * size_of::<gf8b::Elem>(), 0, || {
        poly_ring::linearized_roots(&polynomial, b(0))
    });
    assert!(matches!(
        result,
        Err(poly_ring::RootError::Polynomial(PolynomialError::Config(
            ConfigError::AllocationFailed { .. }
        )))
    ));
    assert_eq!(
        poly_ring::linearized_roots(&polynomial, b(0)).unwrap(),
        vec![b(0), b(1)]
    );
}

#[cfg(not(feature = "fft"))]
#[test]
fn portable_subspace_point_reservation_is_fallible() {
    let basis = [b(1), b(2), b(4), b(8)];
    let result = reject_matching(16, 0, || {
        poly_ring::EvaluationDomain::<Gf8B>::additive_subspace_with_basis(16, &basis)
    });
    assert!(matches!(
        result,
        Err(poly_ring::DomainError::Config(
            ConfigError::AllocationFailed { .. }
        ))
    ));
    let domain =
        poly_ring::EvaluationDomain::<Gf8B>::additive_subspace_with_basis(16, &basis).unwrap();
    assert_eq!(domain.points(), &(0..16).map(b).collect::<Vec<_>>());
}

#[test]
fn jet_scratch_late_reservations_leave_plan_usable() {
    let plan = poly_ring::JetPlan::<Gf8B>::new(b(3), 1, 17).unwrap();
    for (size, skip) in [(20 * size_of::<Vec<u8>>(), 0), (17, 0), (17, 1)] {
        let result = reject_matching(size, skip, || plan.scratch(1));
        assert!(matches!(
            result,
            Err(poly_ring::HasseError::AllocationFailed { .. })
        ));
        let mut scratch = plan.scratch(1).unwrap();
        let input = [b(5), b(7)];
        let mut output = [b(0)];
        plan.evaluate_into(&input, &mut scratch, &mut output)
            .unwrap();
        assert_eq!(output, [b(5).add(b(7).mul(b(3)))]);
    }
}

#[test]
fn jet_power_reservation_failure_is_recoverable() {
    let result = reject_matching(5 * size_of::<Vec<u8>>(), 0, || {
        poly_ring::JetPlan::<Gf8B>::new(b(3), 1, 17)
    });
    assert!(matches!(
        result,
        Err(poly_ring::HasseError::AllocationFailed { .. })
    ));
    let plan = poly_ring::JetPlan::<Gf8B>::new(b(3), 1, 17).unwrap();
    let mut scratch = plan.scratch(1).unwrap();
    let mut output = [b(0)];
    plan.evaluate_into(&[b(5), b(7)], &mut scratch, &mut output)
        .unwrap();
    assert_eq!(output, [b(5).add(b(7).mul(b(3)))]);
}

#[test]
fn multipoint_output_reservation_failure_is_recoverable() {
    let polynomial = noise_poly::<Gf8B>(7, 0xA302);
    let points: Vec<_> = (0..32).map(b).collect();
    let expected: Vec<_> = points
        .iter()
        .map(|&point| polynomial.evaluate(point))
        .collect();
    let mut scratch = poly_ring::MultipointScratch::new();
    let mut output = Vec::new();
    let result = with_threshold(0, || {
        poly_ring::evaluate_multipoint_into(&mut output, &polynomial, &points, &mut scratch)
    });
    assert!(matches!(
        result,
        Err(PolynomialError::Config(
            ConfigError::AllocationFailed { .. }
        ))
    ));
    poly_ring::evaluate_multipoint_into(&mut output, &polynomial, &points, &mut scratch).unwrap();
    assert_eq!(output, expected);
}

#[cfg(feature = "fft")]
#[test]
fn affine_row_workspace_reservations_recover() {
    let rows = [
        Polynomial::<Gf8B>::constant(b(3)).unwrap(),
        Polynomial::one().unwrap(),
    ];
    let prefix = Polynomial::constant(b(5)).unwrap();
    let expected = vec![
        Polynomial::constant(b(6)).unwrap(),
        Polynomial::from_coefficients(&[b(0), b(1)]).unwrap(),
    ];
    for (size, skip) in [
        (2 * size_of::<Polynomial<Gf8B>>(), 0),
        (2 * size_of::<(usize, usize)>(), 0),
        (4 * size_of::<Polynomial<Gf8B>>(), 0),
    ] {
        let mut scratch = poly_ring::PolynomialProductScratch::new();
        let mut output = Vec::new();
        let mut pool = Vec::new();
        let result = reject_matching(size, skip, || {
            poly_ring::internals::substitute_y_affine_rows_truncated_into(
                &rows,
                &prefix,
                1,
                3,
                &mut scratch,
                &mut output,
                &mut pool,
            )
        });
        assert!(
            matches!(
                result,
                Err(poly_ring::ProductError::Config(
                    ConfigError::AllocationFailed { .. }
                ))
            ),
            "size={size}, skip={skip}: {result:?}"
        );
        poly_ring::internals::substitute_y_affine_rows_truncated_into(
            &rows,
            &prefix,
            1,
            3,
            &mut scratch,
            &mut output,
            &mut pool,
        )
        .unwrap();
        assert_eq!(output, expected);
    }
}

#[test]
fn hermite_crt_allocation_failure_recovers_interpolation() {
    let points = [b(1), b(2)];
    let result = reject_matching(2 * size_of::<Polynomial<Gf8B>>(), 1, || {
        HermitePlan::<Gf8B>::new(&points, &[1, 1])
    });
    assert!(matches!(
        result,
        Err(poly_ring::HermiteError::Config(
            ConfigError::AllocationFailed { .. }
        ))
    ));
    let plan = HermitePlan::<Gf8B>::new(&points, &[1, 1]).unwrap();
    let polynomial = plan.interpolate(&[b(7), b(9)]).unwrap();
    assert_eq!(polynomial.evaluate(b(1)), b(7));
    assert_eq!(polynomial.evaluate(b(2)), b(9));
}

#[test]
fn wide_multipoint_output_failure_recovers_values() {
    let polynomial = noise_poly::<Gf16>(7, 0xA303);
    let points: Vec<_> = (0..32).map(fgf::gf16::Elem::from_raw).collect();
    let expected: Vec<_> = points
        .iter()
        .map(|&point| polynomial.evaluate(point))
        .collect();
    let mut scratch = poly_ring::MultipointScratch::new();
    let mut output = Vec::new();
    let result = with_threshold(0, || {
        poly_ring::evaluate_multipoint_into(&mut output, &polynomial, &points, &mut scratch)
    });
    assert!(matches!(
        result,
        Err(PolynomialError::Config(
            ConfigError::AllocationFailed { .. }
        ))
    ));
    poly_ring::evaluate_multipoint_into(&mut output, &polynomial, &points, &mut scratch).unwrap();
    assert_eq!(output, expected);
}

#[test]
fn derivative_factor_reservation_failure_is_recoverable() {
    let result = with_threshold(0, || poly_ring::DerivativePlan::<Mersenne31>::new(1, 3));
    assert!(matches!(
        result,
        Err(poly_ring::HasseError::AllocationFailed { .. })
    ));
    let plan = poly_ring::DerivativePlan::<Mersenne31>::new(1, 3).unwrap();
    let m = fgf::mersenne31::Elem::from_raw;
    let mut output = [m(0); 2];
    plan.apply_into(&[m(2), m(3), m(4)], &mut output).unwrap();
    assert_eq!(output, [m(3), m(8)]);
}

#[test]
fn remainder_descent_reservation_failure_recovers_reduction() {
    let modulus = Polynomial::<Gf8B>::from_coefficients(&[b(1), b(1)]).unwrap();
    let tree = poly_ring::RemainderTree::new(&[modulus], 4).unwrap();
    let result = reject_matching(2 * size_of::<Vec<u8>>(), 0, || tree.scratch(1));
    assert!(matches!(
        result,
        Err(poly_ring::ProductError::Config(
            ConfigError::AllocationFailed { .. }
        ))
    ));
    let mut scratch = tree.scratch(1).unwrap();
    let mut output = [0];
    tree.remainders_into(&[2, 3, 4, 5], 4, 1, &mut scratch, &mut output)
        .unwrap();
    assert_eq!(output, [2 ^ 3 ^ 4 ^ 5]);
}
