//! Factorization edge inputs, prepared quotient-ring in-place routes, and
//! the reservation failures the factorization stages report.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use fgf::field::Elem;
use fgf::{Gf8B, Gf16, Mersenne31, gf8b, mersenne31};
use poly_ring::{
    ConfigError, DistinctDegreeFactor, FactorizationError, IrreducibleFactor, ModulusPlan,
    ModulusScratch, Polynomial, PolynomialError, SquareFreeFactor,
};

// A test-only failing allocator refuses chosen allocations, so the public
// factorization paths exercise their `try_reserve` failure branches
// deterministically. The gate is thread-local: parallel tests on other
// threads are never charged to it, and a panicking thread always gets its
// memory so a failure cannot hide behind an allocation abort.
struct FailingAllocator;

thread_local! {
    static MATCHING_ALLOCATION: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let refused = !std::thread::panicking()
            && MATCHING_ALLOCATION
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

/// Arm the gate to refuse the allocation after `skip` admitted allocations
/// of exactly `size` bytes, and disarm it however the operation leaves.
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

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn gf8_poly(coefficients: &[u8]) -> Polynomial<Gf8B> {
    Polynomial::from_coefficients(&coefficients.iter().copied().map(b).collect::<Vec<_>>())
        .expect("binary polynomial")
}

fn m31_poly(coefficients: &[u32]) -> Polynomial<Mersenne31> {
    Polynomial::from_coefficients(&coefficients.iter().copied().map(m31).collect::<Vec<_>>())
        .expect("prime-field polynomial")
}

fn power<F: fgf::kernel::FieldKernels>(factor: &Polynomial<F>, exponent: usize) -> Polynomial<F> {
    let mut result = Polynomial::one().expect("one");
    for _ in 0..exponent {
        result = result.multiply(factor).expect("power");
    }
    result
}

#[test]
fn distinct_degree_stage_rejects_zero_and_maps_units_to_no_groups() {
    assert_eq!(
        Polynomial::<Gf8B>::zero().distinct_degree_factorization(),
        Err(FactorizationError::ZeroPolynomial)
    );
    let groups = gf8_poly(&[7])
        .distinct_degree_factorization()
        .expect("a unit has no irreducible factors");
    assert!(groups.is_empty());
}

#[test]
fn equal_degree_stage_rejects_zero_and_maps_units_to_no_factors() {
    assert_eq!(
        Polynomial::<Gf8B>::zero().equal_degree_factorization(1),
        Err(FactorizationError::ZeroPolynomial)
    );
    for unit in [gf8_poly(&[5]), Polynomial::<Gf8B>::one().expect("one")] {
        let split = unit
            .equal_degree_factorization(2)
            .expect("a unit has no equal-degree factors");
        assert!(split.is_empty());
    }
}

#[test]
fn equal_degree_stage_rejects_mixed_and_mismatched_degrees() {
    // M31 is congruent to seven modulo eight, so -1 and -2 are quadratic
    // nonresidues and both X² + 1 and X² + 2 are irreducible.
    let linear = m31_poly(&[3, 1]);
    let quadratics = m31_poly(&[1, 0, 1])
        .multiply(&m31_poly(&[2, 0, 1]))
        .expect("square-free product of quadratics");
    let mixed = linear.multiply(&quadratics).expect("mixed degrees");

    // A product of degree-one and degree-two factors has no single degree.
    assert_eq!(
        mixed.equal_degree_factorization(1),
        Err(FactorizationError::NotEqualDegree { factor_degree: 1 })
    );
    // A single degree-two group split at the wrong requested degree.
    assert_eq!(
        quadratics.equal_degree_factorization(3),
        Err(FactorizationError::NotEqualDegree { factor_degree: 3 })
    );
}

#[test]
fn irreducibility_rejects_zero_and_repeated_inputs() {
    assert!(
        !Polynomial::<Gf8B>::zero()
            .is_irreducible()
            .expect("zero is not irreducible")
    );
    let linear = gf8_poly(&[1, 1]);
    let repeated = power(&linear, 2);
    assert!(
        !repeated
            .is_irreducible()
            .expect("a repeated factor is not irreducible")
    );
    assert!(
        linear
            .is_irreducible()
            .expect("a linear polynomial is irreducible")
    );
}

#[test]
fn square_free_stage_maps_constants_to_no_factors() {
    let factors = gf8_poly(&[7])
        .square_free_factorization()
        .expect("a unit has no square-free factors");
    assert!(factors.is_empty());
}

fn undefined_stage_inputs_reject_for<F: fgf::kernel::FieldKernels + PartialEq>() {
    assert_eq!(
        Polynomial::<F>::zero().distinct_degree_factorization(),
        Err(FactorizationError::ZeroPolynomial)
    );
    assert!(
        Polynomial::<F>::from_coefficients(&[F::Elem::ONE])
            .expect("unit")
            .distinct_degree_factorization()
            .expect("a unit has no irreducible factors")
            .is_empty()
    );
    assert_eq!(
        Polynomial::<F>::zero().equal_degree_factorization(1),
        Err(FactorizationError::ZeroPolynomial)
    );
    assert!(
        Polynomial::<F>::one()
            .expect("one")
            .equal_degree_factorization(2)
            .expect("a unit has no equal-degree factors")
            .is_empty()
    );
    assert!(
        Polynomial::<F>::from_coefficients(&[F::Elem::ONE])
            .expect("unit")
            .square_free_factorization()
            .expect("a unit has no square-free factors")
            .is_empty()
    );
}

#[test]
fn undefined_stage_inputs_reject_across_binary_and_prime_fields() {
    undefined_stage_inputs_reject_for::<Gf8B>();
    undefined_stage_inputs_reject_for::<Gf16>();
    undefined_stage_inputs_reject_for::<Mersenne31>();
}

#[test]
fn binary_factorization_reconstructs_inputs_with_repeated_factors() {
    let first = gf8_poly(&[1, 1]);
    let second = gf8_poly(&[2, 1]);
    let input = power(&first, 2)
        .multiply(&power(&second, 3))
        .expect("repeated input");

    let factors = input.factor().expect("complete factorization");
    assert_eq!(factors.len(), 2);
    let mut reconstructed = Polynomial::<Gf8B>::one().expect("one");
    for entry in &factors {
        assert!(entry.factor.is_irreducible().expect("irreducible"));
        reconstructed = reconstructed
            .multiply(&power(&entry.factor, entry.multiplicity))
            .expect("reconstruct");
    }
    assert_eq!(reconstructed, input.monic());
    assert!(factors.contains(&IrreducibleFactor {
        factor: first,
        multiplicity: 2,
    }));
    assert!(factors.contains(&IrreducibleFactor {
        factor: second,
        multiplicity: 3,
    }));
}

#[test]
fn in_place_quotient_routes_match_allocating_counterparts() {
    // M31 is congruent to three modulo four, so X² + 1 is irreducible.
    let modulus = m31_poly(&[1, 0, 1]);
    let plan = ModulusPlan::new(&modulus).expect("plan");
    let value = m31_poly(&[2, 3]);
    let other = m31_poly(&[5, 7]);

    let mut scratch = plan.scratch();
    // Destinations are write-only, so a dirty output must not survive.
    let mut output = m31_poly(&[9, 9, 9, 9]);

    plan.reduce_into(&value, &mut scratch, &mut output)
        .expect("reduce into");
    assert_eq!(output, value.remainder(&modulus).expect("direct remainder"));

    plan.multiply_into(&value, &other, &mut scratch, &mut output)
        .expect("multiply into");
    assert_eq!(
        output,
        plan.multiply(&value, &other).expect("allocating product")
    );

    plan.square_into(&value, &mut scratch, &mut output)
        .expect("square into");
    let expected_square = value
        .square()
        .expect("direct square")
        .remainder(&modulus)
        .expect("reduced square");
    assert_eq!(output, expected_square);
    assert_eq!(output, plan.square(&value).expect("allocating square"));
    plan.square_into(&value, &mut scratch, &mut output)
        .expect("repeat square into warmed scratch");
    assert_eq!(output, expected_square);

    plan.pow_into(&value, 0, &mut scratch, &mut output)
        .expect("zero power into");
    assert!(output.is_one());

    let mut expected_power = Polynomial::<Mersenne31>::one().expect("one");
    for _ in 0..13 {
        expected_power = expected_power
            .multiply(&value)
            .expect("power product")
            .remainder(&modulus)
            .expect("reduced power product");
    }
    plan.pow_into(&value, 13, &mut scratch, &mut output)
        .expect("power into");
    assert_eq!(output, expected_power);
}

#[test]
fn quotient_inverse_routes_reuse_scratch_and_report_non_coprime_values() {
    let modulus = m31_poly(&[1, 0, 1]);
    let plan = ModulusPlan::new(&modulus).expect("plan");
    let value = m31_poly(&[0, 1]);

    let mut scratch = ModulusScratch::default();
    let mut output = m31_poly(&[4, 4, 4]);
    plan.inverse_into(&value, &mut scratch, &mut output)
        .expect("inverse into");
    let expected = plan.inverse(&value).expect("allocating inverse");
    assert_eq!(output, expected);
    assert!(
        plan.multiply(&value, &output)
            .expect("multiplicative identity")
            .is_one()
    );
    plan.inverse_into(&value, &mut scratch, &mut output)
        .expect("repeat inverse into warmed scratch");
    assert_eq!(output, expected);

    // A non-invertible residue: X + 1 shares a factor with (X + 1)(X + 2).
    let reducible = m31_poly(&[1, 1])
        .multiply(&m31_poly(&[2, 1]))
        .expect("reducible modulus");
    let reducible_plan = ModulusPlan::new(&reducible).expect("plan");
    let mut reducible_scratch = reducible_plan.scratch();
    let mut reducible_output = Polynomial::zero();
    assert_eq!(
        reducible_plan.inverse_into(
            &m31_poly(&[1, 1]),
            &mut reducible_scratch,
            &mut reducible_output
        ),
        Err(PolynomialError::NotInvertibleModulo)
    );
}

#[test]
fn quotient_composition_handles_constant_and_sparse_outer_polynomials() {
    // M31 is congruent to seven modulo eight, so X² + 2 is irreducible.
    let modulus = m31_poly(&[2, 0, 1]);
    let plan = ModulusPlan::new(&modulus).expect("plan");
    let inner = m31_poly(&[5, 1, 1]);
    let mut scratch = plan.scratch();
    let mut output = m31_poly(&[7, 7, 7]);

    // A constant outer polynomial composes to itself.
    plan.compose_into(&m31_poly(&[6]), &inner, &mut scratch, &mut output)
        .expect("constant composition");
    assert_eq!(output, m31_poly(&[6]));

    // A sparse outer polynomial skips its zero coefficients.
    let sparse = m31_poly(&[5, 0, 3]);
    plan.compose_into(&sparse, &inner, &mut scratch, &mut output)
        .expect("sparse composition");
    let expected = sparse
        .compose(&inner)
        .expect("direct composition")
        .remainder(&modulus)
        .expect("reduced composition");
    assert_eq!(output, expected);
    assert_eq!(plan.compose(&sparse, &inner).expect("allocating"), expected);
}

/// A pair of distinct irreducible quadratics over GF(8), found by excluding
/// every polynomial with a root in the field.
fn gf8_irreducible_quadratics() -> Vec<Polynomial<Gf8B>> {
    let mut found = Vec::new();
    'search: for linear in 0_u16..=u8::MAX.into() {
        for constant in 1_u16..=u8::MAX.into() {
            let linear = b(linear as u8);
            let constant = b(constant as u8);
            let has_root = (0_u16..=u8::MAX.into()).any(|raw| {
                let root = b(raw as u8);
                root.mul(root).add(linear.mul(root)).add(constant).is_zero()
            });
            if !has_root {
                found.push(gf8_poly(&[constant.to_raw(), linear.to_raw(), 1]));
                if found.len() == 2 {
                    break 'search;
                }
            }
        }
    }
    assert_eq!(found.len(), 2, "GF(8) has irreducible quadratics");
    found
}

#[test]
fn square_free_and_distinct_degree_stages_report_reservation_failures() {
    let first = gf8_poly(&[1, 1]);
    let second = gf8_poly(&[2, 1]);
    let repeated = power(&first, 2)
        .multiply(&power(&second, 3))
        .expect("repeated input");
    assert_eq!(
        repeated
            .square_free_factorization()
            .expect("unthrottled square-free")
            .len(),
        2
    );
    let square_free = first.multiply(&second).expect("square-free product");
    assert_eq!(
        square_free
            .distinct_degree_factorization()
            .expect("unthrottled distinct-degree")
            .len(),
        1
    );

    // A first push into an empty factor vector reserves four amortized
    // entries of the factor type.
    let entry = 4 * core::mem::size_of::<SquareFreeFactor<Gf8B>>();
    reject_matching(entry, 0, || {
        assert_eq!(
            repeated.square_free_factorization(),
            Err(FactorizationError::Polynomial(PolynomialError::Config(
                ConfigError::AllocationFailed {
                    context: "square-free factors",
                    elements: 1,
                    element_size: core::mem::size_of::<SquareFreeFactor<Gf8B>>(),
                }
            )))
        );
    });
    let entry = 4 * core::mem::size_of::<DistinctDegreeFactor<Gf8B>>();
    reject_matching(entry, 0, || {
        assert_eq!(
            square_free.distinct_degree_factorization(),
            Err(FactorizationError::Polynomial(PolynomialError::Config(
                ConfigError::AllocationFailed {
                    context: "distinct-degree factors",
                    elements: 1,
                    element_size: core::mem::size_of::<DistinctDegreeFactor<Gf8B>>(),
                }
            )))
        );
    });
    assert_eq!(
        repeated
            .square_free_factorization()
            .expect("restored square-free")
            .len(),
        2
    );
}

#[test]
fn equal_degree_stage_reports_both_reservation_failures() {
    let quadratics = gf8_irreducible_quadratics();
    let input = quadratics[0]
        .multiply(&quadratics[1])
        .expect("equal-degree product");
    assert_eq!(
        input
            .equal_degree_factorization(2)
            .expect("unthrottled split")
            .len(),
        2
    );

    // Two factors of degree two reserve two polynomial entries exactly: the
    // pending stack first, then the output vector.
    let stack_entry = 2 * core::mem::size_of::<Polynomial<Gf8B>>();
    reject_matching(stack_entry, 0, || {
        assert_eq!(
            input.equal_degree_factorization(2),
            Err(FactorizationError::Polynomial(PolynomialError::Config(
                ConfigError::AllocationFailed {
                    context: "equal-degree factor stack",
                    elements: 2,
                    element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
                }
            )))
        );
    });
    reject_matching(stack_entry, 1, || {
        assert_eq!(
            input.equal_degree_factorization(2),
            Err(FactorizationError::Polynomial(PolynomialError::Config(
                ConfigError::AllocationFailed {
                    context: "equal-degree factors",
                    elements: 2,
                    element_size: core::mem::size_of::<Polynomial<Gf8B>>(),
                }
            )))
        );
    });
    assert_eq!(
        input
            .equal_degree_factorization(2)
            .expect("restored split")
            .len(),
        2
    );
}

#[test]
fn irreducible_output_reservation_reports_failure() {
    let input = gf8_poly(&[1, 1])
        .multiply(&gf8_poly(&[2, 1]))
        .expect("linear product");
    assert_eq!(input.factor().expect("unthrottled factors").len(), 2);

    // The square-free push, the distinct-degree push, and the validation
    // distinct-degree pass inside the equal-degree stage come first; the
    // fourth amortized entry reservation belongs to the irreducible output.
    let entry = 4 * core::mem::size_of::<IrreducibleFactor<Gf8B>>();
    reject_matching(entry, 3, || {
        assert_eq!(
            input.factor(),
            Err(FactorizationError::Polynomial(PolynomialError::Config(
                ConfigError::AllocationFailed {
                    context: "irreducible factors",
                    elements: 1,
                    element_size: core::mem::size_of::<IrreducibleFactor<Gf8B>>(),
                }
            )))
        );
    });
    assert_eq!(input.factor().expect("restored factors").len(), 2);
}
