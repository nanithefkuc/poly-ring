//! The factorization, quotient-ring, Euclidean, and truncated-series
//! pipelines instantiated over the fields the behavior suites exercise only
//! shallowly or not at all. Binary extensions run the characteristic-two
//! branches (Frobenius square roots, absolute traces) and the odd-order
//! fields the quadratic-character branches, each against independently
//! derived irreducible inputs: `X^2 + X + c` with the absolute trace of `c`
//! equal to one, `X^2 + d` with `d` an odd power of the generator, and
//! `X^3 + c` with `-c` outside the cube subgroup.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{FanPaar8, FanPaar16, FanPaar32, FanPaar64, Gf8D, Gf16, Goldilocks, QuadMersenne31};
use poly_ring::{
    ConfigError, DistinctDegreeFactor, FactorizationError, ModulusPlan, ModulusScratch, Polynomial,
    PolynomialError, SquareFreeFactor, series_divide, truncated_eea,
};

// A test-only allocator both records and refuses chosen allocations, so the
// prepared and Euclidean routes exercise their `try_reserve` failure
// branches deterministically. The gates are thread-local: parallel tests on
// other threads are never charged, and a panicking thread always gets its
// memory so a failure cannot hide behind an allocation abort.
struct GatedAllocator;
thread_local! {
    static REFUSAL: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

unsafe impl GlobalAlloc for GatedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let refused = !std::thread::panicking()
            && REFUSAL
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
static GLOBAL: GatedAllocator = GatedAllocator;

/// Arm the gate to refuse the allocation after `skip` admitted allocations
/// of exactly `size` bytes, and disarm it however the operation leaves.
fn reject_matching<R>(size: usize, skip: usize, operation: impl FnOnce() -> R) -> R {
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            REFUSAL.with(|gate| gate.set(None));
        }
    }
    REFUSAL.with(|gate| gate.set(Some((size, skip))));
    let _disarm = Disarm;
    operation()
}

/// The generator raised to `exponent`, the only element constructor every
/// field here shares.
fn element<F: Field>(exponent: u64) -> F::Elem {
    F::GENERATOR.pow(exponent)
}

fn poly<F: FieldKernels>(coefficients: &[F::Elem]) -> Polynomial<F> {
    Polynomial::from_coefficients(coefficients).expect("coefficients build a polynomial")
}

/// The monic linear factor `X - root`.
fn linear<F: FieldKernels>(root: F::Elem) -> Polynomial<F> {
    poly(&[root.neg(), F::Elem::ONE])
}

/// `factor^exponent` by repeated multiplication.
fn power<F: FieldKernels>(factor: &Polynomial<F>, exponent: usize) -> Polynomial<F> {
    let mut result = Polynomial::one().expect("one");
    for _ in 0..exponent {
        result = result.multiply(factor).expect("power product");
    }
    result
}

fn monic_product<F: FieldKernels>(factors: &[Polynomial<F>]) -> Polynomial<F> {
    let mut product = Polynomial::one().expect("one");
    for factor in factors {
        product = product.multiply(factor).expect("product");
    }
    product
}

/// The monic monomial `X^degree`.
fn monomial<F: FieldKernels>(degree: usize) -> Polynomial<F> {
    let mut coefficients = vec![F::Elem::ZERO; degree];
    coefficients.push(F::Elem::ONE);
    poly(&coefficients)
}

/// Whether the absolute trace of a binary-field element is one, evaluated as
/// the sum of its conjugates.
fn has_absolute_trace_one<F: Field>(value: F::Elem) -> bool {
    let extension = u64::from(F::ORDER.trailing_zeros());
    let mut trace = value;
    let mut conjugate = value;
    for _ in 1..extension {
        conjugate = conjugate.pow(2);
        trace = trace.add(conjugate);
    }
    trace.is_one()
}

/// Distinct monic irreducible quadratics. In characteristic two,
/// `X^2 + bX + c` is irreducible exactly when `c/b^2` has absolute trace
/// one; the first two linear coefficients are chosen with different
/// absolute traces, because the equal-degree trace separator separates two
/// quadratic factors precisely through that difference. In odd
/// characteristic `X^2 + d` is irreducible exactly when `d` is a quadratic
/// nonresidue, and every odd power of a generator is one.
fn irreducible_quadratics<F: FieldKernels>(count: usize) -> Vec<Polynomial<F>> {
    let mut found = Vec::new();
    if F::CHARACTERISTIC == 2 {
        let one_trace = has_absolute_trace_one::<F>(F::Elem::ONE);
        // Generator powers with absolute trace one, one needed per factor.
        let mut trace_one_constants = Vec::new();
        let mut exponent = 0_u64;
        while trace_one_constants.len() < count {
            let candidate = element::<F>(exponent);
            exponent += 1;
            if has_absolute_trace_one::<F>(candidate) {
                trace_one_constants.push(candidate);
            }
        }

        // The first factor takes b = 1; the second takes the first
        // generator power whose absolute trace differs from 1's, so the
        // trace separator splits the leading pair; any further factors
        // (used only where no splitting follows) need only distinct b.
        let mut linear_coefficients = vec![F::Elem::ONE];
        let mut exponent = 1_u64;
        while linear_coefficients.len() < count {
            let candidate = element::<F>(exponent);
            exponent += 1;
            if linear_coefficients.contains(&candidate) {
                continue;
            }
            let distinct_enough = linear_coefficients.len() >= 2
                || has_absolute_trace_one::<F>(candidate) != one_trace;
            if distinct_enough {
                linear_coefficients.push(candidate);
            }
        }

        for (index, linear_coefficient) in linear_coefficients.iter().enumerate() {
            // c = b^2 * t keeps c/b^2 = t at trace one, so the quadratic
            // stays irreducible for every choice of b.
            let constant = linear_coefficient
                .mul(*linear_coefficient)
                .mul(trace_one_constants[index]);
            found.push(poly::<F>(&[constant, *linear_coefficient, F::Elem::ONE]));
        }
    } else {
        for exponent in [1_u64, 3, 5, 7] {
            if found.len() == count {
                break;
            }
            found.push(poly::<F>(&[
                element::<F>(exponent),
                F::Elem::ZERO,
                F::Elem::ONE,
            ]));
        }
    }
    found
}

/// Distinct monic irreducible cubics `X^3 + c`, available only when three
/// divides the group order so that non-cubes exist. A cubic is reducible
/// exactly when it has a root, and the roots of `X^3 + c` are the cube roots
/// of `-c`. In characteristic two the two constants are additionally chosen
/// with different absolute traces: the trace separator of two cubics
/// evaluates to the trace of `c` plus the candidate's constant term, so
/// equal traces would never separate.
fn irreducible_cubics<F: FieldKernels>() -> Vec<Polynomial<F>> {
    if (F::ORDER - 1) % 3 != 0 {
        return Vec::new();
    }
    let cube_exponent = u64::try_from((F::ORDER - 1) / 3).expect("cube exponent fits u64");
    let mut constants = Vec::new();
    let mut exponent = 1_u64;
    while constants.len() < 2 {
        let constant = element::<F>(exponent);
        exponent += 1;
        if constant.neg().pow(cube_exponent).is_one() {
            continue;
        }
        if F::CHARACTERISTIC == 2
            && constants.len() == 1
            && has_absolute_trace_one::<F>(constant) == has_absolute_trace_one::<F>(constants[0])
        {
            continue;
        }
        constants.push(constant);
    }
    constants
        .iter()
        .map(|constant| poly::<F>(&[*constant, F::Elem::ZERO, F::Elem::ZERO, F::Elem::ONE]))
        .collect()
}

fn square_free_multiplicities_recover_for<F: FieldKernels + PartialEq>() {
    let first = linear::<F>(element::<F>(0));
    let second = linear::<F>(element::<F>(5));
    let input = power(&first, 2)
        .multiply(&power(&second, 3))
        .expect("repeated input");

    let factors = input
        .square_free_factorization()
        .expect("square-free decomposition");
    assert_eq!(factors.len(), 2, "{}", core::any::type_name::<F>());
    assert_eq!(
        factors[0],
        SquareFreeFactor {
            factor: first.clone(),
            multiplicity: 2,
        }
    );
    assert_eq!(
        factors[1],
        SquareFreeFactor {
            factor: second.clone(),
            multiplicity: 3,
        }
    );
    let mut reconstructed = Polynomial::<F>::one().expect("one");
    for entry in &factors {
        reconstructed = reconstructed
            .multiply(&power(&entry.factor, entry.multiplicity))
            .expect("reconstruction product");
    }
    assert_eq!(reconstructed, input.monic());

    assert!(
        poly::<F>(&[element::<F>(7)])
            .square_free_factorization()
            .expect("a unit has no square-free factors")
            .is_empty()
    );
    assert_eq!(
        Polynomial::<F>::zero().square_free_factorization(),
        Err(FactorizationError::ZeroPolynomial)
    );
}

/// Frobenius squares and fourth powers have zero derivative, so the
/// decomposition extracts p-th roots before splitting multiplicities.
fn frobenius_powers_decompose_for<F: FieldKernels + PartialEq>() {
    let square_free = linear::<F>(element::<F>(0))
        .multiply(&linear::<F>(element::<F>(5)))
        .expect("square-free base");
    let input = square_free.square().expect("Frobenius square");
    assert!(input.formal_derivative().expect("derivative").is_zero());
    assert_eq!(
        input.square_free_factorization().expect("square root"),
        vec![SquareFreeFactor {
            factor: square_free.clone(),
            multiplicity: 2,
        }]
    );

    // A fourth power recurses through the root extraction twice.
    let deep = power(&square_free, 4);
    let factors = deep.square_free_factorization().expect("nested p-th roots");
    assert_eq!(factors.len(), 1);
    assert_eq!(factors[0].multiplicity, 4);
    let mut reconstructed = Polynomial::<F>::one().expect("one");
    for entry in &factors {
        reconstructed = reconstructed
            .multiply(&power(&entry.factor, entry.multiplicity))
            .expect("reconstruction product");
    }
    assert_eq!(reconstructed, deep.monic());
}

fn distinct_degree_groups_recover_for<F: FieldKernels + PartialEq>() {
    let quadratics = irreducible_quadratics::<F>(3);
    let cubics = irreducible_cubics::<F>();
    let linears = [
        linear::<F>(element::<F>(0)),
        linear::<F>(element::<F>(5)),
        linear::<F>(element::<F>(11)),
    ];

    // Only linear factors: one degree-one group and the loop breaks once the
    // remaining cofactor is one.
    let product = monic_product(&linears);
    assert_eq!(
        product
            .distinct_degree_factorization()
            .expect("linear groups"),
        vec![DistinctDegreeFactor {
            factor: product,
            factor_degree: 1,
        }]
    );

    // A linear times an irreducible quadratic: the quadratic stays as the
    // final group once the degree bound excludes it.
    let mixed = linears[0].multiply(&quadratics[0]).expect("mixed product");
    assert_eq!(
        mixed.distinct_degree_factorization().expect("mixed groups"),
        vec![
            DistinctDegreeFactor {
                factor: linears[0].clone(),
                factor_degree: 1,
            },
            DistinctDegreeFactor {
                factor: quadratics[0].clone(),
                factor_degree: 2,
            },
        ]
    );

    // Two and three irreducible quadratics: no degree-one factors, so the
    // first gcd hit arrives at degree two.
    for count in [2, 3] {
        let product = monic_product(&quadratics[..count]);
        assert_eq!(
            product
                .distinct_degree_factorization()
                .expect("quadratic groups"),
            vec![DistinctDegreeFactor {
                factor: product,
                factor_degree: 2,
            }]
        );
    }

    // Two irreducible cubics where the field has them: degrees one and two
    // miss before degree three hits.
    if !cubics.is_empty() {
        let product = monic_product(&cubics);
        assert_eq!(
            product
                .distinct_degree_factorization()
                .expect("cubic groups"),
            vec![DistinctDegreeFactor {
                factor: product,
                factor_degree: 3,
            }]
        );
    }

    assert_eq!(
        Polynomial::<F>::zero().distinct_degree_factorization(),
        Err(FactorizationError::ZeroPolynomial)
    );
    assert!(
        poly::<F>(&[element::<F>(9)])
            .distinct_degree_factorization()
            .expect("a unit has no groups")
            .is_empty()
    );
    assert_eq!(
        power(&linears[0], 2).distinct_degree_factorization(),
        Err(FactorizationError::NotSquareFree)
    );
}

fn equal_degree_splits_recover_for<F: FieldKernels + PartialEq>() {
    let quadratics = irreducible_quadratics::<F>(3);
    let cubics = irreducible_cubics::<F>();

    let pair = monic_product(&quadratics[..2]);
    let split = pair
        .equal_degree_factorization(2)
        .expect("split the quadratic pair");
    assert_eq!(split.len(), 2);
    assert!(split.contains(&quadratics[0]));
    assert!(split.contains(&quadratics[1]));
    assert_eq!(monic_product(&split), pair);

    // In odd characteristic three factors exercise the pending stack across
    // two split levels. The quadratic-character separator tracks the
    // candidate digits, so enumeration is immediate; characteristic two
    // partitions by trace instead, and a two-value trace cannot separate
    // every subset of three.
    if F::CHARACTERISTIC != 2 {
        let triple = monic_product(&quadratics);
        let split = triple
            .equal_degree_factorization(2)
            .expect("split the quadratic triple");
        assert_eq!(split.len(), 3);
        for quadratic in &quadratics {
            assert!(split.contains(quadratic));
        }
        assert_eq!(monic_product(&split), triple);
    }

    // Linear factors split at degree one.
    let first = linear::<F>(element::<F>(0));
    let second = linear::<F>(element::<F>(5));
    let linear_pair = first.multiply(&second).expect("linear pair");
    let split = linear_pair
        .equal_degree_factorization(1)
        .expect("split the linear pair");
    assert_eq!(split.len(), 2);
    assert!(split.contains(&first));
    assert!(split.contains(&second));

    // Cubic pairs split at degree three where the field has them.
    if !cubics.is_empty() {
        let pair = monic_product(&cubics);
        let split = pair
            .equal_degree_factorization(3)
            .expect("split the cubic pair");
        assert_eq!(split.len(), 2);
        assert!(split.contains(&cubics[0]));
        assert!(split.contains(&cubics[1]));
        assert_eq!(monic_product(&split), pair);
    }

    assert_eq!(
        Polynomial::<F>::zero().equal_degree_factorization(1),
        Err(FactorizationError::ZeroPolynomial)
    );
    assert_eq!(
        linear_pair.equal_degree_factorization(0),
        Err(FactorizationError::ZeroFactorDegree)
    );
    assert!(
        Polynomial::<F>::one()
            .expect("one")
            .equal_degree_factorization(2)
            .expect("a unit has no factors")
            .is_empty()
    );
    let mixed = first.multiply(&quadratics[0]).expect("mixed degrees");
    assert_eq!(
        mixed.equal_degree_factorization(1),
        Err(FactorizationError::NotEqualDegree { factor_degree: 1 })
    );
    assert_eq!(
        quadratics[0].equal_degree_factorization(3),
        Err(FactorizationError::NotEqualDegree { factor_degree: 3 })
    );
}

fn complete_factorization_recovers_for<F: FieldKernels + PartialEq>() {
    let quadratics = irreducible_quadratics::<F>(2);
    let cubics = irreducible_cubics::<F>();
    let first = linear::<F>(element::<F>(0));
    let input = power(&first, 2)
        .multiply(&quadratics[0])
        .expect("mixed input")
        .multiply(&quadratics[1])
        .expect("mixed input");

    let factors = input.factor().expect("complete factorization");
    assert_eq!(factors.len(), 3);
    assert!(
        factors
            .iter()
            .any(|entry| entry.factor == first && entry.multiplicity == 2)
    );
    assert_eq!(
        factors
            .iter()
            .filter(|entry| entry.multiplicity == 1)
            .count(),
        2
    );
    for entry in &factors {
        assert!(
            entry
                .factor
                .is_irreducible()
                .expect("factor irreducibility"),
            "{}",
            core::any::type_name::<F>()
        );
    }
    let reconstructed = factors.iter().fold(
        Polynomial::<F>::one().expect("one"),
        |accumulator, entry| {
            accumulator
                .multiply(&power(&entry.factor, entry.multiplicity))
                .expect("reconstruction product")
        },
    );
    assert_eq!(reconstructed, input.monic());

    // Irreducibility of the constructed inputs and of their products.
    assert!(first.is_irreducible().expect("linear"));
    for quadratic in &quadratics {
        assert!(quadratic.is_irreducible().expect("quadratic"));
    }
    for cubic in &cubics {
        assert!(cubic.is_irreducible().expect("cubic"));
    }
    let reducible = quadratics[0]
        .multiply(&quadratics[1])
        .expect("quadratic product");
    assert!(!reducible.is_irreducible().expect("reducible"));
    assert!(
        !quadratics[0]
            .square()
            .expect("square")
            .is_irreducible()
            .expect("a square is reducible")
    );
    assert!(!Polynomial::<F>::zero().is_irreducible().expect("zero"));
    assert!(
        !poly::<F>(&[element::<F>(4)])
            .is_irreducible()
            .expect("constant")
    );
}

fn quotient_ring_routes_hold_for<F: FieldKernels + PartialEq>() {
    let quadratics = irreducible_quadratics::<F>(2);
    let modulus = quadratics[0].clone();
    let plan = ModulusPlan::new(&modulus).expect("plan");
    assert_eq!(plan.modulus(), &modulus);
    assert_eq!(
        ModulusPlan::new(&Polynomial::<F>::zero()),
        Err(PolynomialError::DivisionByZero)
    );
    assert_eq!(
        ModulusPlan::new(&poly::<F>(&[element::<F>(6)])),
        Err(PolynomialError::ConstantModulus)
    );

    let value = poly::<F>(&[element::<F>(1), element::<F>(2), F::Elem::ONE]);
    let other = poly::<F>(&[element::<F>(3), F::Elem::ONE]);
    let mut scratch = plan.scratch();
    // Destinations are write-only, so a dirty output must not survive.
    let mut output = poly::<F>(&[element::<F>(9), element::<F>(9)]);

    plan.reduce_into(&value, &mut scratch, &mut output)
        .expect("reduce into");
    assert_eq!(output, value.remainder(&modulus).expect("direct remainder"));
    assert_eq!(output, plan.reduce(&value).expect("allocating reduce"));

    plan.multiply_into(&value, &other, &mut scratch, &mut output)
        .expect("multiply into");
    let expected_product = value
        .multiply(&other)
        .expect("direct product")
        .remainder(&modulus)
        .expect("reduced product");
    assert_eq!(output, expected_product);
    assert_eq!(
        output,
        plan.multiply(&value, &other).expect("allocating product")
    );

    plan.square_into(&value, &mut scratch, &mut output)
        .expect("square into");
    assert_eq!(output, plan.square(&value).expect("allocating square"));

    plan.pow_into(&value, 0, &mut scratch, &mut output)
        .expect("zero power into");
    assert!(output.is_one());
    let mut expected_power = Polynomial::<F>::one().expect("one");
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

    // The modulus is irreducible of degree two, so the q^2-Frobenius fixes
    // every residue class; fields whose squared order leaves the exponent
    // range skip the check.
    if let Some(exponent) = F::ORDER.checked_mul(F::ORDER) {
        plan.pow_into(&value, exponent, &mut scratch, &mut output)
            .expect("Frobenius power into");
        assert_eq!(output, value.remainder(&modulus).expect("reduced value"));
    }

    // A constant outer composes to itself; sparse and dense outers agree
    // with the direct composition reduced.
    plan.compose_into(
        &poly::<F>(&[element::<F>(7)]),
        &other,
        &mut scratch,
        &mut output,
    )
    .expect("constant composition");
    assert_eq!(output, poly::<F>(&[element::<F>(7)]));
    let sparse = poly::<F>(&[element::<F>(5), F::Elem::ZERO, F::Elem::ONE]);
    plan.compose_into(&sparse, &other, &mut scratch, &mut output)
        .expect("sparse composition");
    let expected_composition = sparse
        .compose(&other)
        .expect("direct composition")
        .remainder(&modulus)
        .expect("reduced composition");
    assert_eq!(output, expected_composition);
    assert_eq!(
        plan.compose(&sparse, &other).expect("allocating compose"),
        expected_composition
    );

    // X is a unit modulo the irreducible quadratic.
    let unit = poly::<F>(&[F::Elem::ZERO, F::Elem::ONE]);
    plan.inverse_into(&unit, &mut scratch, &mut output)
        .expect("inverse into");
    assert_eq!(output, plan.inverse(&unit).expect("allocating inverse"));
    assert!(
        plan.multiply(&unit, &output)
            .expect("identity product")
            .is_one()
    );
    plan.inverse_into(&unit, &mut scratch, &mut output)
        .expect("inverse into with warmed scratch");
    assert!(
        plan.multiply(&unit, &output)
            .expect("identity product again")
            .is_one()
    );

    // A factor of a reducible modulus is not a unit.
    let reducible = linear::<F>(element::<F>(0))
        .multiply(&linear::<F>(element::<F>(5)))
        .expect("reducible modulus");
    let reducible_plan = ModulusPlan::new(&reducible).expect("reducible plan");
    let mut reducible_scratch = ModulusScratch::default();
    let mut sink = Polynomial::zero();
    assert_eq!(
        reducible_plan.inverse_into(
            &linear::<F>(element::<F>(0)),
            &mut reducible_scratch,
            &mut sink
        ),
        Err(PolynomialError::NotInvertibleModulo)
    );
}

fn euclidean_identities_hold_for<F: FieldKernels + PartialEq>() {
    let quadratics = irreducible_quadratics::<F>(2);
    let left = linear::<F>(element::<F>(0))
        .multiply(&linear::<F>(element::<F>(5)))
        .expect("left input")
        .multiply(&quadratics[0])
        .expect("left input");
    let right = linear::<F>(element::<F>(11))
        .multiply(&quadratics[0])
        .expect("right input")
        .multiply(&quadratics[1])
        .expect("right input");

    assert_eq!(left.gcd(&right).expect("gcd"), quadratics[0]);
    assert_eq!(
        left.gcd(&Polynomial::<F>::zero()).expect("gcd with zero"),
        left.monic()
    );
    assert_eq!(
        Polynomial::<F>::zero()
            .gcd(&Polynomial::<F>::zero())
            .expect("gcd of zeros"),
        Polynomial::<F>::zero()
    );

    let relation = left.extended_gcd(&right).expect("Bézout relation");
    assert_eq!(relation.gcd, quadratics[0]);
    assert_eq!(
        relation
            .a_cofactor
            .multiply(&left)
            .expect("cofactor product")
            .add(
                &relation
                    .b_cofactor
                    .multiply(&right)
                    .expect("cofactor product")
            )
            .expect("Bézout sum"),
        relation.gcd.clone()
    );
    // The cofactors are the minimal-degree ones.
    let left_degree = left.degree().expect("left degree");
    let right_degree = right.degree().expect("right degree");
    let gcd_degree = relation.gcd.degree().expect("gcd degree");
    assert!(
        relation
            .a_cofactor
            .degree()
            .is_none_or(|degree| degree + gcd_degree <= right_degree)
    );
    assert!(
        relation
            .b_cofactor
            .degree()
            .is_none_or(|degree| degree + gcd_degree <= left_degree)
    );

    // An exact division leaves the divisible operand's cofactor at zero.
    let base = linear::<F>(element::<F>(5));
    let exact = power(&base, 3);
    let relation = exact.extended_gcd(&base).expect("divisible pair");
    assert_eq!(relation.gcd, base);
    assert!(relation.a_cofactor.is_zero());
    assert!(relation.b_cofactor.is_one());

    // gcd(0, 0) keeps the identity at zero.
    let relation = Polynomial::<F>::zero()
        .extended_gcd(&Polynomial::<F>::zero())
        .expect("zero pair");
    assert!(relation.gcd.is_zero());
    assert!(relation.a_cofactor.is_zero());
    assert!(relation.b_cofactor.is_zero());

    // The truncated algorithm stops at the key-equation bound and keeps the
    // same identity; a bound above every degree stops immediately.
    let dividend = monomial::<F>(8);
    let series = poly::<F>(&[
        F::Elem::ONE,
        element::<F>(1),
        F::Elem::ZERO,
        element::<F>(4),
        element::<F>(2),
        F::Elem::ONE,
        element::<F>(3),
        element::<F>(7),
    ]);
    let stopped = truncated_eea(&dividend, &series, 4).expect("truncated relation");
    assert!(
        stopped.remainder.degree().is_none_or(|degree| degree < 4),
        "the truncation bound holds"
    );
    assert_eq!(
        stopped
            .a_cofactor
            .multiply(&dividend)
            .expect("cofactor product")
            .add(
                &stopped
                    .b_cofactor
                    .multiply(&series)
                    .expect("cofactor product")
            )
            .expect("truncated sum"),
        stopped.remainder
    );

    let immediate = truncated_eea(&dividend, &series, 9).expect("immediate stop");
    assert_eq!(immediate.remainder, series);
    assert!(immediate.a_cofactor.is_zero());
    assert!(immediate.b_cofactor.is_one());
}

fn truncated_series_routes_hold_for<F: FieldKernels + PartialEq>() {
    let series = poly::<F>(&[
        F::Elem::ONE,
        element::<F>(1),
        F::Elem::ZERO,
        element::<F>(4),
        F::Elem::ONE,
        element::<F>(2),
    ]);
    for truncation in [1, 3, 6, 9] {
        let newton = series
            .inverse_mod_x_power(truncation)
            .expect("Newton inverse");
        let naive = series
            .inverse_mod_x_power_naive(truncation)
            .expect("naive inverse");
        assert_eq!(
            newton,
            naive,
            "the inversion forms agree at {truncation} for {}",
            core::any::type_name::<F>()
        );
        assert!(newton.coefficient_count() <= truncation);
        let product = series
            .multiply_truncated(&newton, truncation)
            .expect("truncated product");
        assert_eq!(product, Polynomial::<F>::one().expect("one"));
    }
    assert_eq!(
        series.inverse_mod_x_power(0).expect("empty precision"),
        Polynomial::<F>::zero()
    );
    assert_eq!(
        series
            .inverse_mod_x_power_naive(0)
            .expect("empty precision"),
        Polynomial::<F>::zero()
    );

    let zero_constant = poly::<F>(&[F::Elem::ZERO, F::Elem::ONE, element::<F>(3)]);
    assert!(matches!(
        zero_constant.inverse_mod_x_power(4),
        Err(PolynomialError::ZeroConstantTerm { .. })
    ));
    assert!(matches!(
        zero_constant.inverse_mod_x_power_naive(4),
        Err(PolynomialError::ZeroConstantTerm { .. })
    ));
    assert!(matches!(
        series_divide(&series, &zero_constant, 4),
        Err(PolynomialError::ZeroConstantTerm { .. })
    ));

    // Division inverts the denominator: the product with the quotient
    // matches the numerator coefficient by coefficient.
    let numerator = poly::<F>(&[
        element::<F>(2),
        F::Elem::ONE,
        element::<F>(5),
        F::Elem::ZERO,
        element::<F>(1),
    ]);
    let quotient = series_divide(&numerator, &series, 5).expect("series quotient");
    let product = series
        .multiply_truncated(&quotient, 5)
        .expect("recovered numerator");
    for degree in 0..5 {
        assert_eq!(
            product.coefficient(degree),
            numerator.coefficient(degree),
            "coefficient {degree} survives for {}",
            core::any::type_name::<F>()
        );
    }

    // Truncation drops shifted terms and skips zero coefficients of the
    // shorter operand.
    let dense = poly::<F>(&[
        F::Elem::ONE,
        element::<F>(1),
        element::<F>(2),
        element::<F>(3),
    ]);
    let sparse = poly::<F>(&[F::Elem::ONE, F::Elem::ZERO, element::<F>(4)]);
    let product = dense
        .multiply_truncated(&sparse, 2)
        .expect("truncated product");
    assert_eq!(product.coefficient_count(), 2);
    assert_eq!(product.coefficient(0), F::Elem::ONE);
    assert_eq!(product.coefficient(1), element::<F>(1));

    let reversed = series.reverse();
    assert_eq!(reversed.degree(), Some(5));
    assert_eq!(reversed.coefficient(0), element::<F>(2));
    assert_eq!(reversed.reverse(), series);
    assert!(Polynomial::<F>::zero().reverse().is_zero());
}

/// Every factorization stage over one field of characteristic two,
/// including the Frobenius root extractions only binary fields admit.
fn binary_factorization_stages_for<F: FieldKernels + PartialEq>() {
    square_free_multiplicities_recover_for::<F>();
    frobenius_powers_decompose_for::<F>();
    distinct_degree_groups_recover_for::<F>();
    equal_degree_splits_recover_for::<F>();
    complete_factorization_recovers_for::<F>();
}

#[test]
fn binary_extensions_instantiate_every_factorization_stage() {
    binary_factorization_stages_for::<Gf8D>();
    binary_factorization_stages_for::<Gf16>();
    binary_factorization_stages_for::<FanPaar8>();
    binary_factorization_stages_for::<FanPaar16>();
    binary_factorization_stages_for::<FanPaar32>();
    binary_factorization_stages_for::<FanPaar64>();
}

#[test]
fn odd_order_fields_instantiate_every_factorization_stage() {
    square_free_multiplicities_recover_for::<Goldilocks>();
    distinct_degree_groups_recover_for::<Goldilocks>();
    equal_degree_splits_recover_for::<Goldilocks>();
    complete_factorization_recovers_for::<Goldilocks>();

    square_free_multiplicities_recover_for::<QuadMersenne31>();
    distinct_degree_groups_recover_for::<QuadMersenne31>();
    equal_degree_splits_recover_for::<QuadMersenne31>();
    complete_factorization_recovers_for::<QuadMersenne31>();
}

#[test]
fn prepared_quotient_routes_instantiate_across_every_field() {
    quotient_ring_routes_hold_for::<Gf8D>();
    quotient_ring_routes_hold_for::<Gf16>();
    quotient_ring_routes_hold_for::<FanPaar8>();
    quotient_ring_routes_hold_for::<FanPaar16>();
    quotient_ring_routes_hold_for::<FanPaar32>();
    quotient_ring_routes_hold_for::<FanPaar64>();
    quotient_ring_routes_hold_for::<Goldilocks>();
    quotient_ring_routes_hold_for::<QuadMersenne31>();
}

#[test]
fn euclidean_and_series_routes_instantiate_across_every_field() {
    euclidean_identities_hold_for::<Gf8D>();
    truncated_series_routes_hold_for::<Gf8D>();
    euclidean_identities_hold_for::<Gf16>();
    truncated_series_routes_hold_for::<Gf16>();
    euclidean_identities_hold_for::<FanPaar8>();
    truncated_series_routes_hold_for::<FanPaar8>();
    euclidean_identities_hold_for::<FanPaar16>();
    truncated_series_routes_hold_for::<FanPaar16>();
    euclidean_identities_hold_for::<FanPaar32>();
    truncated_series_routes_hold_for::<FanPaar32>();
    euclidean_identities_hold_for::<FanPaar64>();
    truncated_series_routes_hold_for::<FanPaar64>();
    euclidean_identities_hold_for::<Goldilocks>();
    truncated_series_routes_hold_for::<Goldilocks>();
    euclidean_identities_hold_for::<QuadMersenne31>();
    truncated_series_routes_hold_for::<QuadMersenne31>();
}

/// Refusing any single fallible allocation in the odd-characteristic routes surfaces `ConfigError::AllocationFailed` through that route
///
/// Each entry names one recorded allocation by its exact byte size and the
/// number of same-size allocations admitted before it; the geometries are
/// the ones the other tests in this file already established.
#[test]
fn quad_mersenne_routes_report_refused_allocations() {
    let quadratics = irreducible_quadratics::<QuadMersenne31>(2);
    let a = linear::<QuadMersenne31>(element::<QuadMersenne31>(0))
        .multiply(&linear::<QuadMersenne31>(element::<QuadMersenne31>(5)))
        .expect("left input")
        .multiply(&quadratics[0])
        .expect("left input");
    let b = linear::<QuadMersenne31>(element::<QuadMersenne31>(11))
        .multiply(&quadratics[0])
        .expect("right input")
        .multiply(&quadratics[1])
        .expect("right input");
    let dividend = monomial::<QuadMersenne31>(8);
    let series = poly::<QuadMersenne31>(&[
        <QuadMersenne31 as Field>::Elem::ONE,
        element::<QuadMersenne31>(1),
        <QuadMersenne31 as Field>::Elem::ZERO,
        element::<QuadMersenne31>(4),
        element::<QuadMersenne31>(2),
        <QuadMersenne31 as Field>::Elem::ONE,
        element::<QuadMersenne31>(3),
        element::<QuadMersenne31>(7),
    ]);
    let plan = ModulusPlan::new(&quadratics[0]).expect("plan");
    let value = poly::<QuadMersenne31>(&[
        element::<QuadMersenne31>(1),
        element::<QuadMersenne31>(2),
        <QuadMersenne31 as Field>::Elem::ONE,
    ]);
    let other = poly::<QuadMersenne31>(&[
        element::<QuadMersenne31>(3),
        <QuadMersenne31 as Field>::Elem::ONE,
    ]);
    let unit = poly::<QuadMersenne31>(&[
        <QuadMersenne31 as Field>::Elem::ZERO,
        <QuadMersenne31 as Field>::Elem::ONE,
    ]);
    let sparse = poly::<QuadMersenne31>(&[
        element::<QuadMersenne31>(5),
        <QuadMersenne31 as Field>::Elem::ZERO,
        <QuadMersenne31 as Field>::Elem::ONE,
    ]);
    let mixed = linear::<QuadMersenne31>(element::<QuadMersenne31>(0))
        .multiply(&quadratics[0])
        .expect("mixed product");
    let repeated = power(&linear::<QuadMersenne31>(element::<QuadMersenne31>(0)), 2)
        .multiply(&power(
            &linear::<QuadMersenne31>(element::<QuadMersenne31>(5)),
            3,
        ))
        .expect("repeated input");
    let pair = monic_product(&quadratics);
    let input = power(&linear::<QuadMersenne31>(element::<QuadMersenne31>(0)), 2)
        .multiply(&quadratics[0])
        .expect("mixed input")
        .multiply(&quadratics[1])
        .expect("mixed input");
    // the extended gcd
    for &(size, skip) in &[
        (8, 0),
        (16, 0),
        (16, 1),
        (16, 3),
        (16, 4),
        (24, 0),
        (24, 1),
        (16, 7),
        (16, 8),
        (32, 1),
        (32, 2),
        (24, 3),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    a.extended_gcd(&b),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the extended gcd reports the refused allocation"
            );
        });
    }
    // the truncated Euclidean algorithm
    for &(size, skip) in &[
        (8, 0),
        (16, 0),
        (16, 2),
        (16, 4),
        (16, 6),
        (24, 0),
        (16, 9),
        (24, 3),
        (32, 1),
        (16, 12),
        (32, 3),
        (40, 1),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    truncated_eea(&dividend, &series, 4),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the truncated Euclidean algorithm reports the refused allocation"
            );
        });
    }
    // Newton series inversion
    for &(size, skip) in &[
        (8, 0),
        (8, 1),
        (16, 0),
        (16, 1),
        (16, 2),
        (32, 0),
        (32, 1),
        (32, 2),
        (48, 0),
        (48, 1),
        (48, 2),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    series.inverse_mod_x_power(6),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "Newton series inversion reports the refused allocation"
            );
        });
    }
    // the linear series inversion
    for &(size, skip) in &[(48, 0), (48, 1)] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    series.inverse_mod_x_power_naive(6),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the linear series inversion reports the refused allocation"
            );
        });
    }
    // series division
    for &(size, skip) in &[
        (8, 0),
        (8, 1),
        (16, 0),
        (16, 1),
        (16, 2),
        (32, 0),
        (32, 1),
        (32, 2),
        (48, 0),
        (48, 1),
        (48, 2),
        (48, 3),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    series_divide(&a, &series, 6),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "series division reports the refused allocation"
            );
        });
    }
    // the prepared remainder
    {
        let &(size, skip) = &(8, 0);
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    plan.reduce(&value),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the prepared remainder reports the refused allocation"
            );
        });
    }
    // the prepared product
    for &(size, skip) in &[(32, 0), (16, 0)] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    plan.multiply(&value, &other),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the prepared product reports the refused allocation"
            );
        });
    }
    // the prepared power
    for &(size, skip) in &[(8, 0), (8, 1), (16, 0), (24, 1)] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    plan.pow(&value, 13),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the prepared power reports the refused allocation"
            );
        });
    }
    // the prepared inverse
    for &(size, skip) in &[(8, 0), (16, 1), (16, 2), (16, 3), (24, 2), (24, 3)] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    plan.inverse(&unit),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the prepared inverse reports the refused allocation"
            );
        });
    }
    // the prepared composition
    for &(size, skip) in &[(8, 0), (16, 1), (24, 0), (8, 1)] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    plan.compose(&sparse, &other),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the prepared composition reports the refused allocation"
            );
        });
    }
    // distinct-degree grouping
    for &(size, skip) in &[
        (24, 0),
        (16, 21),
        (16, 39),
        (16, 56),
        (32, 3),
        (40, 78),
        (16, 107),
        (16, 125),
        (16, 142),
        (40, 148),
        (40, 166),
        (8, 6),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    mixed.distinct_degree_factorization(),
                    Err(FactorizationError::Polynomial(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    )))
                ),
                "distinct-degree grouping reports the refused allocation"
            );
        });
    }
    // irreducibility testing
    for &(size, skip) in &[
        (16, 0),
        (16, 13),
        (8, 10),
        (8, 32),
        (8, 52),
        (16, 23),
        (16, 33),
        (16, 43),
        (16, 53),
        (8, 110),
        (8, 120),
        (128, 0),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    quadratics[0].is_irreducible(),
                    Err(FactorizationError::Polynomial(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    )))
                ),
                "irreducibility testing reports the refused allocation"
            );
        });
    }
    // square-free decomposition
    for &(size, skip) in &[
        (40, 0),
        (16, 0),
        (16, 3),
        (24, 1),
        (16, 5),
        (16, 6),
        (16, 9),
        (16, 13),
        (8, 2),
        (16, 16),
        (16, 21),
        (8, 8),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    repeated.square_free_factorization(),
                    Err(FactorizationError::Polynomial(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    )))
                ),
                "square-free decomposition reports the refused allocation"
            );
        });
    }
    // equal-degree splitting
    for &(size, skip) in &[
        (32, 0),
        (8, 16),
        (8, 38),
        (40, 62),
        (8, 74),
        (40, 86),
        (48, 30),
        (8, 107),
        (16, 66),
        (8, 134),
        (8, 156),
        (8, 179),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    pair.equal_degree_factorization(2),
                    Err(FactorizationError::Polynomial(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    )))
                ),
                "equal-degree splitting reports the refused allocation"
            );
        });
    }
    // complete factorization
    for &(size, skip) in &[
        (48, 0),
        (16, 25),
        (40, 24),
        (40, 46),
        (40, 68),
        (16, 44),
        (8, 92),
        (16, 65),
        (8, 114),
        (16, 87),
        (8, 140),
        (8, 162),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    input.factor(),
                    Err(FactorizationError::Polynomial(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    )))
                ),
                "complete factorization reports the refused allocation"
            );
        });
    }
}

/// Refusing any single fallible allocation in the characteristic-two routes surfaces `ConfigError::AllocationFailed` through that route
///
/// Each entry names one recorded allocation by its exact byte size and the
/// number of same-size allocations admitted before it; the geometries are
/// the ones the other tests in this file already established.
#[test]
fn gf8d_routes_report_refused_allocations() {
    let quadratics = irreducible_quadratics::<Gf8D>(2);
    let a = linear::<Gf8D>(element::<Gf8D>(0))
        .multiply(&linear::<Gf8D>(element::<Gf8D>(5)))
        .expect("left input")
        .multiply(&quadratics[0])
        .expect("left input");
    let b = linear::<Gf8D>(element::<Gf8D>(11))
        .multiply(&quadratics[0])
        .expect("right input")
        .multiply(&quadratics[1])
        .expect("right input");
    let dividend = monomial::<Gf8D>(8);
    let series = poly::<Gf8D>(&[
        <Gf8D as Field>::Elem::ONE,
        element::<Gf8D>(1),
        <Gf8D as Field>::Elem::ZERO,
        element::<Gf8D>(4),
        element::<Gf8D>(2),
        <Gf8D as Field>::Elem::ONE,
        element::<Gf8D>(3),
        element::<Gf8D>(7),
    ]);
    let plan = ModulusPlan::new(&quadratics[0]).expect("plan");
    let value = poly::<Gf8D>(&[
        element::<Gf8D>(1),
        element::<Gf8D>(2),
        <Gf8D as Field>::Elem::ONE,
    ]);
    let other = poly::<Gf8D>(&[element::<Gf8D>(3), <Gf8D as Field>::Elem::ONE]);
    let unit = poly::<Gf8D>(&[<Gf8D as Field>::Elem::ZERO, <Gf8D as Field>::Elem::ONE]);
    let sparse = poly::<Gf8D>(&[
        element::<Gf8D>(5),
        <Gf8D as Field>::Elem::ZERO,
        <Gf8D as Field>::Elem::ONE,
    ]);
    let mixed = linear::<Gf8D>(element::<Gf8D>(0))
        .multiply(&quadratics[0])
        .expect("mixed product");
    let repeated = power(&linear::<Gf8D>(element::<Gf8D>(0)), 2)
        .multiply(&power(&linear::<Gf8D>(element::<Gf8D>(5)), 3))
        .expect("repeated input");
    let pair = monic_product(&quadratics);
    let input = power(&linear::<Gf8D>(element::<Gf8D>(0)), 2)
        .multiply(&quadratics[0])
        .expect("mixed input")
        .multiply(&quadratics[1])
        .expect("mixed input");
    // the extended gcd
    for &(size, skip) in &[
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 3),
        (2, 4),
        (3, 0),
        (3, 1),
        (2, 7),
        (2, 8),
        (4, 1),
        (4, 2),
        (3, 3),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    a.extended_gcd(&b),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the extended gcd reports the refused allocation"
            );
        });
    }
    // the truncated Euclidean algorithm
    for &(size, skip) in &[
        (1, 0),
        (2, 0),
        (2, 2),
        (2, 4),
        (2, 6),
        (3, 0),
        (2, 9),
        (3, 3),
        (4, 1),
        (2, 12),
        (4, 3),
        (5, 1),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    truncated_eea(&dividend, &series, 4),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the truncated Euclidean algorithm reports the refused allocation"
            );
        });
    }
    // Newton series inversion
    for &(size, skip) in &[
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (3, 0),
        (4, 0),
        (6, 0),
        (6, 1),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    series.inverse_mod_x_power(6),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "Newton series inversion reports the refused allocation"
            );
        });
    }
    // the linear series inversion
    for &(size, skip) in &[(6, 0), (6, 1)] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    series.inverse_mod_x_power_naive(6),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the linear series inversion reports the refused allocation"
            );
        });
    }
    // series division
    for &(size, skip) in &[
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (3, 0),
        (4, 0),
        (6, 0),
        (6, 1),
        (6, 2),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    series_divide(&a, &series, 6),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "series division reports the refused allocation"
            );
        });
    }
    // the prepared remainder
    {
        let &(size, skip) = &(1, 0);
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    plan.reduce(&value),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the prepared remainder reports the refused allocation"
            );
        });
    }
    // the prepared product
    for &(size, skip) in &[(4, 0), (2, 0)] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    plan.multiply(&value, &other),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the prepared product reports the refused allocation"
            );
        });
    }
    // the prepared power
    for &(size, skip) in &[(1, 0), (1, 1), (2, 0), (3, 0)] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    plan.pow(&value, 13),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the prepared power reports the refused allocation"
            );
        });
    }
    // the prepared inverse
    for &(size, skip) in &[(1, 0), (2, 0), (2, 1), (2, 2), (3, 0)] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    plan.inverse(&unit),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the prepared inverse reports the refused allocation"
            );
        });
    }
    // the prepared composition
    for &(size, skip) in &[(1, 0), (2, 0), (3, 0), (1, 1)] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    plan.compose(&sparse, &other),
                    Err(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    ))
                ),
                "the prepared composition reports the refused allocation"
            );
        });
    }
    // distinct-degree grouping
    for &(size, skip) in &[
        (3, 0),
        (3, 3),
        (1, 1),
        (2, 6),
        (2, 8),
        (5, 6),
        (5, 8),
        (2, 15),
        (2, 17),
        (2, 21),
        (128, 0),
        (1, 4),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    mixed.distinct_degree_factorization(),
                    Err(FactorizationError::Polynomial(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    )))
                ),
                "distinct-degree grouping reports the refused allocation"
            );
        });
    }
    // irreducibility testing
    for &(size, skip) in &[
        (2, 0),
        (3, 4),
        (3, 9),
        (3, 10),
        (3, 12),
        (1, 10),
        (1, 12),
        (1, 15),
        (1, 17),
        (1, 19),
        (2, 7),
        (128, 0),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    quadratics[0].is_irreducible(),
                    Err(FactorizationError::Polynomial(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    )))
                ),
                "irreducibility testing reports the refused allocation"
            );
        });
    }
    // square-free decomposition
    for &(size, skip) in &[
        (5, 0),
        (2, 2),
        (4, 1),
        (4, 3),
        (1, 3),
        (2, 14),
        (2, 19),
        (3, 6),
        (1, 5),
        (2, 28),
        (2, 34),
        (1, 13),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    repeated.square_free_factorization(),
                    Err(FactorizationError::Polynomial(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    )))
                ),
                "square-free decomposition reports the refused allocation"
            );
        });
    }
    // equal-degree splitting
    for &(size, skip) in &[
        (4, 0),
        (7, 0),
        (7, 8),
        (2, 9),
        (3, 26),
        (3, 34),
        (2, 17),
        (4, 10),
        (3, 47),
        (3, 55),
        (3, 63),
        (3, 70),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    pair.equal_degree_factorization(2),
                    Err(FactorizationError::Polynomial(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    )))
                ),
                "equal-degree splitting reports the refused allocation"
            );
        });
    }
    // complete factorization
    for &(size, skip) in &[
        (6, 0),
        (1, 2),
        (7, 4),
        (2, 28),
        (3, 44),
        (1, 24),
        (2, 45),
        (3, 80),
        (3, 88),
        (3, 100),
        (3, 115),
        (24, 1),
    ] {
        reject_matching(size, skip, || {
            assert!(
                matches!(
                    input.factor(),
                    Err(FactorizationError::Polynomial(PolynomialError::Config(
                        ConfigError::AllocationFailed { .. }
                    )))
                ),
                "complete factorization reports the refused allocation"
            );
        });
    }
}
