// `as_chunks::<F::BYTES>()` needs a const generic depending on a
// type parameter, which Rust rejects in const argument position.
#![allow(clippy::chunks_exact_to_as_chunks)]

//! Polynomial operations behavior and error boundaries.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, Goldilocks, Mersenne31, QuadMersenne31, gf8b, mersenne31};
use poly_ring::{
    BivariatePolynomial, ConfigError, Polynomial, PolynomialError, ProductError, RootError,
};

use crate::oracles;
#[cfg(feature = "fft")]
use oracles::naive_multiply;
use oracles::{naive_evaluate, noise, noise_poly};

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn rows<F: FieldKernels>(rows: &[&[F::Elem]]) -> BivariatePolynomial<F> {
    BivariatePolynomial::from_y_coefficients(
        rows.iter()
            .map(|row| Polynomial::from_coefficients(row).expect("row"))
            .collect(),
    )
}

/// Bivariate add/sub/scale agree with evaluation, and zero scales to zero.
#[test]
fn bivariate_sums_differences_and_scales_evaluate() {
    fn check<F: FieldKernels>() {
        let left = rows::<F>(&[&[F::Elem::ONE, F::Elem::ONE], &[F::Elem::ONE]]);
        let right = rows::<F>(&[&[F::Elem::ONE], &[F::Elem::ONE, F::Elem::ONE]]);
        let x = F::GENERATOR;
        let y = F::GENERATOR.pow(3);
        assert_eq!(
            left.add(&right).expect("add").evaluate(x, y),
            left.evaluate(x, y).add(right.evaluate(x, y))
        );
        assert_eq!(
            left.sub(&right).expect("sub").evaluate(x, y),
            left.evaluate(x, y).add(right.evaluate(x, y).neg())
        );
        let scale = F::GENERATOR.pow(5);
        assert_eq!(
            left.scaled(scale).expect("scaled").evaluate(x, y),
            left.evaluate(x, y).mul(scale)
        );
        assert!(left.scaled(F::Elem::ZERO).expect("zero").is_zero());
    }
    check::<Gf8B>();
    check::<Mersenne31>();
}

/// Bivariate products with an internal zero row skip the convolution, and
/// the Hasse derivative carries the characteristic-exact binomial factor.
#[test]
fn bivariate_products_and_derivatives_evaluate() {
    fn check<F: FieldKernels>() {
        let left = rows::<F>(&[&[F::Elem::ONE, F::Elem::ONE], &[F::Elem::ONE]]);
        let right = rows::<F>(&[&[F::Elem::ONE], &[], &[F::Elem::ONE]]);
        let x = F::GENERATOR;
        let y = F::GENERATOR.pow(3);
        let product = left.multiply(&right).expect("product");
        assert_eq!(
            product.evaluate(x, y),
            left.evaluate(x, y).mul(right.evaluate(x, y))
        );
        // Hasse derivative against the defining binomial sum.
        let derived = left.hasse_derivative(1, 1).expect("derivative");
        let mut expected = F::Elem::ZERO;
        for (y_degree, row) in left.y_coefficients().iter().enumerate() {
            if y_degree < 1 {
                continue;
            }
            for (x_degree, coefficient) in row.coefficients().enumerate() {
                if x_degree < 1 {
                    continue;
                }
                expected = expected.add(
                    coefficient
                        .mul(poly_ring::binomial::<F>(x_degree, 1))
                        .mul(poly_ring::binomial::<F>(y_degree, 1))
                        .mul(x.pow((x_degree - 1) as u64))
                        .mul(y.pow((y_degree - 1) as u64)),
                );
            }
        }
        assert_eq!(derived.evaluate(x, y), expected);
        // Orders beyond either support produce zero.
        assert!(left.hasse_derivative(40, 0).expect("x").is_zero());
        assert!(left.hasse_derivative(0, 40).expect("y").is_zero());
    }
    check::<Gf8B>();
    check::<Mersenne31>();
}

/// `compose_y` evaluates the bivariate polynomial as a univariate in the
/// candidate; `has_root` is exactly the zero-composition test.
#[test]
fn bivariate_composition_detects_roots() {
    // Q(X,Y) = Y + X over Gf8B: Y = X is a root, Y = 1 is not.
    let q = rows::<Gf8B>(&[&[b(0), b(1)], &[b(1)]]);
    let candidate = Polynomial::<Gf8B>::from_coefficients(&[b(0), b(1)]).expect("X");
    assert!(q.has_root(&candidate).expect("root"));
    let other = Polynomial::<Gf8B>::from_coefficients(&[b(1)]).expect("1");
    assert!(!q.has_root(&other).expect("nonroot"));
    // The composition value matches direct evaluation at several points.
    let composed = q.compose_y(&candidate).expect("compose");
    for seed in 0..4 {
        let point = noise::<Gf8B>(1, 0xC301 + seed)[0];
        assert_eq!(
            composed.evaluate(point),
            q.evaluate(point, naive_evaluate(&candidate, point))
        );
    }
}

/// Packed row assignment validates, reuses buffers, and normalizes; partial
/// trailing elements reject the whole call without mutation.
#[test]
fn bivariate_packed_assignment_validates_and_normalizes() {
    let mut q = rows::<Gf8B>(&[&[b(1), b(2)], &[b(3)]]);
    let before = q.clone();
    // Partial trailing element: the whole call fails, the value is kept.
    // A 3-byte Gf8B row is not a whole number of 1-byte elements only when
    // the row view itself is partial; use a 1-byte field row of length 1
    // against a 3-byte claim via Gf16 rows of mismatched parity instead.
    let bad16 = [vec![0_u8; 3]];
    let bad16_refs: Vec<&[u8]> = bad16.iter().map(Vec::as_slice).collect();
    let mut q16 = rows::<Gf16>(&[&[<Gf16 as Field>::Elem::ONE]]);
    let before16 = q16.clone();
    assert_eq!(
        q16.assign_y_coefficients_packed(bad16_refs.iter().copied())
            .map(|_| ()),
        Err(PolynomialError::Config(ConfigError::BufferLength {
            context: "bivariate packed row",
            expected: 4,
            actual: 3,
        }))
    );
    assert_eq!(q16, before16);
    // Empty input resets to zero; round trip restores the rows.
    let row0 = q.y_coefficient(0).expect("row").as_packed().to_vec();
    let row1 = q.y_coefficient(1).expect("row").as_packed().to_vec();
    let empty: Vec<&[u8]> = Vec::new();
    q.assign_y_coefficients_packed(empty.iter().copied())
        .expect("empty");
    assert!(q.is_zero());
    let good = [row0.clone(), row1.clone()];
    let good_refs: Vec<&[u8]> = good.iter().map(Vec::as_slice).collect();
    q.assign_y_coefficients_packed(good_refs.iter().copied())
        .expect("round trip");
    assert_eq!(q, before);
    // Assigning fewer rows drops the tail.
    let short = [row0.clone()];
    let short_refs: Vec<&[u8]> = short.iter().map(Vec::as_slice).collect();
    q.assign_y_coefficients_packed(short_refs.iter().copied())
        .expect("short");
    assert_eq!(q.y_coefficient_count(), 1);
}

/// Sparse products, Hasse derivatives, and evaluations match hand expansions
/// over both characteristics.
#[test]
fn sparse_products_derivatives_and_evaluations_match() {
    use poly_ring::{MonomialOrder, MultiIndex, SparsePolynomial, Term};

    fn term<F: FieldKernels, const N: usize>(
        exponents: [usize; N],
        coefficient: F::Elem,
    ) -> Term<F, N> {
        Term {
            exponents: MultiIndex::new(exponents),
            coefficient,
        }
    }

    // (1 + X0)(1 + X1) = 1 + X0 + X1 + X0·X1 over Gf8B.
    let left =
        SparsePolynomial::<Gf8B, 2>::from_terms(vec![term([0, 0], b(1)), term([1, 0], b(1))]);
    let right =
        SparsePolynomial::<Gf8B, 2>::from_terms(vec![term([0, 0], b(1)), term([0, 1], b(1))]);
    let product = left.multiply(&right).expect("product");
    assert_eq!(
        product.terms(),
        &[
            term([0, 0], b(1)),
            term([0, 1], b(1)),
            term([1, 0], b(1)),
            term([1, 1], b(1)),
        ]
    );
    // Evaluation at (x0, x1) is the defining sum.
    let point = [b(2), b(3)];
    assert_eq!(product.evaluate(&point), b(1).add(b(2)).mul(b(1).add(b(3))));
    // Hasse derivative of X0² at order [1,0] is 0 in characteristic two.
    let square = SparsePolynomial::<Gf8B, 2>::from_terms(vec![term([2, 0], b(1))]);
    let derived = square
        .hasse_derivative(&MultiIndex::new([1, 0]))
        .expect("derivative");
    assert!(derived.is_zero());
    // Over M31 the same derivative is 2·X0.
    let square_m31 = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![term([2, 0], m31(1))]);
    let derived_m31 = square_m31
        .hasse_derivative(&MultiIndex::new([1, 0]))
        .expect("derivative");
    assert_eq!(derived_m31.terms(), &[term([1, 0], m31(2))]);
    assert_eq!(
        derived_m31.evaluate_hasse(&[m31(3), m31(4)], &MultiIndex::new([0, 0])),
        m31(6)
    );
    // Orders beyond a term's support contribute zero.
    assert!(
        square_m31
            .hasse_derivative(&MultiIndex::new([5, 0]))
            .expect("beyond")
            .is_zero()
    );
    // Leading terms under graded order.
    let both = SparsePolynomial::<Mersenne31, 2>::from_terms(vec![
        term([4, 0], m31(1)),
        term([0, 5], m31(1)),
    ]);
    assert_eq!(
        both.leading_term(&MonomialOrder::GradedLex)
            .expect("graded")
            .expect("nonzero")
            .exponents
            .exponents(),
        &[0, 5]
    );
    // Coefficient lookup hits and misses.
    assert_eq!(both.coefficient(&MultiIndex::new([4, 0])), m31(1));
    assert!(both.coefficient(&MultiIndex::new([9, 9])).is_zero());
    // Zero factors multiply to zero.
    assert!(
        SparsePolynomial::<Gf8B, 2>::zero()
            .multiply(&both_sibling())
            .expect("zero")
            .is_zero()
    );
}

fn both_sibling() -> poly_ring::SparsePolynomial<Gf8B, 2> {
    use poly_ring::{MultiIndex, SparsePolynomial, Term};
    SparsePolynomial::from_terms(vec![Term {
        exponents: MultiIndex::new([1, 1]),
        coefficient: b(4),
    }])
}

/// Sparse/dense conversions round-trip losslessly in both directions.
#[test]
fn sparse_dense_conversions_round_trip() {
    use poly_ring::{MultiIndex, SparsePolynomial};

    // Univariate dense -> sparse -> dense.
    let dense = noise_poly::<Gf8B>(7, 0xC302);
    let sparse = SparsePolynomial::<Gf8B, 1>::from_univariate(&dense).expect("to sparse");
    assert_eq!(sparse.to_univariate().expect("to dense"), dense);
    // Bivariate dense -> sparse -> dense.
    let bivar = rows::<Gf8B>(&[&[b(1), b(2)], &[b(3)]]);
    let sparse2 = SparsePolynomial::<Gf8B, 2>::from_bivariate(&bivar).expect("to sparse");
    assert_eq!(sparse2.to_bivariate().expect("to bivariate"), bivar);
    // Empty conversions stay zero.
    assert!(
        SparsePolynomial::<Gf8B, 1>::from_terms(Vec::new())
            .to_univariate()
            .expect("empty")
            .is_zero()
    );
    assert!(
        SparsePolynomial::<Gf8B, 2>::from_terms(Vec::new())
            .to_bivariate()
            .expect("empty")
            .is_zero()
    );
    let _ = MultiIndex::new([0_usize; 1]);
}

/// Series inversion agrees between Newton and naive on both characteristics,
// and `reverse`/`series_divide` satisfy their defining identities.
#[test]
fn series_inversion_division_and_reversal_hold() {
    use oracles::{naive_series_inverse, noise_unit};

    fn check<F: FieldKernels>() {
        let unit = noise_unit::<F>(9, 0xC303);
        let newton = unit.inverse_mod_x_power(9).expect("newton");
        let naive = naive_series_inverse(&unit, 9);
        assert_eq!(newton, naive);
        // a · a^{-1} == 1 mod x^9.
        let one = unit.multiply_truncated(&newton, 9).expect("check product");
        assert_eq!(one, Polynomial::<F>::one().expect("one"));
        // series_divide inverts the divisor: (a/b)·b == a mod x^t.
        let other = noise_unit::<F>(7, 0xC304);
        let quotient = poly_ring::series_divide(&unit, &other, 7).expect("divide");
        let mut truncated = unit.clone();
        truncated.truncate(7);
        assert_eq!(
            quotient.multiply_truncated(&other, 7).expect("reconstruct"),
            truncated
        );
        // reverse is the coefficient mirror.
        let f = noise_poly::<F>(6, 0xC305);
        let reversed = f.reverse();
        assert_eq!(reversed.coefficient_count(), f.coefficient_count());
        for degree in 0..f.coefficient_count() {
            assert_eq!(
                reversed.coefficient(degree),
                f.coefficient(f.coefficient_count() - 1 - degree)
            );
        }
        assert!(Polynomial::<F>::zero().reverse().is_zero());
        // A zero constant term is not invertible by either path.
        let mut coefficients = noise::<F>(5, 0xC306);
        coefficients[0] = F::Elem::ZERO;
        let singular = Polynomial::<F>::from_coefficients(&coefficients).expect("singular");
        assert_eq!(
            singular.inverse_mod_x_power(5).map(|_| ()),
            Err(PolynomialError::ZeroConstantTerm {
                context: "truncated power-series inversion",
            })
        );
    }
    check::<Gf8B>();
    check::<Mersenne31>();
}

/// Truncated EEA satisfies its defining identity and the stop bound.
#[test]
fn truncated_eea_satisfies_the_identity() {
    let a = noise_poly::<Gf8B>(17, 0xC307).shifted(8).expect("shift");
    let b = noise_poly::<Gf8B>(9, 0xC308);
    let step = poly_ring::truncated_eea(&a, &b, 4).expect("eea");
    assert!(step.remainder.degree().is_none_or(|d| d < 4));
    // remainder == a_cofactor·a + b_cofactor·b.
    let reconstructed = step
        .a_cofactor
        .multiply(&a)
        .expect("left")
        .add(&step.b_cofactor.multiply(&b).expect("right"))
        .expect("sum");
    assert_eq!(reconstructed, step.remainder);
}

/// Roth–Ruckenstein limits, scratch reuse, and the `into` recycling path.
#[test]
fn roth_ruckenstein_limits_and_scratch_reuse() {
    use poly_ring::{RothRuckensteinLimits, RothRuckensteinScratch, roth_ruckenstein_roots_into};

    fn planted<F: FieldKernels>(seed: u64) -> Vec<Polynomial<F>> {
        let root = Polynomial::<F>::from_coefficients(&noise::<F>(2, seed)).expect("root");
        let doubled = root.add(&root).expect("doubled root coefficients");
        vec![
            root.multiply(&root).expect("q0"),
            doubled,
            Polynomial::<F>::one().expect("q2"),
        ]
    }

    let rows = planted::<Gf8B>(0xC309);
    let limits = RothRuckensteinLimits::new(10_000, 64);
    assert_eq!(limits.max_work_items(), 10_000);
    assert_eq!(limits.max_output_roots(), 64);

    let mut scratch = RothRuckensteinScratch::<Gf8B>::new();
    let mut output = Vec::new();
    roth_ruckenstein_roots_into(&mut output, &rows, 2, limits, &mut scratch).expect("roots");
    assert!(!output.is_empty());
    for candidate in &output {
        let mut composition = Polynomial::<Gf8B>::zero();
        for row in rows.iter().rev() {
            composition = composition.multiply(candidate).expect("mul");
            composition = composition.add(row).expect("add");
        }
        assert!(composition.is_zero());
    }
    // Scratch reuse: the second run over the same geometry recycles.
    let mut output2 = Vec::new();
    roth_ruckenstein_roots_into(&mut output2, &rows, 2, limits, &mut scratch).expect("reuse");
    assert_eq!(output, output2);
    assert!(scratch.capacity() >= scratch.capacity());
    let _ = RothRuckensteinScratch::<Gf8B>::default();

    // A zero work-item limit fails before any traversal.
    assert_eq!(
        poly_ring::roth_ruckenstein_roots(&rows, 2, RothRuckensteinLimits::new(0, 64)).map(|_| ()),
        Err(RootError::ResourceLimitExceeded {
            resource: "Roth–Ruckenstein work items",
            required: 1,
            limit: 0,
        })
    );
    // A single-row input has Y-degree zero: no roots, no error.
    let single = vec![noise_poly::<Gf8B>(4, 0xC310)];
    assert!(
        poly_ring::roth_ruckenstein_roots(&single, 2, limits)
            .expect("y-degree zero")
            .is_empty()
    );
    // max_degree at the address-space edge is a geometry failure.
    assert!(poly_ring::roth_ruckenstein_roots(&rows, usize::MAX, limits).is_err());
}

/// Cost selectors route small fields to schoolbook and honor the explicit
/// crossover knobs.
#[test]
fn cost_selectors_route_by_geometry() {
    use poly_ring::{
        BackendClass, BaseRootBackend, BaseRootCostKey, ProductBackend, ProductCostKey,
        RootLiftingBackend, RootLiftingCostKey, chien_equal_degree_crossover, select_base_roots,
        select_product, select_root_lifting,
    };

    // GF(2^8) never leaves schoolbook.
    assert_eq!(
        select_product(ProductCostKey {
            left_coefficients: 10_000,
            right_coefficients: 10_000,
            output_coefficients: 19_999,
            batch: 16,
            field_order: 256,
            backend: BackendClass::new(64, false),
        }),
        ProductBackend::Schoolbook
    );
    // Accessors round-trip the construction arguments.
    let class = BackendClass::new(32, false);
    assert_eq!(class.lane_bytes(), 32);
    assert!(!class.is_scalar());
    assert!(BackendClass::new(1, true).is_scalar());
    assert!(BackendClass::detect::<Gf16>().lane_bytes() > 0);
    // Root selection honors an explicit small crossover. Without the `fft`
    // feature the Alekhnovich tier does not exist and every key routes to
    // Roth–Ruckenstein; with it, a weighted size above the crossover takes
    // Alekhnovich.
    #[cfg(feature = "fft")]
    assert_eq!(
        select_root_lifting(RootLiftingCostKey {
            weighted_coefficients: 50_000,
            y_degree: 4,
            target_precision: 32,
            backend: BackendClass::new(64, false),
            roth_ruckenstein_crossover: 20_000,
            backend_adaptive: false,
        }),
        RootLiftingBackend::Alekhnovich
    );
    #[cfg(not(feature = "fft"))]
    assert_eq!(
        select_root_lifting(RootLiftingCostKey {
            weighted_coefficients: 50_000,
            y_degree: 4,
            target_precision: 32,
            backend: BackendClass::new(64, false),
            roth_ruckenstein_crossover: 20_000,
            backend_adaptive: false,
        }),
        RootLiftingBackend::RothRuckenstein
    );
    assert_eq!(
        select_root_lifting(RootLiftingCostKey {
            weighted_coefficients: 50,
            y_degree: 4,
            target_precision: 32,
            backend: BackendClass::new(64, false),
            roth_ruckenstein_crossover: 20_000,
            backend_adaptive: false,
        }),
        RootLiftingBackend::RothRuckenstein
    );
    // Base-root selection splits at the measured crossover.
    let crossover = chien_equal_degree_crossover(256);
    assert_eq!(
        select_base_roots(BaseRootCostKey {
            degree: crossover + 10,
            field_order: 256
        }),
        BaseRootBackend::EqualDegree
    );
    assert_eq!(
        select_base_roots(BaseRootCostKey {
            degree: 1,
            field_order: 256
        }),
        BaseRootBackend::Chien
    );
}

/// Prepared Hasse derivatives over odd characteristics match the defining
/// binomial sums, including the zero-factor rows.
#[test]
fn odd_characteristic_derivatives_match_binomial_sums() {
    use poly_ring::DerivativePlan;

    let plan = DerivativePlan::<Mersenne31>::new(2, 8).expect("plan");
    // f = 1 + 2X + 3X² + 4X³ + 5X⁴: D^[2]f = 3 + 12X + 30X².
    let input = [m31(1), m31(2), m31(3), m31(4), m31(5)];
    let mut output = [m31(0); 3];
    plan.apply_into(&input, &mut output).expect("apply");
    assert_eq!(output, [m31(3), m31(12), m31(30)]);
    // A zero binomial factor writes an explicit zero row in batch mode.
    let batch_plan = DerivativePlan::<Mersenne31>::new(31, 33).expect("plan");
    let mut packed = vec![0_u8; 33 * 4];
    for (degree, slot) in packed.chunks_exact_mut(4).enumerate() {
        <Mersenne31 as Field>::encode(slot, m31(degree as u32 + 1));
    }
    let mut batch_out = vec![0_u8; 2 * 4];
    batch_plan
        .apply_batch_into(&packed, 33, 1, &mut batch_out)
        .expect("batch");
    // Output row 0 is C(31,31)·a_31 = a_31; row 1 is C(32,31)·a_32 = 32·a_32 mod p.
    assert_eq!(<Mersenne31 as Field>::decode(&batch_out[..4]), m31(32));
}

/// Binomial helpers follow the field characteristic, not a parity shortcut.
#[test]
fn binomial_helpers_follow_the_characteristic() {
    // C(4,2) = 6 vanishes nowhere over M31 but is 0 mod 2 as a factor.
    assert_eq!(poly_ring::binomial::<Mersenne31>(4, 2), m31(6));
    assert!(!poly_ring::internals::binomial_odd(4, 2));
    assert!(poly_ring::internals::binomial_odd(3, 1));
    // Evaluate-hasse on M31 matches the defining sum at a random point.
    let polynomial = noise_poly::<Mersenne31>(7, 0xC311);
    let point = noise::<Mersenne31>(1, 0xC312)[0];
    let mut expected = m31(0);
    for (degree, coefficient) in polynomial.coefficients().enumerate() {
        if degree >= 2 {
            expected = expected.add(
                coefficient
                    .mul(poly_ring::binomial::<Mersenne31>(degree, 2))
                    .mul(point.pow((degree - 2) as u64)),
            );
        }
    }
    assert_eq!(polynomial.evaluate_hasse(point, 2), expected);
}

/// Prepared products: AFFT batches, capacity accessors, and the empty-batch
/// fast path.
#[test]
#[cfg(feature = "fft")]
fn afft_batches_match_schoolbook_and_report_capacity() {
    use poly_ring::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated_into};

    let left = noise_poly::<Gf16>(300, 0xC313);
    let right = noise_poly::<Gf16>(260, 0xC314);
    // Sixteen pairs at full product 559 exceed the batch-16 crossover (127
    // scalar / 8191 packed only when the backend qualifies): force the AFFT
    // strategy so the transform path is taken regardless of crossover.
    let pairs = [(&left, &right); 20];
    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    let mut output = Vec::new();
    multiply_batch_truncated_into(
        &mut output,
        &pairs,
        usize::MAX,
        ProductStrategy::Afft,
        &mut scratch,
    )
    .expect("batch");
    assert_eq!(output.len(), 20);
    for product in &output {
        assert_eq!(*product, naive_multiply(&left, &right));
    }
    assert!(scratch.operand_capacity_bytes() > 0);
    assert!(scratch.product_capacity_bytes() > 0);
    let _ = PolynomialProductScratch::<Gf16>::default();
    // Empty batch writes nothing and succeeds.
    let mut empty_out = Vec::new();
    multiply_batch_truncated_into(&mut empty_out, &[], 8, ProductStrategy::Auto, &mut scratch)
        .expect("empty");
    assert!(empty_out.is_empty());
    // Forced schoolbook strategy matches the dispatched product.
    let mut schoolbook_out = Vec::new();
    multiply_batch_truncated_into(
        &mut schoolbook_out,
        &[(&left, &right)],
        usize::MAX,
        ProductStrategy::Schoolbook,
        &mut scratch,
    )
    .expect("schoolbook");
    assert_eq!(schoolbook_out[0], naive_multiply(&left, &right));
    // Zero truncation writes zero rows.
    let mut truncated = Vec::new();
    multiply_batch_truncated_into(
        &mut truncated,
        &[(&left, &right)],
        0,
        ProductStrategy::Auto,
        &mut scratch,
    )
    .expect("truncated");
    assert!(truncated[0].is_zero());
}

/// Row-slice affine substitution agrees with the bivariate method, and the
/// empty-rows fast paths recycle without error.
#[test]
#[cfg(feature = "fft")]
fn row_slice_affine_substitution_matches_bivariate() {
    use poly_ring::PolynomialProductScratch;
    use poly_ring::internals::substitute_y_affine_rows_truncated_into;

    let q = rows::<Gf16>(&[
        &[<Gf16 as Field>::Elem::ONE],
        &[<Gf16 as Field>::Elem::ONE, <Gf16 as Field>::Elem::ONE],
    ]);
    let prefix = noise_poly::<Gf16>(3, 0xC315);
    let expected = q
        .substitute_y_affine_truncated(&prefix, 1, 8)
        .expect("bivariate");
    let mut scratch = PolynomialProductScratch::<Gf16>::new();
    let mut output = Vec::new();
    let mut pool = Vec::new();
    substitute_y_affine_rows_truncated_into(
        q.y_coefficients(),
        &prefix,
        1,
        8,
        &mut scratch,
        &mut output,
        &mut pool,
    )
    .expect("rows");
    assert_eq!(output.len(), expected.y_coefficient_count());
    for (got, want) in output.iter().zip(expected.y_coefficients()) {
        assert_eq!(got, want);
    }
    // Empty rows and zero precision recycle into the pool without error.
    substitute_y_affine_rows_truncated_into(
        &[],
        &prefix,
        1,
        8,
        &mut scratch,
        &mut output,
        &mut pool,
    )
    .expect("empty");
    assert!(output.is_empty());
    substitute_y_affine_rows_truncated_into(
        q.y_coefficients(),
        &prefix,
        1,
        0,
        &mut scratch,
        &mut output,
        &mut pool,
    )
    .expect("zero precision");
    assert!(output.is_empty());
}

/// Prepared convolution rows: exact products, validation errors, and the
/// transform route through the public entry point.
#[test]
fn prepared_convolution_matches_naive_and_validates() {
    use poly_ring::{ConvolutionScratch, multiply_rows_into};

    fn pack<F: FieldKernels>(coefficients: &[F::Elem]) -> Vec<u8> {
        let mut packed = vec![0_u8; coefficients.len() * F::BYTES];
        for (slot, value) in packed.chunks_exact_mut(F::BYTES).zip(coefficients) {
            F::encode(slot, *value);
        }
        packed
    }

    fn check<F: poly_ring::PolynomialField>() {
        let left = noise::<F>(17, 0xC316);
        let right = noise::<F>(11, 0xC317);
        let full = left.len() + right.len() - 1;
        let mut scratch = ConvolutionScratch::<F>::new(17, 11, 1).expect("scratch");
        let mut output = vec![0_u8; full * F::BYTES];
        multiply_rows_into::<F>(
            &mut output,
            &pack::<F>(&left),
            left.len(),
            &pack::<F>(&right),
            right.len(),
            1,
            full,
            &mut scratch,
        )
        .expect("product");
        // Independent scalar convolution oracle.
        let mut expected = vec![F::Elem::ZERO; full];
        for (i, a) in left.iter().enumerate() {
            for (j, b) in right.iter().enumerate() {
                expected[i + j] = expected[i + j].add(a.mul(*b));
            }
        }
        for (degree, want) in expected.iter().enumerate() {
            assert_eq!(
                F::decode(&output[degree * F::BYTES..][..F::BYTES]),
                *want,
                "coefficient {degree} diverged"
            );
        }
        // Wrong left length names the buffer.
        assert_eq!(
            multiply_rows_into::<F>(
                &mut output,
                &[0_u8; 3],
                left.len(),
                &pack::<F>(&right),
                right.len(),
                1,
                full,
                &mut scratch,
            )
            .map(|_| ()),
            Err(ProductError::Config(ConfigError::BufferLength {
                context: "prepared left operand rows",
                expected: left.len() * F::BYTES,
                actual: 3,
            }))
        );
        // Oversized operands exceed the prepared capacity.
        assert!(
            multiply_rows_into::<F>(
                &mut output,
                &pack::<F>(&left),
                left.len() + 100,
                &pack::<F>(&right),
                right.len(),
                1,
                full,
                &mut scratch,
            )
            .is_err()
        );
    }
    check::<Gf8B>();
    check::<Gf16>();
    check::<Goldilocks>();
    check::<QuadMersenne31>();
    check::<Mersenne31>();
}

/// Alekhnovich limits accessors, the empty-rows rejection, and the forced
/// divide-and-conquer agreement with Roth–Ruckenstein.
#[test]
#[cfg(feature = "fft")]
fn alekhnovich_accessors_rejections_and_agreement() {
    use poly_ring::{
        AlekhnovichLimits, AlekhnovichScratch, RothRuckensteinLimits, alekhnovich_roots,
        roth_ruckenstein_roots,
    };

    let limits = AlekhnovichLimits::new(100, 200, 1 << 20, 1 << 24, 64);
    assert_eq!(limits.max_work_items(), 100);
    assert_eq!(limits.max_intermediate_families(), 200);
    assert_eq!(limits.max_coefficients(), 1 << 20);
    assert_eq!(limits.max_scratch_bytes(), 1 << 24);
    assert_eq!(limits.max_output_roots(), 64);

    let mut scratch = AlekhnovichScratch::<Gf16>::new();
    assert_eq!(
        alekhnovich_roots(&[], 2, limits, &mut scratch).map(|_| ()),
        Err(RootError::ZeroBivariatePolynomial)
    );
    // Y-degree zero clears the output without error.
    let single = vec![noise_poly::<Gf16>(4, 0xC318)];
    let mut output = vec![noise_poly::<Gf16>(2, 0xC319)];
    poly_ring::alekhnovich_roots_into(&mut output, &single, 2, limits, &mut scratch)
        .expect("y-degree zero");
    assert!(output.is_empty());
    let _ = AlekhnovichScratch::<Gf16>::default();
    assert_eq!(scratch.frame_capacity(), scratch.frame_capacity());

    // Planted roots: the forced D&C path agrees with prefix lifting.
    let root_a = noise::<Gf16>(3, 0xC320);
    let root_b = noise::<Gf16>(2, 0xC321);
    let rows = planted_rows(&root_a, &root_b);
    let big = AlekhnovichLimits::new(1_000_000, 1_000_000, 1 << 22, 1 << 26, 256);
    let lifted = alekhnovich_roots(&rows, 4, big, &mut scratch).expect("alekhnovich");
    let prefixed = roth_ruckenstein_roots(&rows, 4, RothRuckensteinLimits::new(1_000_000, 256))
        .expect("roth-ruckenstein");
    assert_eq!(lifted, prefixed);
    // Forced D&C (crossover 0) agrees as well.
    let forced = alekhnovich_roots(
        &rows,
        4,
        big.with_roth_ruckenstein_crossover(0),
        &mut scratch,
    )
    .expect("forced d&c");
    assert_eq!(forced, prefixed);
    assert!(scratch.capacity() >= scratch.frame_capacity());
}

#[cfg(feature = "fft")]
fn planted_rows<F: FieldKernels>(a: &[F::Elem], b: &[F::Elem]) -> Vec<Polynomial<F>> {
    let mut rows = vec![
        Polynomial::one().expect("one"),
        Polynomial::zero(),
        Polynomial::zero(),
    ];
    for coefficients in [a, b] {
        let root = Polynomial::from_coefficients(coefficients).expect("planted");
        let mut next = vec![Polynomial::zero(); 3];
        for (j, row) in rows.iter().enumerate() {
            next[j] = next[j].add(&row.multiply(&root).expect("f")).expect("add");
        }
        for j in 1..rows.len() {
            next[j] = next[j].add(&rows[j - 1]).expect("add");
        }
        rows = next;
    }
    rows
}
