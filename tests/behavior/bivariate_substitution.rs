//! Bivariate substitution families: scalar, affine-truncated, and batched.
//!
//! The agreement suite checks one substitution per characteristic; every
//! test here takes a branch it never reaches — vanishing binomial factors,
//! zero scales, overflowing or out-of-range shifts, fully truncated rows,
//! packed-row validation, divide-by-`X` propagation, and the fast batched
//! form — each against exact rows or an independent evaluation check.

#[cfg(feature = "fft")]
use fgf::field::Elem;
use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::{BivariatePolynomial, ConfigError, Polynomial, PolynomialError};

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

fn b(value: u8) -> gf8b::Elem {
    gf8b::Elem::from_raw(value)
}

fn rows<F: FieldKernels>(rows: &[&[F::Elem]]) -> BivariatePolynomial<F> {
    BivariatePolynomial::from_y_coefficients(
        rows.iter()
            .map(|row| Polynomial::from_coefficients(row).expect("row"))
            .collect(),
    )
}

/// `Y = c + XZ` distributes through the binomial expansion: `Y^2` with
/// `c = 1` over M31 yields rows `1`, `2X`, `X^2` scaled by the row.
#[test]
fn linear_substitution_distributes_binomial_rows() {
    let q = rows::<Mersenne31>(&[&[], &[], &[m31(3)]]);
    let substituted = q.substitute_y_linear(m31(1)).expect("substitution");
    assert_eq!(substituted.y_coefficient_count(), 3);
    assert_eq!(
        substituted.y_coefficient(0).expect("row").coefficient(0),
        m31(3)
    );
    // Row 1 carries C(2,1)·c·q_2 shifted by one: coefficient 6 at degree 1.
    assert_eq!(
        substituted.y_coefficient(1).expect("row").coefficient(1),
        m31(6)
    );
    assert_eq!(
        substituted.y_coefficient(2).expect("row").coefficient(2),
        m31(3)
    );
    // Evaluation commutes with the substitution at every probe point.
    let x = m31(5);
    let z = m31(7);
    assert_eq!(
        substituted.evaluate(x, z),
        q.evaluate(x, m31(1).add(x.mul(z)))
    );
}

/// In characteristic two the even binomial factors vanish: `Y^2` with
/// `c = 1` keeps only rows 0 and 2, and the middle row is the stored zero.
#[test]
fn binary_linear_substitution_drops_even_factors() {
    let q = rows::<Gf8B>(&[&[], &[], &[b(1), b(1)]]);
    let substituted = q.substitute_y_linear(b(1)).expect("substitution");
    assert_eq!(substituted.y_coefficient_count(), 3);
    assert!(substituted.y_coefficient(1).expect("row").is_zero());
    let x = b(0x1b);
    let z = b(0x2d);
    assert_eq!(
        substituted.evaluate(x, z),
        q.evaluate(x, b(1).add(x.mul(z)))
    );
}

/// A zero `Y^1` row is skipped, never convolved: only the `Y^2` row feeds
/// the output, and evaluation still commutes.
#[test]
fn linear_substitution_skips_zero_rows() {
    let q = rows::<Mersenne31>(&[&[m31(4)], &[], &[m31(0), m31(1)]]);
    let substituted = q.substitute_y_linear(m31(2)).expect("substitution");
    assert_eq!(substituted.y_coefficient_count(), 3);
    let x = m31(6);
    let z = m31(8);
    assert_eq!(
        substituted.evaluate(x, z),
        q.evaluate(x, m31(2).add(x.mul(z)))
    );
}

/// The affine truncation keeps every product below the precision: with a
/// generous bound the affine form agrees with the linear substitution at
/// `prefix = c` and `tail_degree = 1`.
#[test]
fn affine_truncation_matches_linear_at_generous_precision() {
    let q = rows::<Mersenne31>(&[&[m31(1), m31(2)], &[m31(3)], &[m31(0), m31(0), m31(1)]]);
    let prefix = Polynomial::from_coefficients(&[m31(2)]).expect("prefix");
    let affine = q
        .substitute_y_affine_truncated(&prefix, 1, 32)
        .expect("affine");
    let linear = q.substitute_y_linear(m31(2)).expect("linear");
    assert_eq!(affine.truncated_x(32), linear.truncated_x(32));
    assert_eq!(affine, linear);
}

/// A zero tail shift collapses every row onto the prefix powers: `Y =
/// prefix + Z` with `tail_degree = 0` still commutes with evaluation.
#[test]
fn affine_zero_tail_shift_still_commutes() {
    let q = rows::<Mersenne31>(&[&[m31(1)], &[m31(2)], &[m31(3)]]);
    let prefix = Polynomial::from_coefficients(&[m31(4), m31(5)]).expect("prefix");
    let affine = q
        .substitute_y_affine_truncated(&prefix, 0, 16)
        .expect("affine");
    let x = m31(6);
    let z = m31(7);
    // Y = prefix(x) + x^0·z.
    let y = prefix.evaluate(x).add(z);
    assert_eq!(affine.evaluate(x, z), q.evaluate(x, y));
}

/// An overflowing tail product drops its rows: with `tail_degree` at the
/// address-space edge and a `Y^2` row present, the `target_y = 2` shift
/// overflows the multiplication itself.
#[test]
fn affine_substitution_drops_overflowing_products() {
    let q = rows::<Mersenne31>(&[&[m31(1)], &[m31(2)], &[m31(3)]]);
    let prefix = Polynomial::from_coefficients(&[m31(1), m31(1)]).expect("prefix");
    let result = q
        .substitute_y_affine_truncated(&prefix, usize::MAX, 8)
        .expect("substitution");
    // target_y = 0 survives unshifted; target_y = 1 shifts past precision;
    // target_y = 2 overflows the shift product. Only row 0 remains.
    assert_eq!(result.y_coefficient_count(), 1);
    let x = m31(6);
    let z = m31(7);
    assert_eq!(
        result.evaluate(x, z),
        result.y_coefficient(0).expect("row").evaluate(x)
    );
}

/// The fast form drops overflowing shifts identically to the scalar one.
#[cfg(feature = "fft")]
#[test]
fn fast_substitution_drops_overflowing_products() {
    use butterfly_fft::kernel::ButterflyKernels;
    use fgf::Gf16;
    use poly_ring::PolynomialProductScratch;

    fn check<F>()
    where
        F: FieldKernels + ButterflyKernels,
    {
        let q = BivariatePolynomial::<F>::from_y_coefficients(vec![
            Polynomial::from_coefficients(&[F::Elem::ONE]).expect("row"),
            Polynomial::from_coefficients(&[F::GENERATOR]).expect("row"),
            Polynomial::from_coefficients(&[F::Elem::ONE]).expect("row"),
        ]);
        let prefix =
            Polynomial::<F>::from_coefficients(&[F::Elem::ONE, F::GENERATOR]).expect("prefix");
        let mut scratch = PolynomialProductScratch::<F>::new();
        let scalar = q
            .substitute_y_affine_truncated(&prefix, usize::MAX, 8)
            .expect("scalar");
        let batched = q
            .substitute_y_affine_truncated_fast(&prefix, usize::MAX, 8, &mut scratch)
            .expect("batched");
        assert_eq!(scalar, batched);
        assert_eq!(scalar.y_coefficient_count(), 1);
    }
    check::<Gf8B>();
    check::<Gf16>();
}

/// Truncation drops high `X` rows coefficient-wise: a one-row precision
/// keeps only constants, and the affine result stays canonical.
#[test]
fn affine_truncation_to_one_row_keeps_constants() {
    let q = rows::<Mersenne31>(&[&[m31(1), m31(9)], &[m31(2), m31(8)]]);
    let prefix = Polynomial::from_coefficients(&[m31(3), m31(4)]).expect("prefix");
    let affine = q
        .substitute_y_affine_truncated(&prefix, 0, 1)
        .expect("affine");
    assert_eq!(affine.y_coefficient_count(), 2);
    // Row 0 is Q_0(0) + Q_1(0)·prefix(0) = 1 + 2·3 = 7; row 1 keeps the
    // unshifted `Y^1` constant Q_1(0) = 2.
    assert_eq!(affine.y_coefficient(0).expect("row").coefficient(0), m31(7));
    assert_eq!(affine.y_coefficient(1).expect("row").coefficient(0), m31(2));
}

/// Packed-row assignment validates every row before mutating: a partial
/// trailing element rejects the whole call and leaves the polynomial
/// untouched.
#[test]
fn packed_rows_reject_partial_elements_without_mutation() {
    let mut q = rows::<Mersenne31>(&[&[m31(1)]]);
    let before = q.clone();
    let partial = [0x01_u8, 0x02];
    let rows_vec = [partial.as_slice()].into_iter();
    assert_eq!(
        q.assign_y_coefficients_packed(rows_vec).map(|_| ()),
        Err(PolynomialError::Config(ConfigError::BufferLength {
            context: "bivariate packed row",
            expected: 4,
            actual: 2,
        }))
    );
    assert_eq!(q, before);
    // Empty input resets to zero; a successful replacement normalizes.
    q.assign_y_coefficients_packed(core::iter::empty::<&[u8]>())
        .expect("empty");
    assert!(q.is_zero());
}

/// Packed rows reuse buffers and drop surplus rows: shrinking from two
/// rows to one leaves one normalized row.
#[test]
fn packed_rows_reuse_buffers_and_drop_surplus() {
    let mut q = rows::<Gf8B>(&[&[b(1), b(2)], &[b(3)]]);
    let replacement = Polynomial::<Gf8B>::from_coefficients(&[b(9)]).expect("row");
    q.assign_y_coefficients_packed([replacement.as_packed()].into_iter())
        .expect("assign");
    assert_eq!(q.y_coefficient_count(), 1);
    assert_eq!(q.y_coefficient(0).expect("row"), &replacement);
}

/// `divide_by_x_power` propagates a non-exact row division instead of
/// inventing a quotient.
#[test]
fn divide_by_x_power_propagates_non_exact_rows() {
    let q = rows::<Mersenne31>(&[&[m31(1)], &[m31(0), m31(1)]]);
    assert_eq!(
        q.divide_by_x_power(1).map(|_| ()),
        Err(PolynomialError::NonExactDivision)
    );
    let even = rows::<Mersenne31>(&[&[m31(0), m31(5)], &[m31(0), m31(0), m31(7)]]);
    let divided = even.divide_by_x_power(1).expect("exact");
    assert_eq!(
        divided.y_coefficient(0).expect("row").coefficient(0),
        m31(5)
    );
    assert_eq!(
        divided.y_coefficient(1).expect("row").coefficient(1),
        m31(7)
    );
}

/// `has_root` is the composition test: `Y - f` vanishes on its own factor
/// over the prime field, where the sign is observable.
#[test]
fn has_root_detects_own_linear_factor_over_prime_field() {
    let f = Polynomial::<Mersenne31>::from_coefficients(&[m31(2), m31(3)]).expect("f");
    // Q = Y - f: rows [-f, 1].
    let negated = f.scaled(m31((<Mersenne31 as Field>::ORDER - 1) as u32));
    let q =
        BivariatePolynomial::from_y_coefficients(vec![negated, Polynomial::one().expect("one")]);
    assert!(q.has_root(&f).expect("root"));
    let g = Polynomial::<Mersenne31>::from_coefficients(&[m31(5)]).expect("g");
    assert!(!q.has_root(&g).expect("nonroot"));
}

/// `add_scaled_x_shifted_assign` widens to the larger row count and shifts
/// every source row in `X` before accumulating.
#[test]
fn scaled_shifted_add_widens_and_shifts_rows() {
    let mut left = rows::<Mersenne31>(&[&[m31(1)]]);
    let right = rows::<Mersenne31>(&[&[m31(2)], &[m31(3)]]);
    left.add_scaled_x_shifted_assign(m31(1), &right, 2)
        .expect("add");
    assert_eq!(left.y_coefficient_count(), 2);
    // Row 0 is 1 + X^2·2; row 1 is X^2·3.
    assert_eq!(left.y_coefficient(0).expect("row").coefficient(2), m31(2));
    assert_eq!(left.y_coefficient(1).expect("row").coefficient(2), m31(3));
    let x = m31(6);
    let y = m31(7);
    assert_eq!(
        left.evaluate(x, y),
        m31(1)
            .add(m31(2).mul(x.pow(2)))
            .add(m31(3).mul(x.pow(2)).mul(y))
    );
}

/// The fast batched substitution agrees with the scalar one over the
/// binary fields it is bounded to.
#[cfg(feature = "fft")]
#[test]
fn fast_substitution_matches_scalar_over_binary_fields() {
    use butterfly_fft::kernel::ButterflyKernels;
    use fgf::Gf16;
    use poly_ring::PolynomialProductScratch;

    fn check<F>()
    where
        F: FieldKernels + ButterflyKernels,
    {
        let q = BivariatePolynomial::<F>::from_y_coefficients(vec![
            Polynomial::from_coefficients(&[F::GENERATOR, F::Elem::ONE]).expect("row"),
            Polynomial::from_coefficients(&[F::GENERATOR.pow(3)]).expect("row"),
            Polynomial::from_coefficients(&[F::Elem::ONE]).expect("row"),
        ]);
        let prefix =
            Polynomial::<F>::from_coefficients(&[F::GENERATOR.pow(2), F::GENERATOR.pow(5)])
                .expect("prefix");
        let mut scratch = PolynomialProductScratch::<F>::new();
        for tail_degree in [0_usize, 1, 2] {
            for coefficient_count in [3_usize, 5, 9] {
                let scalar = q
                    .substitute_y_affine_truncated(&prefix, tail_degree, coefficient_count)
                    .expect("scalar");
                let batched = q
                    .substitute_y_affine_truncated_fast(
                        &prefix,
                        tail_degree,
                        coefficient_count,
                        &mut scratch,
                    )
                    .expect("batched");
                assert_eq!(
                    scalar, batched,
                    "tail {tail_degree} prec {coefficient_count}"
                );
            }
        }
    }
    check::<Gf8B>();
    check::<Gf16>();
}

/// The fast form on degenerate inputs: zero and zero-precision return zero
/// without touching the scratch.
#[cfg(feature = "fft")]
#[test]
fn fast_substitution_degenerates_to_zero() {
    use poly_ring::PolynomialProductScratch;

    let mut scratch = PolynomialProductScratch::<Gf8B>::new();
    let prefix = Polynomial::<Gf8B>::from_coefficients(&[b(1)]).expect("prefix");
    assert!(
        BivariatePolynomial::<Gf8B>::zero()
            .substitute_y_affine_truncated_fast(&prefix, 1, 4, &mut scratch)
            .expect("zero")
            .is_zero()
    );
    let q = rows::<Gf8B>(&[&[b(1)], &[b(2)]]);
    assert!(
        q.substitute_y_affine_truncated_fast(&prefix, 1, 0, &mut scratch)
            .expect("empty precision")
            .is_zero()
    );
}
