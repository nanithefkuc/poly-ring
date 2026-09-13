//! Greatest common divisors, Bézout cofactors, and the truncated Euclidean
//! algorithm.

use fgf::field::Elem as _;
use fgf::kernel::FieldKernels;

use crate::error::PolynomialError;

use super::dense::Polynomial;

/// The Bézout relation `gcd = a_cofactor * a + b_cofactor * b`.
///
/// `gcd` is monic. For nonzero inputs the cofactors are the minimal-degree
/// ones the extended Euclidean algorithm produces, so
/// `deg(a_cofactor) <= deg(b) - deg(gcd)` and
/// `deg(b_cofactor) <= deg(a) - deg(gcd)`; a zero cofactor appears only when
/// its polynomial divides the other exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BezoutRelation<F: FieldKernels> {
    /// The monic greatest common divisor.
    pub gcd: Polynomial<F>,
    /// The cofactor of the first operand.
    pub a_cofactor: Polynomial<F>,
    /// The cofactor of the second operand.
    pub b_cofactor: Polynomial<F>,
}

/// One stopped step of the extended Euclidean algorithm.
///
/// `remainder = a_cofactor * a + b_cofactor * b` with
/// `deg(remainder) < stop_degree`, where `stop_degree` is the caller's
/// truncation bound. This is the Padé / key-equation primitive: run on
/// `(x^t, S(x))` with `stop_degree = t/2`, the `b_cofactor` is the connection
/// polynomial and the `remainder` the discrepancy (equivalent to
/// Berlekamp–Massey on the same sequence).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TruncatedEea<F: FieldKernels> {
    /// The remainder at the stop, degree below `stop_degree`.
    pub remainder: Polynomial<F>,
    /// The cofactor of the first operand at the stop.
    pub a_cofactor: Polynomial<F>,
    /// The cofactor of the second operand at the stop (the connection
    /// polynomial when the first operand is a power of `x`).
    pub b_cofactor: Polynomial<F>,
}

impl<F: FieldKernels> Polynomial<F> {
    /// Return the monic greatest common divisor.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::DivisionByZero`] never (the loop guards
    /// it) and [`PolynomialError::Config`] when an intermediate division
    /// buffer cannot be reserved.
    pub fn gcd(&self, other: &Self) -> Result<Self, PolynomialError> {
        let mut left = self.clone();
        let mut right = other.clone();
        while !right.is_zero() {
            let (_, remainder) = left.div_rem(&right)?;
            left = right;
            right = remainder;
        }
        Ok(left.monic())
    }

    /// Return the monic gcd together with its Bézout cofactors.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when an intermediate division or
    /// product buffer cannot be reserved.
    ///
    /// # Panics
    ///
    /// The internal leading-coefficient expectation holds for every nonzero
    /// remainder.
    pub fn gcd_ext(&self, other: &Self) -> Result<BezoutRelation<F>, PolynomialError> {
        // r_{-1} = a, r_0 = b; cofactors s (of a) and t (of b).
        let mut r_old = self.clone();
        let mut r = other.clone();
        let mut s_old = Self::one()?;
        let mut s = Self::zero();
        let mut t_old = Self::zero();
        let mut t = Self::one()?;

        while !r.is_zero() {
            let (quotient, remainder) = r_old.div_rem(&r)?;
            r_old = r;
            r = remainder;
            // Characteristic two: subtraction is addition, so the cofactor
            // updates are plain sums of scaled terms.
            let s_next = s_old.add(&quotient.multiply(&s)?)?;
            let t_next = t_old.add(&quotient.multiply(&t)?)?;
            s_old = s;
            s = s_next;
            t_old = t;
            t = t_next;
        }

        if r_old.is_zero() {
            // gcd(0, 0) with cofactors that keep the identity at zero.
            return Ok(BezoutRelation {
                gcd: Self::zero(),
                a_cofactor: Self::zero(),
                b_cofactor: Self::zero(),
            });
        }
        let inverse = r_old
            .leading_coefficient()
            .expect("nonzero remainder has a leading coefficient")
            .inv();
        Ok(BezoutRelation {
            gcd: r_old.scaled(inverse),
            a_cofactor: s_old.scaled(inverse),
            b_cofactor: t_old.scaled(inverse),
        })
    }
}

/// Run the extended Euclidean algorithm, stopping at the first remainder of
/// degree below `stop_degree`.
///
/// The returned triple satisfies `remainder = a_cofactor * a + b_cofactor *
/// b` with `deg(remainder) < stop_degree`; when both inputs already qualify
/// through `b`, the result is `(b, 0, 1)`. When the first operand is `x^t`
/// and the second a power series `S`, the identity reads
/// `remainder ≡ b_cofactor * S (mod x^t)` — the Padé approximant relation
/// that solves the key equation, equivalent to Berlekamp–Massey on the
/// sequence of `S`'s coefficients.
///
/// # Errors
///
/// Returns [`PolynomialError::Config`] when an intermediate division or
/// product buffer cannot be reserved.
pub fn truncated_eea<F: FieldKernels>(
    dividend: &Polynomial<F>,
    divisor: &Polynomial<F>,
    stop_degree: usize,
) -> Result<TruncatedEea<F>, PolynomialError> {
    let mut r_old = dividend.clone();
    let mut r = divisor.clone();
    let mut u_old = Polynomial::<F>::one()?;
    let mut u = Polynomial::<F>::zero();
    let mut v_old = Polynomial::<F>::zero();
    let mut v = Polynomial::<F>::one()?;

    while r.degree().is_some_and(|degree| degree >= stop_degree) {
        let (quotient, remainder) = r_old.div_rem(&r)?;
        r_old = r;
        r = remainder;
        let u_next = u_old.add(&quotient.multiply(&u)?)?;
        let v_next = v_old.add(&quotient.multiply(&v)?)?;
        u_old = u;
        u = u_next;
        v_old = v;
        v = v_next;
    }

    Ok(TruncatedEea {
        remainder: r,
        a_cofactor: u,
        b_cofactor: v,
    })
}
