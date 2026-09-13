//! Monomial structure: exponents, terms, and monomial orders.
//!
//! This module owns the exponent vocabulary of the ring's multivariate
//! surfaces — [`MultiIndex`] addresses a monomial in `N` variables, [`Term`]
//! pairs it with a field coefficient, and [`MonomialOrder`] ranks monomials —
//! plus the crate-private integer embedding shared by the binomial
//! computation and the prepared Hasse factor recurrences.

use core::cmp::Ordering;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;

use crate::error::ConfigError;

/// The exponent vector of one monomial: `X0^e0 · … · Xn^{en−1}`.
///
/// The arity is a const generic, so a `MultiIndex<3>` never mixes with a
/// `MultiIndex<2>`; `N = 0` is the constant ring, not an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MultiIndex<const N: usize>([usize; N]);

impl<const N: usize> MultiIndex<N> {
    /// The exponent vector `exponents`, variable 0 first.
    #[must_use]
    pub const fn new(exponents: [usize; N]) -> Self {
        Self(exponents)
    }

    /// The stored exponents, variable 0 first.
    #[must_use]
    pub const fn exponents(&self) -> &[usize; N] {
        &self.0
    }

    /// The componentwise sum, rejecting overflow instead of wrapping.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::GeometryOverflow`] when any component sum
    /// exceeds `usize`.
    pub fn checked_add(&self, rhs: &Self) -> Result<Self, ConfigError> {
        let mut summed = [0_usize; N];
        for (index, (low, high)) in self.0.iter().zip(rhs.0.iter()).enumerate() {
            summed[index] = low
                .checked_add(*high)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "multivariate exponent sum",
                })?;
        }
        Ok(Self(summed))
    }

    /// The sum of all exponents: the total degree of the monomial.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::GeometryOverflow`] when the sum exceeds
    /// `usize`.
    pub fn total_degree(&self) -> Result<usize, ConfigError> {
        let mut total = 0_usize;
        for exponent in &self.0 {
            total = total
                .checked_add(*exponent)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "multivariate total degree",
                })?;
        }
        Ok(total)
    }
}

/// One monomial term: an exponent vector and its field coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Term<F: FieldKernels, const N: usize> {
    /// Exponent of each variable, variable 0 first.
    pub exponents: MultiIndex<N>,
    /// Field coefficient. Zero terms never survive a constructor.
    pub coefficient: F::Elem,
}

/// A monomial ranking.
///
/// Storage order inside [`crate::poly::SparsePolynomial`] never depends on
/// this choice; orders matter only where a caller asks for a leading term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonomialOrder<const N: usize> {
    /// Pure lexicographic from variable 0: `X0` dominates everything.
    Lex,
    /// Total degree first, then lexicographic from variable 0.
    GradedLex,
    /// The dot product of these weights with the exponents first, then the
    /// exponent array from variable `N − 1` down to 0 — so weights
    /// `[1, y_weight]` rank exactly like [`crate::poly::WeightedTerm`], the
    /// larger `Y` degree winning ties. Zero weights are allowed.
    Weighted([usize; N]),
}

impl<const N: usize> MonomialOrder<N> {
    /// Rank `a` against `b`: [`Ordering::Greater`] means `a` is the larger
    /// monomial under this order.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::GeometryOverflow`] when a weighted degree or a
    /// total degree cannot be represented.
    pub(crate) fn compare(
        &self,
        a: &MultiIndex<N>,
        b: &MultiIndex<N>,
    ) -> Result<Ordering, ConfigError> {
        match self {
            Self::Lex => Ok(a.exponents().cmp(b.exponents())),
            Self::GradedLex => {
                let a_degree = a.total_degree()?;
                let b_degree = b.total_degree()?;
                Ok(a_degree
                    .cmp(&b_degree)
                    .then_with(|| a.exponents().cmp(b.exponents())))
            }
            Self::Weighted(weights) => {
                let a_degree = weighted_degree(a, weights)?;
                let b_degree = weighted_degree(b, weights)?;
                Ok(a_degree
                    .cmp(&b_degree)
                    .then_with(|| a.exponents().iter().rev().cmp(b.exponents().iter().rev())))
            }
        }
    }
}

/// The checked dot product `weights · exponents`.
fn weighted_degree<const N: usize>(
    exponents: &MultiIndex<N>,
    weights: &[usize; N],
) -> Result<usize, ConfigError> {
    let mut total = 0_usize;
    for (weight, exponent) in weights.iter().zip(exponents.exponents()) {
        total = weight
            .checked_mul(*exponent)
            .and_then(|product| total.checked_add(product))
            .ok_or(ConfigError::GeometryOverflow {
                context: "weighted monomial degree",
            })?;
    }
    Ok(total)
}

/// Embed the integer `n` into the field by double-and-add over `ONE`.
///
/// The only portable integer embedding: a raw byte pattern is an integer in
/// a prime field but *not* in a binary extension field, where `3` is
/// `X + 1`, not the value three.
pub(crate) fn embed_integer<F: Field>(mut n: u64) -> F::Elem {
    let mut term = F::Elem::ONE;
    let mut total = F::Elem::ZERO;
    while n != 0 {
        if n & 1 != 0 {
            total = total.add(term);
        }
        term = term.add(term);
        n >>= 1;
    }
    total
}
