//! Sparse multivariate polynomials over `fgf` fields.
//!
//! A [`SparsePolynomial`] stores exactly its nonzero terms, sorted ascending
//! lexicographically on the exponent arrays (variable 0 first), duplicates
//! merged, canonical field coefficients, no zero terms; the zero polynomial
//! is the empty vector. Storage order never changes when a caller ranks
//! monomials under a different [`MonomialOrder`] — orders matter only for
//! [`SparsePolynomial::leading_term`]. Conversions to and from the dense
//! univariate and bivariate representations are explicit and lossless;
//! storage never switches implicitly.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use super::bivariate::BivariatePolynomial;
use super::dense::Polynomial;
use super::monomial::{MonomialOrder, MultiIndex, Term};
use crate::error::{ConfigError, PolynomialError};

/// A sparse multivariate polynomial in `N` variables.
///
/// Scalar term-pair arithmetic, exact over every characteristic: products
/// are the defining convolution, Hasse derivatives carry
/// `C(a_i, r_i)` binomial factors taken in the field's characteristic, and
/// no optimized strategy stands in for them yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparsePolynomial<F: FieldKernels, const N: usize> {
    terms: Vec<Term<F, N>>,
}

impl<F: FieldKernels, const N: usize> SparsePolynomial<F, N> {
    /// The zero polynomial: no terms.
    #[must_use]
    pub const fn zero() -> Self {
        Self { terms: Vec::new() }
    }

    /// Normalize raw terms into the canonical sparse form.
    ///
    /// Consumes the vector, sorts it in place, merges duplicate exponents by
    /// field addition — duplicates cancel in small characteristic exactly as
    /// any other coefficient sum — and drops zero coefficients. No second
    /// allocation is performed, so normalization is infallible.
    #[must_use]
    pub fn from_terms(mut terms: Vec<Term<F, N>>) -> Self {
        terms.sort_by_key(|term| *term.exponents.exponents());
        let mut kept = 0_usize;
        for read in 0..terms.len() {
            if kept > 0 && terms[kept - 1].exponents == terms[read].exponents {
                terms[kept - 1].coefficient =
                    terms[kept - 1].coefficient.add(terms[read].coefficient);
            } else {
                terms[kept] = terms[read];
                kept += 1;
            }
        }
        terms.truncate(kept);
        terms.retain(|term| !term.coefficient.is_zero());
        Self { terms }
    }

    /// The stored terms, ascending lexicographically on exponents.
    #[must_use]
    pub fn terms(&self) -> &[Term<F, N>] {
        &self.terms
    }

    /// Whether the polynomial has no terms.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// The coefficient of one monomial, zero beyond the stored support.
    #[must_use]
    pub fn coefficient(&self, exponents: &MultiIndex<N>) -> F::Elem {
        self.terms
            .binary_search_by(|term| term.exponents.exponents().cmp(exponents.exponents()))
            .ok()
            .map_or(F::Elem::ZERO, |index| self.terms[index].coefficient)
    }

    /// The row-wise sum with `rhs`, merged in storage order.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::GeometryOverflow`] when the term count cannot
    /// be represented and [`ConfigError::AllocationFailed`] when the merged
    /// storage cannot be reserved.
    pub fn add(&self, rhs: &Self) -> Result<Self, ConfigError> {
        Ok(Self {
            terms: merge_terms::<F, N>(&self.terms, &rhs.terms, false)?,
        })
    }

    /// The difference `self - rhs`, scaling `rhs` by the additive inverse.
    ///
    /// # Errors
    ///
    /// As [`Self::add`].
    pub fn sub(&self, rhs: &Self) -> Result<Self, ConfigError> {
        Ok(Self {
            terms: merge_terms::<F, N>(&self.terms, &rhs.terms, true)?,
        })
    }

    /// Copy with every coefficient scaled by `scale`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] reserved for later allocation-tracking
    /// strategies; a zero scale yields the canonical zero polynomial.
    pub fn scaled(&self, scale: F::Elem) -> Result<Self, ConfigError> {
        let mut terms = self.terms.clone();
        if scale.is_zero() {
            terms.clear();
        } else {
            for term in &mut terms {
                term.coefficient = term.coefficient.mul(scale);
            }
        }
        Ok(Self { terms })
    }

    /// The term-pair convolution `self · rhs`, normalized.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::GeometryOverflow`] when a pair count or an
    /// exponent sum cannot be represented and
    /// [`ConfigError::AllocationFailed`] when the product storage cannot be
    /// reserved.
    pub fn multiply(&self, rhs: &Self) -> Result<Self, ConfigError> {
        if self.is_zero() || rhs.is_zero() {
            return Ok(Self::zero());
        }
        let pair_count =
            self.terms
                .len()
                .checked_mul(rhs.terms.len())
                .ok_or(ConfigError::GeometryOverflow {
                    context: "multivariate product term count",
                })?;
        let mut products = Vec::new();
        products
            .try_reserve_exact(pair_count)
            .map_err(|_| ConfigError::AllocationFailed {
                context: "multivariate product terms",
                elements: pair_count,
                element_size: core::mem::size_of::<Term<F, N>>(),
            })?;
        for left in &self.terms {
            for right in &rhs.terms {
                products.push(Term {
                    exponents: left.exponents.checked_add(&right.exponents)?,
                    coefficient: left.coefficient.mul(right.coefficient),
                });
            }
        }
        Ok(Self::from_terms(products))
    }

    /// Evaluate at `point` as the defining sum of monomials.
    #[must_use]
    pub fn evaluate(&self, point: &[F::Elem; N]) -> F::Elem {
        let mut total = F::Elem::ZERO;
        for term in &self.terms {
            total = total.add(
                term.coefficient
                    .mul(monomial_value::<F, N>(point, &term.exponents)),
            );
        }
        total
    }

    /// The Hasse derivative of one multi-order `order`.
    ///
    /// A term with exponent `a` contributes zero when any `r_i > a_i`;
    /// otherwise it contributes coefficient `c · ∏_i C(a_i, r_i)` — the
    /// binomials taken in the field's characteristic, never a factorial —
    /// with exponent `a − order`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::GeometryOverflow`] when the term count cannot
    /// be represented and [`ConfigError::AllocationFailed`] when the
    /// derivative storage cannot be reserved.
    pub fn hasse_derivative(&self, order: &MultiIndex<N>) -> Result<Self, ConfigError> {
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(self.terms.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "multivariate derivative terms",
                elements: self.terms.len(),
                element_size: core::mem::size_of::<Term<F, N>>(),
            })?;
        for term in &self.terms {
            let exponents = *term.exponents.exponents();
            let mut factor = F::Elem::ONE;
            let mut reduced = [0_usize; N];
            let mut within_support = true;
            for index in 0..N {
                if order.exponents()[index] > exponents[index] {
                    within_support = false;
                    break;
                }
                reduced[index] = exponents[index] - order.exponents()[index];
                factor = factor.mul(super::binomial::<F>(
                    exponents[index],
                    order.exponents()[index],
                ));
            }
            if !within_support {
                continue;
            }
            let coefficient = term.coefficient.mul(factor);
            if coefficient.is_zero() {
                continue;
            }
            terms.push(Term {
                exponents: MultiIndex::new(reduced),
                coefficient,
            });
        }
        Ok(Self { terms })
    }

    /// Evaluate the Hasse derivative of one multi-order at `point`.
    ///
    /// The defining scalar sum: no intermediate derivative polynomial is
    /// built. Orders beyond a term's support contribute zero.
    #[must_use]
    pub fn evaluate_hasse(&self, point: &[F::Elem; N], order: &MultiIndex<N>) -> F::Elem {
        let mut total = F::Elem::ZERO;
        for term in &self.terms {
            let exponents = *term.exponents.exponents();
            let mut factor = F::Elem::ONE;
            let mut monomial = F::Elem::ONE;
            let mut within_support = true;
            for index in 0..N {
                if order.exponents()[index] > exponents[index] {
                    within_support = false;
                    break;
                }
                factor = factor.mul(super::binomial::<F>(
                    exponents[index],
                    order.exponents()[index],
                ));
                let mut power = F::Elem::ONE;
                for _ in 0..(exponents[index] - order.exponents()[index]) {
                    power = power.mul(point[index]);
                }
                monomial = monomial.mul(power);
            }
            if !within_support {
                continue;
            }
            total = total.add(term.coefficient.mul(factor).mul(monomial));
        }
        total
    }

    /// Evaluate the Hasse derivatives of `orders` at `point`, in exactly
    /// the caller's order, preserving repeats, into `output`.
    ///
    /// This is an ordered derivative buffer, not a claim that arbitrary
    /// requested orders form a quotient algebra. Empty requests succeed.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::BufferLength`] (context
    /// `"multivariate jet output"`, byte counts) when `output` does not hold
    /// exactly `orders.len()` elements; nothing is written. Returns
    /// [`ConfigError::GeometryOverflow`] when a byte count cannot be
    /// represented.
    pub fn evaluate_jet_into(
        &self,
        point: &[F::Elem; N],
        orders: &[MultiIndex<N>],
        output: &mut [F::Elem],
    ) -> Result<(), ConfigError> {
        let unit = core::mem::size_of::<F::Elem>();
        if output.len() != orders.len() {
            let expected = orders
                .len()
                .checked_mul(unit)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "multivariate jet output",
                })?;
            let actual = output
                .len()
                .checked_mul(unit)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "multivariate jet output",
                })?;
            return Err(ConfigError::BufferLength {
                context: "multivariate jet output",
                expected,
                actual,
            });
        }
        for (slot, order) in output.iter_mut().zip(orders) {
            *slot = self.evaluate_hasse(point, order);
        }
        Ok(())
    }

    /// The leading term under `order`, if any term exists.
    ///
    /// Storage order is untouched; only the answer is ranked.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::GeometryOverflow`] when a weighted or total
    /// degree comparison cannot be represented.
    pub fn leading_term(
        &self,
        order: &MonomialOrder<N>,
    ) -> Result<Option<&Term<F, N>>, ConfigError> {
        let mut leading: Option<&Term<F, N>> = None;
        for term in &self.terms {
            let is_larger = match leading {
                None => true,
                Some(current) => order.compare(&term.exponents, &current.exponents)?.is_gt(),
            };
            if is_larger {
                leading = Some(term);
            }
        }
        Ok(leading)
    }
}

impl<F: FieldKernels, const N: usize> Default for SparsePolynomial<F, N> {
    fn default() -> Self {
        Self::zero()
    }
}

/// Merge two sorted term lists, negating the right side when asked, into
/// canonical storage: duplicates combined, zero coefficients dropped.
fn merge_terms<F: FieldKernels, const N: usize>(
    left: &[Term<F, N>],
    right: &[Term<F, N>],
    negate_right: bool,
) -> Result<Vec<Term<F, N>>, ConfigError> {
    let capacity = left
        .len()
        .checked_add(right.len())
        .ok_or(ConfigError::GeometryOverflow {
            context: "multivariate sum term count",
        })?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(capacity)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "multivariate sum terms",
            elements: capacity,
            element_size: core::mem::size_of::<Term<F, N>>(),
        })?;
    let mut left_index = 0_usize;
    let mut right_index = 0_usize;
    while left_index < left.len() && right_index < right.len() {
        let left_term = &left[left_index];
        let right_term = &right[right_index];
        match left_term
            .exponents
            .exponents()
            .cmp(right_term.exponents.exponents())
        {
            core::cmp::Ordering::Less => {
                terms.push(*left_term);
                left_index += 1;
            }
            core::cmp::Ordering::Greater => {
                let coefficient = right_term.coefficient;
                terms.push(Term {
                    exponents: right_term.exponents,
                    coefficient: if negate_right {
                        coefficient.neg()
                    } else {
                        coefficient
                    },
                });
                right_index += 1;
            }
            core::cmp::Ordering::Equal => {
                let right_value = if negate_right {
                    right_term.coefficient.neg()
                } else {
                    right_term.coefficient
                };
                let coefficient = left_term.coefficient.add(right_value);
                if !coefficient.is_zero() {
                    terms.push(Term {
                        exponents: left_term.exponents,
                        coefficient,
                    });
                }
                left_index += 1;
                right_index += 1;
            }
        }
    }
    for term in &left[left_index..] {
        terms.push(*term);
    }
    for term in &right[right_index..] {
        terms.push(Term {
            exponents: term.exponents,
            coefficient: if negate_right {
                term.coefficient.neg()
            } else {
                term.coefficient
            },
        });
    }
    Ok(terms)
}

/// The monomial `point^exponents` as a plain product of powers.
fn monomial_value<F: FieldKernels, const N: usize>(
    point: &[F::Elem; N],
    exponents: &MultiIndex<N>,
) -> F::Elem {
    let mut value = F::Elem::ONE;
    for (variable, &exponent) in point.iter().zip(exponents.exponents()) {
        for _ in 0..exponent {
            value = value.mul(*variable);
        }
    }
    value
}

impl<F: FieldKernels> SparsePolynomial<F, 2> {
    /// Enumerate a dense bivariate polynomial's stored coefficients as
    /// sparse terms. Variable 0 is `X`, variable 1 is `Y`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::GeometryOverflow`] when the term count cannot
    /// be represented and [`ConfigError::AllocationFailed`] when the term
    /// storage cannot be reserved.
    pub fn from_bivariate(value: &BivariatePolynomial<F>) -> Result<Self, ConfigError> {
        let term_count = value
            .y_coefficients()
            .iter()
            .try_fold(0_usize, |total, row| {
                total
                    .checked_add(row.coefficient_count())
                    .ok_or(ConfigError::GeometryOverflow {
                        context: "sparse term count",
                    })
            })?;
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(term_count)
            .map_err(|_| ConfigError::AllocationFailed {
                context: "sparse bivariate terms",
                elements: term_count,
                element_size: core::mem::size_of::<Term<F, 2>>(),
            })?;
        for (y_degree, row) in value.y_coefficients().iter().enumerate() {
            for (x_degree, coefficient) in row.coefficients().enumerate() {
                terms.push(Term {
                    exponents: MultiIndex::new([x_degree, y_degree]),
                    coefficient,
                });
            }
        }
        Ok(Self::from_terms(terms))
    }

    /// Materialize the dense bivariate form: row `j` is the coefficient of
    /// `Y^j`. Maximum row and column indices are checked before any dense
    /// allocation, so an unrepresentable exponent is an error rather than a
    /// wrapped allocation.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] for geometry overflow or a
    /// failed row or row-buffer reservation.
    pub fn to_bivariate(&self) -> Result<BivariatePolynomial<F>, PolynomialError> {
        if self.is_zero() {
            return Ok(BivariatePolynomial::zero());
        }
        let mut y_max = 0_usize;
        let mut x_max = 0_usize;
        for term in &self.terms {
            let [x_degree, y_degree] = *term.exponents.exponents();
            y_max = y_max.max(y_degree);
            x_max = x_max.max(x_degree);
        }
        let row_count = y_max.checked_add(1).ok_or(ConfigError::GeometryOverflow {
            context: "dense bivariate row count",
        })?;
        // Column bound: rows grow lazily up to x_max + 1 coefficients, so
        // the check is the allocation guard.
        x_max.checked_add(1).ok_or(ConfigError::GeometryOverflow {
            context: "dense bivariate column count",
        })?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(row_count).map_err(|_| {
            PolynomialError::Config(ConfigError::AllocationFailed {
                context: "dense bivariate rows",
                elements: row_count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })
        })?;
        rows.resize_with(row_count, Polynomial::zero);
        for term in &self.terms {
            let [x_degree, y_degree] = *term.exponents.exponents();
            rows[y_degree].set_coefficient(x_degree, term.coefficient)?;
        }
        Ok(BivariatePolynomial::from_y_coefficients(rows))
    }
}

impl<F: FieldKernels> SparsePolynomial<F, 1> {
    /// Enumerate a univariate polynomial's stored coefficients as sparse
    /// terms of the single variable `X`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::AllocationFailed`] when the term storage
    /// cannot be reserved.
    pub fn from_univariate(value: &Polynomial<F>) -> Result<Self, ConfigError> {
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(value.coefficient_count())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "sparse univariate terms",
                elements: value.coefficient_count(),
                element_size: core::mem::size_of::<Term<F, 1>>(),
            })?;
        for (degree, coefficient) in value.coefficients().enumerate() {
            terms.push(Term {
                exponents: MultiIndex::new([degree]),
                coefficient,
            });
        }
        Ok(Self::from_terms(terms))
    }

    /// Materialize the dense univariate form.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the dense buffer cannot be
    /// reserved.
    pub fn to_univariate(&self) -> Result<Polynomial<F>, PolynomialError> {
        let mut result = Polynomial::zero();
        for term in &self.terms {
            result.set_coefficient(term.exponents.exponents()[0], term.coefficient)?;
        }
        Ok(result)
    }
}
