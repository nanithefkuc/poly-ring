//! Hermite interpolation: one polynomial from every point's local jet.
//!
//! The inverse of multiplicity-weighted evaluation. Where
//! [`MultiplicityPlan`](crate::eval::MultiplicityPlan) turns a polynomial
//! into the packed jet vector `D^[j]f(a_i)` for `j < s_i`, this module turns
//! that vector back into the unique polynomial of degree below the total
//! weight `W = Σ s_i` matching every constraint. The bridge is a prepared
//! scalar CRT over the moduli `(X − a_i)^{s_i}`, assembled from the ring's
//! own multiplication, exact division, extended gcd, and jet translation —
//! no Gaussian elimination, no matrix machinery.

use alloc::vec::Vec;

use crate::error::{ConfigError, HermiteError, PolynomialError};
use crate::jet::JetPlan;
use crate::poly::{Polynomial, PolynomialField};
use fgf::field::{Elem, Field};

/// A prepared Hermite interpolation request: points with multiplicities.
///
/// Construction expands the weighted moduli, builds their product, and
/// prepares one CRT blending polynomial per active entry plus one jet
/// translation per active entry. Execution is a translation and a modular
/// sum per entry; the API is deliberately allocating and makes no
/// steady-state zero-allocation claim.
///
/// Entries with zero multiplicity are retained as empty offset ranges and
/// ignored in the algebra. Repeated points are legal while at most one of
/// them carries a positive weight; two positive weights on the same point
/// conflict and are rejected at construction, never merged. The total
/// weight is not capped by the field order — multiplicity is not a count
/// of distinct field elements.
pub struct HermitePlan<F: PolynomialField> {
    points: Vec<F::Elem>,
    multiplicities: Vec<usize>,
    offsets: Vec<usize>,
    total_weight: usize,
    /// Per active entry, in caller order: the modulus `(X − a_i)^{s_i}`.
    moduli: Vec<Polynomial<F>>,
    /// The product of all active moduli.
    product: Polynomial<F>,
    /// Per active entry: `B_i = (M_i · (M_i^{-1} mod m_i)) mod M`,
    /// congruent to one at `a_i` and zero at every other active point.
    weights: Vec<Polynomial<F>>,
    /// Per active entry: the translation `g(T) ↦ g(X − a_i)` prepared at
    /// the negated point.
    translations: Vec<JetPlan<F>>,
}

impl<F: PolynomialField> HermitePlan<F> {
    /// Prepare the request `points[i]` with multiplicity
    /// `multiplicities[i]`.
    ///
    /// Points are canonicalized before storage and comparison, so a
    /// noncanonical prime lane matches its field value.
    ///
    /// # Errors
    ///
    /// Returns [`HermiteError::LengthMismatch`] when the slices disagree in
    /// length, [`HermiteError::DuplicatePoint`] when two entries share a
    /// point and both weights are positive, [`HermiteError::Config`] for
    /// geometry overflow or a failed reservation, and the polynomial or jet
    /// errors of the CRT preparation.
    pub fn new(points: &[F::Elem], multiplicities: &[usize]) -> Result<Self, HermiteError> {
        if points.len() != multiplicities.len() {
            return Err(HermiteError::LengthMismatch {
                expected: points.len(),
                actual: multiplicities.len(),
            });
        }

        // Canonicalize once: a noncanonical prime lane must never reach a
        // packed kernel, a stored modulus, or a point comparison.
        let canonical = canonical_points::<F>(points)?;

        // Checked offsets and total weight.
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(points.len() + 1).map_err(|_| {
            HermiteError::Config(ConfigError::AllocationFailed {
                context: "Hermite offsets",
                elements: points.len() + 1,
                element_size: core::mem::size_of::<usize>(),
            })
        })?;
        let mut total_weight = 0_usize;
        offsets.push(0);
        for &weight in multiplicities {
            total_weight = total_weight
                .checked_add(weight)
                .ok_or(HermiteError::Config(ConfigError::GeometryOverflow {
                    context: "total multiplicity weight",
                }))?;
            offsets.push(total_weight);
        }

        // Conflicting duplicates are a construction error, compared by
        // field value; zero weights never conflict.
        for second in 1..canonical.len() {
            if multiplicities[second] == 0 {
                continue;
            }
            if let Some(first) = (0..second)
                .find(|&first| multiplicities[first] > 0 && canonical[first] == canonical[second])
            {
                return Err(HermiteError::DuplicatePoint { first, second });
            }
        }

        let moduli = weighted_moduli::<F>(&canonical, multiplicities)?;

        // M = ∏ m_i; empty and all-zero requests interpolate to zero, so
        // the product stays the empty modulus there.
        let mut product = Polynomial::one()?;
        for modulus in &moduli {
            product = product.multiply(modulus)?;
        }

        let weights = crt_weights::<F>(&moduli, &product)?;

        // Jet translations: g(T) ↦ g(X − a_i), exact because
        // deg g_i < s_i. A translation plan computes q(p + T), so the
        // negated point turns the local jet back into the local remainder.
        let mut translations = Vec::new();
        translations.try_reserve_exact(moduli.len()).map_err(|_| {
            HermiteError::Config(ConfigError::AllocationFailed {
                context: "Hermite jet translations",
                elements: moduli.len(),
                element_size: core::mem::size_of::<JetPlan<F>>(),
            })
        })?;
        for (point, &weight) in canonical.iter().zip(multiplicities) {
            if weight == 0 {
                continue;
            }
            translations.push(JetPlan::new(point.neg(), weight, weight)?);
        }

        let multiplicities = {
            let mut stored = Vec::new();
            stored
                .try_reserve_exact(multiplicities.len())
                .map_err(|_| {
                    HermiteError::Config(ConfigError::AllocationFailed {
                        context: "Hermite multiplicities",
                        elements: multiplicities.len(),
                        element_size: core::mem::size_of::<usize>(),
                    })
                })?;
            stored.extend_from_slice(multiplicities);
            stored
        };

        Ok(Self {
            points: canonical,
            multiplicities,
            offsets,
            total_weight,
            moduli,
            product,
            weights,
            translations,
        })
    }

    /// The prepared points, in caller order.
    #[must_use]
    pub fn points(&self) -> &[F::Elem] {
        &self.points
    }

    /// The prepared multiplicities, in caller order.
    #[must_use]
    pub fn multiplicities(&self) -> &[usize] {
        &self.multiplicities
    }

    /// Output offsets: `offsets[0] = 0`, `offsets[i+1] = offsets[i] +
    /// multiplicities[i]`. Input position `offsets[i] + j` is `D^[j]f(a_i)`,
    /// the same layout [`MultiplicityPlan`](crate::eval::MultiplicityPlan)
    /// writes.
    #[must_use]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// The total input weight `W = Σ multiplicities[i]`.
    #[must_use]
    pub fn total_weight(&self) -> usize {
        self.total_weight
    }

    /// Reconstruct the unique polynomial of degree below the total weight
    /// whose local jets at the prepared points are `values`.
    ///
    /// `values` uses exactly the
    /// [`MultiplicityPlan`](crate::eval::MultiplicityPlan) output order:
    /// input position `offsets[i] + j` is the value of `D^[j]f(a_i)`, and
    /// the length must equal [`Self::total_weight`]. Empty and all-zero
    /// requests reconstruct the zero polynomial.
    ///
    /// # Errors
    ///
    /// Returns [`HermiteError::LengthMismatch`] when `values` does not hold
    /// exactly the total weight, and the polynomial, jet, geometry, or
    /// allocation errors of the reconstruction.
    pub fn interpolate(&self, values: &[F::Elem]) -> Result<Polynomial<F>, HermiteError> {
        if values.len() != self.total_weight {
            return Err(HermiteError::LengthMismatch {
                expected: self.total_weight,
                actual: values.len(),
            });
        }
        if self.total_weight == 0 {
            return Ok(Polynomial::zero());
        }
        let mut result = Polynomial::zero();
        // Walk caller entries so zero-weight points are skipped in lockstep
        // with the prepared per-entry tables.
        let mut slot = 0_usize;
        for (index, &weight) in self.multiplicities.iter().enumerate() {
            if weight == 0 {
                continue;
            }
            let offset = self.offsets[index];
            let translation = &self.translations[slot];
            let mut scratch = translation.scratch(1)?;
            let mut translated = alloc::vec![F::Elem::ZERO; weight];
            // The jet ingress canonicalizes raw prime lanes itself.
            translation.evaluate_into(
                &values[offset..offset + weight],
                &mut scratch,
                &mut translated,
            )?;
            let local = Polynomial::from_coefficients(&translated)?;
            result = result.add(
                &local
                    .multiply(&self.weights[slot])?
                    .remainder(&self.product)?,
            )?;
            slot += 1;
        }
        result.normalize();
        Ok(result)
    }
}

/// Canonical representatives of the caller's points.
fn canonical_points<F: PolynomialField>(points: &[F::Elem]) -> Result<Vec<F::Elem>, HermiteError> {
    let mut canonical = Vec::new();
    canonical.try_reserve_exact(points.len()).map_err(|_| {
        HermiteError::Config(ConfigError::AllocationFailed {
            context: "Hermite points",
            elements: points.len(),
            element_size: core::mem::size_of::<F::Elem>(),
        })
    })?;
    for &point in points {
        canonical.push(point.add(F::Elem::ZERO));
    }
    Ok(canonical)
}

/// The active moduli `(X − a_i)^{s_i}`, by repeated multiplication by
/// `X − a_i = X + (−a_i)`; zero-weight entries build nothing.
fn weighted_moduli<F: PolynomialField>(
    points: &[F::Elem],
    multiplicities: &[usize],
) -> Result<Vec<Polynomial<F>>, HermiteError> {
    let mut moduli = Vec::new();
    moduli.try_reserve_exact(points.len()).map_err(|_| {
        HermiteError::Config(ConfigError::AllocationFailed {
            context: "Hermite moduli",
            elements: points.len(),
            element_size: core::mem::size_of::<Polynomial<F>>(),
        })
    })?;
    for (point, &weight) in points.iter().zip(multiplicities) {
        if weight == 0 {
            continue;
        }
        let negated = point.neg();
        let mut modulus = Polynomial::one()?;
        for _ in 0..weight {
            modulus = modulus.multiply_x_plus(negated)?;
        }
        moduli.push(modulus);
    }
    Ok(moduli)
}

/// The CRT blending polynomials `B_i = (M_i · (M_i^{-1} mod m_i)) mod M`.
///
/// `M_i = M / m_i` is coprime to `m_i` (distinct positive weights imply
/// distinct points), so the Bézout relation of `(M_i, m_i)` has a constant
/// gcd; its cofactor rescaled by the inverse of that constant is
/// `M_i^{-1} mod m_i`. A non-constant or zero gcd would mean the moduli
/// share a factor and the division contract is broken — that is a
/// [`PolynomialError::NonExactDivision`] failure, never an invented
/// inverse.
fn crt_weights<F: PolynomialField>(
    moduli: &[Polynomial<F>],
    product: &Polynomial<F>,
) -> Result<Vec<Polynomial<F>>, HermiteError> {
    let mut weights = Vec::new();
    weights.try_reserve_exact(moduli.len()).map_err(|_| {
        HermiteError::Config(ConfigError::AllocationFailed {
            context: "Hermite CRT weights",
            elements: moduli.len(),
            element_size: core::mem::size_of::<Polynomial<F>>(),
        })
    })?;
    for modulus in moduli {
        let cofactor = product.exact_divide(modulus)?;
        let relation = cofactor.gcd_ext(modulus)?;
        let unit: <F as Field>::Elem = match relation.gcd.degree() {
            Some(0) => relation.gcd.coefficient(0),
            _ => return Err(PolynomialError::NonExactDivision.into()),
        };
        if unit.is_zero() {
            return Err(PolynomialError::NonExactDivision.into());
        }
        let inverse = relation.a_cofactor.scaled(unit.inv());
        let weight = cofactor.multiply(&inverse)?.remainder(product)?;
        weights.push(weight);
    }
    Ok(weights)
}

impl<F: PolynomialField> core::fmt::Debug for HermitePlan<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("HermitePlan")
            .field("points", &self.points.len())
            .field("total_weight", &self.total_weight)
            .field("active_entries", &self.moduli.len())
            .finish_non_exhaustive()
    }
}
