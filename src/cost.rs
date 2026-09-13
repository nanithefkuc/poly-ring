//! Backend-explicit cost keys and pure strategy selectors.
//!
//! Every automatic strategy decision in this crate resolves through this
//! module. A selector is a *pure* function of a small cost key: it performs
//! no CPU detection and reads no environment variable, so it is
//! deterministic and directly testable across hypothetical backends.
//! Detection happens exactly once at a stage boundary via
//! [`BackendClass::detect`]; the resulting class is then threaded into the
//! key.
//!
//! The crossover constants these selectors compare against, and the exact
//! measurement commands and hardware that set them, are recorded in
//! `BENCHMARKS.md`; source carries only a one-line pointer.

use fgf::kernel::{Backend, FieldKernels, backend_for};

/// Backend capability class consumed by cost keys.
///
/// Derived once from the upstream `fgf`/`simdispatch` selected backend. It
/// does not perform detection itself; [`BackendClass::detect`] is the single
/// seam that queries the host, and the selectors below take a
/// [`BackendClass`] by value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendClass {
    lane_bytes: usize,
    scalar: bool,
}

impl BackendClass {
    /// Build a class from an explicit lane width and scalar flag.
    ///
    /// Prefer [`BackendClass::detect`] on a real decode path; this
    /// constructor exists so selectors can be exercised for held-out
    /// backends in tests.
    #[must_use]
    pub const fn new(lane_bytes: usize, scalar: bool) -> Self {
        Self { lane_bytes, scalar }
    }

    /// Classify the backend `fgf` selected for field `F`.
    ///
    /// This is the only cost-model function that observes the host.
    #[must_use]
    pub fn detect<F: FieldKernels>() -> Self {
        let backend = backend_for::<F>();
        Self {
            lane_bytes: backend.lane_bytes(),
            scalar: backend == Backend::Scalar,
        }
    }

    /// SIMD lane width in bytes reported by the selected backend.
    #[must_use]
    pub const fn lane_bytes(self) -> usize {
        self.lane_bytes
    }

    /// Whether the selected backend has no wide SIMD kernels.
    #[must_use]
    pub const fn is_scalar(self) -> bool {
        self.scalar
    }
}

/// Polynomial-product backend choice for one batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductBackend {
    /// Truncated schoolbook multiplication.
    Schoolbook,
    /// Packed additive-FFT multiplication.
    Afft,
}

/// Cost key for choosing a polynomial-product backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductCostKey {
    /// Left operand coefficient count of the widest pair.
    pub left_coefficients: usize,
    /// Right operand coefficient count of the widest pair.
    pub right_coefficients: usize,
    /// Full (untruncated) product coefficient count of the widest pair.
    pub output_coefficients: usize,
    /// Number of pairs in the batch.
    pub batch: usize,
    /// Number of elements in the field.
    pub field_order: u128,
    /// Selected backend class.
    pub backend: BackendClass,
}

/// Root-extraction backend choice for one interpolation polynomial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootBackend {
    /// Coefficient-prefix Roth–Ruckenstein lifting.
    RothRuckenstein,
    /// Divide-and-conquer Alekhnovich lifting.
    Alekhnovich,
}

/// Base-field root-finding backend choice for one univariate polynomial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseRootBackend {
    /// Classical Chien domain scan, `O(|F| * deg)`.
    Chien,
    /// `gcd(p, X^|F| + X)` plus trace splitting.
    EqualDegree,
}

/// Cost key for choosing a root-extraction backend.
///
/// `roth_ruckenstein_crossover`/`backend_adaptive` carry the caller's root
/// policy so the selector stays pure while honoring explicit overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootCostKey {
    /// Weighted input size (precision times `Y` rows).
    pub weighted_coefficients: usize,
    /// `Y`-coefficient row count.
    pub y_degree: usize,
    /// Target root precision (maximum root degree plus one).
    pub target_precision: usize,
    /// Selected backend class.
    pub backend: BackendClass,
    /// Crossover at or below which Roth–Ruckenstein runs.
    pub roth_ruckenstein_crossover: usize,
    /// Whether the selector forces Roth–Ruckenstein on scalar backends.
    pub backend_adaptive: bool,
}

/// Cost key for choosing a base-field root backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseRootCostKey {
    /// Polynomial degree.
    pub degree: usize,
    /// Number of elements in the field.
    pub field_order: u128,
}

/// Degree at or below which the Chien scan wins over equal-degree
/// factorization for base-field roots, on GF(2^8). Scales with the field
/// order because Chien's cost is `O(|F| * deg)` while the equal-degree route
/// is `O(deg^2)`-dominated. Measured; see `BENCHMARKS.md`.
#[must_use]
pub fn chien_equal_degree_crossover(field_order: u128) -> usize {
    // The scan visits every field element; the split route performs
    // O(m) modular Frobenius chains of length log2(|F|) per split level.
    // Chien wins while the polynomial is small relative to the field.
    let bits = u128::from(field_order.checked_ilog2().map_or(0, u32::from));
    ((field_order / (bits * bits)).max(8)) as usize
}

/// AFFT product crossover in full-product coefficients for a batch bucket.
#[must_use]
#[cfg(feature = "fft")]
pub fn product_crossover(scalar: bool, batch: usize) -> usize {
    use crate::poly::{
        AFFT_BATCH4_CROSSOVER, AFFT_BATCH8_CROSSOVER, AFFT_BATCH16_CROSSOVER,
        AFFT_PRODUCT_CROSSOVER, SCALAR_AFFT_BATCH4_CROSSOVER, SCALAR_AFFT_BATCH8_CROSSOVER,
        SCALAR_AFFT_BATCH16_CROSSOVER, SCALAR_AFFT_PRODUCT_CROSSOVER,
    };
    match (scalar, batch) {
        (true, 0..=3) => SCALAR_AFFT_PRODUCT_CROSSOVER,
        (true, 4..=7) => SCALAR_AFFT_BATCH4_CROSSOVER,
        (true, 8..=15) => SCALAR_AFFT_BATCH8_CROSSOVER,
        (true, _) => SCALAR_AFFT_BATCH16_CROSSOVER,
        (false, 0..=3) => AFFT_PRODUCT_CROSSOVER,
        (false, 4..=7) => AFFT_BATCH4_CROSSOVER,
        (false, 8..=15) => AFFT_BATCH8_CROSSOVER,
        (false, _) => AFFT_BATCH16_CROSSOVER,
    }
}

/// Choose the polynomial-product backend. Pure. See `BENCHMARKS.md`.
///
/// Without the `fft` feature the AFFT tier does not exist and the selector
/// degrades cleanly to schoolbook.
#[must_use]
pub fn select_product(key: ProductCostKey) -> ProductBackend {
    #[cfg(not(feature = "fft"))]
    {
        let _ = key;
        ProductBackend::Schoolbook
    }
    #[cfg(feature = "fft")]
    {
        if key.field_order <= 256 {
            return ProductBackend::Schoolbook;
        }
        if key.output_coefficients >= product_crossover(key.backend.is_scalar(), key.batch) {
            ProductBackend::Afft
        } else {
            ProductBackend::Schoolbook
        }
    }
}

/// Choose the root-extraction backend. Pure. See `BENCHMARKS.md`.
///
/// Without the `fft` feature the Alekhnovich tier does not exist and the
/// selector degrades cleanly to Roth–Ruckenstein.
#[must_use]
pub fn select_root(key: RootCostKey) -> RootBackend {
    #[cfg(not(feature = "fft"))]
    {
        let _ = key;
        RootBackend::RothRuckenstein
    }
    #[cfg(feature = "fft")]
    {
        let crossover = if key.backend_adaptive && key.backend.is_scalar() {
            usize::MAX
        } else {
            key.roth_ruckenstein_crossover
        };
        if key.weighted_coefficients <= crossover {
            RootBackend::RothRuckenstein
        } else {
            RootBackend::Alekhnovich
        }
    }
}

/// Choose the base-field root backend. Pure. See `BENCHMARKS.md`.
#[must_use]
pub fn select_base_roots(key: BaseRootCostKey) -> BaseRootBackend {
    if key.degree >= chien_equal_degree_crossover(key.field_order) {
        BaseRootBackend::EqualDegree
    } else {
        BaseRootBackend::Chien
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_covers_transform_free_small_fields() {
        // GF(2^8) always stays schoolbook: the transform cannot cover the
        // full product length for degree >= 255 without hitting the field
        // size, so the selector short-circuits.
        let key = ProductCostKey {
            left_coefficients: 1000,
            right_coefficients: 1000,
            output_coefficients: 1999,
            batch: 8,
            field_order: 256,
            backend: BackendClass::new(32, false),
        };
        assert_eq!(select_product(key), ProductBackend::Schoolbook);
    }

    #[cfg(feature = "fft")]
    #[test]
    fn scalar_single_products_enter_afft_above_their_crossover() {
        let key = ProductCostKey {
            left_coefficients: 600,
            right_coefficients: 600,
            output_coefficients: 1199,
            batch: 1,
            field_order: 65_536,
            backend: BackendClass::new(1, true),
        };
        assert_eq!(select_product(key), ProductBackend::Afft);
        let small = ProductCostKey {
            output_coefficients: 510,
            ..key
        };
        assert_eq!(select_product(small), ProductBackend::Schoolbook);
    }

    #[cfg(feature = "fft")]
    #[test]
    fn root_selection_honors_the_adaptive_scalar_guard() {
        let key = RootCostKey {
            weighted_coefficients: 100_000,
            y_degree: 4,
            target_precision: 128,
            backend: BackendClass::new(1, true),
            roth_ruckenstein_crossover: 20_000,
            backend_adaptive: true,
        };
        assert_eq!(select_root(key), RootBackend::RothRuckenstein);
        assert_eq!(
            select_root(RootCostKey {
                backend_adaptive: false,
                ..key
            }),
            RootBackend::Alekhnovich
        );
    }

    #[test]
    fn base_roots_pick_chien_for_small_locators() {
        assert_eq!(
            select_base_roots(BaseRootCostKey {
                degree: 3,
                field_order: 256
            }),
            BaseRootBackend::Chien
        );
        assert_eq!(
            select_base_roots(BaseRootCostKey {
                degree: 40,
                field_order: 256
            }),
            BaseRootBackend::EqualDegree
        );
    }
}
