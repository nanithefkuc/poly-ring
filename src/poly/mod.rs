//! Dense univariate polynomials over `fgf` fields.

#[cfg(feature = "fft")]
mod afft;
mod bivariate;
pub mod convolution;
mod dense;
mod divide;
mod gcd;
mod karatsuba;
pub(crate) mod monomial;
mod multivariate;
mod ring;
mod series;

#[cfg(feature = "fft")]
pub use afft::{
    AFFT_BATCH4_CROSSOVER, AFFT_BATCH8_CROSSOVER, AFFT_BATCH16_CROSSOVER, AFFT_PRODUCT_CROSSOVER,
    PolynomialProductScratch, ProductStrategy, SCALAR_AFFT_BATCH4_CROSSOVER,
    SCALAR_AFFT_BATCH8_CROSSOVER, SCALAR_AFFT_BATCH16_CROSSOVER, SCALAR_AFFT_PRODUCT_CROSSOVER,
    multiply_batch_truncated, multiply_batch_truncated_with,
    substitute_y_affine_rows_truncated_into,
};
pub use bivariate::{BivariatePolynomial, WeightedTerm};
pub use convolution::{ConvolutionScratch, PolynomialField, multiply_rows_into};
#[cfg(feature = "internals")]
pub use convolution::{ProductRoute, multiply_rows_route_into};
pub use dense::Polynomial;
pub use gcd::{BezoutRelation, TruncatedEea, truncated_eea};
pub use karatsuba::{KARATSUBA_CROSSOVER, karatsuba_multiply};
pub use monomial::{MonomialOrder, MultiIndex, Term};
pub use multivariate::SparsePolynomial;
pub use ring::{binomial, binomial_odd};
pub use series::series_divide;
