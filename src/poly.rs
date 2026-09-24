//! Dense univariate polynomials over `fgf` fields.

#[cfg(feature = "fft")]
pub(crate) mod afft;
mod bivariate;
pub(crate) mod convolution;
mod dense;
mod divide;
mod factor;
mod gcd;
pub(crate) mod karatsuba;
pub(crate) mod monomial;
mod multivariate;
mod quotient;
pub(crate) mod ring;
mod series;

#[cfg(feature = "fft")]
pub(crate) use afft::substitute_y_affine_rows_truncated_into;
#[cfg(feature = "fft")]
pub use afft::{
    AFFT_BATCH4_CROSSOVER, AFFT_BATCH8_CROSSOVER, AFFT_BATCH16_CROSSOVER, AFFT_PRODUCT_CROSSOVER,
    PolynomialProductScratch, ProductStrategy, SCALAR_AFFT_BATCH4_CROSSOVER,
    SCALAR_AFFT_BATCH8_CROSSOVER, SCALAR_AFFT_BATCH16_CROSSOVER, SCALAR_AFFT_PRODUCT_CROSSOVER,
    multiply_batch_truncated_into,
};
pub use bivariate::{BivariatePolynomial, WeightedTerm};
pub use convolution::{ConvolutionScratch, PolynomialField, multiply_rows_into};
pub use dense::Polynomial;
pub use factor::{DistinctDegreeFactor, IrreducibleFactor, SquareFreeFactor};
pub use gcd::{BezoutRelation, TruncatedEea, truncated_eea};
pub use monomial::{MonomialOrder, MultiIndex, Term};
pub use multivariate::SparsePolynomial;
pub use quotient::{ModulusPlan, ModulusScratch};
pub use ring::binomial;
pub(crate) use ring::binomial_odd;
pub use series::series_divide;
