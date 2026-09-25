//! Unstable implementation details for benchmarks and experiments.
//!
//! This module changes without compatibility guarantees. Stable public APIs do
//! not depend on it.

#[cfg(feature = "fft")]
pub use crate::poly::afft::substitute_y_affine_rows_truncated_into;
pub use crate::poly::convolution::{ProductRoute, multiply_rows_route_into};
pub use crate::poly::karatsuba::{KARATSUBA_CROSSOVER, karatsuba_multiply};
pub use crate::poly::ring::binomial_odd;
pub use crate::roots::equal_degree::element_key;
