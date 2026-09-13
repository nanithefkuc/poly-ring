//! Polynomial roots: Chien search, equal-degree splitting, linearized
//! solving, and power-series lifting.

mod chien;
mod equal_degree;
mod lift;
mod linearized;

pub use chien::{ChienScratch, chien_roots, chien_roots_into};
pub use equal_degree::{FieldRootScratch, base_field_roots, base_field_roots_into, element_key};
#[cfg(feature = "fft")]
pub use lift::{
    AffineRootFamily, AlekhnovichLimits, AlekhnovichScratch, DEFAULT_ROTH_RUCKENSTEIN_CROSSOVER,
    alekhnovich_roots, alekhnovich_roots_into,
};
pub use lift::{
    RothRuckensteinLimits, RothRuckensteinScratch, roth_ruckenstein_roots,
    roth_ruckenstein_roots_into,
};
pub use linearized::linearized_roots;

use alloc::vec::Vec;

/// The roots of a univariate polynomial over its coefficient field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BaseFieldRoots<E> {
    /// The zero polynomial vanishes at every field element.
    All,
    /// A sorted, deduplicated finite root list.
    Finite(Vec<E>),
}

impl<E> BaseFieldRoots<E> {
    /// Borrow the finite root list, or return `None` for the zero polynomial.
    #[must_use]
    pub fn as_slice(&self) -> Option<&[E]> {
        match self {
            Self::All => None,
            Self::Finite(roots) => Some(roots),
        }
    }

    /// Consume the result, returning `None` when every field element is a root.
    #[must_use]
    pub fn into_finite(self) -> Option<Vec<E>> {
        match self {
            Self::All => None,
            Self::Finite(roots) => Some(roots),
        }
    }
}
