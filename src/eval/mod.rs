//! Evaluation, interpolation, and the point-set domain abstraction.

mod domain;
mod hermite;
mod multiplicity;
mod multipoint;
mod newton;
mod remainder;
#[cfg(feature = "fft")]
mod transform;
pub(crate) mod tree;

pub use crate::error::DomainError;
pub use domain::{DomainScratch, EvaluationBackend, EvaluationDomain};
pub use hermite::HermitePlan;
pub use multiplicity::{MultiplicityPlan, MultiplicityScratch};
pub use multipoint::{
    MULTIPOINT_EVAL_CROSSOVER, MultipointScratch, evaluate_multipoint, evaluate_multipoint_into,
    interpolate_lagrange,
};
pub use newton::{
    MODULE_INTERPOLATION_CROSSOVER, NewtonBasis, interpolate_newton, interpolate_newton_into,
};
pub use remainder::{RemainderScratch, RemainderTree};
#[cfg(feature = "fft")]
pub use transform::{
    TransformScratch, evaluate_coset_into, evaluate_subspace, evaluate_subspace_into,
    interpolate_subspace, interpolate_subspace_into,
};

use fgf::kernel::FieldKernels;

/// Find the first pair of equal points, if any.
pub(crate) fn find_duplicate<F: FieldKernels>(points: &[F::Elem]) -> Option<(usize, usize)> {
    for second in 1..points.len() {
        if let Some(first) = points[..second]
            .iter()
            .position(|point| point == &points[second])
        {
            return Some((first, second));
        }
    }
    None
}
