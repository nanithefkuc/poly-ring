//! Polynomial behavior, boundary conditions, and recoverable allocation failures.

#[path = "behavior/formatting.rs"]
mod formatting;
mod oracles;

#[path = "behavior/affine_roots.rs"]
mod affine_roots;
#[path = "behavior/alekhnovich_edges.rs"]
mod alekhnovich_edges;
#[path = "behavior/alloc_resistance.rs"]
mod alloc_resistance;
#[path = "behavior/bivariate.rs"]
mod bivariate;
#[path = "behavior/bivariate_edges.rs"]
mod bivariate_edges;
#[path = "behavior/bivariate_substitution.rs"]
mod bivariate_substitution;
#[path = "behavior/capacity_limits.rs"]
mod capacity_limits;
#[path = "behavior/chien_invariants.rs"]
mod chien_invariants;
#[path = "behavior/convolution_routes.rs"]
mod convolution_routes;
#[path = "behavior/derivative_edges.rs"]
mod derivative_edges;
#[path = "behavior/equal_degree_paths.rs"]
mod equal_degree_paths;
#[path = "behavior/eval.rs"]
mod eval;
#[path = "behavior/eval_branches.rs"]
mod eval_branches;
#[path = "behavior/eval_edges.rs"]
mod eval_edges;
#[path = "behavior/factor_splitting.rs"]
mod factor_splitting;
#[path = "behavior/factorization.rs"]
mod factorization;
#[path = "behavior/forced_products.rs"]
mod forced_products;
#[path = "behavior/hermite.rs"]
mod hermite;
#[path = "behavior/hermite_edges.rs"]
mod hermite_edges;
#[path = "behavior/hermite_substitution.rs"]
mod hermite_substitution;
#[path = "behavior/hermite_weights.rs"]
mod hermite_weights;
#[path = "behavior/jet_edges.rs"]
mod jet_edges;
#[path = "behavior/jets.rs"]
mod jets;
#[path = "behavior/karatsuba_ring_extra.rs"]
mod karatsuba_ring_extra;
#[path = "behavior/lift_branches.rs"]
mod lift_branches;
#[path = "behavior/lift_divide_conquer.rs"]
mod lift_divide_conquer;
#[path = "behavior/lifting_reuse.rs"]
mod lifting_reuse;
#[path = "behavior/linearized_edges.rs"]
mod linearized_edges;
#[path = "behavior/multiplicity.rs"]
mod multiplicity;
#[path = "behavior/multiplicity_edges.rs"]
mod multiplicity_edges;
#[path = "behavior/multivariate.rs"]
mod multivariate;
#[path = "behavior/poly.rs"]
mod poly;
#[path = "behavior/poly_edges.rs"]
mod poly_edges;
#[path = "behavior/polynomial_operations.rs"]
mod polynomial_operations;
#[path = "behavior/portable_domains.rs"]
mod portable_domains;
#[path = "behavior/prepared_evaluation.rs"]
mod prepared_evaluation;
#[path = "behavior/prepared_products.rs"]
mod prepared_products;
#[path = "behavior/prime_derivatives.rs"]
mod prime_derivatives;
#[path = "behavior/properties.rs"]
mod properties;
#[path = "behavior/quotient.rs"]
mod quotient;
#[path = "behavior/recursive_lifting.rs"]
mod recursive_lifting;
#[path = "behavior/ring_and_roots.rs"]
mod ring_and_roots;
#[path = "behavior/root_edges.rs"]
mod root_edges;
#[path = "behavior/root_lifting_paths.rs"]
mod root_lifting_paths;
#[path = "behavior/roots.rs"]
mod roots;
#[path = "behavior/series.rs"]
mod series;
#[path = "behavior/sparse_edges.rs"]
mod sparse_edges;
#[path = "behavior/sparse_operations.rs"]
mod sparse_operations;
#[path = "behavior/sparse_paths.rs"]
mod sparse_paths;
#[path = "behavior/transform_branches.rs"]
mod transform_branches;
#[path = "behavior/transform_limits.rs"]
mod transform_limits;
#[path = "behavior/transform_products.rs"]
mod transform_products;
#[path = "behavior/translation_scratch.rs"]
mod translation_scratch;
#[path = "behavior/weighted_batches.rs"]
mod weighted_batches;
