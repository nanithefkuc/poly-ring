//! Monomial structure and shared scalar helpers.
//!
//! This module owns the exponent/index vocabulary shared by the ring's
//! multivariate surfaces and the crate-private integer embedding used by
//! both the binomial computation and the prepared Hasse factor recurrences.

use fgf::field::{Elem, Field};

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
