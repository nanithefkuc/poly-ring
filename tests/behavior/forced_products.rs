// `as_chunks::<F::BYTES>()` needs a const generic depending on a
// type parameter, which Rust rejects in const argument position.
#![allow(clippy::chunks_exact_to_as_chunks)]

//! Forced products behavior and error boundaries.

#[cfg(feature = "internals")]
use fgf::field::{Elem, Field};
#[cfg(feature = "internals")]
use fgf::kernel::FieldKernels;
#[cfg(feature = "internals")]
use fgf::{Gf8B, Gf8D, Gf16, Goldilocks, Mersenne31, QuadMersenne31};
#[cfg(feature = "internals")]
use poly_ring::{
    ConfigError, ConvolutionScratch, ProductError,
    internals::{ProductRoute, multiply_rows_route_into},
};

#[cfg(feature = "internals")]
use crate::oracles;

#[cfg(feature = "internals")]
fn pack<F: FieldKernels>(coefficients: &[F::Elem]) -> Vec<u8> {
    let mut packed = vec![0_u8; coefficients.len() * F::BYTES];
    for (slot, value) in packed.chunks_exact_mut(F::BYTES).zip(coefficients) {
        F::encode(slot, *value);
    }
    packed
}

#[cfg(feature = "internals")]
fn scalar_product<F: poly_ring::PolynomialField>(
    left: &[F::Elem],
    right: &[F::Elem],
) -> Vec<F::Elem> {
    let full = left.len() + right.len() - 1;
    let mut product = vec![F::Elem::ZERO; full];
    for (i, a) in left.iter().enumerate() {
        for (j, b) in right.iter().enumerate() {
            product[i + j] = product[i + j].add(a.mul(*b));
        }
    }
    product
}

#[cfg(feature = "internals")]
fn check_route<F: poly_ring::PolynomialField>(route: ProductRoute) {
    for (left_len, right_len) in [(1_usize, 1), (5, 9), (33, 64), (100, 7)] {
        let left = oracles::noise::<F>(left_len, 0xD001 + left_len as u64);
        let right = oracles::noise::<F>(right_len, 0xD002 + right_len as u64);
        let full = left_len + right_len - 1;
        let mut scratch = ConvolutionScratch::<F>::new(left_len, right_len, 1).expect("scratch");
        // Warm every plan the run can request so the forced route never
        // allocates mid-run.
        scratch.prepare_transform(full, 1).expect("prepare");
        let mut output = vec![0_u8; full * F::BYTES];
        multiply_rows_route_into::<F>(
            &mut output,
            &pack::<F>(&left),
            left_len,
            &pack::<F>(&right),
            right_len,
            1,
            full,
            route,
            &mut scratch,
        )
        .expect("forced product");
        let expected = scalar_product::<F>(&left, &right);
        for (degree, want) in expected.iter().enumerate() {
            assert_eq!(
                F::decode(&output[degree * F::BYTES..][..F::BYTES]),
                *want,
                "coefficient {degree} diverged at {left_len}x{right_len} on {}",
                core::any::type_name::<F>()
            );
        }
    }
}

/// Forced schoolbook and Karatsuba routes match the scalar oracle on every
/// field, including the route-less Gf8D.
#[cfg(feature = "internals")]
#[test]
fn forced_fallback_routes_match_the_oracle() {
    check_route::<Gf8B>(ProductRoute::Schoolbook);
    check_route::<Gf8B>(ProductRoute::Karatsuba);
    check_route::<Gf16>(ProductRoute::Schoolbook);
    check_route::<Gf16>(ProductRoute::Karatsuba);
    check_route::<Goldilocks>(ProductRoute::Schoolbook);
    check_route::<Goldilocks>(ProductRoute::Karatsuba);
    check_route::<QuadMersenne31>(ProductRoute::Schoolbook);
    check_route::<QuadMersenne31>(ProductRoute::Karatsuba);
    check_route::<Mersenne31>(ProductRoute::Schoolbook);
    check_route::<Mersenne31>(ProductRoute::Karatsuba);
    check_route::<Gf8D>(ProductRoute::Schoolbook);
    check_route::<Gf8D>(ProductRoute::Karatsuba);
    check_route::<Gf8D>(ProductRoute::Auto);
}

/// Forced auto routes match the oracle through each field's default path.
#[cfg(feature = "internals")]
#[test]
fn forced_auto_routes_match_the_oracle() {
    check_route::<Gf8B>(ProductRoute::Auto);
    check_route::<Gf16>(ProductRoute::Auto);
    check_route::<Goldilocks>(ProductRoute::Auto);
    check_route::<QuadMersenne31>(ProductRoute::Auto);
    check_route::<Mersenne31>(ProductRoute::Auto);
}

/// Forced transform routes match the oracle on every transform field.
#[cfg(feature = "internals")]
#[test]
fn forced_transform_routes_match_the_oracle() {
    check_route::<Gf8B>(ProductRoute::Transform);
    check_route::<Gf16>(ProductRoute::Transform);
    check_route::<Goldilocks>(ProductRoute::Transform);
    check_route::<QuadMersenne31>(ProductRoute::Transform);
    check_route::<Mersenne31>(ProductRoute::Transform);
}

/// A forced transform on a route-less field is an error carrying the full
/// product size, never a silent fallback.
#[cfg(feature = "internals")]
#[test]
fn forced_transform_on_routeless_field_is_an_error() {
    let left = oracles::noise::<Gf8D>(5, 0xD010);
    let right = oracles::noise::<Gf8D>(7, 0xD011);
    let full = 5 + 7 - 1;
    let mut scratch = ConvolutionScratch::<Gf8D>::new(5, 7, 1).expect("scratch");
    let mut output = vec![0_u8; full * <Gf8D as Field>::BYTES];
    assert_eq!(
        multiply_rows_route_into::<Gf8D>(
            &mut output,
            &pack::<Gf8D>(&left),
            5,
            &pack::<Gf8D>(&right),
            7,
            1,
            full,
            ProductRoute::Transform,
            &mut scratch,
        )
        .map(|_| ()),
        Err(ProductError::Config(ConfigError::ScratchTooSmall {
            context: "prepared transform plan",
            required: full,
            available: 0,
        }))
    );
}

/// Validation errors name the offending buffer with both lengths.
#[cfg(feature = "internals")]
#[test]
fn forced_routes_validate_buffers_and_capacities() {
    let left = oracles::noise::<Gf8B>(5, 0xD020);
    let right = oracles::noise::<Gf8B>(7, 0xD021);
    let full = 5 + 7 - 1;
    let mut scratch = ConvolutionScratch::<Gf8B>::new(5, 7, 1).expect("scratch");
    let mut output = vec![0_u8; full];
    // Short left buffer names expected vs actual.
    assert_eq!(
        multiply_rows_route_into::<Gf8B>(
            &mut output,
            &[0_u8; 2],
            5,
            &pack::<Gf8B>(&right),
            7,
            1,
            full,
            ProductRoute::Schoolbook,
            &mut scratch,
        )
        .map(|_| ()),
        Err(ProductError::Config(ConfigError::BufferLength {
            context: "prepared left operand rows",
            expected: 5,
            actual: 2,
        }))
    );
    // Short right buffer names expected vs actual.
    assert_eq!(
        multiply_rows_route_into::<Gf8B>(
            &mut output,
            &pack::<Gf8B>(&left),
            5,
            &[0_u8; 2],
            7,
            1,
            full,
            ProductRoute::Schoolbook,
            &mut scratch,
        )
        .map(|_| ()),
        Err(ProductError::Config(ConfigError::BufferLength {
            context: "prepared right operand rows",
            expected: 7,
            actual: 2,
        }))
    );
    // Short output buffer names expected vs actual.
    assert_eq!(
        multiply_rows_route_into::<Gf8B>(
            &mut [0_u8; 2],
            &pack::<Gf8B>(&left),
            5,
            &pack::<Gf8B>(&right),
            7,
            1,
            full,
            ProductRoute::Schoolbook,
            &mut scratch,
        )
        .map(|_| ()),
        Err(ProductError::Config(ConfigError::BufferLength {
            context: "prepared output rows",
            expected: full,
            actual: 2,
        }))
    );
    // Oversized operands exceed the prepared capacity.
    assert_eq!(
        multiply_rows_route_into::<Gf8B>(
            &mut output,
            &pack::<Gf8B>(&left),
            500,
            &pack::<Gf8B>(&right),
            7,
            1,
            full,
            ProductRoute::Schoolbook,
            &mut scratch,
        )
        .map(|_| ()),
        Err(ProductError::Config(ConfigError::BufferLength {
            context: "prepared left operand rows",
            expected: 500,
            actual: full - 6,
        }))
    );
    // Oversized lanes exceed the prepared lane capacity.
    let mut wide_output = vec![0_u8; full * 4];
    assert_eq!(
        multiply_rows_route_into::<Gf8B>(
            &mut wide_output,
            &pack::<Gf8B>(&left),
            5,
            &pack::<Gf8B>(&right),
            7,
            4,
            full,
            ProductRoute::Schoolbook,
            &mut scratch,
        )
        .map(|_| ()),
        Err(ProductError::Config(ConfigError::BufferLength {
            context: "prepared left operand rows",
            expected: 20,
            actual: full - 6,
        }))
    );
}

/// Empty geometries write nothing and succeed on every forced route.
#[cfg(feature = "internals")]
#[test]
fn forced_routes_accept_empty_geometries() {
    for route in [
        ProductRoute::Auto,
        ProductRoute::Schoolbook,
        ProductRoute::Karatsuba,
        ProductRoute::Transform,
    ] {
        let mut scratch = ConvolutionScratch::<Gf8B>::new(4, 4, 1).expect("scratch");
        let mut output = Vec::new();
        multiply_rows_route_into::<Gf8B>(&mut output, &[], 0, &[], 0, 1, 0, route, &mut scratch)
            .expect("empty product");
        assert!(output.is_empty());
    }
}
