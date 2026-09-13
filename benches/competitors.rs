//! Competitor benchmarks: `poly_ring` against its named non-copyleft
//! competitors from `BENCHMARKS.md` (ark-poly, lambdaworks-math).
//!
//! Neither competitor can instantiate GF(2^m) (prime-base fields only), so
//! both sides run element-size-matched 8-byte fields: this crate over
//! `fgf::Gf64`, the competitors over the Goldilocks prime
//! `2^64 - 2^32 + 1` (which ark needs anyway for FFT-based products).
//! Numbers are indicative of throughput at matched element width, not of
//! identical domains; `BENCHMARKS.md` records the caveats.
//!
//! Problem sizes are matched across sides; algorithms are each library's
//! own default path.

use ark_poly::{DenseUVPolynomial, EvaluationDomain, Polynomial};
use criterion::{Criterion, criterion_group, criterion_main};
use fgf::Gf64;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::montgomery_backed_prime_fields::{IsModulus, U64PrimeField};
use lambdaworks_math::polynomial::Polynomial as LwPolynomial;
use lambdaworks_math::unsigned_integer::element::U64;

type Gf64Elem = <Gf64 as fgf::field::Field>::Elem;
type ArkPoly = ark_poly::polynomial::univariate::DensePolynomial<ArkFp64>;
type LwFe = FieldElement<Lw64>;
type LwPoly = LwPolynomial<LwFe>;

/// ark-ff Goldilocks prime, one 64-bit limb, 2^32 | p - 1.
#[derive(ark_ff::MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
pub struct ArkFp64Config;

pub type ArkFp64 = ark_ff::Fp<ark_ff::MontBackend<ArkFp64Config, 1>, 1>;

/// Goldilocks modulus for lambdaworks' Montgomery-backed `U64` field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldiModulus;

impl IsModulus<U64> for GoldiModulus {
    const MODULUS: U64 = U64::from_u64(18_446_744_069_414_584_321);
}

type Lw64 = U64PrimeField<GoldiModulus>;

/// Fixed-seed LCG in fgf's noise shape: deterministic coefficients.
fn noise_elem(seed: u64) -> [u8; 8] {
    let state = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    state.to_le_bytes()
}

fn noise_elems(len: usize, seed: u64) -> Vec<[u8; 8]> {
    (0..len)
        .map(|i| noise_elem(seed.wrapping_add(i as u64)))
        .collect()
}

fn noise_poly(len: usize, seed: u64) -> poly_ring::Polynomial<Gf64> {
    let coefficients: Vec<Gf64Elem> = noise_elems(len, seed)
        .iter()
        .map(|bytes| <Gf64 as fgf::field::Field>::read(bytes))
        .collect();
    poly_ring::Polynomial::from_coefficients(&coefficients).expect("bench polynomial")
}

fn ark_dense(len: usize, seed: u64) -> ArkPoly {
    let coefficients: Vec<ArkFp64> = noise_elems(len, seed)
        .iter()
        .map(|bytes| ArkFp64::from(u64::from_le_bytes(*bytes)))
        .collect();
    ark_poly::polynomial::univariate::DensePolynomial::from_coefficients_vec(coefficients)
}

fn noise_lw(len: usize, seed: u64) -> LwPoly {
    let coefficients: Vec<LwFe> = noise_elems(len, seed)
        .iter()
        .map(|bytes| LwFe::new(U64::from_u64(u64::from_le_bytes(*bytes))))
        .collect();
    LwPoly::new(&coefficients)
}

fn gf_elem(bytes: &[u8; 8]) -> Gf64Elem {
    <Gf64 as fgf::field::Field>::read(bytes)
}

fn lw_elem(bytes: &[u8; 8]) -> LwFe {
    LwFe::new(U64::from_u64(u64::from_le_bytes(*bytes)))
}

fn multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("competitor_multiply");
    for len in [64_usize, 256, 1024, 4096] {
        let left = noise_poly(len, 0x5EED_1000);
        let right = noise_poly(len, 0x5EED_2000);
        let ark_left = ark_dense(len, 0x5EED_1000);
        let ark_right = ark_dense(len, 0x5EED_2000);
        let lw_left = noise_lw(len, 0x5EED_1000);
        let lw_right = noise_lw(len, 0x5EED_2000);

        group.bench_function(format!("ours_dispatch/{len}"), |b| {
            b.iter(|| left.multiply(&right).expect("product"))
        });
        group.bench_function(format!("ours_schoolbook/{len}"), |b| {
            b.iter(|| {
                left.multiply_truncated(&right, 2 * len - 1)
                    .expect("product")
            })
        });
        group.bench_function(format!("ark_fft_default/{len}"), |b| {
            b.iter(|| &ark_left * &ark_right)
        });
        group.bench_function(format!("ark_schoolbook/{len}"), |b| {
            b.iter(|| ark_left.naive_mul(&ark_right))
        });
        group.bench_function(format!("lambdaworks_schoolbook/{len}"), |b| {
            b.iter(|| lw_left.mul_with_ref(&lw_right))
        });
    }
    group.finish();
}

fn divide(c: &mut Criterion) {
    let mut group = c.benchmark_group("competitor_divide");
    let dividend = noise_poly(512, 0x5EED_3000);
    let divisor = noise_poly(128, 0x5EED_3001);
    let ark_dividend = ark_dense(512, 0x5EED_3000);
    let ark_divisor = ark_dense(128, 0x5EED_3001);
    let lw_dividend = noise_lw(512, 0x5EED_3000);
    let lw_divisor = noise_lw(128, 0x5EED_3001);

    group.bench_function("ours/512x128", |b| {
        b.iter(|| dividend.div_rem(&divisor).expect("division"))
    });
    group.bench_function("ark_naive/512x128", |b| {
        b.iter(|| {
            ark_poly::polynomial::univariate::DenseOrSparsePolynomial::from(&ark_dividend)
                .divide_with_q_and_r(
                    &ark_poly::polynomial::univariate::DenseOrSparsePolynomial::from(&ark_divisor),
                )
                .expect("division")
        })
    });
    group.bench_function("lambdaworks/512x128", |b| {
        b.iter(|| {
            lw_dividend
                .clone()
                .long_division_with_remainder(&lw_divisor)
        })
    });
    group.finish();
}

/// Both sides compute the Bézout triple (ours: `gcd_ext`, lambdaworks:
/// `xgcd`). ark-poly offers no polynomial gcd or EEA at all, so it has no
/// row here; that gap is recorded in `BENCHMARKS.md`.
fn gcd_eea(c: &mut Criterion) {
    let mut group = c.benchmark_group("competitor_gcd_eea");
    let left = noise_poly(400, 0x5EED_4000);
    let right = noise_poly(300, 0x5EED_4001);
    let lw_left = noise_lw(400, 0x5EED_4000);
    let lw_right = noise_lw(300, 0x5EED_4001);

    group.bench_function("ours_ext/400x300", |b| {
        b.iter(|| left.gcd_ext(&right).expect("extended gcd"))
    });
    group.bench_function("lambdaworks_xgcd/400x300", |b| {
        b.iter(|| lw_left.xgcd(&lw_right))
    });
    group.finish();
}

fn evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("competitor_evaluate");
    let poly = noise_poly(4096, 0x5EED_5000);
    let ark_poly_4096 = ark_dense(4096, 0x5EED_5000);
    let lw_poly = noise_lw(4096, 0x5EED_5000);
    let point = noise_elem(0x5EED_5001);

    group.bench_function("ours_horner/4096", |b| {
        b.iter(|| poly.evaluate(gf_elem(&point)))
    });
    group.bench_function("ark/4096", |b| {
        b.iter(|| ark_poly_4096.evaluate(&ArkFp64::from(u64::from_le_bytes(point))))
    });
    group.bench_function("lambdaworks/4096", |b| {
        b.iter(|| lw_poly.evaluate(&lw_elem(&point)))
    });
    group.finish();
}

/// Lagrange interpolation at 64 arbitrary points. ark-poly has no
/// arbitrary-point interpolation, only domain-based
/// (`Evaluations::interpolate`), so its row runs over a radix-2 domain of
/// the same size and is labeled as such.
fn interpolate(c: &mut Criterion) {
    let n = 64;
    let points: Vec<Gf64Elem> = (0..n)
        .map(|i| {
            let mut bytes = [0_u8; 8];
            bytes[0] = i as u8 + 1;
            gf_elem(&bytes)
        })
        .collect();
    let values: Vec<Gf64Elem> = (0..n)
        .map(|i| gf_elem(&noise_elem(0x5EED_6000 + i as u64)))
        .collect();
    let lw_xs: Vec<LwFe> = points.iter().map(|e| lw_elem(&e.to_bytes())).collect();
    let lw_ys: Vec<LwFe> = values.iter().map(|e| lw_elem(&e.to_bytes())).collect();
    let ark_ys: Vec<ArkFp64> = values
        .iter()
        .map(|e| ArkFp64::from(u64::from_le_bytes(e.to_bytes())))
        .collect();
    let ark_domain = ark_poly::Radix2EvaluationDomain::<ArkFp64>::new(n).expect("domain");

    let mut group = c.benchmark_group("competitor_interpolate");
    group.bench_function("ours_lagrange/64", |b| {
        b.iter(|| {
            poly_ring::interpolate_lagrange::<Gf64>(&points, &values).expect("interpolation")
        })
    });
    group.bench_function("ark_domain_ifft/64", |b| {
        b.iter(|| {
            ark_poly::Evaluations::from_vec_and_domain(ark_ys.clone(), ark_domain).interpolate()
        })
    });
    group.bench_function("lambdaworks_lagrange/64", |b| {
        b.iter(|| LwPoly::interpolate(&lw_xs, &lw_ys).expect("interpolation"))
    });
    group.finish();
}

criterion_group!(benches, multiply, divide, gcd_eea, evaluate, interpolate);
criterion_main!(benches);
