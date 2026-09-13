//! Product, division, gcd/EEA, series, root, and evaluation benchmarks.

use criterion::{Criterion, criterion_group, criterion_main};
use fgf::{Gf8B, Gf16};
use poly_ring::Polynomial;

fn noise_poly<F: fgf::kernel::FieldKernels>(len: usize, seed: u64) -> Polynomial<F> {
    // Fixed-seed LCG in fgf's noise shape: deterministic coefficients.
    let mut state = seed;
    let coefficients: Vec<F::Elem> = (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bytes = state.to_le_bytes();
            F::read(&bytes[..F::BYTES])
        })
        .collect();
    Polynomial::from_coefficients(&coefficients).expect("bench polynomial")
}

fn multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiply");
    for len in [16_usize, 64, 256, 1024, 2048] {
        let left: Polynomial<Gf8B> = noise_poly(len, 0x5EED_0001);
        let right: Polynomial<Gf8B> = noise_poly(len, 0x5EED_0002);
        group.bench_function(format!("schoolbook/{len}"), |b| {
            b.iter(|| {
                left.multiply_truncated(&right, 2 * len - 1)
                    .expect("product")
            })
        });
        group.bench_function(format!("karatsuba/{len}"), |b| {
            b.iter(|| poly_ring::karatsuba_multiply(&left, &right).expect("product"))
        });
    }
    group.finish();
}

fn divide_and_gcd(c: &mut Criterion) {
    let mut group = c.benchmark_group("divide");
    let dividend: Polynomial<Gf16> = noise_poly(256, 0x5EED_0003);
    let divisor: Polynomial<Gf16> = noise_poly(64, 0x5EED_0004);
    group.bench_function("div_rem/256x64", |b| {
        b.iter(|| dividend.div_rem(&divisor).expect("division"))
    });
    group.finish();

    let mut group = c.benchmark_group("gcd");
    let left: Polynomial<Gf16> = noise_poly(200, 0x5EED_0005);
    let right: Polynomial<Gf16> = noise_poly(150, 0x5EED_0006);
    group.bench_function("plain/200x150", |bencher| {
        bencher.iter(|| left.gcd(&right).expect("gcd"))
    });
    group.bench_function("ext/200x150", |bencher| {
        bencher.iter(|| left.gcd_ext(&right).expect("extended gcd"))
    });
    group.finish();
}

fn series(c: &mut Criterion) {
    let mut group = c.benchmark_group("series");
    let mut unit = noise_poly::<Gf16>(512, 0x5EED_0007);
    unit.set_coefficient(0, <Gf16 as fgf::field::Field>::Elem::ONE)
        .expect("unit constant");
    group.bench_function("inverse_mod_x/512", |b| {
        b.iter(|| unit.inverse_mod_x_power(512).expect("inverse"))
    });
    group.finish();
}

criterion_group!(benches, multiply, divide_and_gcd, series);
criterion_main!(benches);
