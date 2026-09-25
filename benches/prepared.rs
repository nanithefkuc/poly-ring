//! Prepared product benchmarks: `multiply_rows_into` across routes, sizes,
//! batches, and fields, with scratch construction separated from
//! steady-state execution.
//!
//! Full product sizes 64 through 65536 by powers of two, batch sizes 1, 4,
//! and 16. Fields: Gf8B, Gf16, Mersenne31, Goldilocks, QuadMersenne31.
//! Mathematically unsupported forced transforms are labelled and omitted,
//! never faked: under `internals`, the forced transform route runs only for
//! the fields whose multiplicative (or embedded) domain covers the size.
//!
//! `prepared_tuning_goldilocks` compares automatic, Karatsuba, and transform
//! routes over equal and asymmetric Goldilocks products. Inputs and scratch
//! stay outside the timed region, and every route is checked before timing.
//!
//! Run with `cargo bench --bench prepared --features internals`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fgf::kernel::FieldKernels;
use fgf::{Field, Gf8B, Gf16, Goldilocks, Mersenne31, QuadMersenne31};
use poly_ring::PolynomialField;

#[cfg(feature = "internals")]
use poly_ring::internals::{ProductRoute, multiply_rows_route_into};
use poly_ring::{ConvolutionScratch, multiply_rows_into};

fn noise<F: FieldKernels>(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut buffer = vec![0_u8; len * F::BYTES];
    for degree in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        F::encode(
            &mut buffer[degree * F::BYTES..(degree + 1) * F::BYTES],
            F::decode(&state.to_le_bytes()[..F::BYTES]),
        );
    }
    buffer
}

fn lanes<F: FieldKernels>(count: usize, batch: usize, seed: u64) -> Vec<u8> {
    let mut buffer = vec![0_u8; count * batch * F::BYTES];
    for lane in 0..batch {
        let coefficients = noise::<F>(count, seed.wrapping_add(lane as u64));
        for degree in 0..count {
            let offset = (degree * batch + lane) * F::BYTES;
            buffer[offset..offset + F::BYTES]
                .copy_from_slice(&coefficients[degree * F::BYTES..(degree + 1) * F::BYTES]);
        }
    }
    buffer
}

fn panel<F: PolynomialField>(criterion: &mut Criterion, name: &str, transform_cap: usize) {
    let mut group = criterion.benchmark_group(format!("prepared/{name}"));

    for size in [64_usize, 256, 1024, 4096, 16384, 65536] {
        for batch in [1_usize, 4, 16] {
            let left = lanes::<F>(size, batch, 0x5EED_1000 + size as u64);
            let right = lanes::<F>(size, batch, 0x5EED_2000 + size as u64);
            let full = 2 * size - 1;
            let mut output = vec![0_u8; full * batch * F::BYTES];

            group.throughput(Throughput::Elements(
                (size as u64) * (size as u64) * batch as u64,
            ));
            let geometry = format!("size={size}/batch={batch}");

            // Scratch construction, including the transform-plan cache.
            group.bench_with_input(
                BenchmarkId::new("construct-scratch", &geometry),
                &(),
                |bench, _| {
                    bench.iter(|| {
                        let mut scratch =
                            ConvolutionScratch::<F>::new(size, size, batch).expect("scratch");
                        scratch
                            .prepare_transform(2 * size - 1, batch)
                            .expect("prepare");
                        scratch
                    });
                },
            );

            let mut scratch = ConvolutionScratch::<F>::new(size, size, batch).expect("scratch");
            scratch
                .prepare_transform(2 * size - 1, batch)
                .expect("prepare");

            group.bench_with_input(
                BenchmarkId::new("execute/auto", &geometry),
                &(),
                |bench, _| {
                    bench.iter(|| {
                        multiply_rows_into::<F>(
                            &mut output,
                            &left,
                            size,
                            &right,
                            size,
                            batch,
                            full,
                            &mut scratch,
                        )
                        .expect("product");
                    });
                },
            );

            // Forced reference modes: the same measurement under an explicit
            // route, so an Auto switch is always a comparison, not a guess.
            #[cfg(feature = "internals")]
            {
                group.bench_with_input(
                    BenchmarkId::new("execute/karatsuba", &geometry),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            multiply_rows_route_into(
                                &mut output,
                                &left,
                                size,
                                &right,
                                size,
                                batch,
                                full,
                                ProductRoute::Karatsuba,
                                &mut scratch,
                            )
                            .expect("product");
                        });
                    },
                );
                // The forced transform route is labelled and omitted beyond
                // the field's domain cap (e.g. Gf8B caps at 256 points),
                // never faked and never silently rerouted.
                let transformable = full
                    .checked_next_power_of_two()
                    .is_some_and(|size| size <= transform_cap);
                if transformable {
                    group.bench_with_input(
                        BenchmarkId::new("execute/transform", &geometry),
                        &(),
                        |bench, _| {
                            bench.iter(|| {
                                multiply_rows_route_into(
                                    &mut output,
                                    &left,
                                    size,
                                    &right,
                                    size,
                                    batch,
                                    full,
                                    ProductRoute::Transform,
                                    &mut scratch,
                                )
                                .expect("product");
                            });
                        },
                    );
                }
            }
        }
    }
    group.finish();
}

fn prepared(c: &mut Criterion) {
    // Gf8B/Gf16: the additive transform domain is the field itself.
    panel::<Gf8B>(c, "gf8b", 256);
    panel::<Gf16>(c, "gf16", 65_536);
    // Mersenne31 (embedded), Goldilocks, QuadMersenne31: the multiplicative
    // route covers every measured size (2^32 | p² − 1 shapes hold to 2^20).
    panel::<Mersenne31>(c, "mersenne31", 1 << 20);
    panel::<Goldilocks>(c, "goldilocks", 1 << 20);
    panel::<QuadMersenne31>(c, "quad-mersenne31", 1 << 20);
}

/// Goldilocks-only internal tuning group: Auto versus forced Karatsuba and
/// forced Transform (`internals`) over the tuning geometries. Not a public
/// record panel — internal A/B evidence for the route selectors.
fn prepared_tuning_goldilocks(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("prepared_tuning/goldilocks");

    let equal: Vec<(usize, usize)> = [64_usize, 96, 128, 160, 192, 256, 384, 512, 768, 1024]
        .into_iter()
        .map(|len| (len, len))
        .collect();
    let geometries: Vec<(usize, usize)> = equal
        .into_iter()
        .chain([(128, 512), (256, 1024), (512, 2048)])
        .collect();

    for (left_len, right_len) in geometries {
        for batch in [1_usize, 4, 16] {
            let left = lanes::<Goldilocks>(left_len, batch, 0x5EED_1000 + left_len as u64);
            let right = lanes::<Goldilocks>(right_len, batch, 0x5EED_2000 + left_len as u64);
            let full = left_len + right_len - 1;

            let mut auto_output = vec![0_u8; full * batch * Goldilocks::BYTES];
            #[cfg(feature = "internals")]
            let mut karatsuba_output = auto_output.clone();
            #[cfg(feature = "internals")]
            let mut transform_output = auto_output.clone();

            // Scratch and inputs built outside the timer.
            let mut scratch =
                ConvolutionScratch::<Goldilocks>::new(left_len, right_len, batch).expect("scratch");
            scratch.prepare_transform(full, batch).expect("prepare");

            multiply_rows_into::<Goldilocks>(
                &mut auto_output,
                &left,
                left_len,
                &right,
                right_len,
                batch,
                full,
                &mut scratch,
            )
            .expect("product");

            #[cfg(feature = "internals")]
            {
                let mut run_forced = |output: &mut Vec<u8>, route: ProductRoute| {
                    multiply_rows_route_into(
                        output,
                        &left,
                        left_len,
                        &right,
                        right_len,
                        batch,
                        full,
                        route,
                        &mut scratch,
                    )
                    .expect("product");
                };
                run_forced(&mut karatsuba_output, ProductRoute::Karatsuba);
                run_forced(&mut transform_output, ProductRoute::Transform);

                // Correctness before timing: every route byte-identical.
                assert_eq!(
                    karatsuba_output, auto_output,
                    "karatsuba must match auto at {left_len}x{right_len}/batch={batch}"
                );
                assert_eq!(
                    transform_output, auto_output,
                    "transform must match auto at {left_len}x{right_len}/batch={batch}"
                );
            }

            let geometry = format!("{left_len}x{right_len}/batch={batch}");
            group.throughput(Throughput::Elements(
                (left_len as u64) * (right_len as u64) * batch as u64,
            ));

            group.bench_with_input(
                BenchmarkId::new("execute/auto", &geometry),
                &(),
                |bench, _| {
                    bench.iter(|| {
                        multiply_rows_into::<Goldilocks>(
                            &mut auto_output,
                            &left,
                            left_len,
                            &right,
                            right_len,
                            batch,
                            full,
                            &mut scratch,
                        )
                        .expect("product");
                    });
                },
            );

            #[cfg(feature = "internals")]
            {
                group.bench_with_input(
                    BenchmarkId::new("execute/karatsuba", &geometry),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            multiply_rows_route_into(
                                &mut karatsuba_output,
                                &left,
                                left_len,
                                &right,
                                right_len,
                                batch,
                                full,
                                ProductRoute::Karatsuba,
                                &mut scratch,
                            )
                            .expect("product");
                        });
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("execute/transform", &geometry),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            multiply_rows_route_into(
                                &mut transform_output,
                                &left,
                                left_len,
                                &right,
                                right_len,
                                batch,
                                full,
                                ProductRoute::Transform,
                                &mut scratch,
                            )
                            .expect("product");
                        });
                    },
                );
            }
        }
    }
    group.finish();
}
criterion_group!(benches, prepared, prepared_tuning_goldilocks);
criterion_main!(benches);
