//! Hasse panel: plan construction, scratch construction, and steady-state
//! weighted evaluation, separated.
//!
//! Fields: Gf8B, Gf16, Mersenne31, Goldilocks, QuadMersenne31. Point
//! counts 16, 64, 256, 1024, skipped — never silently shrunk — where the
//! count exceeds the field's element count (Gf8B tops out at 256, so its
//! 1024-point geometry does not run); uniform multiplicities 1, 2, 4, 8,
//! 16, 64; input degree counts around W/2, W, and 2W; plus a nonuniform
//! [1, 2, 4, 8]-repeat panel and a single heavy leaf. Inputs use every
//! coefficient; outputs are consumed through Criterion's black_box.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, Goldilocks, Mersenne31, QuadMersenne31};
use poly_ring::MultiplicityPlan;
use std::hint::black_box;

fn noise<F: FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bytes = state.to_le_bytes();
            F::read(&bytes[..F::BYTES])
        })
        .collect()
}

fn points<F: FieldKernels>(count: usize) -> Vec<F::Elem> {
    // Deterministic distinct points from the fixed LCG, canonicalized; the
    // stream is extended until `count` distinct values are gathered, so
    // every labelled count is the count actually measured.
    let mut state = 0xC0FFEE_u64;
    let mut distinct = Vec::with_capacity(count);
    while distinct.len() < count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bytes = state.to_le_bytes();
        let value = F::read(&bytes[..F::BYTES]).add(F::Elem::ZERO);
        if !distinct.contains(&value) {
            distinct.push(value);
        }
    }
    distinct
}

fn panel<F: poly_ring::PolynomialField>(criterion: &mut Criterion, name: &str) {
    let mut group = criterion.benchmark_group(format!("hasse/{name}"));

    for point_count in [16_usize, 64, 256, 1024] {
        // Distinct points cannot exceed the field's element count; skip —
        // never silently shrink — geometries the field cannot supply
        // (Gf8B's 1024-point panel).
        if point_count as u128 > F::ORDER {
            continue;
        }
        let point_set = points::<F>(point_count);
        assert_eq!(
            point_set.len(),
            point_count,
            "the LCG stream must supply every count below the field's order"
        );
        let Some(&first) = point_set.first() else {
            continue;
        };
        for multiplicity in [1_usize, 2, 4, 8, 16, 64] {
            let plan = MultiplicityPlan::<F>::uniform(&point_set, multiplicity, 2 * 1024 * 64)
                .expect("plan");
            let weight = plan.total_weight();

            group.bench_with_input(
                BenchmarkId::new(
                    format!("construct/points={point_count}/s={multiplicity}"),
                    "plan",
                ),
                &weight,
                |bench, _| {
                    bench.iter(|| {
                        black_box(
                            MultiplicityPlan::<F>::uniform(
                                black_box(&point_set),
                                black_box(multiplicity),
                                black_box(2 * 1024 * 64),
                            )
                            .expect("plan"),
                        )
                    });
                },
            );

            let mut scratch = plan.scratch(1).expect("scratch");
            group.bench_with_input(
                BenchmarkId::new(
                    format!("construct/points={point_count}/s={multiplicity}"),
                    "scratch",
                ),
                &weight,
                |bench, _| {
                    bench.iter(|| black_box(plan.scratch(black_box(1)).expect("scratch")));
                },
            );

            for factor in [2_usize, 1, 3] {
                let count = (weight * factor / 2).max(1);
                let coefficients = noise::<F>(count, 0x5EED_0000 + count as u64);
                let mut output = vec![first; weight];
                group.throughput(Throughput::Elements(count as u64 + weight as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("evaluate/points={point_count}/s={multiplicity}/D={count}"),
                        "scalar",
                    ),
                    &count,
                    |bench, _| {
                        bench.iter(|| {
                            plan.evaluate_into(
                                black_box(&coefficients),
                                black_box(&mut scratch),
                                black_box(&mut output),
                            )
                            .expect("evaluate");
                        });
                    },
                );
                // Black-box the outputs so the run cannot be elided.
                black_box(&output);
            }
        }
    }

    // Nonuniform [1, 2, 4, 8] repeats and one heavy single leaf.
    let point_set = points::<F>(64);
    let weights: Vec<usize> = (0..64).map(|index| [1, 2, 4, 8][index % 4]).collect();
    let plan = MultiplicityPlan::<F>::new(&point_set, &weights, 4096).expect("nonuniform plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let coefficients = noise::<F>(plan.total_weight(), 0x5EED_9999);
    let mut output = vec![point_set[0]; plan.total_weight()];
    group.bench_function("evaluate/nonuniform-[1,2,4,8]x16/D=W", |bench| {
        bench.iter(|| {
            plan.evaluate_into(
                black_box(&coefficients),
                black_box(&mut scratch),
                black_box(&mut output),
            )
            .expect("evaluate");
        });
    });

    let mut heavy = vec![1_usize; 63];
    heavy.push(64);
    let plan = MultiplicityPlan::<F>::new(&point_set, &heavy, 4096).expect("heavy plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let coefficients = noise::<F>(plan.total_weight(), 0x5EED_8888);
    let mut output = vec![point_set[0]; plan.total_weight()];
    group.bench_function("evaluate/heavy-leaf-s=64/D=W", |bench| {
        bench.iter(|| {
            plan.evaluate_into(
                black_box(&coefficients),
                black_box(&mut scratch),
                black_box(&mut output),
            )
            .expect("evaluate");
        });
    });

    group.finish();
}

fn criterion_config(c: &mut Criterion) {
    panel::<Gf8B>(c, "gf8b");
    panel::<Gf16>(c, "gf16");
    panel::<Mersenne31>(c, "mersenne31");
    panel::<Goldilocks>(c, "goldilocks");
    panel::<QuadMersenne31>(c, "quad-mersenne31");
}

criterion_group!(benches, criterion_config);
criterion_main!(benches);
