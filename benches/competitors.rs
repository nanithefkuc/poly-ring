//! Competitor benchmarks for `poly_ring` (`BENCHMARKS.md` panels).
//!
//! Two panels, one file:
//!
//! - `ring_panel`: ring operations (multiply, divide, gcd/EEA, evaluate,
//!   interpolate) against ark-poly and lambdaworks-math over 8-byte fields.
//!   Neither competitor can instantiate GF(2^m) (prime-base fields only), so
//!   both sides run element-size-matched fields: this crate over `fgf::Gf64`,
//!   the competitors over the Goldilocks prime `2^64 - 2^32 + 1`. Numbers
//!   are indicative of throughput at matched element width, not of identical
//!   domains; `BENCHMARKS.md` records the caveats.
//! - `hasse_panel`: derivative/jet evaluation against ark-poly,
//!   lambdaworks-math, and winter-math over the shared Goldilocks prime.
//!   For multiplicity above one no direct competitor found — none of the
//!   three exposes a weighted Hasse-jet API — so the panel records an
//!   explicitly labelled composed baseline (independent `u128` binomials,
//!   then each library's per-order evaluation loop), asserted equal to this
//!   crate's jets before any timing.
//!
//! All competitors are dev-only (ground rule 3 exception); copyleft NTL,
//! FLINT, and gf2x stay out of the build and the measurement set.

use criterion::{criterion_group, criterion_main};

mod ring_panel {

    use ark_poly::{DenseUVPolynomial, EvaluationDomain, Polynomial};
    use criterion::Criterion;
    use fgf::Gf64;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::montgomery_backed_prime_fields::{
        IsModulus, U64PrimeField,
    };
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

    pub fn multiply(c: &mut Criterion) {
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

    pub fn divide(c: &mut Criterion) {
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
                        &ark_poly::polynomial::univariate::DenseOrSparsePolynomial::from(
                            &ark_divisor,
                        ),
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
    pub fn gcd_eea(c: &mut Criterion) {
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

    pub fn evaluate(c: &mut Criterion) {
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
    pub fn interpolate(c: &mut Criterion) {
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
}

mod hasse_panel {
    use ark_ff::PrimeField;
    use ark_poly::{DenseUVPolynomial, Polynomial};
    use criterion::{BenchmarkId, Criterion, Throughput};
    use fgf::{Goldilocks, goldilocks};
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::montgomery_backed_prime_fields::{
        IsModulus, U64PrimeField,
    };
    use lambdaworks_math::polynomial::Polynomial as LwPolynomial;
    use lambdaworks_math::unsigned_integer::element::U64;
    use poly_ring::MultiplicityPlan;
    use std::hint::black_box;
    use winter_math::fields::f64::BaseElement;
    use winter_math::polynom;

    /// The Goldilocks prime every side shares.
    const MODULUS: u128 = 18_446_744_069_414_584_321;

    type ArkPoly = ark_poly::polynomial::univariate::DensePolynomial<ArkFp64>;
    type LwFe = FieldElement<Lw64>;
    type LwPoly = LwPolynomial<LwFe>;

    /// ark-ff Goldilocks prime, one 64-bit limb, `2^32 | p − 1`.
    #[derive(ark_ff::MontConfig)]
    #[modulus = "18446744069414584321"]
    #[generator = "7"]
    pub struct ArkFp64Config;

    pub type ArkFp64 = ark_ff::Fp<ark_ff::MontBackend<ArkFp64Config, 1>, 1>;

    /// Goldilocks modulus for lambdaworks' Montgomery-backed `U64` field.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct GoldiModulus;

    impl IsModulus<U64> for GoldiModulus {
        const MODULUS: U64 = U64::from_u64(18_446_744_069_414_584_321_u64);
    }

    type Lw64 = U64PrimeField<GoldiModulus>;

    /// One canonical Goldilocks value from the fixed LCG, as a reduced `u128`.
    fn lcg(state: &mut u64) -> u128 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u128::from(*state) % MODULUS
    }

    fn canonical_values(len: usize, seed: u64) -> Vec<u128> {
        let mut state = seed;
        (0..len).map(|_| lcg(&mut state)).collect()
    }

    fn distinct_points(count: usize, seed: u64) -> Vec<u128> {
        let mut state = seed;
        let mut points = Vec::with_capacity(count);
        while points.len() < count {
            let value = lcg(&mut state);
            if !points.contains(&value) {
                points.push(value);
            }
        }
        points
    }

    fn fgf_elem(value: u128) -> goldilocks::Elem {
        goldilocks::Elem::from_raw(value as u64)
    }

    fn ark_elem(value: u128) -> ArkFp64 {
        ArkFp64::from(value as u64)
    }

    fn lw_elem(value: u128) -> LwFe {
        LwFe::new(U64::from_u64(value as u64))
    }

    /// The canonical Goldilocks value of a lambdaworks element.
    ///
    /// The Montgomery-backed field stores residues scaled by `R = 2^64`, so
    /// reading `.value()` alone returns the residue, not the value; multiplying
    /// by the element `R^{-1}` unscales before the read.
    fn lw_canonical(element: &LwFe) -> u128 {
        let r_inverse = mod_inverse(1_u128 << 64);
        u128::from((element.clone() * lw_elem(r_inverse)).value().limbs[0])
    }

    fn winter_elem(value: u128) -> BaseElement {
        BaseElement::new(value as u64)
    }

    /// `C(n, k) mod p` in `u128` arithmetic — valid while `n < p`, which every
    /// benchmark degree satisfies.
    fn binomial_mod(n: usize, k: usize) -> u128 {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        let mut result: u128 = 1;
        for i in 0..k {
            let numerator = (n - i) as u128;
            let denominator = (i + 1) as u128;
            // Divide through with modular inverses; all factors are below p and
            // nonzero, so invertible.
            result = result * numerator % MODULUS;
            result = result * mod_inverse(denominator) % MODULUS;
        }
        result
    }

    fn mod_inverse(value: u128) -> u128 {
        // p is prime and value < p, nonzero: Fermat.
        let mut result: u128 = 1;
        let mut base = value % MODULUS;
        let mut exponent = MODULUS - 2;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result * base % MODULUS;
            }
            base = base * base % MODULUS;
            exponent >>= 1;
        }
        result
    }

    /// The composed baseline's independent Hasse derivative, `u128` binomials.
    fn derivative_mod(coefficients: &[u128], order: usize) -> Vec<u128> {
        (0..coefficients.len().saturating_sub(order))
            .map(|output| {
                (binomial_mod(output + order, order) * coefficients[output + order]) % MODULUS
            })
            .collect()
    }

    fn multiplicity_one(c: &mut Criterion) {
        let mut group = c.benchmark_group("competitors/goldilocks/multiplicity-1");

        for point_count in [16_usize, 64, 256, 1024] {
            for degree in [16_usize, 64, 256, 1024] {
                let coefficient_values = canonical_values(degree + 1, 0xBEEF_0000 + degree as u64);
                let point_values = distinct_points(point_count, 0xC0FF_0000 + point_count as u64);

                // hasse side.
                let coefficients: Vec<goldilocks::Elem> = coefficient_values
                    .iter()
                    .map(|&value| fgf_elem(value))
                    .collect();
                let points: Vec<goldilocks::Elem> =
                    point_values.iter().map(|&value| fgf_elem(value)).collect();
                let plan = MultiplicityPlan::<Goldilocks>::uniform(&points, 1, coefficients.len())
                    .expect("plan");
                let mut scratch = plan.scratch(1).expect("scratch");
                let mut output = vec![goldilocks::Elem::ZERO; point_count];
                plan.evaluate_into(&coefficients, &mut scratch, &mut output)
                    .expect("evaluate");
                let expected: Vec<u128> = output
                    .iter()
                    .map(|element| u128::from(element.to_raw()))
                    .collect();

                // ark side: per-point public evaluator loop.
                let ark = ArkPoly::from_coefficients_vec(
                    coefficient_values
                        .iter()
                        .map(|&value| ark_elem(value))
                        .collect(),
                );
                let ark_points: Vec<ArkFp64> =
                    point_values.iter().map(|&value| ark_elem(value)).collect();
                let ark_values: Vec<u128> = ark_points
                    .iter()
                    .map(|point| u128::from(ark.evaluate(point).into_bigint().0[0]))
                    .collect();
                assert_eq!(ark_values, expected, "ark must agree before timing");

                // lambdaworks side: `Polynomial::new` stores ascending
                // coefficients, the same low-to-high order as this panel.
                let lw = LwPoly::new(
                    &coefficient_values
                        .iter()
                        .map(|&value| lw_elem(value))
                        .collect::<Vec<_>>(),
                );
                let lw_points: Vec<LwFe> =
                    point_values.iter().map(|&value| lw_elem(value)).collect();
                let lw_values: Vec<u128> = lw_points
                    .iter()
                    .map(|point| lw_canonical(&lw.evaluate(point)))
                    .collect();
                assert_eq!(lw_values, expected, "lambdaworks must agree before timing");

                // winter-math side.
                let winter_coefficients: Vec<BaseElement> = coefficient_values
                    .iter()
                    .map(|&value| winter_elem(value))
                    .collect();
                let winter_points: Vec<BaseElement> = point_values
                    .iter()
                    .map(|&value| winter_elem(value))
                    .collect();
                let winter_values: Vec<u128> = winter_points
                    .iter()
                    .map(|point| u128::from(polynom::eval(&winter_coefficients, *point).as_int()))
                    .collect();
                assert_eq!(
                    winter_values, expected,
                    "winter-math must agree before timing"
                );

                let geometry = format!("points={point_count}/D={degree}");
                group.throughput(Throughput::Elements(
                    (degree as u64 + 1) * point_count as u64,
                ));

                group.bench_with_input(
                    BenchmarkId::new("hasse/multiplicity-plan", &geometry),
                    &(),
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
                group.bench_with_input(
                    BenchmarkId::new("hasse/plan-construction", &geometry),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            black_box(
                                MultiplicityPlan::<Goldilocks>::uniform(
                                    black_box(&points),
                                    black_box(1),
                                    black_box(coefficients.len()),
                                )
                                .expect("plan"),
                            );
                        });
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("ark-poly/per-point-evaluate", &geometry),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for point in black_box(&ark_points) {
                                black_box(ark.evaluate(point));
                            }
                        });
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("lambdaworks/per-point-evaluate", &geometry),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for point in black_box(&lw_points) {
                                black_box(lw.evaluate(point));
                            }
                        });
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("winter-math/eval-many", &geometry),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            black_box(polynom::eval_many(
                                black_box(&winter_coefficients),
                                black_box(&winter_points),
                            ));
                        });
                    },
                );
            }
        }
        group.finish();
    }

    fn composed_baseline(c: &mut Criterion) {
        let mut group = c.benchmark_group("competitors/goldilocks/composed-Hasse-baseline");

        for multiplicity in [2_usize, 4, 8, 16] {
            let point_count = 64_usize;
            let degree = 64_usize;
            let coefficient_values =
                canonical_values(degree + 1, 0xDEAD_0000 + multiplicity as u64);
            let point_values = distinct_points(point_count, 0xF00D_0000 + multiplicity as u64);

            let coefficients: Vec<goldilocks::Elem> = coefficient_values
                .iter()
                .map(|&value| fgf_elem(value))
                .collect();
            let points: Vec<goldilocks::Elem> =
                point_values.iter().map(|&value| fgf_elem(value)).collect();
            let plan =
                MultiplicityPlan::<Goldilocks>::uniform(&points, multiplicity, coefficients.len())
                    .expect("plan");
            let mut scratch = plan.scratch(1).expect("scratch");
            let mut output = vec![goldilocks::Elem::ZERO; plan.total_weight()];
            plan.evaluate_into(&coefficients, &mut scratch, &mut output)
                .expect("evaluate");

            let geometry = format!("s={multiplicity}/points={point_count}/D={degree}");
            group.throughput(Throughput::Elements(
                ((degree + 1) * point_count * multiplicity) as u64,
            ));

            // hasse one-shot: a first-time caller builds the plan, the scratch,
            // and the output buffer, then evaluates — the same shape as the
            // libraries' one-shot column, which builds every derivative
            // polynomial before evaluating.
            group.bench_with_input(
                BenchmarkId::new("hasse/composed-one-shot", &geometry),
                &(),
                |bench, _| {
                    bench.iter(|| {
                        let plan = MultiplicityPlan::<Goldilocks>::uniform(
                            black_box(&points),
                            black_box(multiplicity),
                            black_box(coefficients.len()),
                        )
                        .expect("plan");
                        let mut scratch = plan.scratch(black_box(1)).expect("scratch");
                        let mut output = vec![goldilocks::Elem::ZERO; plan.total_weight()];
                        plan.evaluate_into(
                            black_box(&coefficients),
                            black_box(&mut scratch),
                            black_box(&mut output),
                        )
                        .expect("evaluate");
                        black_box(&output);
                    });
                },
            );
            // hasse prepared: plan and scratch built outside the timer.
            group.bench_with_input(
                BenchmarkId::new("hasse/composed-prepared", &geometry),
                &(),
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

            // Composed baselines: independent u128-binomial derivative
            // preparation, then each library's evaluator once per order,
            // materializing every per-order/per-point result. Timed one-shot
            // (preparation included) and prepared (excluded). The Montgomery
            // unscale element is computed once: reading a lambdaworks value
            // canonically costs one unscale multiply, never a per-point
            // Fermat inversion.
            let lw_unscale = lw_elem(mod_inverse(1_u128 << 64));
            for (library, name) in [
                ("ark", "ark-poly"),
                ("lw", "lambdaworks"),
                ("winter", "winter-math"),
            ] {
                let evaluate_all = |derivatives: &[Vec<u128>]| -> Vec<Vec<u128>> {
                    match library {
                        "ark" => derivatives
                            .iter()
                            .map(|derivative| {
                                let poly = ArkPoly::from_coefficients_vec(
                                    derivative.iter().map(|&value| ark_elem(value)).collect(),
                                );
                                point_values
                                    .iter()
                                    .map(|&value| {
                                        u128::from(
                                            poly.evaluate(&ark_elem(value)).into_bigint().0[0],
                                        )
                                    })
                                    .collect::<Vec<u128>>()
                            })
                            .collect(),
                        "lw" => derivatives
                            .iter()
                            .map(|derivative| {
                                let poly = LwPoly::new(
                                    &derivative
                                        .iter()
                                        .map(|&value| lw_elem(value))
                                        .collect::<Vec<_>>(),
                                );
                                point_values
                                    .iter()
                                    .map(|&value| {
                                        u128::from(
                                            (poly.evaluate(&lw_elem(value)) * lw_unscale.clone())
                                                .value()
                                                .limbs[0],
                                        )
                                    })
                                    .collect::<Vec<u128>>()
                            })
                            .collect(),
                        _ => derivatives
                            .iter()
                            .map(|derivative| {
                                let polynomial: Vec<BaseElement> =
                                    derivative.iter().map(|&value| winter_elem(value)).collect();
                                point_values
                                    .iter()
                                    .map(|&value| {
                                        u128::from(
                                            polynom::eval(&polynomial, winter_elem(value)).as_int(),
                                        )
                                    })
                                    .collect::<Vec<u128>>()
                            })
                            .collect(),
                    }
                };
                let one_shot = || {
                    let derivatives: Vec<Vec<u128>> = (0..multiplicity)
                        .map(|order| derivative_mod(&coefficient_values, order))
                        .collect();
                    evaluate_all(&derivatives)
                };
                let prepared_derivatives: Vec<Vec<u128>> = (0..multiplicity)
                    .map(|order| derivative_mod(&coefficient_values, order))
                    .collect();

                // Cross-check before any timing: result[order][point] must be
                // hasse's D^[order]f(a_point) at output position
                // point * multiplicity + order — every value, both paths.
                let checked = one_shot();
                assert_eq!(checked.len(), multiplicity, "{name} order count");
                for (order, row) in checked.iter().enumerate() {
                    assert_eq!(row.len(), point_count, "{name} point count");
                    for (point, &value) in row.iter().enumerate() {
                        assert_eq!(
                            value,
                            u128::from(output[point * multiplicity + order].to_raw()),
                            "{name} D^[{order}] at point {point} must match hasse"
                        );
                    }
                }
                assert_eq!(
                    evaluate_all(&prepared_derivatives),
                    checked,
                    "{name} prepared path must match its own one-shot results"
                );

                group.bench_with_input(
                    BenchmarkId::new(format!("{name}/composed-one-shot"), &geometry),
                    &(),
                    |bench, _| {
                        bench.iter(|| black_box(one_shot()));
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("{name}/composed-prepared"), &geometry),
                    &(),
                    |bench, _| {
                        bench.iter(|| black_box(evaluate_all(&prepared_derivatives)));
                    },
                );
            }
        }
        group.finish();
    }

    pub fn competitors(c: &mut Criterion) {
        multiplicity_one(c);
        composed_baseline(c);
    }
}

criterion_group!(
    benches,
    ring_panel::multiply,
    ring_panel::divide,
    ring_panel::gcd_eea,
    ring_panel::evaluate,
    ring_panel::interpolate,
    hasse_panel::competitors
);
criterion_main!(benches);
