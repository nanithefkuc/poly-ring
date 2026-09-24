//! Steady-state zero allocation for every `*_into` / scratch-owning path,
//! proven under a counting global allocator (U5).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use fgf::field::{Elem, Field};
use fgf::{Gf8B, Gf16, Goldilocks, Mersenne31, mersenne31};
use poly_ring::{
    BinaryRootScratch, ChienScratch, DomainScratch, EvaluationDomain, MultiplicityPlan,
    MultipointScratch, Polynomial, RemainderTree, RothRuckensteinLimits, RothRuckensteinScratch,
    truncated_eea,
};

struct CountingAllocator;

// The counter is scoped to the counting thread: sibling tests running in
// parallel must not be charged to this measurement.
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.with(Cell::get) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn count_allocations<F>(mut operation: F) -> usize
where
    F: FnMut(),
{
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.with(|counting| counting.set(true));
    operation();
    COUNTING.with(|counting| counting.set(false));
    ALLOCATIONS.load(Ordering::Relaxed)
}

fn noise<F: fgf::kernel::FieldKernels>(len: usize, seed: u64) -> Polynomial<F> {
    let mut state = seed;
    let coefficients: Vec<F::Elem> = (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bytes = state.to_le_bytes();
            F::decode(&bytes[..F::BYTES])
        })
        .collect();
    Polynomial::from_coefficients(&coefficients).expect("noise polynomial")
}

#[test]
fn steady_state_paths_do_not_allocate() {
    let left = noise::<Gf8B>(64, 0x51);
    let right = noise::<Gf8B>(48, 0x52);
    let divisor = noise::<Gf8B>(16, 0x53);
    let mut product = Polynomial::<Gf8B>::zero();
    let mut quotient = Polynomial::<Gf8B>::zero();
    let mut remainder = Polynomial::<Gf8B>::zero();
    let mut square = Polynomial::<Gf8B>::zero();

    // Warm every buffer.
    left.multiply_truncated_into(&right, 111, &mut product)
        .unwrap();
    left.div_rem_into(&divisor, &mut quotient, &mut remainder)
        .unwrap();
    left.square_into(&mut square).unwrap();

    assert_eq!(
        count_allocations(|| {
            left.multiply_truncated_into(&right, 111, &mut product)
                .unwrap();
            left.div_rem_into(&divisor, &mut quotient, &mut remainder)
                .unwrap();
            left.square_into(&mut square).unwrap();
        }),
        0,
        "steady-state product/division/square must not allocate"
    );

    // Root extraction paths. Two warm-up rounds: the equal-degree route
    // ping-pongs two buffer pairs, so the steady state starts on the third
    // call (the same convention gs-engine's decode-allocation tests use).
    let locator = noise::<Gf16>(9, 0x61);
    let mut chien_scratch = ChienScratch::new();
    let mut field_scratch = BinaryRootScratch::new();
    let mut roots = Vec::new();
    for _ in 0..2 {
        poly_ring::chien_roots_into(&mut roots, &locator, &mut chien_scratch).unwrap();
        poly_ring::binary_field_roots_into(&mut roots, &locator, &mut field_scratch).unwrap();
    }
    let chien_count = count_allocations(|| {
        poly_ring::chien_roots_into(&mut roots, &locator, &mut chien_scratch).unwrap();
    });
    let equal_degree_count = count_allocations(|| {
        poly_ring::binary_field_roots_into(&mut roots, &locator, &mut field_scratch).unwrap();
    });
    assert_eq!(chien_count, 0, "warmed Chien scan must not allocate");
    assert_eq!(
        equal_degree_count, 0,
        "warmed equal-degree factorization must not allocate"
    );

    // Multipoint evaluation over a fixed point set.
    let points: Vec<<Gf16 as Field>::Elem> = (1..=40)
        .map(|exponent| <Gf16 as Field>::GENERATOR.pow(exponent as u64))
        .collect();
    let mut multipoint = MultipointScratch::new();
    let mut values = Vec::new();
    // Warm to convergence: whatever route the step count selects, buffer
    // capacities migrate to their steady-state slots over the first
    // rounds.
    for _ in 0..3 {
        poly_ring::evaluate_multipoint_into(&mut values, &locator, &points, &mut multipoint)
            .unwrap();
    }
    assert_eq!(
        count_allocations(|| {
            poly_ring::evaluate_multipoint_into(&mut values, &locator, &points, &mut multipoint)
                .unwrap();
        }),
        0,
        "warmed multipoint evaluation must not allocate"
    );

    // Domain evaluation over a subspace (transform path under `fft`).
    let domain = EvaluationDomain::<Gf16>::additive_subspace(32).unwrap();
    let mut domain_scratch = DomainScratch::new();
    let mut domain_values = Vec::new();
    domain
        .evaluate_into(&locator, &mut domain_scratch, &mut domain_values)
        .unwrap();
    assert_eq!(
        count_allocations(|| {
            domain
                .evaluate_into(&locator, &mut domain_scratch, &mut domain_values)
                .unwrap();
        }),
        0,
        "warmed domain evaluation must not allocate"
    );

    // Roth–Ruckenstein lifting over a fixed geometry.
    let rows = vec![
        noise::<Gf8B>(6, 0x71),
        noise::<Gf8B>(4, 0x72),
        noise::<Gf8B>(3, 0x73),
    ];
    let mut lift_scratch = RothRuckensteinScratch::new();
    let mut lifted = Vec::new();
    // Two warm-up rounds: the lifted base-field factorization ping-pongs
    // buffer pairs internally, so its steady state starts on the third
    // call.
    for _ in 0..2 {
        poly_ring::roth_ruckenstein_roots_into(
            &mut lifted,
            &rows,
            3,
            RothRuckensteinLimits::new(100_000, 256),
            &mut lift_scratch,
        )
        .unwrap();
    }
    assert_eq!(
        count_allocations(|| {
            poly_ring::roth_ruckenstein_roots_into(
                &mut lifted,
                &rows,
                3,
                RothRuckensteinLimits::new(100_000, 256),
                &mut lift_scratch,
            )
            .unwrap();
        }),
        0,
        "warmed Roth–Ruckenstein extraction must not allocate"
    );
}

#[cfg(feature = "fft")]
#[test]
fn afft_product_scratch_reuse_does_not_allocate() {
    use poly_ring::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated_into};

    let left = noise::<Gf16>(80, 0x81);
    let right = noise::<Gf16>(70, 0x82);
    let mut scratch = PolynomialProductScratch::new();
    let mut output = Vec::new();
    multiply_batch_truncated_into(
        &mut output,
        &[(&left, &right); 4].map(|(left, right)| (left, right)),
        149,
        ProductStrategy::Afft,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(
        count_allocations(|| {
            multiply_batch_truncated_into(
                &mut output,
                &[(&left, &right); 4].map(|(left, right)| (left, right)),
                149,
                ProductStrategy::Afft,
                &mut scratch,
            )
            .unwrap();
        }),
        0,
        "warmed AFFT batches must not allocate"
    );
}

#[test]
fn truncated_eea_reports_its_cost_honestly() {
    // The truncated EEA is the allocating form; this test pins that it
    // completes and satisfies its identity, not its allocation profile.
    let a = noise::<Gf8B>(20, 0x91);
    let b = noise::<Gf8B>(14, 0x92);
    let step = truncated_eea(&a, &b, 4).expect("truncated eea");
    let identity = step
        .a_cofactor
        .multiply(&a)
        .expect("u·a")
        .add(&step.b_cofactor.multiply(&b).expect("v·b"))
        .expect("sum");
    assert_eq!(identity, step.remainder);
}

// ---------------------------------------------------------------------------
// Weighted multipoint evaluation: first call and steady state.
//
// The plan, scratch, and output buffers are the same machinery the
// multiplicity suites warm; here the counting allocator proves both regimes
// allocation-free across zero → dense → short coefficient transitions,
// binary and prime fields, batch 1 and 8, scalar and packed paths.

fn noise_values<F: fgf::kernel::FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bytes = state.to_le_bytes();
            F::decode(&bytes[..F::BYTES])
        })
        .collect()
}

fn multiplicity_points<F: fgf::kernel::FieldKernels>(count: usize) -> Vec<F::Elem> {
    (0..count)
        .map(|index| {
            let mut value = F::Elem::ONE;
            for _ in 0..index {
                value = value.add(F::Elem::ONE);
            }
            value
        })
        .collect()
}

fn assert_zero_alloc<F>(plan: &MultiplicityPlan<F>, batch: usize, max_count: usize, seeds: [u64; 3])
where
    F: poly_ring::PolynomialField,
{
    let weight = plan.total_weight();
    let mut scratch = plan.scratch(batch).expect("scratch");
    let row_bytes = batch * F::BYTES;
    let mut packed = vec![0_u8; max_count * row_bytes];
    let mut output = vec![0_u8; weight * row_bytes];

    let write_input = |packed: &mut [u8], coefficients: &[F::Elem]| {
        for (degree, value) in coefficients.iter().enumerate() {
            for lane in 0..batch {
                let offset = (degree * batch + lane) * F::BYTES;
                F::encode(&mut packed[offset..offset + F::BYTES], *value);
            }
        }
    };

    // Warm-up outside the counted region: initializes runtime dispatch and
    // any first-touch state, and proves the geometry runs at all.
    let warm = noise_values::<F>(max_count, seeds[0]);
    write_input(&mut packed, &warm);
    plan.evaluate_batch_into(&packed, max_count, batch, &mut scratch, &mut output)
        .expect("warm-up evaluation");

    for (seed, count) in [
        (seeds[0], max_count),
        (seeds[1], 1),
        (seeds[2], max_count / 2),
    ] {
        // zero → dense → short coefficient transitions, outputs compared.
        let coefficients = noise_values::<F>(count, seed);
        for byte in packed.iter_mut() {
            *byte = 0;
        }
        write_input(&mut packed, &coefficients);
        for byte in output.iter_mut() {
            *byte = 0xAA;
        }
        let allocations = count_allocations(|| {
            plan.evaluate_batch_into(
                &packed[..count * row_bytes],
                count,
                batch,
                &mut scratch,
                &mut output,
            )
            .expect("counted evaluation");
        });
        assert_eq!(allocations, 0, "evaluation must not allocate");

        // The scalar path must be equally allocation-free.
        if batch == 1 {
            let mut scalar_scratch = plan.scratch(1).expect("scalar scratch");
            let mut scalar_output = vec![F::Elem::ZERO; weight];
            let coefficients_vec: Vec<F::Elem> = coefficients.clone();
            let allocations = count_allocations(|| {
                plan.evaluate_into(&coefficients_vec, &mut scalar_scratch, &mut scalar_output)
                    .expect("scalar evaluation");
            });
            assert_eq!(allocations, 0, "scalar evaluation must not allocate");
            // And it agrees with the batch run, row for row.
            for row in 0..weight {
                assert_eq!(
                    F::decode(&output[row * F::BYTES..][..F::BYTES]),
                    scalar_output[row]
                );
            }
        }
    }
}

#[test]
fn weighted_evaluation_is_steady_state_zero_alloc() {
    fn run<F: poly_ring::PolynomialField>() {
        let plan = MultiplicityPlan::<F>::new(&multiplicity_points::<F>(4), &[1, 2, 4, 3], 97)
            .expect("plan");
        assert_zero_alloc::<F>(&plan, 1, 97, [0x5EED_1000, 0x5EED_1001, 0x5EED_1002]);
        assert_zero_alloc::<F>(&plan, 8, 97, [0x5EED_2000, 0x5EED_2001, 0x5EED_2002]);
    }
    run::<Gf8B>();
    run::<Goldilocks>();
}

/// First-call counting: every coefficient set is evaluated by a freshly
/// built scratch's **first** call, with no warm-up anywhere in between —
/// the constructors pre-cache their transform plans, so even the first
/// call must allocate nothing. The scalar lane gets its own fresh scratch.
fn assert_first_call_zero_alloc<F>(
    plan: &MultiplicityPlan<F>,
    batch: usize,
    max_count: usize,
    seeds: [u64; 2],
) where
    F: poly_ring::PolynomialField,
{
    let weight = plan.total_weight();
    let row_bytes = batch * F::BYTES;
    let mut packed = vec![0_u8; max_count * row_bytes];
    let mut output = vec![0_u8; weight * row_bytes];

    let write_input = |packed: &mut [u8], coefficients: &[F::Elem]| {
        for (degree, value) in coefficients.iter().enumerate() {
            for lane in 0..batch {
                let offset = (degree * batch + lane) * F::BYTES;
                F::encode(&mut packed[offset..offset + F::BYTES], *value);
            }
        }
    };

    // zero → dense → short, each from its own fresh scratch.
    let zero = vec![F::Elem::ZERO; max_count];
    let dense = noise_values::<F>(max_count, seeds[0]);
    let short = noise_values::<F>(1, seeds[1]);
    for (label, coefficients) in [("zero", &zero), ("dense", &dense), ("short", &short)] {
        let count = coefficients.len();
        for byte in packed.iter_mut() {
            *byte = 0;
        }
        write_input(&mut packed, coefficients);
        for byte in output.iter_mut() {
            *byte = 0xAA;
        }

        let mut scratch = plan.scratch(batch).expect("scratch");
        let allocations = count_allocations(|| {
            plan.evaluate_batch_into(
                &packed[..count * row_bytes],
                count,
                batch,
                &mut scratch,
                &mut output,
            )
            .expect("first batch evaluation");
        });
        assert_eq!(
            allocations, 0,
            "first batch call ({label}) must not allocate"
        );

        let mut scalar_scratch = plan.scratch(1).expect("scalar scratch");
        let coefficients_vec = coefficients.clone();
        let mut scalar_output = vec![F::Elem::ZERO; weight];
        let allocations = count_allocations(|| {
            plan.evaluate_into(&coefficients_vec, &mut scalar_scratch, &mut scalar_output)
                .expect("first scalar evaluation");
        });
        assert_eq!(
            allocations, 0,
            "first scalar call ({label}) must not allocate"
        );

        // The first-call lanes agree with each other, row for row (lane 0).
        for row in 0..weight {
            assert_eq!(
                F::decode(&output[row * row_bytes..][..F::BYTES]),
                scalar_output[row]
            );
        }
    }
}

#[test]
fn first_call_from_a_fresh_scratch_is_zero_alloc() {
    fn run<F: poly_ring::PolynomialField>() {
        let plan = MultiplicityPlan::<F>::new(&multiplicity_points::<F>(4), &[1, 2, 4, 3], 97)
            .expect("plan");
        assert_first_call_zero_alloc::<F>(&plan, 1, 97, [0x5EED_3000, 0x5EED_3001]);
        assert_first_call_zero_alloc::<F>(&plan, 8, 97, [0x5EED_4000, 0x5EED_4001]);
    }
    run::<Gf8B>();
    run::<Mersenne31>();
}

/// Warmed lane-parallel Horner evaluation. The multipoint surface routes
/// above the per-point crossover and below the lane step crossover; the
/// single-weight multiplicity plan at batch one routes through the lane
/// evaluator. Both scratches hold their lane buffers from construction, so
/// the counted calls allocate nothing.
#[test]
fn lane_routes_are_steady_state_zero_alloc() {
    fn check_multipoint<F: fgf::kernel::FieldKernels>() {
        let points = multiplicity_points::<F>(40);
        let polynomial = noise::<F>(20, 0x5EED_5000);
        let mut scratch = MultipointScratch::new();
        let mut values = Vec::new();
        poly_ring::evaluate_multipoint_into(&mut values, &polynomial, &points, &mut scratch)
            .unwrap();
        assert_eq!(
            count_allocations(|| {
                poly_ring::evaluate_multipoint_into(
                    &mut values,
                    &polynomial,
                    &points,
                    &mut scratch,
                )
                .unwrap();
            }),
            0,
            "warmed lane-route evaluation must not allocate"
        );
    }
    check_multipoint::<Gf8B>();
    check_multipoint::<Goldilocks>();

    fn check_weighted<F: poly_ring::PolynomialField>() {
        let points = multiplicity_points::<F>(4);
        let plan = MultiplicityPlan::<F>::new(&points, &[1, 1, 1, 1], 97).expect("plan");
        let mut scratch = plan.scratch(1).expect("scratch");
        let coefficients = noise_values::<F>(12, 0x5EED_5100);
        let mut output = vec![<F as Field>::Elem::ZERO; plan.total_weight()];
        plan.evaluate_into(&coefficients, &mut scratch, &mut output)
            .expect("warm-up evaluation");
        assert_eq!(
            count_allocations(|| {
                plan.evaluate_into(&coefficients, &mut scratch, &mut output)
                    .expect("counted evaluation");
            }),
            0,
            "warmed weighted lane route must not allocate"
        );
    }
    check_weighted::<Gf8B>();
    check_weighted::<Goldilocks>();
}

/// Warmed bulk row ingress: with the same nonzero row geometry, the
/// second `assign_y_coefficients_packed` reuses the row vector and every
/// row buffer, so the counted call allocates nothing. The first, shape-
/// establishing call is outside the counted region.
#[test]
fn packed_row_assignment_reuses_row_buffers() {
    use poly_ring::BivariatePolynomial;

    let element = Mersenne31::BYTES;
    let row_bytes = 3 * element;
    let mut packed = vec![0_u8; 2 * row_bytes];
    for lane in 0..2 {
        for coefficient in 0..3 {
            let offset = lane * row_bytes + coefficient * element;
            <Mersenne31 as fgf::field::Field>::encode(
                &mut packed[offset..offset + element],
                m31_from(coefficient as u64 + 1),
            );
        }
    }

    let mut q = BivariatePolynomial::<Mersenne31>::zero();
    q.assign_y_coefficients_packed([&packed[..row_bytes], &packed[row_bytes..]].into_iter())
        .expect("warm assignment");

    let allocations = count_allocations(|| {
        q.assign_y_coefficients_packed([&packed[..row_bytes], &packed[row_bytes..]].into_iter())
            .expect("counted assignment");
    });
    assert_eq!(allocations, 0, "warmed packed assignment must not allocate");
}

/// Warmed remainder-tree descent over short dividends under an oversized
/// declared capacity: mixed modulus degrees including a constant and two
/// heavies, batch one and a batched lane count, nothing allocated after the
/// plan and scratch are built.
#[test]
fn short_dividend_descents_do_not_allocate() {
    fn run<F: poly_ring::PolynomialField>() {
        let short = noise_values::<F>(5, 0x5EED_D000);
        let mut heavy_small = noise_values::<F>(33, 0x5EED_D001);
        heavy_small.push(F::Elem::ONE);
        let mut heavy_large = noise_values::<F>(40, 0x5EED_D002);
        heavy_large.push(F::Elem::ONE);
        let quadratic = [F::Elem::ZERO, F::Elem::ZERO, F::Elem::ONE];
        let linear = [F::Elem::ZERO, F::Elem::ONE];
        let constant = [F::Elem::ONE];

        let mut moduli: Vec<Polynomial<F>> = vec![
            Polynomial::from_coefficients(&heavy_small).expect("modulus"),
            Polynomial::from_coefficients(&linear).expect("modulus"),
            Polynomial::from_coefficients(&constant).expect("modulus"),
            Polynomial::from_coefficients(&quadratic).expect("modulus"),
            Polynomial::from_coefficients(&heavy_large).expect("modulus"),
        ];
        // Many linears push the descent several levels deep, so the short
        // dividend is shorter than the tree itself, level by level.
        for index in 0..64 {
            let point = noise_values::<F>(1, 0x5EED_D100 + index as u64);
            moduli.push(Polynomial::from_coefficients(&[point[0], F::Elem::ONE]).expect("modulus"));
        }
        let capacity = 512_usize;
        let tree = RemainderTree::new(&moduli, capacity).expect("tree");

        for batch in [1_usize, 8] {
            let count = short.len();
            let row_bytes = batch * F::BYTES;
            let mut scratch = tree.scratch(batch).expect("scratch");
            let mut packed = vec![0_u8; count * row_bytes];
            let mut output =
                vec![0_u8; tree.leaf_offsets().last().copied().unwrap_or(0) * row_bytes];
            let fill = |packed: &mut [u8], lanes: &[Vec<F::Elem>]| {
                for (lane, coefficients) in lanes.iter().enumerate() {
                    for (degree, value) in coefficients.iter().enumerate() {
                        let offset = (degree * batch + lane) * F::BYTES;
                        F::encode(&mut packed[offset..offset + F::BYTES], *value);
                    }
                }
            };

            // Warm-up outside the counted region, same shape as the counted
            // calls. Distinct lane contents per run.
            let lanes: Vec<Vec<F::Elem>> = (0..batch)
                .map(|lane| noise_values::<F>(count, 0x5EED_D200 + lane as u64))
                .collect();
            fill(&mut packed, &lanes);
            tree.remainders_into(&packed, count, batch, &mut scratch, &mut output)
                .expect("warm-up");

            let lanes: Vec<Vec<F::Elem>> = (0..batch)
                .map(|lane| noise_values::<F>(count, 0x5EED_D300 + lane as u64))
                .collect();
            fill(&mut packed, &lanes);
            let allocations = count_allocations(|| {
                tree.remainders_into(&packed, count, batch, &mut scratch, &mut output)
                    .expect("counted run");
            });
            assert_eq!(
                allocations, 0,
                "warmed short-dividend descent must not allocate"
            );
        }
    }
    run::<Gf8B>();
    run::<Mersenne31>();
}

/// Wrap `value` in the M31 element type.
fn m31_from(value: u64) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value as u32)
}
