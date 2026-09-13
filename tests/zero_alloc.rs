//! Steady-state zero allocation for every `*_into` / scratch-owning path,
//! proven under a counting global allocator (U5).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use fgf::field::Field;
use fgf::{Gf8B, Gf16};
use poly_ring::{
    ChienScratch, DomainScratch, EvaluationDomain, FieldRootScratch, MultipointScratch, Polynomial,
    RothRuckensteinLimits, RothRuckensteinScratch, truncated_eea,
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
            F::read(&bytes[..F::BYTES])
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
    let mut field_scratch = FieldRootScratch::new();
    let mut roots = Vec::new();
    for _ in 0..2 {
        poly_ring::chien_roots_into(&locator, &mut chien_scratch, &mut roots).unwrap();
        poly_ring::base_field_roots_into(&locator, &mut field_scratch, &mut roots).unwrap();
    }
    let chien_count = count_allocations(|| {
        poly_ring::chien_roots_into(&locator, &mut chien_scratch, &mut roots).unwrap();
    });
    let equal_degree_count = count_allocations(|| {
        poly_ring::base_field_roots_into(&locator, &mut field_scratch, &mut roots).unwrap();
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
    // Warm to convergence: the recycled subproduct nodes are drawn from the
    // pool in reverse construction order, so buffer capacities migrate to
    // their steady-state slots over the first rounds.
    for _ in 0..3 {
        poly_ring::evaluate_multipoint_into(&locator, &points, &mut multipoint, &mut values)
            .unwrap();
    }
    assert_eq!(
        count_allocations(|| {
            poly_ring::evaluate_multipoint_into(&locator, &points, &mut multipoint, &mut values)
                .unwrap();
        }),
        0,
        "warmed subproduct-tree evaluation must not allocate"
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
            &rows,
            3,
            RothRuckensteinLimits::new(100_000, 256),
            &mut lift_scratch,
            &mut lifted,
        )
        .unwrap();
    }
    assert_eq!(
        count_allocations(|| {
            poly_ring::roth_ruckenstein_roots_into(
                &rows,
                3,
                RothRuckensteinLimits::new(100_000, 256),
                &mut lift_scratch,
                &mut lifted,
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
    use poly_ring::{PolynomialProductScratch, ProductStrategy, multiply_batch_truncated};

    let left = noise::<Gf16>(80, 0x81);
    let right = noise::<Gf16>(70, 0x82);
    let mut scratch = PolynomialProductScratch::new();
    let mut output = Vec::new();
    multiply_batch_truncated(
        &[(&left, &right); 4].map(|(left, right)| (left, right)),
        149,
        ProductStrategy::Afft,
        &mut scratch,
        &mut output,
    )
    .unwrap();
    assert_eq!(
        count_allocations(|| {
            multiply_batch_truncated(
                &[(&left, &right); 4].map(|(left, right)| (left, right)),
                149,
                ProductStrategy::Afft,
                &mut scratch,
                &mut output,
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
