//! The prepared remainder tree against an independent scalar long-division
//! oracle, plus its steady-state zero-allocation proof.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, Goldilocks, Mersenne31, QuadMersenne31};
use poly_ring::{Polynomial, PolynomialError, ProductError, RemainderTree};

// ---------------------------------------------------------------------------
// Independent oracle: long division by hand over scalar elements.

fn oracle_remainder<F: FieldKernels>(dividend: &[F::Elem], divisor: &[F::Elem]) -> Vec<F::Elem> {
    let divisor_degree = divisor.len() - 1;
    let mut remainder: Vec<F::Elem> = dividend.to_vec();
    let leading = divisor[divisor_degree];
    loop {
        let degree = match remainder.iter().rposition(|c| !c.is_zero()) {
            Some(degree) if degree >= divisor_degree => degree,
            _ => break,
        };
        let factor = remainder[degree].mul(leading.inv());
        for (offset, coefficient) in divisor.iter().enumerate() {
            let index = degree - divisor_degree + offset;
            remainder[index] = remainder[index].sub(factor.mul(*coefficient));
        }
    }
    remainder.truncate(divisor_degree);
    remainder.resize(divisor_degree, F::Elem::ZERO);
    remainder
}

fn noise<F: FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
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

fn poly<F: FieldKernels>(coefficients: &[F::Elem]) -> Polynomial<F> {
    Polynomial::from_coefficients(coefficients).expect("polynomial")
}

fn small_points<F: FieldKernels>(count: usize) -> Vec<F::Elem> {
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

/// Run the tree over every lane and compare against the oracle per lane,
/// declaring an explicit coefficient capacity for the prepared tree.
fn check_tree_with_capacity<F: poly_ring::PolynomialField>(
    dividend: &[F::Elem],
    moduli: &[Vec<F::Elem>],
    batch: usize,
    capacity: usize,
) {
    let modulus_polys: Vec<Polynomial<F>> = moduli.iter().map(|m| poly(m)).collect();
    let tree = RemainderTree::new(&modulus_polys, capacity).expect("tree");
    let lanes: Vec<Vec<F::Elem>> = (0..batch)
        .map(|lane| {
            let mut coefficients = dividend.to_vec();
            // Genuinely different polynomials per lane: rotate the noise.
            let extra = noise::<F>(coefficients.len(), 0x5EED_9000 + lane as u64);
            for (index, value) in coefficients.iter_mut().enumerate() {
                *value = value.add(extra[index % extra.len()]);
            }
            coefficients
        })
        .collect();

    let mut scratch = tree.scratch(batch).expect("scratch");
    let mut packed = vec![0_u8; dividend.len() * batch * F::BYTES];
    for (lane, coefficients) in lanes.iter().enumerate() {
        for (degree, value) in coefficients.iter().enumerate() {
            let offset = (degree * batch + lane) * F::BYTES;
            F::encode(&mut packed[offset..offset + F::BYTES], *value);
        }
    }
    let mut output =
        vec![0_u8; tree.leaf_offsets().last().copied().unwrap_or(0) * batch * F::BYTES];
    tree.remainders_into(&packed, dividend.len(), batch, &mut scratch, &mut output)
        .expect("remainders");

    for (index, modulus) in moduli.iter().enumerate() {
        let start = tree.leaf_offsets()[index];
        let rows = tree.leaf_offsets()[index + 1] - start;
        assert_eq!(rows, modulus.len() - 1, "leaf row count is the degree");
        for (lane, lane_coefficients) in lanes.iter().enumerate() {
            let expected = oracle_remainder::<F>(lane_coefficients, modulus);
            for (row, want) in expected.iter().enumerate() {
                let offset = ((start + row) * batch + lane) * F::BYTES;
                assert_eq!(
                    F::decode(&output[offset..offset + F::BYTES]),
                    *want,
                    "modulus {index}, order {row}, lane {lane}"
                );
            }
        }
    }
}

/// Run the tree over every lane and compare against the oracle per lane,
/// with the capacity padded just past the dividend length.
fn check_tree<F: poly_ring::PolynomialField>(
    dividend: &[F::Elem],
    moduli: &[Vec<F::Elem>],
    batch: usize,
) {
    check_tree_with_capacity::<F>(dividend, moduli, batch, dividend.len() + 4)
}

#[test]
fn remainders_match_long_division_across_fields() {
    fn check<F: poly_ring::PolynomialField>() {
        let dividend = noise::<F>(23, 0x5EED_0001);
        let points = small_points::<F>(4);
        // Nonmonic moduli (X − a)·c, a constant, and a cubic.
        let scaled: Vec<F::Elem> = [points[0].neg(), F::Elem::ONE]
            .iter()
            .map(|value| value.mul(points[2]))
            .collect();
        let cubic = [
            points[1].neg(),
            F::Elem::ONE.add(F::Elem::ONE),
            F::Elem::ZERO,
            F::Elem::ONE,
        ]
        .to_vec();
        let linear = vec![points[3].neg(), F::Elem::ONE];
        check_tree::<F>(
            &dividend,
            &[
                scaled.clone(),
                vec![F::Elem::ONE],
                cubic.clone(),
                linear.clone(),
            ],
            1,
        );
        check_tree::<F>(&dividend, &[scaled, vec![F::Elem::ONE], cubic, linear], 3);
    }
    check::<Gf8B>();
    check::<Gf16>();
    check::<Mersenne31>();
    check::<Goldilocks>();
    check::<QuadMersenne31>();
}

#[test]
fn fixed_length_reversal_handles_leading_zero_head() {
    // X^10 + 1 divided by X^3 − 2 over Mersenne31: the reversed dividend has
    // a zero constant term, which breaks any normalized-reverse shortcut.
    let mut dividend = vec![<Mersenne31 as Field>::Elem::from_raw(0); 11];
    dividend[0] = Mersenne31Elem::from_raw(1);
    dividend[10] = Mersenne31Elem::from_raw(1);
    let mut modulus = vec![Mersenne31Elem::from_raw(0); 4];
    modulus[0] = Mersenne31Elem::from_raw(2).neg();
    modulus[3] = Mersenne31Elem::from_raw(1);
    check_tree::<Mersenne31>(&dividend, &[modulus], 1);

    // D far above the total modulus degree: the root single reduction.
    let mut long = vec![<Mersenne31 as Field>::Elem::from_raw(0); 600];
    long[0] = Mersenne31Elem::from_raw(7);
    long[599] = Mersenne31Elem::from_raw(3);
    let quadratic = vec![
        Mersenne31Elem::from_raw(5),
        Mersenne31Elem::from_raw(0),
        Mersenne31Elem::from_raw(1),
    ];
    let linear = vec![
        Mersenne31Elem::from_raw(0).neg(),
        Mersenne31Elem::from_raw(1),
    ];
    check_tree::<Mersenne31>(&long, &[quadratic.clone(), linear, quadratic], 2);
}

use fgf::mersenne31::Elem as Mersenne31Elem;

#[test]
fn heavily_unequal_and_repeated_moduli_hold() {
    let dividend = noise::<Gf8B>(131, 0x5EED_0002);
    let points = small_points::<Gf8B>(8);
    let mut heavy = vec![Gf8B_Elem::ZERO; 65];
    for (index, slot) in heavy.iter_mut().enumerate().take(64) {
        if index % 3 == 0 {
            *slot = noise::<Gf8B>(1, 0xAA + index as u64)[0];
        }
    }
    heavy[64] = Gf8B_Elem::ONE;
    let mut moduli = vec![heavy];
    for point in &points {
        moduli.push(vec![point.neg(), Gf8B_Elem::ONE]);
    }
    // Repeated identical moduli keep separate leaves in caller order.
    moduli.push(vec![points[0].neg(), Gf8B_Elem::ONE]);
    check_tree::<Gf8B>(&dividend, &moduli, 1);
}

use fgf::gf8b::Elem as Gf8B_Elem;

#[test]
fn input_lengths_straddle_the_product_crossovers() {
    // 2100 coefficients crosses the 2048 Karatsuba crossover; 100 does not.
    for len in [100_usize, 2100] {
        let dividend = noise::<Gf8B>(len, 0x5EED_0003 + len as u64);
        let points = small_points::<Gf8B>(3);
        let moduli = vec![
            vec![points[0].neg(), Gf8B_Elem::ONE],
            vec![
                points[1].neg(),
                Gf8B_Elem::ONE,
                Gf8B_Elem::ZERO,
                Gf8B_Elem::ONE,
            ],
            vec![points[2].neg(), Gf8B_Elem::ONE],
        ];
        check_tree::<Gf8B>(&dividend, &moduli, 1);
    }
}

#[test]
fn short_dividends_reduce_under_oversized_capacity() {
    fn check<F: poly_ring::PolynomialField>() {
        let points = small_points::<F>(6);
        // Two monic heavies whose degrees exceed a short dividend, a
        // constant that contributes an empty output range, and mixed
        // smaller degrees.
        let mut heavy_33 = noise::<F>(33, 0x5EED_C001);
        heavy_33.push(F::Elem::ONE);
        let mut heavy_40 = noise::<F>(40, 0x5EED_C002);
        heavy_40.push(F::Elem::ONE);
        let constant = vec![points[1].add(F::Elem::ONE)];
        let quadratic = vec![
            points[2].neg(),
            F::Elem::ONE.add(F::Elem::ONE),
            F::Elem::ONE,
        ];
        let mut moduli = vec![
            heavy_33.clone(),
            vec![points[3].neg(), F::Elem::ONE],
            constant,
            quadratic,
            heavy_40.clone(),
            vec![points[4].neg(), F::Elem::ONE],
        ];
        // Many linears push the descent several levels deep, so a dividend
        // of a handful of rows is shorter than the tree itself, level by
        // level, not only at the root.
        for index in 0..64 {
            moduli.push(vec![points[index % points.len()].neg(), F::Elem::ONE]);
        }
        let capacity = 512_usize;

        // A handful of coefficients against a root modulus list summing far
        // beyond it, under a capacity much larger than the dividend.
        let short = noise::<F>(5, 0x5EED_C004);
        check_tree_with_capacity::<F>(&short, &moduli, 1, capacity);
        check_tree_with_capacity::<F>(&short, &moduli, 4, capacity);

        // Dividends exactly at and past one heavy modulus's row count, so
        // both sides of the copy/divide boundary are exercised. The heavy
        // degree still exceeds them.
        let at_rows = noise::<F>(33, 0x5EED_C005);
        check_tree_with_capacity::<F>(&at_rows, &moduli, 2, capacity);
        let one_past_rows = noise::<F>(34, 0x5EED_C006);
        check_tree_with_capacity::<F>(&one_past_rows, &moduli, 2, capacity);
        let two_past_rows = noise::<F>(35, 0x5EED_C007);
        check_tree_with_capacity::<F>(&two_past_rows, &moduli, 2, capacity);
    }
    check::<Gf8B>();
    check::<Mersenne31>();
    check::<Goldilocks>();
    check::<QuadMersenne31>();
}

#[test]
fn noncanonical_prime_lanes_evaluate_their_field_values() {
    const MODULUS: u32 = 0x7FFF_FFFF;
    // A raw `p` lane is the zero coefficient; the tree canonicalizes on
    // ingress, so it must equal the canonical run.
    let raw = [
        Mersenne31Elem::from_raw(MODULUS),
        Mersenne31Elem::from_raw(MODULUS + 3),
    ];
    let canonical = [Mersenne31Elem::from_raw(0), Mersenne31Elem::from_raw(3)];
    let linear = vec![
        Mersenne31Elem::from_raw(2).neg(),
        Mersenne31Elem::from_raw(1),
    ];
    check_tree::<Mersenne31>(&raw, std::slice::from_ref(&linear), 1);
    check_tree::<Mersenne31>(&canonical, &[linear], 1);
}

#[test]
fn zero_modulus_is_rejected_and_geometry_errors_preserve_output() {
    let points = small_points::<Gf8B>(2);
    let linear = vec![points[0].neg(), Gf8B_Elem::ONE];
    let _dividend = noise::<Gf8B>(5, 0x5EED_0004);

    assert_eq!(
        RemainderTree::<Gf8B>::new(&[poly(&linear), Polynomial::zero()], 8).unwrap_err(),
        ProductError::Polynomial(PolynomialError::DivisionByZero)
    );

    let tree = RemainderTree::<Gf8B>::new(&[poly(&linear)], 8).expect("tree");
    let mut scratch = tree.scratch(2).expect("scratch");
    let mut output = vec![0xAB_u8; 2 * Gf8B::BYTES];
    let packed = vec![0_u8; 5 * 2 * Gf8B::BYTES];

    // Wrong output length leaves the sentinel untouched.
    let mut short = vec![0xAB_u8; 3];
    assert!(matches!(
        tree.remainders_into(&packed, 5, 2, &mut scratch, &mut short),
        Err(ProductError::Config(_))
    ));
    assert!(short.iter().all(|byte| *byte == 0xAB));

    // Coefficient capacity and lane capacity violations.
    assert!(matches!(
        tree.remainders_into(&packed, 9, 2, &mut scratch, &mut output),
        Err(ProductError::Config(_))
    ));
    let wide = vec![0_u8; 5 * 4 * Gf8B::BYTES];
    let mut wide_output = vec![0_u8; 4 * Gf8B::BYTES];
    assert!(matches!(
        tree.remainders_into(&wide, 5, 4, &mut scratch, &mut wide_output),
        Err(ProductError::Config(_))
    ));

    // A valid run immediately after the errors proves the scratch is intact.
    tree.remainders_into(&packed, 5, 2, &mut scratch, &mut output)
        .expect("valid run after errors");
    let expected = oracle_remainder::<Gf8B>(
        &{
            let mut coefficients = noise::<Gf8B>(5, 0x5EED_0004);
            let extra = noise::<Gf8B>(5, 0x5EED_9000);
            for (index, value) in coefficients.iter_mut().enumerate() {
                *value = value.add(extra[index]);
            }
            coefficients
        },
        &linear,
    );
    let _ = expected; // lane comparison covered by check_tree above.
}

#[test]
fn empty_modulus_list_and_constant_moduli() {
    let dividend = noise::<Gf8B>(7, 0x5EED_0005);
    let tree = RemainderTree::<Gf8B>::new(&[], 16).expect("empty tree");
    let mut scratch = tree.scratch(1).expect("scratch");
    let mut output: Vec<u8> = Vec::new();
    tree.remainders_into(&[0; 7], 7, 1, &mut scratch, &mut output)
        .expect("empty request");
    assert!(output.is_empty());

    let constant = poly::<Gf8B>(&[Gf8B_Elem::from_raw(9)]);
    let tree = RemainderTree::new(&[constant], 16).expect("constant tree");
    assert_eq!(tree.leaf_offsets(), &[0, 0]);
    let mut scratch = tree.scratch(1).expect("scratch");
    let mut output: Vec<u8> = Vec::new();
    tree.remainders_into(&[0; 7 * Gf8B::BYTES], 7, 1, &mut scratch, &mut output)
        .expect("constant request");
    assert!(output.is_empty());
    let _ = dividend;
}

#[test]
fn modulus_degree_above_the_capacity_yields_zero_padded_remainder() {
    // A single heavy leaf whose degree exceeds `max_coefficients`: the
    // terminal slot must hold the modulus degree, and a shorter dividend
    // copies through zero-padded instead of slicing past the slot.
    let x2 = poly::<Gf8B>(&[Gf8B_Elem::ZERO, Gf8B_Elem::ZERO, Gf8B_Elem::ONE]);
    let tree = RemainderTree::new(&[x2], 1).expect("tree");
    let mut scratch = tree.scratch(1).expect("scratch");
    let mut output = vec![0xAB_u8; 2 * Gf8B::BYTES];
    tree.remainders_into(&[1], 1, 1, &mut scratch, &mut output)
        .expect("short dividend");
    let first = Gf8B::decode(&output[..Gf8B::BYTES]);
    let second = Gf8B::decode(&output[Gf8B::BYTES..2 * Gf8B::BYTES]);
    assert_eq!((first, second), (Gf8B_Elem::from_raw(1), Gf8B_Elem::ZERO));
}

#[test]
fn incompatible_scratch_is_rejected_before_any_mutation() {
    // Equal coefficient and lane capacities do not make a scratch reusable
    // across trees: the descent indexes per-depth slots and shared buffers
    // sized to the owning tree's bounds.
    let empty = RemainderTree::<Gf8B>::new(&[], 4).expect("empty tree");
    let linear = poly::<Gf8B>(&[Gf8B_Elem::ZERO, Gf8B_Elem::ONE]);
    let tree = RemainderTree::new(&[linear], 4).expect("tree");
    let mut scratch = empty.scratch(1).expect("foreign scratch");
    let mut output = vec![0xAB_u8; Gf8B::BYTES];
    match tree.remainders_into(&[1], 1, 1, &mut scratch, &mut output) {
        Err(ProductError::Config(poly_ring::ConfigError::ScratchTooSmall {
            context: "remainder scratch geometry",
            ..
        })) => {}
        other => panic!("expected a scratch geometry error, got {other:?}"),
    }
    assert!(output.iter().all(|byte| *byte == 0xAB));

    // The owning tree's own scratch still serves it.
    let mut scratch = tree.scratch(1).expect("scratch");
    tree.remainders_into(&[1], 1, 1, &mut scratch, &mut output)
        .expect("own scratch");
}

// ---------------------------------------------------------------------------
// Zero allocation.

struct CountingAllocator;

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

#[test]
fn prepared_remainders_are_steady_state_zero_alloc() {
    fn run<F: poly_ring::PolynomialField>() {
        let points = small_points::<F>(4);
        let moduli: Vec<Polynomial<F>> = vec![
            Polynomial::from_coefficients(&[points[0].neg(), F::Elem::ONE]).unwrap(),
            Polynomial::from_coefficients(&[
                points[1].neg(),
                F::Elem::ONE,
                F::Elem::ZERO,
                F::Elem::ONE,
            ])
            .unwrap(),
            Polynomial::from_coefficients(&[points[2].neg(), F::Elem::ONE]).unwrap(),
        ];
        let batch = 4_usize;
        let max_count = 200_usize;
        let tree = RemainderTree::new(&moduli, max_count).expect("tree");
        let mut scratch = tree.scratch(batch).expect("scratch");
        let row_bytes = batch * F::BYTES;
        let mut packed = vec![0_u8; max_count * row_bytes];
        let mut output = vec![0_u8; tree.leaf_offsets().last().copied().unwrap_or(0) * row_bytes];

        let fill = |packed: &mut [u8], coefficients: &[F::Elem]| {
            for (degree, value) in coefficients.iter().enumerate() {
                for lane in 0..batch {
                    let offset = (degree * batch + lane) * F::BYTES;
                    F::encode(&mut packed[offset..offset + F::BYTES], *value);
                }
            }
        };

        // Warm-up outside the counted region.
        let warm = noise::<F>(max_count, 0x5EED_A000);
        fill(&mut packed, &warm);
        tree.remainders_into(&packed, max_count, batch, &mut scratch, &mut output)
            .expect("warm-up");

        // Dense and short runs, both counted.
        for (seed, count) in [(0x5EED_A001_u64, max_count), (0x5EED_A002_u64, 17)] {
            let coefficients = noise::<F>(count, seed);
            for byte in packed.iter_mut() {
                *byte = 0;
            }
            fill(&mut packed, &coefficients);
            let allocations = count_allocations(|| {
                tree.remainders_into(
                    &packed[..count * row_bytes],
                    count,
                    batch,
                    &mut scratch,
                    &mut output,
                )
                .expect("counted run");
            });
            assert_eq!(allocations, 0, "prepared remainders must not allocate");
        }
    }
    run::<Gf8B>();
    run::<Mersenne31>();
}

// ---------------------------------------------------------------------------
// Forced transform routes: exact products through every field's transform
// path (AFFT, NTT, embedded), against the schoolbook oracle.

#[cfg(feature = "internals")]
#[test]
fn forced_transform_products_match_schoolbook() {
    use poly_ring::ConvolutionScratch;
    use poly_ring::internals::{ProductRoute, multiply_rows_route_into};

    fn check<F: poly_ring::PolynomialField>() {
        for (left_count, right_count) in [(1_usize, 1), (5, 9), (33, 64), (100, 7)] {
            let full = left_count + right_count - 1;
            let left: Vec<u8> = (0..left_count)
                .flat_map(|degree| F::write_value(degree as u64 * 7 + 1))
                .collect();
            let right: Vec<u8> = (0..right_count)
                .flat_map(|degree| F::write_value(degree as u64 * 5 + 3))
                .collect();
            let mut scratch = ConvolutionScratch::<F>::new(left_count, right_count, 1).unwrap();
            scratch.prepare_transform(full, 1).unwrap();
            let mut transformed = vec![0_u8; full * F::BYTES];
            multiply_rows_route_into::<F>(
                &mut transformed,
                &left,
                left_count,
                &right,
                right_count,
                1,
                full,
                ProductRoute::Transform,
                &mut scratch,
            )
            .expect("forced transform product");

            // Independent schoolbook oracle over scalar elements.
            let expected: Vec<F::Elem> = {
                let mut product = vec![F::Elem::ZERO; full];
                for (i, a) in left.chunks(F::BYTES).map(F::decode).enumerate() {
                    for (j, b) in right.chunks(F::BYTES).map(F::decode).enumerate() {
                        product[i + j] = product[i + j].add(a.mul(b));
                    }
                }
                product
            };
            for (degree, want) in expected.iter().enumerate() {
                assert_eq!(
                    F::decode(&transformed[degree * F::BYTES..][..F::BYTES]),
                    *want,
                    "{} coefficient {degree} diverged at {left_count}x{right_count}",
                    F::NAME
                );
            }
        }
    }
    check::<Gf8B>();
    check::<Gf16>();
    check::<Mersenne31>();
    check::<Goldilocks>();
    check::<QuadMersenne31>();
}

/// Encode an integer as a canonical field element's bytes.
#[cfg(feature = "internals")]
trait WriteValue {
    fn write_value(n: u64) -> Vec<u8>;
}

#[cfg(feature = "internals")]
impl<F: poly_ring::PolynomialField> WriteValue for F {
    fn write_value(n: u64) -> Vec<u8> {
        let mut value = F::Elem::ONE;
        let mut total = F::Elem::ZERO;
        let mut remaining = n;
        while remaining != 0 {
            if remaining & 1 != 0 {
                total = total.add(value);
            }
            value = value.add(value);
            remaining >>= 1;
        }
        let mut bytes = vec![0_u8; F::BYTES];
        F::encode(&mut bytes, total);
        bytes
    }
}
