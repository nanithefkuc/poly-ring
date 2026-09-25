//! Prime derivatives behavior and error boundaries.

use fgf::field::Field;
use fgf::{Mersenne31, mersenne31};
use poly_ring::DerivativePlan;

fn m31(value: u32) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(value)
}

/// Batched odd-characteristic rows match the scalar path row for row, and
/// the per-row canonicalization keeps every batch lane exact.
#[test]
fn odd_characteristic_batches_match_scalar() {
    // Order 2 over 40 coefficients, two lanes with distinct values.
    let plan = DerivativePlan::<Mersenne31>::new(2, 40).expect("plan");
    let mut packed = vec![0_u8; 40 * 2 * 4];
    for degree in 0..40 {
        for lane in 0..2 {
            let value = m31(((degree * 3 + lane * 7) % 30 + 1) as u32);
            let offset = (degree * 2 + lane) * 4;
            <Mersenne31 as Field>::encode(&mut packed[offset..offset + 4], value);
        }
    }
    let mut batch_output = vec![0_u8; 38 * 2 * 4];
    plan.apply_batch_into(&packed, 40, 2, &mut batch_output)
        .expect("batch");
    for lane in 0..2 {
        let mut scalar = vec![m31(0); 40];
        for (degree, slot) in scalar.iter_mut().enumerate() {
            let offset = (degree * 2 + lane) * 4;
            *slot = <Mersenne31 as Field>::decode(&packed[offset..offset + 4]);
        }
        let mut scalar_output = vec![m31(0); 38];
        plan.apply_into(&scalar, &mut scalar_output).expect("apply");
        for (degree, expected) in scalar_output.iter().enumerate() {
            let offset = (degree * 2 + lane) * 4;
            assert_eq!(
                <Mersenne31 as Field>::decode(&batch_output[offset..offset + 4]),
                *expected,
                "lane {lane} row {degree} diverged"
            );
        }
    }
    // Spot-check one row against the defining binomial: row 4 is
    // C(6,2)·a_6 = 15·a_6.
    let lane0_row4 = <Mersenne31 as Field>::decode(&batch_output[(4 * 2) * 4..][..4]);
    let lane0_in6 = <Mersenne31 as Field>::decode(&packed[(6 * 2) * 4..][..4]);
    assert_eq!(lane0_row4, lane0_in6.mul(m31(15)));
}

/// The scalar odd-characteristic path with a prime-power valuation: order
/// 31 over 33 coefficients exercises the p-adic recurrence past the first
/// multiple of the characteristic.
#[test]
fn prime_power_valuation_recurrence_matches_definition() {
    let plan = DerivativePlan::<Mersenne31>::new(31, 33).expect("plan");
    // f = sum_{d} X^d: D^[31]f[j] = C(j+31,31) in M31 (characteristic
    // 2^31-1, so every small binomial is its integer value).
    let input = [m31(1); 33];
    let mut output = [m31(0); 2];
    plan.apply_into(&input, &mut output).expect("apply");
    let mut expected = [m31(0); 2];
    for (j, slot) in expected.iter_mut().enumerate() {
        *slot = poly_ring::binomial::<Mersenne31>(j + 31, 31);
    }
    assert_eq!(output, expected);
    // C(31,31) = 1 and C(32,31) = 32: small values stay small.
    assert_eq!(output, [m31(1), m31(32)]);
}
