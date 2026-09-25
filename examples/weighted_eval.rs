//! End-to-end consumer proof: the two exact jet fixtures through the public
//! `MultiplicityPlan` surface, plus one two-polynomial packed batch whose
//! lanes are asserted equal to their scalar evaluations.

use fgf::field::Field;
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::MultiplicityPlan;

fn m31(n: u64) -> mersenne31::Elem {
    mersenne31::Elem::from_raw(n as u32)
}

fn main() {
    // Fixture 1 (binary): f = 1 + X^2 over GF(2^8), points [0, 1],
    // multiplicities [3, 4]. f(T) = 1 + T^2 and f(1 + T) = T^2.
    let binary = [gf8b::Elem::ONE, gf8b::Elem::ZERO, gf8b::Elem::ONE];
    let binary_points = [gf8b::Elem::ZERO, gf8b::Elem::ONE];
    let plan = MultiplicityPlan::<Gf8B>::new(&binary_points, &[3, 4], 8).expect("plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = [gf8b::Elem::ZERO; 7];
    plan.evaluate_into(&binary, &mut scratch, &mut output)
        .expect("evaluate");
    println!("GF(2^8), f = 1 + X^2, points [0, 1], multiplicities [3, 4]:");
    println!(
        "  {:?}",
        output
            .iter()
            .map(|element| element.to_raw())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        output,
        [
            gf8b::Elem::ONE,
            gf8b::Elem::ZERO,
            gf8b::Elem::ONE,
            gf8b::Elem::ZERO,
            gf8b::Elem::ZERO,
            gf8b::Elem::ONE,
            gf8b::Elem::ZERO,
        ]
    );

    // Fixture 2 (prime): f = 1 + 2X + 3X^2 + 4X^3 over Mersenne31, points
    // [0, 2], multiplicities [2, 4]: [1, 2, 49, 62, 27, 4].
    let prime = [m31(1), m31(2), m31(3), m31(4)];
    let prime_points = [m31(0), m31(2)];
    let plan = MultiplicityPlan::<Mersenne31>::new(&prime_points, &[2, 4], 8).expect("plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut output = [m31(0); 6];
    plan.evaluate_into(&prime, &mut scratch, &mut output)
        .expect("evaluate");
    println!("Mersenne31, f = 1 + 2X + 3X^2 + 4X^3, points [0, 2], multiplicities [2, 4]:");
    println!(
        "  {:?}",
        output
            .iter()
            .map(|element| element.to_raw())
            .collect::<Vec<_>>()
    );
    assert_eq!(output, [m31(1), m31(2), m31(49), m31(62), m31(27), m31(4)]);

    // A two-polynomial packed batch: two genuinely different polynomials in
    // adjacent lanes, asserted against their own scalar evaluations.
    let batch = 2_usize;
    let count = 6_usize;
    let lanes = [
        vec![m31(1), m31(2), m31(3), m31(4), m31(5), m31(6)],
        vec![m31(7), m31(0), m31(2), m31(0), m31(9), m31(0)],
    ];
    let mut packed = vec![0_u8; count * batch * Mersenne31::BYTES];
    for (lane, coefficients) in lanes.iter().enumerate() {
        for (degree, value) in coefficients.iter().enumerate() {
            let offset = (degree * batch + lane) * Mersenne31::BYTES;
            Mersenne31::encode(&mut packed[offset..offset + Mersenne31::BYTES], *value);
        }
    }
    let mut scratch = plan.scratch(batch).expect("batch scratch");
    let mut batched = vec![0_u8; plan.total_weight() * batch * Mersenne31::BYTES];
    plan.evaluate_batch_into(&packed, count, batch, &mut scratch, &mut batched)
        .expect("batch evaluate");
    for (lane, coefficients) in lanes.iter().enumerate() {
        let mut scalar = vec![m31(0); plan.total_weight()];
        plan.evaluate_into(coefficients, &mut scratch, &mut scalar)
            .expect("scalar evaluate");
        for (row, expected) in scalar.iter().enumerate() {
            let offset = (row * batch + lane) * Mersenne31::BYTES;
            assert_eq!(
                Mersenne31::decode(&batched[offset..offset + Mersenne31::BYTES]),
                *expected,
                "lane {lane} diverged from its scalar evaluation"
            );
        }
    }
    println!("packed two-lane batch agrees with both scalar evaluations");
}
