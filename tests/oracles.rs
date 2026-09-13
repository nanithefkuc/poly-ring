//! Naive reference implementations, written once and never optimized.
//!
//! These are the oracles the suite leans on: structurally unrelated to every
//! dispatched path in the crate (scalar element loops over `fgf::field::Elem`
//! only), deliberately slow, and shared by the per-surface test files.

#![allow(dead_code)]

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use poly_ring::Polynomial;

/// Fixed-seed LCG coefficients in `fgf`'s `noise` shape.
pub fn noise<F: FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
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

/// A noise polynomial with a nonzero constant coefficient (a unit).
pub fn noise_unit<F: FieldKernels>(len: usize, seed: u64) -> Polynomial<F> {
    let mut coefficients = noise::<F>(len, seed);
    if coefficients.is_empty() || coefficients[0].is_zero() {
        coefficients[0] = F::Elem::ONE;
    }
    Polynomial::from_coefficients(&coefficients).expect("noise polynomial")
}

/// A plain noise polynomial (possibly zero constant).
pub fn noise_poly<F: FieldKernels>(len: usize, seed: u64) -> Polynomial<F> {
    Polynomial::from_coefficients(&noise::<F>(len, seed)).expect("noise polynomial")
}

/// Naive per-coefficient convolution over scalar elements.
pub fn naive_multiply<F: FieldKernels>(
    left: &Polynomial<F>,
    right: &Polynomial<F>,
) -> Polynomial<F> {
    let mut product = vec![F::Elem::ZERO; left.coefficient_count() + right.coefficient_count()];
    for (i, a) in left.coefficients().enumerate() {
        for (j, b) in right.coefficients().enumerate() {
            product[i + j] = product[i + j].add(a.mul(b));
        }
    }
    Polynomial::from_coefficients(&product).expect("naive product")
}

/// Naive Horner evaluation.
pub fn naive_evaluate<F: FieldKernels>(polynomial: &Polynomial<F>, point: F::Elem) -> F::Elem {
    let mut value = F::Elem::ZERO;
    for coefficient in polynomial.coefficients().rev() {
        value = value.mul(point).add(coefficient);
    }
    value
}

/// Long division by hand over scalar coefficients: repeatedly cancel the
/// leading term.
pub fn naive_div_rem<F: FieldKernels>(
    dividend: &Polynomial<F>,
    divisor: &Polynomial<F>,
) -> (Polynomial<F>, Polynomial<F>) {
    let mut quotient = vec![F::Elem::ZERO; dividend.coefficient_count()];
    let mut remainder: Vec<F::Elem> = dividend.coefficients().collect();
    let divisor_degree = divisor.degree().expect("nonzero divisor");
    let divisor_leading = divisor.coefficient(divisor_degree);
    while remainder.len() > divisor_degree && remainder.iter().any(|c| !c.is_zero()) {
        let degree = remainder.len() - 1;
        let factor = remainder[degree].mul(divisor_leading.inv());
        quotient[degree - divisor_degree] = factor;
        for (offset, coefficient) in divisor.coefficients().enumerate() {
            if !coefficient.is_zero() {
                let index = degree - divisor_degree + offset;
                remainder[index] = remainder[index].sub(factor.mul(coefficient));
            }
        }
        while remainder
            .last()
            .is_some_and(|coefficient| coefficient.is_zero())
        {
            remainder.pop();
        }
    }
    (
        Polynomial::from_coefficients(&quotient).expect("naive quotient"),
        Polynomial::from_coefficients(&remainder).expect("naive remainder"),
    )
}

/// Plain textbook extended Euclid over scalar coefficient vectors.
pub fn naive_gcd_ext<F: FieldKernels>(
    a: &Polynomial<F>,
    b: &Polynomial<F>,
) -> (Polynomial<F>, Polynomial<F>, Polynomial<F>) {
    let to_vec = |p: &Polynomial<F>| -> Vec<F::Elem> { p.coefficients().collect() };
    let from_vec = |v: &[F::Elem]| -> Polynomial<F> {
        Polynomial::from_coefficients(v).expect("naive cofactor")
    };
    let mut r_old = to_vec(a);
    let mut r = to_vec(b);
    let mut s_old = vec![F::Elem::ONE];
    let mut s = vec![];
    let mut t_old = vec![];
    let mut t = vec![F::Elem::ONE];

    let trim = |v: &mut Vec<F::Elem>| {
        while v.last().is_some_and(|coefficient| coefficient.is_zero()) {
            v.pop();
        }
    };
    while !r.is_empty() {
        // r_old = q * r + rem, computed by hand.
        let mut rem = r_old.clone();
        let mut q = vec![F::Elem::ZERO; r_old.len().max(1)];
        let r_degree = r.len() - 1;
        let r_leading = r[r_degree];
        while rem.len() > r_degree {
            let degree = rem.len() - 1;
            let factor = rem[degree].mul(r_leading.inv());
            q[degree - r_degree] = factor;
            for (offset, coefficient) in r.iter().enumerate() {
                rem[degree - r_degree + offset] =
                    rem[degree - r_degree + offset].sub(factor.mul(*coefficient));
            }
            trim(&mut rem);
            if rem.len() <= r_degree {
                break;
            }
        }
        trim(&mut q);
        r_old = r;
        r = rem;
        // s_next = s_old − q * s and t_next = t_old − q * t, all by hand.
        // The signs are explicit so the oracle is exact in every
        // characteristic; in characteristic two they collapse to XOR.
        let mut qs = vec![F::Elem::ZERO; q.len() + s.len()];
        for (i, q_coefficient) in q.iter().enumerate() {
            for (j, s_coefficient) in s.iter().enumerate() {
                qs[i + j] = qs[i + j].add(q_coefficient.mul(*s_coefficient));
            }
        }
        trim(&mut qs);
        let mut s_next = s_old.clone();
        for (index, coefficient) in qs.iter().enumerate() {
            while s_next.len() <= index {
                s_next.push(F::Elem::ZERO);
            }
            s_next[index] = s_next[index].sub(*coefficient);
        }
        trim(&mut s_next);
        let mut qt = vec![F::Elem::ZERO; q.len() + t.len()];
        for (i, q_coefficient) in q.iter().enumerate() {
            for (j, t_coefficient) in t.iter().enumerate() {
                qt[i + j] = qt[i + j].add(q_coefficient.mul(*t_coefficient));
            }
        }
        trim(&mut qt);
        let mut t_next = t_old.clone();
        for (index, coefficient) in qt.iter().enumerate() {
            while t_next.len() <= index {
                t_next.push(F::Elem::ZERO);
            }
            t_next[index] = t_next[index].sub(*coefficient);
        }
        trim(&mut t_next);
        s_old = s;
        s = s_next;
        t_old = t;
        t = t_next;
    }
    let leading = r_old.last().copied();
    let scale = match leading {
        Some(value) if !value.is_zero() => value.inv(),
        _ => F::Elem::ONE,
    };
    let scale_vec = |v: &mut Vec<F::Elem>| {
        for coefficient in v.iter_mut() {
            *coefficient = coefficient.mul(scale);
        }
        trim(v);
    };
    scale_vec(&mut r_old);
    scale_vec(&mut s_old);
    scale_vec(&mut t_old);
    (from_vec(&r_old), from_vec(&s_old), from_vec(&t_old))
}

/// Full-domain Chien-style scan by plain Horner: every field element of a
/// GF(2^8) field.
pub fn naive_roots_small_field<F: FieldKernels>(polynomial: &Polynomial<F>) -> Vec<F::Elem> {
    let mut roots = Vec::new();
    for key in 0..F::ORDER {
        let bytes = key.to_le_bytes();
        let element = F::read(&bytes[..F::BYTES]);
        if naive_evaluate(polynomial, element).is_zero() {
            roots.push(element);
        }
    }
    roots
}

/// Reference Berlekamp–Massey over a scalar sequence, never optimized.
///
/// Returns the minimal connection polynomial `C(x) = 1 + c_1 x + … + c_L x^L`
/// (constant term one) and its degree `L`.
pub fn reference_berlekamp_massey<F: FieldKernels>(sequence: &[F::Elem]) -> (Vec<F::Elem>, usize) {
    let mut c = vec![F::Elem::ONE];
    let mut b = vec![F::Elem::ONE];
    let mut l = 0_usize;
    let mut m = 1_usize;
    let mut bb = F::Elem::ONE;
    for n in 0..sequence.len() {
        let mut discrepancy = sequence[n];
        for i in 1..=l {
            discrepancy = discrepancy.add(c[i].mul(sequence[n - i]));
        }
        if discrepancy.is_zero() {
            m += 1;
        } else if 2 * l <= n {
            let t = c.clone();
            while c.len() < b.len() + m {
                c.push(F::Elem::ZERO);
            }
            for (index, coefficient) in b.iter().enumerate() {
                c[index + m] = c[index + m].sub(discrepancy.mul(bb.inv()).mul(*coefficient));
            }
            l = n + 1 - l;
            b = t;
            bb = discrepancy;
            m = 1;
        } else {
            while c.len() < b.len() + m {
                c.push(F::Elem::ZERO);
            }
            for (index, coefficient) in b.iter().enumerate() {
                c[index + m] = c[index + m].sub(discrepancy.mul(bb.inv()).mul(*coefficient));
            }
            m += 1;
        }
    }
    c.truncate(l + 1);
    (c, l)
}

/// Linear schoolbook series inverse: solve one coefficient at a time.
pub fn naive_series_inverse<F: FieldKernels>(
    polynomial: &Polynomial<F>,
    t: usize,
) -> Polynomial<F> {
    let mut coefficients = vec![F::Elem::ZERO; t];
    coefficients[0] = polynomial.coefficient(0).inv();
    for degree in 1..t {
        let mut discrepancy = F::Elem::ZERO;
        for j in 1..=degree {
            discrepancy = discrepancy.add(polynomial.coefficient(j).mul(coefficients[degree - j]));
        }
        coefficients[degree] = discrepancy.neg().mul(coefficients[0]);
    }
    Polynomial::from_coefficients(&coefficients).expect("naive series inverse")
}
