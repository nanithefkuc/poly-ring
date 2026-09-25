//! Instantiation sweep for the prepared evaluation surfaces.
//!
//! The behavior suites instantiate the plans over a subset of the field
//! types and mostly at sizes below the recursion bases, so the
//! divide-and-conquer bodies, the batched lane paths, and the scratch-reuse
//! routes stay unexecuted for several existing field instantiations. Every
//! test here drives those paths over the fields that already instantiate
//! them — Gf8B, Gf16, Goldilocks, Mersenne31, QuadMersenne31, and the
//! transform-capable Gf32 — and checks each result against an independent
//! scalar oracle: Horner jets, long division, naive convolution, and
//! in-field binomial factors. No result is compared against another
//! production path.

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, Goldilocks, Mersenne31, QuadMersenne31, gf8b, gf16};
use poly_ring::{
    BivariatePolynomial, ChienScratch, DerivativePlan, HermitePlan, JetPlan, MultiIndex,
    MultiplicityPlan, MultipointScratch, Polynomial, PolynomialField, RemainderTree,
    SparsePolynomial, Term, chien_roots, chien_roots_into, evaluate_multipoint,
    evaluate_multipoint_into, interpolate_lagrange, linearized_roots,
};

// ---------------------------------------------------------------------------
// Shared scalar oracles and generators.

/// Fixed-seed LCG in `fgf`'s `noise` shape: the raw state bytes decode to
/// one field element per draw.
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

/// The Horner jet: `D^[j]f(a)` for `j < multiplicity`, one scalar operation
/// at a time through the recurrence `jet ← jet·(a + T) + c`.
fn horner_jet<F: FieldKernels>(
    coefficients: &[F::Elem],
    point: F::Elem,
    multiplicity: usize,
) -> Vec<F::Elem> {
    let mut jet = vec![F::Elem::ZERO; multiplicity];
    if multiplicity == 0 {
        return jet;
    }
    for &coefficient in coefficients.iter().rev() {
        for degree in (1..multiplicity).rev() {
            jet[degree] = jet[degree - 1].add(jet[degree].mul(point));
        }
        jet[0] = jet[0].mul(point).add(coefficient);
    }
    jet
}

/// The integer `n` embedded in the field by repeated doubling of one.
fn embed<F: Field>(n: u64) -> F::Elem {
    let mut term = F::Elem::ONE;
    let mut total = F::Elem::ZERO;
    let mut rest = n;
    while rest != 0 {
        if rest & 1 != 0 {
            total = total.add(term);
        }
        term = term.add(term);
        rest >>= 1;
    }
    total
}

/// The binomial `C(n, k)` as a field element, by Lucas' theorem over the
/// field's characteristic: the digitwise product of small binomials, each
/// taken by the multiplicative recurrence (safe below the characteristic,
/// where every factor is invertible). Exact in every characteristic.
fn field_binomial<F: Field>(n: u64, k: u64) -> F::Elem {
    if k > n {
        return F::Elem::ZERO;
    }
    let characteristic = F::CHARACTERISTIC;
    let mut value = F::Elem::ONE;
    let mut high = n;
    let mut low = k;
    while high > 0 || low > 0 {
        let high_digit = high % characteristic;
        let low_digit = low % characteristic;
        if low_digit > high_digit {
            return F::Elem::ZERO;
        }
        let mut digit = F::Elem::ONE;
        for i in 1..=low_digit {
            digit = digit
                .mul(embed::<F>(high_digit - low_digit + i))
                .mul(embed::<F>(i).inv());
        }
        value = value.mul(digit);
        high /= characteristic;
        low /= characteristic;
    }
    value
}

/// Naive Horner evaluation at every point.
fn horner_values<F: FieldKernels>(polynomial: &Polynomial<F>, points: &[F::Elem]) -> Vec<F::Elem> {
    points
        .iter()
        .map(|point| {
            polynomial
                .coefficients()
                .rev()
                .fold(F::Elem::ZERO, |acc, c| acc.mul(*point).add(c))
        })
        .collect()
}

/// Naive scalar convolution of two coefficient vectors, normalized.
fn naive_convolution<F: FieldKernels>(left: &[F::Elem], right: &[F::Elem]) -> Vec<F::Elem> {
    let mut product = vec![F::Elem::ZERO; left.len() + right.len()];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            product[i + j] = product[i + j].add(a.mul(b));
        }
    }
    normalize_coefficients(&mut product);
    product
}

fn normalize_coefficients<E: Elem>(coefficients: &mut Vec<E>) {
    while coefficients.len() > 1 && coefficients[coefficients.len() - 1].is_zero() {
        coefficients.pop();
    }
}

/// Independent long division: `(quotient, remainder)` over scalar elements.
fn oracle_div_rem<F: FieldKernels>(
    dividend: &[F::Elem],
    divisor: &[F::Elem],
) -> (Vec<F::Elem>, Vec<F::Elem>) {
    let divisor_degree = divisor.len() - 1;
    let leading = divisor[divisor_degree];
    let mut remainder = dividend.to_vec();
    let mut quotient = vec![F::Elem::ZERO; dividend.len().saturating_sub(divisor_degree)];
    loop {
        let degree = match remainder.iter().rposition(|c| !c.is_zero()) {
            Some(degree) if degree >= divisor_degree => degree,
            _ => break,
        };
        let factor = remainder[degree].mul(leading.inv());
        quotient[degree - divisor_degree] = factor;
        for (offset, &coefficient) in divisor.iter().enumerate() {
            let index = degree - divisor_degree + offset;
            remainder[index] = remainder[index].sub(factor.mul(coefficient));
        }
    }
    remainder.truncate(divisor_degree);
    remainder.resize(divisor_degree, F::Elem::ZERO);
    normalize_coefficients(&mut quotient);
    (quotient, remainder)
}

fn poly<F: FieldKernels>(coefficients: &[F::Elem]) -> Polynomial<F> {
    Polynomial::from_coefficients(coefficients).expect("test polynomial")
}

fn coefficients_of<F: FieldKernels>(polynomial: &Polynomial<F>) -> Vec<F::Elem> {
    polynomial.coefficients().collect()
}

/// A coefficient vector whose top coefficient is nonzero, so the length is
/// the degree bound.
fn top_nonzero<F: FieldKernels>(len: usize, seed: u64) -> Vec<F::Elem> {
    let mut coefficients = noise::<F>(len, seed);
    if len > 0 && coefficients[len - 1].is_zero() {
        coefficients[len - 1] = F::Elem::ONE;
    }
    coefficients
}

/// Distinct points by deduplicating draws, keeping caller order.
fn distinct_points<F: FieldKernels>(count: usize, seed: u64) -> Vec<F::Elem> {
    let mut points = Vec::new();
    for point in noise::<F>(2 * count + 4, seed) {
        if !points.contains(&point) {
            points.push(point);
        }
        if points.len() == count {
            break;
        }
    }
    assert_eq!(points.len(), count, "enough distinct draws");
    points
}

/// Encode `lanes` coefficient vectors as coefficient-major packed rows.
fn pack_lanes<F: FieldKernels>(lanes: &[Vec<F::Elem>]) -> Vec<u8> {
    let batch = lanes.len();
    let count = lanes.iter().map(Vec::len).max().unwrap_or(0);
    let mut packed = vec![0_u8; count * batch * F::BYTES];
    for (lane, coefficients) in lanes.iter().enumerate() {
        for (degree, &value) in coefficients.iter().enumerate() {
            let start = (degree * batch + lane) * F::BYTES;
            F::encode(&mut packed[start..start + F::BYTES], value);
        }
    }
    packed
}

// ---------------------------------------------------------------------------
// Jets through the divide-and-conquer recursion.

fn jet_case<F: PolynomialField>(multiplicity: usize, count: usize, seed: u64) {
    let coefficients = noise::<F>(count, seed);
    let mut point_draw = noise::<F>(1, seed ^ 0x5A);
    let drawn = point_draw.pop().unwrap();
    for point in [F::Elem::ZERO, drawn] {
        let plan = JetPlan::<F>::new(point, multiplicity, count).expect("jet plan");
        let mut scratch = plan.scratch(3).expect("jet scratch");
        let mut output = vec![F::Elem::ZERO; multiplicity];
        plan.evaluate_into(&coefficients, &mut scratch, &mut output)
            .expect("scalar jet");
        assert_eq!(output, horner_jet::<F>(&coefficients, point, multiplicity));

        // Three distinct polynomials as lanes, then a warmed second run
        // over the same scratch: the recursion slots must come back intact.
        let lanes = [
            coefficients.clone(),
            noise::<F>(count, seed ^ 0x11),
            noise::<F>(count.saturating_sub(1), seed ^ 0x22),
        ];
        let packed = pack_lanes::<F>(&lanes);
        let live = lanes.iter().map(Vec::len).max().unwrap();
        let mut batched = vec![0_u8; multiplicity * 3 * F::BYTES];
        plan.evaluate_batch_into(&packed, live, 3, &mut scratch, &mut batched)
            .expect("batched jet");
        for (lane, lane_coefficients) in lanes.iter().enumerate() {
            let expected = horner_jet::<F>(lane_coefficients, point, multiplicity);
            for (order, &want) in expected.iter().enumerate() {
                let start = (order * 3 + lane) * F::BYTES;
                let value = F::decode(&batched[start..start + F::BYTES]);
                assert_eq!(value, want, "lane {lane} order {order}");
            }
        }
        plan.evaluate_batch_into(&packed, live, 3, &mut scratch, &mut batched)
            .expect("warmed batched jet");
    }
}

/// Translations above the Horner base recurse for every field that
/// instantiates jets; the multiplicity-seventeen case crosses the base
/// inside a weighted plan too.
#[test]
fn jets_recurse_over_every_instantiated_field() {
    jet_case::<Gf8B>(3, 64, 0x10);
    jet_case::<Gf8B>(17, 33, 0x11);
    jet_case::<Gf16>(3, 64, 0x12);
    jet_case::<Gf16>(17, 65, 0x13);
    jet_case::<Goldilocks>(3, 64, 0x14);
    jet_case::<Goldilocks>(17, 33, 0x15);
    jet_case::<Mersenne31>(3, 64, 0x16);
    jet_case::<Mersenne31>(17, 65, 0x17);
    jet_case::<QuadMersenne31>(3, 64, 0x18);
    jet_case::<QuadMersenne31>(17, 33, 0x19);
}

// ---------------------------------------------------------------------------
// Weighted multipoint evaluation.

fn weighted_case<F: PolynomialField>(seed: u64) {
    let mut points = distinct_points::<F>(4, seed);
    points.insert(0, F::Elem::ZERO);
    // A zero weight, a repeated weight (shared jet scratch slot), and a
    // weight above the jet Horner base.
    let multiplicities = [0_usize, 1, 3, 3, 17];
    let total: usize = multiplicities.iter().sum();
    let count = 64;
    let plan = MultiplicityPlan::<F>::new(&points, &multiplicities, count).expect("plan");
    let mut scratch = plan.scratch(3).expect("scratch");
    let coefficients = noise::<F>(count, seed ^ 0x99);

    let mut output = vec![F::Elem::ZERO; total];
    plan.evaluate_into(&coefficients, &mut scratch, &mut output)
        .expect("scalar weighted evaluation");
    let mut expected = Vec::new();
    for (point, &weight) in points.iter().zip(&multiplicities) {
        expected.extend(horner_jet::<F>(&coefficients, *point, weight));
    }
    assert_eq!(output, expected);

    // Batched lanes, then a warmed second descent through the same scratch.
    let lanes = [
        coefficients.clone(),
        noise::<F>(count, seed ^ 0xAA),
        noise::<F>(count, seed ^ 0xBB),
    ];
    let packed = pack_lanes::<F>(&lanes);
    let mut batched = vec![0_u8; total * 3 * F::BYTES];
    plan.evaluate_batch_into(&packed, count, 3, &mut scratch, &mut batched)
        .expect("batched weighted evaluation");
    for (lane, lane_coefficients) in lanes.iter().enumerate() {
        for (index, (point, &weight)) in points.iter().zip(&multiplicities).enumerate() {
            let jet = horner_jet::<F>(lane_coefficients, *point, weight);
            let offset = plan.offsets()[index];
            for (order, &want) in jet.iter().enumerate() {
                let start = ((offset + order) * 3 + lane) * F::BYTES;
                let value = F::decode(&batched[start..start + F::BYTES]);
                assert_eq!(value, want, "lane {lane} point {index}");
            }
        }
    }
    plan.evaluate_batch_into(&packed, count, 3, &mut scratch, &mut batched)
        .expect("warmed weighted evaluation");

    // The uniform constructor shares the weighted path with every weight
    // equal, and reuses its own scratch across two evaluations.
    let uniform = MultiplicityPlan::<F>::uniform(&points[1..], 2, count).expect("uniform plan");
    let mut uniform_scratch = uniform.scratch(1).expect("uniform scratch");
    let mut uniform_output = vec![F::Elem::ZERO; uniform.total_weight()];
    for _ in 0..2 {
        uniform
            .evaluate_into(&coefficients, &mut uniform_scratch, &mut uniform_output)
            .expect("uniform evaluation");
    }
    let mut uniform_expected = Vec::new();
    for point in &points[1..] {
        uniform_expected.extend(horner_jet::<F>(&coefficients, *point, 2));
    }
    assert_eq!(uniform_output, uniform_expected);
}

#[test]
fn weighted_plans_batch_and_reuse_over_every_instantiated_field() {
    weighted_case::<Gf8B>(0x21);
    weighted_case::<Gf16>(0x22);
    weighted_case::<Goldilocks>(0x23);
    weighted_case::<Mersenne31>(0x24);
    weighted_case::<QuadMersenne31>(0x25);
}

// ---------------------------------------------------------------------------
// Hermite reconstruction.

fn hermite_case<F: PolynomialField>(seed: u64) {
    let points = distinct_points::<F>(4, seed);
    let multiplicities = [2_usize, 3, 1, 4];
    let total: usize = multiplicities.iter().sum();
    let coefficients = noise::<F>(total, seed ^ 0x31);
    let source = poly::<F>(&coefficients);

    let plan = MultiplicityPlan::<F>::new(&points, &multiplicities, total).expect("plan");
    let mut scratch = plan.scratch(1).expect("scratch");
    let mut jets = vec![F::Elem::ZERO; total];
    plan.evaluate_into(&coefficients, &mut scratch, &mut jets)
        .expect("jets");

    let hermite = HermitePlan::<F>::new(&points, &multiplicities).expect("hermite plan");
    let recovered = hermite.interpolate(&jets).expect("hermite interpolation");
    assert_eq!(recovered, source);
}

#[test]
fn hermite_recovers_polynomials_over_every_instantiated_field() {
    hermite_case::<Gf8B>(0x41);
    hermite_case::<Gf16>(0x42);
    hermite_case::<Goldilocks>(0x43);
    hermite_case::<Mersenne31>(0x44);
    hermite_case::<QuadMersenne31>(0x45);
}

// ---------------------------------------------------------------------------
// Subproduct-tree multipoint evaluation and Lagrange interpolation.

fn multipoint_case<F: FieldKernels>(seed: u64) {
    // Above MULTIPOINT_EVAL_CROSSOVER so the tree path runs.
    let first_points = distinct_points::<F>(24, seed);
    let second_points = distinct_points::<F>(24, seed ^ 0x5F);
    let first = poly::<F>(&noise::<F>(20, seed ^ 0x60));
    let second = poly::<F>(&noise::<F>(20, seed ^ 0x61));

    let values = evaluate_multipoint(&first, &first_points).expect("multipoint");
    assert_eq!(values, horner_values(&first, &first_points));

    // Lagrange inverts the evaluation exactly.
    let interpolant = interpolate_lagrange(&first_points, &values).expect("lagrange");
    assert_eq!(interpolant, first);

    // Scratch reuse over a changed point set rebuilds the subproduct tree.
    let mut scratch = MultipointScratch::new();
    let mut reused = Vec::new();
    evaluate_multipoint_into(&mut reused, &second, &second_points, &mut scratch)
        .expect("reused multipoint");
    assert_eq!(reused, horner_values(&second, &second_points));
    let mut again = Vec::new();
    evaluate_multipoint_into(&mut again, &first, &first_points, &mut scratch)
        .expect("rebuilt multipoint");
    assert_eq!(again, values);
}

#[test]
fn multipoint_and_lagrange_round_trip_with_tree_reuse() {
    multipoint_case::<Gf8B>(0x51);
    multipoint_case::<Gf16>(0x52);
    multipoint_case::<Mersenne31>(0x53);
}

// ---------------------------------------------------------------------------
// Prepared remainder trees, batched lanes, scratch reuse.

fn remainder_case<F: PolynomialField>(seed: u64) {
    let moduli: Vec<Polynomial<F>> = [2_usize, 4, 3, 6]
        .iter()
        .enumerate()
        .map(|(index, &len)| poly::<F>(&top_nonzero::<F>(len, seed + index as u64)))
        .collect();
    let degrees: Vec<usize> = moduli.iter().map(|m| m.coefficient_count() - 1).collect();
    let total_rows: usize = degrees.iter().sum();
    let tree = RemainderTree::<F>::new(&moduli, 48).expect("remainder tree");
    assert_eq!(tree.leaf_offsets().last().copied(), Some(total_rows));
    let mut scratch = tree.scratch(3).expect("remainder scratch");

    for round in 0..2 {
        let lanes: Vec<Vec<F::Elem>> = (0..3)
            .map(|lane| top_nonzero::<F>(40, seed ^ (0x70 + round * 16 + lane)))
            .collect();
        let packed = pack_lanes::<F>(&lanes);
        let mut output = vec![0_u8; total_rows * 3 * F::BYTES];
        tree.remainders_into(&packed, 40, 3, &mut scratch, &mut output)
            .expect("batched remainders");
        for (lane, lane_coefficients) in lanes.iter().enumerate() {
            let mut offset = 0;
            for modulus in &moduli {
                let degree = modulus.coefficient_count() - 1;
                let (quotient, remainder) =
                    oracle_div_rem::<F>(lane_coefficients, &coefficients_of(modulus));
                assert!(
                    quotient.len() + degree >= remainder.len(),
                    "oracle shapes stay consistent"
                );
                for (row, &want) in remainder.iter().enumerate() {
                    let start = ((offset + row) * 3 + lane) * F::BYTES;
                    let value = F::decode(&output[start..start + F::BYTES]);
                    assert_eq!(value, want, "lane {lane} round {round}");
                }
                offset += degree;
            }
            assert_eq!(offset, total_rows);
        }
    }
}

#[test]
fn remainder_trees_batch_lanes_and_reuse_scratch() {
    remainder_case::<Gf8B>(0x81);
    remainder_case::<Gf16>(0x82);
    remainder_case::<Goldilocks>(0x83);
    remainder_case::<Mersenne31>(0x84);
    remainder_case::<QuadMersenne31>(0x85);
}

// ---------------------------------------------------------------------------
// Derivative plans.

fn derivative_case<F: PolynomialField>(seed: u64) {
    let count = 40;
    let coefficients = noise::<F>(count, seed);
    for order in [1_usize, 2, 3] {
        let plan = DerivativePlan::<F>::new(order, count).expect("derivative plan");
        assert_eq!(plan.order(), order);
        assert_eq!(plan.max_coefficients(), count);
        let factor = |degree: usize| -> F::Elem {
            coefficients[degree + order]
                .mul(field_binomial::<F>((degree + order) as u64, order as u64))
        };
        let expected: Vec<F::Elem> = (0..count - order).map(factor).collect();
        let mut output = vec![F::Elem::ZERO; count - order];
        plan.apply_into(&coefficients, &mut output).expect("apply");
        assert_eq!(output, expected, "order {order}");

        // Batched lanes through the same prepared factors.
        let lanes = [
            coefficients.clone(),
            noise::<F>(count, seed ^ 0x91),
            noise::<F>(count - order, seed ^ 0x92),
        ];
        let packed = pack_lanes::<F>(&lanes);
        let live = lanes.iter().map(Vec::len).max().unwrap();
        let out_rows = lanes.iter().map(|l| l.len() - order).max().unwrap();
        let mut batched = vec![0_u8; out_rows * 3 * F::BYTES];
        plan.apply_batch_into(&packed, live, 3, &mut batched)
            .expect("batched apply");
        for (lane, lane_coefficients) in lanes.iter().enumerate() {
            let lane_factor = |degree: usize| -> F::Elem {
                lane_coefficients[degree + order]
                    .mul(field_binomial::<F>((degree + order) as u64, order as u64))
            };
            for degree in 0..lane_coefficients.len() - order {
                let start = (degree * 3 + lane) * F::BYTES;
                let value = F::decode(&batched[start..start + F::BYTES]);
                assert_eq!(
                    value,
                    lane_factor(degree),
                    "lane {lane} degree {degree} order {order}"
                );
            }
        }
    }
}

#[test]
fn derivative_batches_match_binomial_factors_over_instantiated_fields() {
    derivative_case::<Gf8B>(0xA1);
    derivative_case::<Goldilocks>(0xA2);
    derivative_case::<Mersenne31>(0xA3);
}

// ---------------------------------------------------------------------------
// Bivariate surfaces.

/// Naive bivariate coefficient table: `rows[j][i]` is the `X^i Y^j`
/// coefficient.
type BivariateTable<F> = Vec<Vec<<F as Field>::Elem>>;

fn bivariate_rows<F: FieldKernels>(value: &BivariatePolynomial<F>) -> BivariateTable<F> {
    (0..value.y_coefficient_count())
        .map(|row| coefficients_of(value.y_coefficient(row).expect("row")))
        .collect()
}

fn normalized_table<E: Elem>(mut table: Vec<Vec<E>>) -> Vec<Vec<E>> {
    while table.len() > 1
        && table
            .last()
            .is_some_and(|row| row.iter().all(|c| c.is_zero()))
    {
        table.pop();
    }
    for row in &mut table {
        normalize_coefficients(row);
    }
    table
}

fn table_add<E: Elem>(left: &[Vec<E>], right: &[Vec<E>], negate: bool) -> Vec<Vec<E>> {
    let rows = left.len().max(right.len());
    let width = left.iter().chain(right).map(Vec::len).max().unwrap_or(0);
    let mut sum = vec![vec![E::ZERO; width]; rows];
    for (table, sign) in [(left, false), (right, negate)] {
        for (j, row) in table.iter().enumerate() {
            for (i, &c) in row.iter().enumerate() {
                sum[j][i] = sum[j][i].add(if sign { c.neg() } else { c });
            }
        }
    }
    sum
}

fn table_mul<E: Elem>(left: &[Vec<E>], right: &[Vec<E>]) -> Vec<Vec<E>> {
    let width = left.iter().map(Vec::len).max().unwrap_or(0)
        + right.iter().map(Vec::len).max().unwrap_or(0);
    let mut product = vec![vec![E::ZERO; width]; left.len() + right.len()];
    for (jx, rowx) in left.iter().enumerate() {
        for (jy, rowy) in right.iter().enumerate() {
            for (ix, &cx) in rowx.iter().enumerate() {
                for (iy, &cy) in rowy.iter().enumerate() {
                    product[jx + jy][ix + iy] = product[jx + jy][ix + iy].add(cx.mul(cy));
                }
            }
        }
    }
    normalized_table(product)
}

fn from_table<F: FieldKernels>(table: &BivariateTable<F>) -> BivariatePolynomial<F> {
    BivariatePolynomial::from_y_coefficients(
        table.iter().map(|row| poly::<F>(row)).collect::<Vec<_>>(),
    )
}

fn assert_tables_equal<F: FieldKernels>(
    actual: &BivariatePolynomial<F>,
    expected: &BivariateTable<F>,
    what: &str,
) {
    assert_eq!(
        bivariate_rows::<F>(actual),
        normalized_table(expected.to_vec()),
        "{what}"
    );
}

fn bivariate_case<F: FieldKernels>(seed: u64) {
    let rows: Vec<Vec<F::Elem>> = (0..3)
        .map(|row| top_nonzero::<F>(3 + row, seed + row as u64))
        .collect();
    let left = from_table::<F>(&rows);

    // Growth through set_y_coefficient reproduces the direct construction,
    // and a zero final row is dropped by normalization.
    let mut grown = BivariatePolynomial::<F>::default();
    for (degree, row) in rows.iter().enumerate() {
        grown
            .set_y_coefficient(degree, poly::<F>(row))
            .expect("set row");
    }
    assert_eq!(grown, left);
    assert_eq!(grown.y_degree(), Some(2));

    // Row arithmetic against the naive tables.
    let other_rows: Vec<Vec<F::Elem>> = (0..2)
        .map(|row| top_nonzero::<F>(2 + row, seed ^ (0xB0 + row as u64)))
        .collect();
    let right = from_table::<F>(&other_rows);
    assert_tables_equal::<F>(
        &left.add(&right).expect("sum"),
        &table_add(&rows, &other_rows, false),
        "sum",
    );
    assert_tables_equal::<F>(
        &left.sub(&right).expect("difference"),
        &table_add(&rows, &other_rows, true),
        "difference",
    );
    let scale = noise::<F>(1, seed ^ 0xB7).pop().unwrap();
    let scaled_table: BivariateTable<F> = rows
        .iter()
        .map(|row| row.iter().map(|&c| c.mul(scale)).collect())
        .collect();
    assert_tables_equal::<F>(
        &left.scaled(scale).expect("scaled"),
        &scaled_table,
        "scaled",
    );
    assert_tables_equal::<F>(
        &left.multiply(&right).expect("product"),
        &table_mul(&rows, &other_rows),
        "product",
    );

    // The weighted order: the leading term names the maximum weighted
    // degree among the table's nonzero coefficients.
    let y_weight = 2;
    let leading = left.weighted_leading_term(y_weight).expect("leading term");
    let mut best: Option<usize> = None;
    for (j, row) in rows.iter().enumerate() {
        for (i, &c) in row.iter().enumerate() {
            if !c.is_zero() {
                let weighted = i + j * y_weight;
                if best.is_none_or(|current| weighted > current) {
                    best = Some(weighted);
                }
            }
        }
    }
    let leading = leading.expect("nonzero leading term");
    assert_eq!(Some(leading.weighted_degree), best);
    assert_eq!(
        left.weighted_degree(y_weight).expect("weighted degree"),
        best
    );

    // The bivariate Hasse derivative against the binomial expansion: row t
    // of `D^[x, y] Q` accumulates C(j, y) · D^[x] row j`, `j >= t + y`. The
    // univariate row derivative is `D^[x] row[i] = C(i + x, x) · row[i+x]`.
    let width = rows.iter().map(Vec::len).max().unwrap();
    let x_hasse = |order: usize, source: &[F::Elem]| -> Vec<F::Elem> {
        (0..source.len().saturating_sub(order))
            .map(|i| source[i + order].mul(field_binomial::<F>((i + order) as u64, order as u64)))
            .collect()
    };
    for (x_order, y_order) in [(0_usize, 1_usize), (1, 1), (1, 0)] {
        let derivative = left
            .hasse_derivative(x_order, y_order)
            .expect("hasse derivative");
        let mut derivative_table: BivariateTable<F> = Vec::new();
        for target in 0..rows.len() - y_order {
            let source = target + y_order;
            let factor = field_binomial::<F>(source as u64, y_order as u64);
            let row: Vec<F::Elem> = if factor.is_zero() {
                vec![F::Elem::ZERO; width]
            } else {
                x_hasse(x_order, &rows[source])
                    .iter()
                    .map(|&c| c.mul(factor))
                    .chain(core::iter::repeat(F::Elem::ZERO))
                    .take(width)
                    .collect()
            };
            derivative_table.push(row);
        }
        assert_tables_equal::<F>(
            &derivative,
            &derivative_table,
            "hasse derivative {x_order},{y_order}",
        );
    }

    // multiply_x_plus: (X + c) · Q against naive convolution per row.
    let constant = noise::<F>(1, seed ^ 0xB9).pop().unwrap();
    let shifted = left.multiply_x_plus(constant).expect("multiply x plus");
    let shifted_table: BivariateTable<F> = rows
        .iter()
        .map(|row| naive_convolution::<F>(row, &[constant, F::Elem::ONE]))
        .collect();
    assert_tables_equal::<F>(&shifted, &shifted_table, "multiply x plus");

    // Y = c substitution: exactly the naive row combination.
    for substitution in [F::Elem::ZERO, noise::<F>(1, seed ^ 0xBA).pop().unwrap()] {
        let substituted = left.substitute_y_linear(substitution).expect("substitute");
        let mut combined = vec![F::Elem::ZERO; width];
        for (j, row) in rows.iter().enumerate() {
            let mut power = F::Elem::ONE;
            for _ in 0..j {
                power = power.mul(substitution);
            }
            for (i, &c) in row.iter().enumerate() {
                combined[i] = combined[i].add(c.mul(power));
            }
        }
        normalize_coefficients(&mut combined);
        assert_eq!(
            coefficients_of(substituted.y_coefficient(0).expect("row")),
            combined
        );
    }

    // Affine truncated substitution against direct polynomial composition
    // of Y = prefix(X) + X^tail · Z, truncated per row. The Z^t row of the
    // t-th power is C(j, t) · prefix^(j−t) shifted by t·tail.
    let prefix_coeffs = top_nonzero::<F>(2, seed ^ 0xBB);
    let tail = 2_usize;
    let keep = 6_usize;
    let truncated = left
        .substitute_y_affine_truncated(&poly::<F>(&prefix_coeffs), tail, keep)
        .expect("affine substitution");
    let trunc_mul = |a: &[F::Elem], b: &[F::Elem]| -> Vec<F::Elem> {
        let mut full = naive_convolution::<F>(a, b);
        full.resize(keep, F::Elem::ZERO);
        full
    };
    let padded = |source: &[F::Elem]| -> Vec<F::Elem> {
        let mut out = vec![F::Elem::ZERO; keep];
        for (i, &c) in source.iter().enumerate().take(keep) {
            out[i] = c;
        }
        out
    };
    let base_const = padded(&prefix_coeffs);
    let mut expected_affine: BivariateTable<F> = vec![vec![F::Elem::ZERO; keep]; rows.len()];
    for (j, row) in rows.iter().enumerate() {
        for (target, slot) in expected_affine.iter_mut().enumerate().take(j + 1) {
            let binom = field_binomial::<F>(j as u64, target as u64);
            if binom.is_zero() {
                continue;
            }
            let mut prefix_power = vec![F::Elem::ZERO; keep];
            prefix_power[0] = F::Elem::ONE;
            for _ in 0..(j - target) {
                prefix_power = trunc_mul(&prefix_power, &base_const);
            }
            let shift = target * tail;
            let mut z_row = vec![F::Elem::ZERO; keep];
            for (i, &c) in prefix_power.iter().enumerate() {
                if i + shift < keep {
                    z_row[i + shift] = c.mul(binom);
                }
            }
            let term = trunc_mul(&padded(row), &z_row);
            for (slot, &value) in slot.iter_mut().zip(&term) {
                *slot = slot.add(value);
            }
        }
    }
    assert_tables_equal::<F>(&truncated, &expected_affine, "affine substitution");

    // Compose Y with a polynomial; the root predicate answers whether that
    // composition vanishes.
    let candidate = poly::<F>(&top_nonzero::<F>(2, seed ^ 0xBC));
    let composed = left.compose_y(&candidate).expect("compose");
    let candidate_coeffs = coefficients_of(&candidate);
    let mut composed_table = vec![F::Elem::ZERO; 2 * width + 4];
    for (j, row) in rows.iter().enumerate() {
        let mut power = vec![F::Elem::ONE];
        for _ in 0..j {
            power = naive_convolution::<F>(&power, &candidate_coeffs);
        }
        let contribution = naive_convolution::<F>(row, &power);
        for (i, &c) in contribution.iter().enumerate() {
            composed_table[i] = composed_table[i].add(c);
        }
    }
    normalize_coefficients(&mut composed_table);
    assert_eq!(coefficients_of(&composed), composed_table, "compose");
    assert_eq!(
        left.has_root(&candidate).expect("has root"),
        composed.is_zero()
    );

    // X-structure helpers against shifted tables.
    let valuation = 3;
    let shifted_source: Vec<Vec<F::Elem>> = rows
        .iter()
        .map(|row| {
            let mut shifted = vec![F::Elem::ZERO; row.len() + valuation];
            shifted[valuation..].copy_from_slice(row);
            shifted
        })
        .collect();
    let shifted_poly = from_table::<F>(&shifted_source);
    assert_eq!(shifted_poly.x_valuation(), Some(valuation));
    assert_tables_equal::<F>(
        &shifted_poly.divide_by_x_power(valuation).expect("divide"),
        &rows,
        "divide by x power",
    );
    // Both sides pass through the canonical form, so an all-zero truncation
    // compares equal to the zero polynomial.
    assert_eq!(
        shifted_poly.truncated_x(2),
        from_table::<F>(
            &shifted_source
                .iter()
                .map(|row| row[..2].to_vec())
                .collect::<Vec<_>>()
        ),
        "truncated x"
    );

    // Scaled shifted accumulation against the naive table.
    let mut accumulator = from_table::<F>(&rows);
    accumulator
        .add_scaled_x_shifted_assign(scale, &right, 2)
        .expect("add scaled shifted");
    let mut accumulated: BivariateTable<F> = rows.clone();
    for (j, row) in other_rows.iter().enumerate() {
        while accumulated.len() <= j {
            accumulated.push(Vec::new());
        }
        for (i, &c) in row.iter().enumerate() {
            let target = i + 2;
            while accumulated[j].len() <= target {
                accumulated[j].push(F::Elem::ZERO);
            }
            accumulated[j][target] = accumulated[j][target].add(c.mul(scale));
        }
    }
    assert_tables_equal::<F>(&accumulator, &accumulated, "add scaled shifted");

    // Packed row assignment round-trips the same table; a partial trailing
    // element rejects the whole call without touching the polynomial.
    let packed_rows: Vec<Vec<u8>> = rows
        .iter()
        .map(|row| {
            let mut bytes = vec![0_u8; row.len() * F::BYTES];
            for (i, &value) in row.iter().enumerate() {
                F::encode(&mut bytes[i * F::BYTES..(i + 1) * F::BYTES], value);
            }
            bytes
        })
        .collect();
    let mut assigned = BivariatePolynomial::<F>::default();
    assigned
        .assign_y_coefficients_packed(packed_rows.iter().map(Vec::as_slice))
        .expect("packed assignment");
    assert_eq!(assigned, left);
    let mut rejected = BivariatePolynomial::<F>::default();
    if F::BYTES > 1 {
        // One byte short of a whole element: the whole call rejects.
        let mut short = packed_rows[0].clone();
        for _ in 0..F::BYTES - 1 {
            short.pop();
        }
        assert!(
            rejected
                .assign_y_coefficients_packed(Some(short.as_slice()).into_iter())
                .is_err()
        );
        assert!(rejected.is_zero());
    }
}

#[test]
fn bivariate_surfaces_over_instantiated_fields() {
    bivariate_case::<Gf8B>(0xC1);
    bivariate_case::<Gf16>(0xC2);
    bivariate_case::<Mersenne31>(0xC3);
}

// ---------------------------------------------------------------------------
// Ring division and products.

fn division_case<F: FieldKernels>(seed: u64) {
    let dividend = top_nonzero::<F>(40, seed);
    let divisor = top_nonzero::<F>(7, seed ^ 0xD0);
    let dividend_poly = poly::<F>(&dividend);
    let divisor_poly = poly::<F>(&divisor);

    let (quotient, remainder) = dividend_poly.div_rem(&divisor_poly).expect("div rem");
    let (oracle_quotient, oracle_remainder) = oracle_div_rem::<F>(&dividend, &divisor);
    assert_eq!(coefficients_of(&quotient), oracle_quotient);
    assert_eq!(coefficients_of(&remainder), oracle_remainder);

    // Exact division holds when the remainder is zero.
    let product = poly::<F>(&naive_convolution::<F>(&dividend, &divisor));
    let exact = product.divide_exact(&divisor_poly).expect("divide exact");
    assert_eq!(exact, dividend_poly);

    // Modular exponentiation against repeated multiplication.
    let modulus = poly::<F>(&top_nonzero::<F>(5, seed ^ 0xD1));
    let base = poly::<F>(&top_nonzero::<F>(4, seed ^ 0xD2));
    let reduced = |value: &Polynomial<F>| -> Polynomial<F> {
        let (_, rem) = oracle_div_rem::<F>(&coefficients_of(value), &coefficients_of(&modulus));
        poly::<F>(&rem)
    };
    let mut expected = Polynomial::<F>::one().expect("one");
    for _ in 0..5 {
        expected = reduced(&expected.multiply(&base).expect("step"));
    }
    assert_eq!(base.pow_mod(5, &modulus).expect("pow mod"), expected);
    assert_eq!(
        base.square_mod(&modulus).expect("square mod"),
        reduced(&base.multiply(&base).expect("square"))
    );
    assert_eq!(
        base.multiply_mod(&divisor_poly, &modulus)
            .expect("multiply mod"),
        reduced(&base.multiply(&divisor_poly).expect("product"))
    );

    // X-valuation, shifted scaling, and shifted division.
    let mut shifted = vec![F::Elem::ZERO; 12];
    shifted[3..].copy_from_slice(&top_nonzero::<F>(9, seed ^ 0xD3));
    let shifted_poly = poly::<F>(&shifted);
    assert_eq!(shifted_poly.x_valuation(), Some(3));
    assert_eq!(
        shifted_poly.divide_by_x_power(3).expect("divide"),
        poly::<F>(&shifted[3..])
    );
    let scale = noise::<F>(1, seed ^ 0xD4).pop().unwrap();
    let scaled_shifted = shifted_poly
        .scaled_shifted(scale, 2)
        .expect("scaled shifted");
    let mut oracle_shifted = vec![F::Elem::ZERO; shifted.len() + 2];
    for (i, &c) in shifted.iter().enumerate() {
        oracle_shifted[i + 2] = c.mul(scale);
    }
    normalize_coefficients(&mut oracle_shifted);
    assert_eq!(coefficients_of(&scaled_shifted), oracle_shifted);
}

#[test]
fn ring_division_round_trips_over_instantiated_fields() {
    division_case::<Gf8B>(0xE1);
    division_case::<Gf16>(0xE2);
    division_case::<Goldilocks>(0xE3);
    division_case::<Mersenne31>(0xE4);
    division_case::<QuadMersenne31>(0xE5);
}

fn karatsuba_case<F: FieldKernels>(seed: u64) {
    // Above the Karatsuba crossover: the recursive product path.
    let left = top_nonzero::<F>(70, seed);
    let right = top_nonzero::<F>(69, seed ^ 0xE9);
    let product = poly::<F>(&left)
        .multiply(&poly::<F>(&right))
        .expect("product");
    assert_eq!(
        coefficients_of(&product),
        naive_convolution::<F>(&left, &right)
    );
}

#[test]
fn large_products_match_naive_convolution_over_instantiated_fields() {
    karatsuba_case::<Gf8B>(0xF1);
    karatsuba_case::<Gf16>(0xF2);
    karatsuba_case::<Goldilocks>(0xF3);
    karatsuba_case::<Mersenne31>(0xF4);
    karatsuba_case::<QuadMersenne31>(0xF5);
    karatsuba_case::<fgf::Gf8D>(0xF6);
}

// ---------------------------------------------------------------------------
// Sparse multivariate surfaces.

/// Structural equality for sparse forms in generic code: same term count,
/// same exponent vectors, coefficients equal by field value.
fn assert_sparse_equal<F: FieldKernels, const N: usize>(
    actual: &SparsePolynomial<F, N>,
    expected: &SparsePolynomial<F, N>,
) {
    assert_eq!(actual.terms().len(), expected.terms().len());
    for (x, y) in actual.terms().iter().zip(expected.terms()) {
        assert_eq!(x.exponents.exponents(), y.exponents.exponents());
        assert!(x.coefficient.sub(y.coefficient).is_zero());
    }
}

fn multivariate_case<F: FieldKernels>(seed: u64) {
    let a = noise::<F>(1, seed).pop().unwrap();
    let b = noise::<F>(1, seed ^ 1).pop().unwrap();
    let c = noise::<F>(1, seed ^ 2).pop().unwrap();
    let d = noise::<F>(1, seed ^ 3).pop().unwrap();
    let left = SparsePolynomial::<F, 2>::from_terms(vec![
        Term {
            coefficient: a,
            exponents: MultiIndex::new([1, 0]),
        },
        Term {
            coefficient: b,
            exponents: MultiIndex::new([0, 2]),
        },
    ]);
    let right = SparsePolynomial::<F, 2>::from_terms(vec![
        Term {
            coefficient: c,
            exponents: MultiIndex::new([2, 0]),
        },
        Term {
            coefficient: d,
            exponents: MultiIndex::new([0, 1]),
        },
    ]);

    // Products against the term-pair convolution.
    let product = left.multiply(&right).expect("sparse product");
    let mut terms: Vec<Term<F, 2>> = Vec::new();
    for x in left.terms() {
        for y in right.terms() {
            let mut exponents = [0_usize; 2];
            for (i, (e1, e2)) in x
                .exponents
                .exponents()
                .iter()
                .zip(y.exponents.exponents())
                .enumerate()
            {
                exponents[i] = e1 + e2;
            }
            terms.push(Term {
                coefficient: x.coefficient.mul(y.coefficient),
                exponents: MultiIndex::new(exponents),
            });
        }
    }
    assert_sparse_equal(&product, &SparsePolynomial::<F, 2>::from_terms(terms));

    // Sum and difference normalize through the same merge.
    assert_sparse_equal(
        &left.add(&right).expect("sum"),
        &SparsePolynomial::<F, 2>::from_terms(
            left.terms()
                .iter()
                .chain(right.terms())
                .map(|t| Term {
                    coefficient: t.coefficient,
                    exponents: MultiIndex::new(*t.exponents.exponents()),
                })
                .collect(),
        ),
    );

    // Point evaluation and Hasse evaluation against the defining sums.
    let point = [
        noise::<F>(1, seed ^ 4).pop().unwrap(),
        noise::<F>(1, seed ^ 5).pop().unwrap(),
    ];
    let evaluate = |value: &SparsePolynomial<F, 2>, order: [usize; 2]| -> F::Elem {
        let mut total = F::Elem::ZERO;
        for term in value.terms() {
            let mut factor = term.coefficient;
            for (i, (&e, &r)) in term
                .exponents
                .exponents()
                .iter()
                .zip(order.iter())
                .enumerate()
            {
                if r > e {
                    factor = F::Elem::ZERO;
                    break;
                }
                factor = factor
                    .mul(field_binomial::<F>(e as u64, r as u64))
                    .mul(point[i].pow(u64::try_from(e - r).expect("small exponent")));
            }
            total = total.add(factor);
        }
        total
    };
    assert_eq!(left.evaluate(&point), evaluate(&left, [0, 0]));
    assert_eq!(
        left.evaluate_hasse(&point, &MultiIndex::new([1, 0])),
        evaluate(&left, [1, 0])
    );
    assert_eq!(
        left.evaluate_hasse(&point, &MultiIndex::new([0, 1])),
        evaluate(&left, [0, 1])
    );
    let mut jet = vec![F::Elem::ZERO; 2];
    left.evaluate_jet_into(
        &point,
        &[MultiIndex::new([0, 0]), MultiIndex::new([1, 0])],
        &mut jet,
    )
    .expect("jet");
    assert_eq!(jet, vec![evaluate(&left, [0, 0]), evaluate(&left, [1, 0])]);

    // The Hasse derivative polynomial against the binomial expansion.
    let derivative = left
        .hasse_derivative(&MultiIndex::new([1, 0]))
        .expect("sparse derivative");
    let mut derivative_terms = Vec::new();
    for term in left.terms() {
        let e = term.exponents.exponents()[0];
        let factor = field_binomial::<F>(e as u64, 1);
        if e >= 1 && !factor.is_zero() {
            derivative_terms.push(Term {
                coefficient: term.coefficient.mul(factor),
                exponents: MultiIndex::new([e - 1, term.exponents.exponents()[1]]),
            });
        }
    }
    assert_sparse_equal(
        &derivative,
        &SparsePolynomial::<F, 2>::from_terms(derivative_terms),
    );

    // Bivariate conversion round-trips through the dense form.
    let dense_rows: BivariateTable<F> = vec![
        {
            let mut row = vec![F::Elem::ZERO; 2];
            row[1] = a;
            row
        },
        Vec::new(),
        {
            let mut row = vec![F::Elem::ZERO; 1];
            row[0] = b;
            row
        },
    ];
    let dense = from_table::<F>(&normalized_table(dense_rows.clone()));
    let sparse = SparsePolynomial::<F, 2>::from_bivariate(&dense).expect("from bivariate");
    assert_eq!(sparse.to_bivariate().expect("to bivariate"), dense);
    assert!(
        sparse
            .coefficient(&MultiIndex::new([1, 0]))
            .sub(a)
            .is_zero(),
        "coefficient lookup"
    );
    assert!(sparse.coefficient(&MultiIndex::new([3, 3])).is_zero());

    // Univariate conversion round-trips in the single-variable form.
    let univariate = poly::<F>(&top_nonzero::<F>(5, seed ^ 6));
    let sparse_univariate =
        SparsePolynomial::<F, 1>::from_univariate(&univariate).expect("from univariate");
    assert_eq!(
        sparse_univariate.to_univariate().expect("to univariate"),
        univariate
    );
}

#[test]
fn multivariate_surfaces_over_instantiated_fields() {
    multivariate_case::<Gf8B>(0x101);
    multivariate_case::<Mersenne31>(0x102);
}

// ---------------------------------------------------------------------------
// Chien scans and linearized solving.

/// The zero polynomial reports every element as a root, and a locator with
/// known roots finds exactly them, reusing one scratch across scans.
#[test]
fn chien_scans_cover_the_all_and_reuse_paths() {
    let zero = Polynomial::<Gf8B>::zero();
    assert!(matches!(
        chien_roots(&zero).expect("chien zero"),
        poly_ring::BaseFieldRoots::All
    ));

    let roots = [0_u8, 1, 2, 3];
    let mut locator_coeffs = vec![gf8b::Elem::from_raw(1)];
    for root in roots {
        let mut next = vec![gf8b::Elem::ZERO; locator_coeffs.len() + 1];
        for (i, &c) in locator_coeffs.iter().enumerate() {
            next[i + 1] = next[i + 1].add(c);
            next[i] = next[i].add(c.mul(gf8b::Elem::from_raw(root)));
        }
        locator_coeffs = next;
    }
    let locator = Polynomial::<Gf8B>::from_coefficients(&locator_coeffs).expect("locator");
    let mut scratch = ChienScratch::new();
    let mut found = Vec::new();
    assert!(!chien_roots_into(&mut found, &locator, &mut scratch).expect("scan"));
    for root in roots {
        assert!(found.contains(&gf8b::Elem::from_raw(root)));
    }
    assert_eq!(found.len(), roots.len());

    // A warmed scan over the zero polynomial reports every element.
    let mut second = Vec::new();
    assert!(chien_roots_into(&mut second, &zero, &mut scratch).expect("warmed scan"));
    assert!(second.is_empty());
}

/// Linearized solving over Gf16: `L(X) = X² + X` against a full enumeration
/// oracle, for a homogeneous and an affine right-hand side.
#[test]
fn linearized_solutions_match_enumeration() {
    let mut coefficients = vec![gf16::Elem::ZERO; 3];
    coefficients[1] = gf16::Elem::ONE;
    coefficients[2] = gf16::Elem::ONE;
    let linearized = Polynomial::<Gf16>::from_coefficients(&coefficients).expect("linearized");
    for affine in [gf16::Elem::ZERO, gf16::Elem::ONE] {
        let solved = linearized_roots(&linearized, affine).expect("solve");
        let enumerated: Vec<gf16::Elem> = (0_u32..1 << 16)
            .map(|raw| gf16::Elem::from_raw(raw as u16))
            .filter(|element| linearized.evaluate(*element).add(affine).is_zero())
            .collect();
        assert_eq!(solved.len(), enumerated.len(), "affine {affine:?}");
        for root in &enumerated {
            assert!(solved.contains(root), "root {root:?} missing");
        }
    }

    // The zero linearized polynomial enumerates the field, and with a
    // nonzero affine constant has no solutions at all.
    let zero = Polynomial::<Gf16>::zero();
    let all = linearized_roots(&zero, gf16::Elem::ZERO).expect("enumerate");
    assert_eq!(
        all.len(),
        usize::try_from(<Gf16 as Field>::ORDER).expect("order")
    );
    assert!(
        linearized_roots(&zero, gf16::Elem::ONE)
            .expect("no solutions")
            .is_empty()
    );
}

// ---------------------------------------------------------------------------
// Structured-domain transforms (fft).

#[cfg(feature = "fft")]
mod transforms {
    use super::{horner_values, noise, poly, top_nonzero};
    use butterfly_fft::kernel::ButterflyKernels;
    use fgf::{Gf16, Gf32};
    use poly_ring::{
        EvaluationDomain, Polynomial, TransformScratch, evaluate_coset_into, evaluate_subspace,
        evaluate_subspace_into, interpolate_subspace,
    };

    fn transform_case<F>(size: usize, seed: u64)
    where
        F: ButterflyKernels,
        Polynomial<F>: PartialEq,
    {
        let domain = EvaluationDomain::<F>::additive_subspace(size).expect("subspace");
        let plan = domain.transform_plan().expect("transform plan");
        let mut scratch = TransformScratch::new();

        // A short polynomial evaluates identically to Horner.
        let short = poly::<F>(&noise::<F>(size - 1, seed));
        let values = evaluate_subspace(&short, plan, &mut scratch).expect("subspace evaluation");
        assert_eq!(values, horner_values(&short, domain.points()));

        // A polynomial longer than the domain reduces first: same values.
        let long = poly::<F>(&top_nonzero::<F>(size * 2 + 9, seed ^ 0x1));
        let mut long_values = Vec::new();
        evaluate_subspace_into(&mut long_values, &long, plan, &mut scratch)
            .expect("long subspace evaluation");
        assert_eq!(long_values, horner_values(&long, domain.points()));

        // Interpolation inverts the evaluation on the same plan.
        let interpolant =
            interpolate_subspace(plan, &values, &mut scratch).expect("subspace interpolation");
        assert_eq!(interpolant, short);

        // The affine coset: a shifted domain evaluates by Horner as well.
        let shift = noise::<F>(1, seed ^ 0x2).pop().unwrap();
        let coset = EvaluationDomain::<F>::affine_coset(size, shift).expect("coset");
        let coset_plan = coset.transform_plan().expect("coset plan");
        let mut coset_values = Vec::new();
        evaluate_coset_into(&mut coset_values, &long, coset_plan, &mut scratch)
            .expect("coset evaluation");
        assert_eq!(coset_values, horner_values(&long, coset.points()));
    }

    #[test]
    fn subspace_and_coset_transforms_round_trip() {
        transform_case::<Gf16>(16, 0x111);
        transform_case::<Gf32>(32, 0x112);
    }
}

// ---------------------------------------------------------------------------
// Reservation-failure surfaces, per field instantiation.
//
// The same closed-gate pattern the Gf8B and Mersenne31 error suites use,
// driven over the field instances they leave out: the first reservation of
// each prepared constructor and evaluation refuses through the checked
// fallible allocation and surfaces its named error.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct FailingAllocator;

thread_local! {
    static THRESHOLD: Cell<usize> = const { Cell::new(usize::MAX) };
}

unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let refused = THRESHOLD
            .try_with(|gate| layout.size() > gate.get())
            .unwrap_or(false)
            && !std::thread::panicking();
        if refused {
            std::ptr::null_mut()
        } else {
            unsafe { System.alloc(layout) }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: FailingAllocator = FailingAllocator;

/// Arms the gate for one operation and disarms it however the operation
/// leaves.
fn with_threshold<R>(bytes: usize, operation: impl FnOnce() -> R) -> R {
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            THRESHOLD.with(|gate| gate.set(usize::MAX));
        }
    }
    THRESHOLD.with(|gate| gate.set(bytes));
    let _disarm = Disarm;
    operation()
}

fn reservation_case<F: PolynomialField>(seed: u64) {
    let points = distinct_points::<F>(4, seed);
    let multiplicities = [2_usize, 3, 1, 4];
    let point = noise::<F>(1, seed ^ 0x1).pop().unwrap();

    let plan = MultiplicityPlan::<F>::new(&points, &multiplicities, 32).expect("plan");
    let jet = JetPlan::<F>::new(point, 5, 32).expect("jet");
    let sample = poly::<F>(&noise::<F>(20, seed ^ 0x2));
    let multipoint_points = distinct_points::<F>(24, seed ^ 0x3);
    let values = horner_values(&sample, &multipoint_points);

    // Validation error arms of the evaluation entry points, per field
    // instance: output length, lane capacity, coefficient capacity, and
    // scratch geometry each name their violation before any mutation.
    let mut scratch = plan.scratch(3).expect("scratch");
    let coefficients = noise::<F>(8, seed ^ 0x4);
    let mut short = vec![F::Elem::ZERO; plan.total_weight() - 1];
    assert!(
        plan.evaluate_into(&coefficients, &mut scratch, &mut short)
            .is_err()
    );
    let packed = pack_lanes::<F>(&[coefficients.clone(), coefficients.clone(), coefficients]);
    let mut batched = vec![0_u8; plan.total_weight() * 3 * F::BYTES];
    assert!(
        plan.evaluate_batch_into(&packed, 33, 3, &mut scratch, &mut batched)
            .is_err()
    );
    assert!(
        plan.evaluate_batch_into(&packed, 8, 4, &mut scratch, &mut batched)
            .is_err()
    );
    let mut mis_sized = vec![0_u8; plan.total_weight() * 3 * F::BYTES - 1];
    assert!(
        plan.evaluate_batch_into(&packed, 8, 3, &mut scratch, &mut mis_sized)
            .is_err()
    );

    // The jet entry points name the same violations per instance.
    let mut jet_scratch = jet.scratch(3).expect("jet scratch");
    let mut jet_output = vec![0_u8; 5 * 3 * F::BYTES - 1];
    assert!(
        jet.evaluate_batch_into(&packed, 8, 3, &mut jet_scratch, &mut jet_output)
            .is_err()
    );
    let mut jet_long = vec![0_u8; 5 * 3 * F::BYTES];
    assert!(
        jet.evaluate_batch_into(&packed, 33, 3, &mut jet_scratch, &mut jet_long)
            .is_err()
    );
    assert!(
        jet.evaluate_batch_into(&packed, 8, 4, &mut jet_scratch, &mut jet_long)
            .is_err()
    );

    // Hermite interpolation rejects short value vectors per instance.
    let hermite = HermitePlan::<F>::new(&points, &multiplicities).expect("hermite");
    assert!(hermite.interpolate(&values[..3]).is_err());
}

#[test]
fn validation_rejects_bad_geometries_over_every_instantiated_field() {
    reservation_case::<Gf16>(0x201);
    reservation_case::<Goldilocks>(0x202);
    reservation_case::<Mersenne31>(0x203);
    reservation_case::<QuadMersenne31>(0x204);
    reservation_case::<Gf8B>(0x205);
}

fn bivariate_reservation_case<F: FieldKernels>(seed: u64) {
    let rows: Vec<Vec<F::Elem>> = (0..3)
        .map(|row| top_nonzero::<F>(3 + row, seed + row as u64))
        .collect();
    let left = from_table::<F>(&rows);
    let prefix = poly::<F>(&top_nonzero::<F>(2, seed ^ 0x4));

    let coefficient = poly::<F>(&rows[0]);
    // Row growth, substitution, products, and composition each refuse
    // through their first checked reservation.
    assert!(
        with_threshold(0, move || {
            BivariatePolynomial::<F>::default().set_y_coefficient(2, coefficient)
        })
        .is_err()
    );
    assert!(with_threshold(0, || left.substitute_y_affine_truncated(&prefix, 2, 8)).is_err());
    assert!(with_threshold(0, || left.substitute_y_linear(F::Elem::ONE)).is_err());
    assert!(with_threshold(0, || left.multiply(&left)).is_err());
    assert!(with_threshold(0, || left.compose_y(&prefix)).is_err());
    assert!(with_threshold(0, || left.hasse_derivative(1, 1)).is_err());
}

#[test]
fn bivariate_reservations_refuse_over_instantiated_fields() {
    bivariate_reservation_case::<Gf16>(0x211);
    bivariate_reservation_case::<QuadMersenne31>(0x212);
    bivariate_reservation_case::<Goldilocks>(0x213);
}

// ---------------------------------------------------------------------------
// Debug rendering of the prepared forms, per field instantiation.

fn debug_case<F: PolynomialField>(seed: u64) {
    let points = distinct_points::<F>(3, seed);
    let multiplicities = [2_usize, 1, 3];
    let plan = MultiplicityPlan::<F>::new(&points, &multiplicities, 16).expect("plan");
    let scratch = plan.scratch(1).expect("scratch");
    let rendered = format!("{plan:?} {scratch:?}");
    assert!(!rendered.is_empty());

    let jet = JetPlan::<F>::new(points[0], 3, 16).expect("jet");
    let jet_scratch = jet.scratch(1).expect("jet scratch");
    let rendered = format!("{jet:?} {jet_scratch:?}");
    assert!(!rendered.is_empty());

    let hermite = HermitePlan::<F>::new(&points, &multiplicities).expect("hermite");
    let rendered = format!("{hermite:?}");
    assert!(!rendered.is_empty());

    let moduli: Vec<Polynomial<F>> = [2_usize, 4]
        .iter()
        .enumerate()
        .map(|(index, &len)| poly::<F>(&top_nonzero::<F>(len, seed + index as u64)))
        .collect();
    let tree = RemainderTree::<F>::new(&moduli, 16).expect("tree");
    let tree_scratch = tree.scratch(1).expect("tree scratch");
    let rendered = format!("{tree:?} {tree_scratch:?}");
    assert!(!rendered.is_empty());

    let derivative = DerivativePlan::<F>::new(2, 16).expect("derivative");
    let rendered = format!("{derivative:?}");
    assert!(!rendered.is_empty());
}

#[test]
fn prepared_forms_render_through_debug_over_every_field() {
    debug_case::<Gf8B>(0x221);
    debug_case::<Gf16>(0x222);
    debug_case::<Goldilocks>(0x223);
    debug_case::<Mersenne31>(0x224);
    debug_case::<QuadMersenne31>(0x225);
}

// ---------------------------------------------------------------------------
// Error display arms reachable through the public API.

#[test]
fn api_errors_format_through_every_display_arm() {
    use poly_ring::{
        ConfigError, DomainError, EvalError, HasseError, HermiteError, interpolate_newton,
    };

    // Empty interpolation supports name the zero parameter.
    let newton = interpolate_newton::<Gf8B>(&[], &[]).expect_err("empty newton");
    assert!(matches!(
        newton,
        EvalError::Domain(DomainError::Config(ConfigError::ZeroParameter { .. }))
    ));
    let _ = format!("{newton}");

    // A support beyond the field order names the capacity.
    let too_many = noise::<Gf8B>(300, 0x121);
    let oversized = interpolate_newton::<Gf8B>(&too_many, &too_many).expect_err("oversized");
    assert!(matches!(
        oversized,
        EvalError::Domain(DomainError::Config(
            ConfigError::FieldCapacityExceeded { .. }
        ))
    ));
    let _ = format!("{oversized}");

    // Duplicate points name both indices through the Lagrange path.
    let duplicate_points = distinct_points::<Gf8B>(3, 0x122);
    let mut repeated = duplicate_points.clone();
    repeated.push(duplicate_points[1]);
    let duplicate = interpolate_lagrange::<Gf8B>(&repeated, &repeated).expect_err("duplicates");
    assert!(matches!(
        duplicate,
        EvalError::Domain(DomainError::DuplicatePoint { .. })
    ));
    let _ = format!("{duplicate}");

    // Zero-sized domains carry the zero parameter through the domain error.
    let zero_domain = poly_ring::EvaluationDomain::<Gf16>::additive_subspace(0)
        .map(|_| ())
        .expect_err("zero domain");
    let _ = format!("{zero_domain}");

    // Hermite construction errors: mismatched lengths and duplicates.
    let mismatch = HermitePlan::<Gf8B>::new(&duplicate_points, &[1, 2]).expect_err("mismatch");
    assert!(matches!(mismatch, HermiteError::LengthMismatch { .. }));
    let _ = format!("{mismatch}");
    let conflicting = HermitePlan::<Gf8B>::new(&repeated, &[1, 1, 1, 1]).expect_err("conflict");
    assert!(matches!(conflicting, HermiteError::DuplicatePoint { .. }));
    let _ = format!("{conflicting}");

    // Weighted evaluation rejects mismatched multiplicity slices.
    let weighted =
        MultiplicityPlan::<Gf8B>::new(&duplicate_points, &[1], 8).expect_err("weighted mismatch");
    assert!(matches!(weighted, HasseError::LengthMismatch { .. }));
    let _ = format!("{weighted}");
}
