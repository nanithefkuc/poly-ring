# Benchmarks

Measured through **2026-09-26**. Paired values are **Lunar Lake / Golden
Cove**; units are in the headers.

## Environment

| Host | CPU | OS | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Lunar Lake | Intel Core Ultra 7 258V | Arch Linux, Linux 7.2.6 | 1.98.0 | 3 |
| Golden Cove | Intel Core i7-12700K | CachyOS, Linux 7.2.6-1-cachyos | 1.98.1 | 8, isolated |

| Setting | Value |
| --- | --- |
| Crate | `poly-ring` 1.0.0 working snapshot, same source on both hosts; fingerprint `e70e1f00f3aaec3f41a59b72abfdb4cd42ab9555e870f27c288ef1f5fef96d61` |
| Dependencies | `fgf` 1.2.1, `butterfly-fft` 1.0.2, Criterion 0.8.2 |
| Build | `--all-features`, thin LTO, one codegen unit, no custom `RUSTFLAGS` |
| Execution | `RAYON_NUM_THREADS=1`; process affinity verified |
| `fgf` backends | `v3_gfni_crypto` for binary fields; `v3` for prime fields on both hosts |
| `butterfly-fft` backends | `v3_gfni_crypto` for `Gf8B`/`Gf16`; `scalar` for `Gf64` |
| Sampling | 10 Criterion samples per case; 0.05 s warmup, 0.1 s requested measurement window |
| Aggregation | Median of five per-run Criterion medians; Goldilocks NTT boundary rows use three |
| Correctness | `just test` passed on both hosts before timing |

## Dense arithmetic

Operands are coefficient counts. Inputs are prepared outside timing; result
allocation and destruction are included.

| Field | Operation | Geometry | Time (µs) |
| --- | --- | --- | ---: |
| `Gf8B` | `multiply_truncated` | 16 × 16 | 0.245/0.192 |
| `Gf8B` | `multiply_truncated` | 64 × 64 | 0.437/0.314 |
| `Gf8B` | `multiply_truncated` | 256 × 256 | 2.27/1.81 |
| `Gf8B` | `multiply_truncated` | 1,024 × 1,024 | 17.3/16.4 |
| `Gf8B` | `multiply_truncated` | 2,048 × 2,048 | 62.0/62.6 |
| `Gf16` | `div_rem` | 256 × 64 | 5.39/4.04 |
| `Gf16` | `gcd` | 200 × 150 | 15.2/13.4 |
| `Gf16` | `extended_gcd` | 200 × 150 | 43.5/39.2 |
| `Gf16` | `inverse_mod_x_power` | precision 512 | 21.7/18.7 |

## Batched products

`multiply_rows_into` uses coefficient-major rows of `n` coefficients per
operand and one polynomial per lane. Product timings reuse prepared
`ConvolutionScratch` and output; construction includes transform preparation.

| Field | n | Construction, 1 lane (µs) | Product, 1 lane (µs) | Product, 16 lanes (µs) |
| --- | ---: | ---: | ---: | ---: |
| `Gf8B` | 64 | 3.51/3.45 | 0.517/0.417 | 8.17/6.59 |
| `Gf8B` | 256 | 6.45/6.27 | 2.54/2.45 | 42.4/39.0 |
| `Gf8B` | 1,024 | 9.22/8.79 | 17.2/18.3 | 281/285 |
| `Gf8B` | 4,096 | 21.7/19.2 | 85.4/69.4 | 1,371/1,133 |
| `Gf8B` | 65,536 | 332/281 | 1,461/1,196 | 24,037/19,256 |
| `Gf16` | 64 | 4.71/4.43 | 0.932/0.814 | 14.6/13.0 |
| `Gf16` | 256 | 14.5/14.0 | 4.72/4.18 | 75.7/72.9 |
| `Gf16` | 1,024 | 53.7/52.7 | 40.7/41.9 | 660/626 |
| `Gf16` | 4,096 | 214/211 | 859/743 | 7,018/6,082 |
| `Gf16` | 65,536 | 2,083/2,001 | 39,319/34,435 | 631,929/546,374 |
| `Mersenne31` | 64 | 12.0/11.3 | 1.02/1.01 | 16.1/16.4 |
| `Mersenne31` | 256 | 32.3/29.4 | 11.5/12.8 | 186/207 |
| `Mersenne31` | 1,024 | 109/96.7 | 153/157 | 939/886 |
| `Mersenne31` | 4,096 | 420/364 | 664/699 | 4,491/4,211 |
| `Mersenne31` | 65,536 | 7,006/6,030 | 13,415/13,999 | 101,398/97,609 |
| `Goldilocks` | 64 | 9.09/8.37 | 3.48/3.95 | 39.9/39.4 |
| `Goldilocks` | 256 | 20.4/19.5 | 34.5/33.9 | 202/200 |
| `Goldilocks` | 1,024 | 67.0/63.6 | 151/149 | 988/971 |
| `Goldilocks` | 4,096 | 266/258 | 669/665 | 4,706/4,519 |
| `Goldilocks` | 65,536 | 5,429/5,135 | 13,313/13,045 | 110,960/106,508 |
| `QuadMersenne31` | 64 | 12.2/11.5 | 2.95/3.27 | 39.3/36.6 |
| `QuadMersenne31` | 256 | 34.2/30.7 | 35.5/36.0 | 191/181 |
| `QuadMersenne31` | 1,024 | 117/104 | 154/157 | 942/859 |
| `QuadMersenne31` | 4,096 | 464/404 | 681/703 | 4,510/4,043 |
| `QuadMersenne31` | 65,536 | 8,298/7,108 | 14,109/13,968 | 100,733/94,452 |

## Weighted evaluation

`MultiplicityPlan::evaluate_into` reuses plan, scratch, and output. For
uniform multiplicity, `P` is the number of distinct points, `s` the
multiplicity, and the input has `D = P·s` coefficients. Plan construction
uses capacity 131,072; scratch is constructed separately.

| Field | P | s | Plan (ms) | Evaluation (µs) |
| --- | ---: | ---: | ---: | ---: |
| `Gf8B` | 64 | 1 | 86.2/86.0 | 1.05/0.794 |
| `Gf8B` | 64 | 4 | 21.2/20.8 | 17.6/17.7 |
| `Gf8B` | 256 | 4 | 0.532/0.506 | 84.0/85.3 |
| `Gf8B` | 256 | 16 | 0.858/0.838 | 510/495 |
| `Gf16` | 64 | 1 | 254/255 | 1.42/1.12 |
| `Gf16` | 64 | 4 | 60.2/61.7 | 27.4/24.9 |
| `Gf16` | 256 | 4 | 62.2/61.2 | 140/128 |
| `Gf16` | 256 | 16 | 17.7/17.0 | 1,065/961 |
| `Mersenne31` | 64 | 1 | 919/1,021 | 1.14/1.02 |
| `Mersenne31` | 64 | 4 | 928/1,029 | 69.4/66.9 |
| `Mersenne31` | 256 | 4 | 954/1,064 | 668/702 |
| `Mersenne31` | 256 | 16 | 1,084/1,200 | 6,893/7,099 |
| `Goldilocks` | 64 | 1 | 4,323/5,058 | 3.59/4.03 |
| `Goldilocks` | 64 | 4 | 4,395/5,105 | 178/198 |
| `Goldilocks` | 256 | 4 | 4,616/5,298 | 1,582/1,658 |
| `Goldilocks` | 256 | 16 | 5,177/5,964 | 10,307/10,528 |
| `QuadMersenne31` | 64 | 1 | 3,554/4,128 | 4.96/5.72 |
| `QuadMersenne31` | 64 | 4 | 3,578/4,187 | 156/173 |
| `QuadMersenne31` | 256 | 4 | 3,672/4,338 | 1,517/1,616 |
| `QuadMersenne31` | 256 | 16 | 4,133/4,870 | 10,294/10,691 |

Nonuniform weights use 64 distinct points and `D =` total weight;
prepared execution only.

| Field | Repeating `[1, 2, 4, 8]`, W=240 (µs) | 63 unit weights + one weight 64, W=127 (µs) |
| --- | ---: | ---: |
| `Gf8B` | 33.2/33.8 | 15.4/15.3 |
| `Gf16` | 60.7/53.5 | 28.6/25.2 |
| `Mersenne31` | 72.3/73.3 | 33.0/32.0 |
| `Goldilocks` | 165/182 | 51.0/54.2 |
| `QuadMersenne31` | 150/169 | 50.7/54.3 |

## NTT boundaries

Public automatic products around the shorter-operand NTT thresholds. Prepared
execution reuses scratch and output; allocating multiplication includes
one-shot preparation. Goldilocks boundary rows use three rounds; the other
fields use five.

| Field | Public path | Operands | Lanes | NTT threshold | Time (µs) |
| --- | --- | --- | ---: | ---: | ---: |
| `Goldilocks` | `multiply_rows_into` | 192 × 192 | 1 | 256 | 29.5/33.9 |
| `Goldilocks` | `multiply_rows_into` | 256 × 256 | 1 | 256 | 33.8/33.9 |
| `Goldilocks` | `multiply_rows_into` | 96 × 96 | 4 | 128 | 29.6/34.7 |
| `Goldilocks` | `multiply_rows_into` | 128 × 128 | 4 | 128 | 34.5/34.1 |
| `Goldilocks` | `multiply_rows_into` | 64 × 64 | 16 | 64 | 38.9/39.7 |
| `Goldilocks` | `Polynomial::multiply` | 256 × 256 | 1 | 512 | 52.3/59.9 |
| `Goldilocks` | `Polynomial::multiply` | 384 × 384 | 1 | 512 | 116/134 |
| `Goldilocks` | `Polynomial::multiply` | 448 × 448 | 1 | 512 | 156/182 |
| `Goldilocks` | `Polynomial::multiply` | 512 × 512 | 1 | 512 | 72.7/69.4 |
| `QuadMersenne31` | `multiply_rows_into` | 256 × 256 | 1 | 256 | 35.2/36.0 |
| `QuadMersenne31` | `multiply_rows_into` | 128 × 128 | 4 | 128 | 39.4/38.9 |
| `QuadMersenne31` | `multiply_rows_into` | 64 × 64 | 16 | 64 | 38.1/36.7 |
| `QuadMersenne31` | `Polynomial::multiply` | 1,024 × 1,024 | 1 | 1,024 | 167/170 |
| `Mersenne31` | `multiply_rows_into` | 1,024 × 1,024 | 1 | 1,024 | 156/158 |
| `Mersenne31` | `multiply_rows_into` | 1,024 × 1,024 | 4 | 1,024 | 390/395 |
| `Mersenne31` | `multiply_rows_into` | 512 × 512 | 16 | 512 | 435/408 |
| `Mersenne31` | `Polynomial::multiply` | 4,096 × 4,096 | 1 | 4,096 | 851/857 |

## Competitors

All competitors are dev-only: `ark-ff`/`ark-poly` 0.6.0,
`lambdaworks-math` 0.13.0, and `winter-math` 0.13.1. Each arm checks
its output before timing.

### Ring operations

Matched **element width, not field**: `poly-ring` uses `Gf64`; the other
libraries use Goldilocks. `-` means no matching API.

| Geometry | Operation | poly-ring (µs) | ark-poly (µs) | lambdaworks (µs) |
| --- | --- | ---: | ---: | ---: |
| 64 × 64 | multiply, automatic | 4.14/4.12 | 4.36/3.73 | 7.67/6.98 |
| 256 × 256 | multiply, automatic | 35.9/37.6 | 20.8/16.4 | 118/110 |
| 1,024 × 1,024 | multiply, automatic | 434/497 | 259/224 | 1,854/1,753 |
| 4,096 × 4,096 | multiply, automatic | 3,608/3,495 | 1,340/1,161 | 29,681/27,860 |
| 512 × 128 | `div_rem` | 56.4/56.2 | 164/150 | 915/884 |
| 400 × 300 | `extended_gcd` | 573/576 | - | 2,914/2,750 |
| degree 4096, 1 point | Horner evaluation | 196/193 | 14.7/11.6 | 19.9/17.5 |
| 64 points | Lagrange interpolation | 107/107 | - | 1,034/999 |

### Multiplication over Goldilocks

Same prime and coefficients across libraries, with exact output agreement.

| Operands | poly-ring (µs) | ark-poly (µs) | lambdaworks (µs) |
| --- | ---: | ---: | ---: |
| 64 × 64 | 3.41/3.89 | 4.35/3.71 | 7.66/6.99 |
| 256 × 256 | 52.0/58.7 | 21.5/16.5 | 118/110 |
| 1,024 × 1,024 | 148/144 | 259/226 | 1,856/1,751 |
| 4,096 × 4,096 | 644/629 | 1,345/1,179 | 30,296/27,866 |

### Goldilocks Hasse evaluation

Multiplicity one, with a reused `MultiplicityPlan`, scratch, and output.

| Points | Degree | poly-ring (µs) | ark-poly (µs) | lambdaworks (µs) | winter-math (µs) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 16 | 0.372/0.342 | 0.793/0.618 | 1.09/0.954 | 0.610/0.532 |
| 16 | 64 | 1.35/1.24 | 3.45/2.80 | 4.88/4.24 | 3.12/2.76 |
| 16 | 256 | 5.21/4.83 | 14.7/11.6 | 19.5/17.5 | 12.7/11.8 |
| 16 | 1,024 | 20.7/19.2 | 89.8/71.3 | 80.8/70.0 | 52.2/47.5 |
| 64 | 16 | 0.996/1.12 | 3.17/2.46 | 4.51/3.80 | 2.50/2.10 |
| 64 | 64 | 3.71/4.23 | 13.9/11.2 | 19.5/16.9 | 12.3/11.0 |
| 64 | 256 | 14.6/16.3 | 91.2/73.9 | 79.1/69.8 | 51.5/47.1 |
| 64 | 1,024 | 57.2/63.7 | 442/387 | 318/280 | 208/190 |
| 256 | 16 | 3.94/4.43 | 12.8/9.87 | 18.2/15.2 | 9.83/8.36 |
| 256 | 64 | 14.8/16.5 | 93.6/71.6 | 78.4/67.8 | 49.6/44.0 |
| 256 | 256 | 57.7/63.8 | 451/393 | 315/279 | 206/187 |
| 256 | 1,024 | 232/254 | 1,828/1,616 | 1,252/1,120 | 832/758 |
| 1,024 | 16 | 15.0/17.5 | 93.5/75.0 | 73.6/60.8 | 39.8/33.4 |
| 1,024 | 64 | 56.6/64.5 | 437/388 | 310/271 | 202/176 |
| 1,024 | 256 | 223/253 | 1,826/1,628 | 1,268/1,116 | 838/751 |
| 1,024 | 1,024 | 901/1,012 | 7,235/6,498 | 5,063/4,479 | 3,291/3,059 |

There is no direct weighted Hasse-jet comparator among these libraries.
The following 64-point, degree-64 baselines compose per-order derivatives;
one-shot includes preparation, while prepared execution reuses it. All jets
were checked for exact equality.

| Multiplicity | Form | poly-ring (µs) | ark-poly (µs) | lambdaworks (µs) | winter-math (µs) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | one-shot | 339/325 | 62.0/51.9 | 66.8/62.7 | 51.5/50.1 |
| 4 | one-shot | 543/528 | 250/227 | 233/229 | 206/204 |
| 8 | one-shot | 986/1,008 | 887/863 | 810/829 | 766/781 |
| 16 | one-shot | 2,299/2,440 | 2,909/2,936 | 2,806/2,832 | 2,603/2,752 |
| 2 | prepared | 37.0/36.5 | 35.7/23.3 | 40.6/34.9 | 25.1/22.5 |
| 4 | prepared | 66.1/66.1 | 84.0/65.7 | 78.7/68.7 | 49.5/44.2 |
| 8 | prepared | 128/128 | 201/167 | 152/133 | 94.8/85.7 |
| 16 | prepared | 285/285 | 408/343 | 288/249 | 189/160 |

## Reproduction

From the crate root, on each host:

```sh
export FEC_GOLDEN_CORE=<cpu> RAYON_NUM_THREADS=1
for run in 1 2 3 4 5; do
    printf '%s\n' poly competitors prepared hasse | shuf | while read -r target; do
        just _bench-run "$target" --noplot --sample-size 10 --warm-up-time 0.05 \
            --measurement-time 0.1 --save-baseline "public-r${run}"
    done
done
```

For the Goldilocks NTT boundary rows, run three rounds on each host:

```sh
for run in 1 2 3; do
    just _bench-run prepared --noplot --sample-size 10 --warm-up-time 0.05 \
        --measurement-time 0.1 --save-baseline "boundary-r${run}" \
        prepared_tuning/goldilocks
    just _bench-run prepared --noplot --sample-size 10 --warm-up-time 0.05 \
        --measurement-time 0.1 --save-baseline "boundary-r${run}" \
        oneshot_tuning/goldilocks
done
```

Values use Criterion's `median.point_estimate` from
`target/criterion/**/<baseline>/estimates.json`, aggregated as specified
above. Criterion extends slow measurement windows. Small cases are timer-
and scheduling-sensitive; construction allocates fresh scratch. Targets
also contain internal tuning cases, whose timings are not published here.
