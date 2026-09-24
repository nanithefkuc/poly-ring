# Benchmarks

Public API timings. Unless a subsection says otherwise, measurements are from
2026-09-24. Every paired cell is **Lunar Lake / Golden Cove**, in microseconds
unless the table header says otherwise. Each value is one pinned run's
Criterion median, rounded to three significant figures. `-` means the case was
not measured; the weighted-evaluation section and the Goldilocks Hasse panel
carry 2026-09-24 Lunar Lake values whose Golden Cove counterparts are not yet
measured.

## Environment

| Host | CPU | Operating system | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Lunar Lake | Intel Core Ultra 7 258V | Arch Linux, Linux 7.2.6 | 1.98.0 | 3 |
| Golden Cove | Intel Core i7-12700K | CachyOS, Linux 7.2.3 | 1.98.1 | 8 |

| Setting | Value |
| --- | --- |
| Crate | `poly-ring` 0.1.0 working snapshot, SHA-256 `f5f9ca6156c438d9c49ad8abd9197a7897c87e05aa88ced2ca3bef3160ff3c8e` over `src/`, `benches/`, `tests/`, `Cargo.toml`, `Cargo.lock` |
| Dependencies | `fgf` 1.1.0, `butterfly-fft` 1.0.0, `simdispatch` 0.1.0, Criterion 0.8.2, all from the registry |
| Build | `--all-features`, `profile.bench` with thin LTO and one codegen unit, no custom `RUSTFLAGS` |
| Threads | `RAYON_NUM_THREADS=1`; benchmark process affinity verified through `/proc` on both hosts |
| Sampling | 10 samples, 0.05 s warmup, 0.1 s requested measurement window; Criterion extends slow cases |
| Aggregation | `median.point_estimate` from `target/criterion/**/public-20260921/estimates.json` |
| Correctness | `just test` passed on both hosts immediately before timing |

Both hosts ran the same source snapshot outside the umbrella patch
configuration, so the dependency graph is registry-only. Resolved upstream
backends are identical on both hosts: `v3_gfni_crypto` for the `fgf` process
selection and the `Gf8B`, `Gf16`, and `Gf64` field kernels; `v3` for
`Mersenne31`, `Goldilocks`, and `QuadMersenne31`; `v3_gfni_crypto` for the
`butterfly-fft` `Gf8B` and `Gf16` butterflies and `scalar` for `Gf64`. These
are the field and butterfly selections; the polynomial routes choose their
own algorithms per call and were not instrumented.

The 2026-09-21 paired cells are independent per-host medians from single runs,
not before/after measurements. That campaign changed no crossover. With ten
samples per case, small nanosecond-scale results are sensitive to timer
resolution and scheduling; the larger product and weighted-evaluation cases
are the headline measurements.

## Dense polynomial arithmetic

Inputs are prepared outside the timed region; the allocating results and
their destruction are inside it. The `multiply` harness label is the public
truncated product, not a forced internal route.

| Field | Operation | Geometry | Time (µs) |
| --- | --- | --- | --- |
| Gf8B | `multiply_truncated` | 16 × 16 | 0.234/0.191 |
| Gf8B | `multiply_truncated` | 64 × 64 | 0.422/0.312 |
| Gf8B | `multiply_truncated` | 256 × 256 | 2.22/1.82 |
| Gf8B | `multiply_truncated` | 1024 × 1024 | 22.7/16.8 |
| Gf8B | `multiply_truncated` | 2048 × 2048 | 57.1/63.5 |
| Gf16 | `div_rem` | 256 × 64 | 5.24/4.02 |
| Gf16 | `gcd` | 200 × 150 | 14.8/13.3 |
| Gf16 | `extended_gcd` | 200 × 150 | 41.9/41.7 |
| Gf16 | `inverse_mod_x_power` | 512, precision 512 | 21.2/19.4 |


## Batched products

Equal-length operands over coefficient-major rows, one independent
polynomial per lane, through `ConvolutionScratch` and `multiply_rows_into`.
Construction includes `ConvolutionScratch::new` and `prepare_transform`;
execution reuses that scratch and a preallocated output. The full product
holds `2n − 1` coefficients per lane. `n` is the per-operand coefficient
count and the lane count is the batch width.

### Gf8B

| n | Lanes | Construction (µs) | Product (µs) |
| --- | --- | --- | --- |
| 64 | 1 | 3.55/3.58 | 0.509/0.43 |
| 64 | 4 | 3.63/3.67 | 2.03/1.79 |
| 64 | 16 | 3.86/3.82 | 8.06/6.81 |
| 256 | 1 | 6.26/6.36 | 2.54/3.17 |
| 256 | 4 | 6.3/6.3 | 10.1/9.76 |
| 256 | 16 | 6.59/6.8 | 40.1/41.7 |
| 1024 | 1 | 8.89/9.01 | 16.4/18.2 |
| 1024 | 4 | 8.96/9.23 | 64.7/72.6 |
| 1024 | 16 | 9.3/9.54 | 276/287 |
| 4096 | 1 | 21/20.5 | 86.8/71.1 |
| 4096 | 4 | 20.7/20.6 | 340/293 |
| 4096 | 16 | 21.1/21 | 1,340/1,120 |
| 16384 | 1 | 74.6/72.5 | 351/278 |
| 16384 | 4 | 74.8/72 | 1,390/1,100 |
| 16384 | 16 | 75.6/72 | 5,630/4,560 |
| 65536 | 1 | 325/301 | 1,420/1,220 |
| 65536 | 4 | 323/301 | 5,820/4,850 |
| 65536 | 16 | 320/300 | 23,100/18,500 |

### Gf16

| n | Lanes | Construction (µs) | Product (µs) |
| --- | --- | --- | --- |
| 64 | 1 | 4.6/4.4 | 0.909/0.815 |
| 64 | 4 | 4.69/4.81 | 3.64/3.3 |
| 64 | 16 | 5.15/5.02 | 14.6/13.1 |
| 256 | 1 | 14.3/14 | 4.7/4.21 |
| 256 | 4 | 14.7/14.3 | 18.8/16.8 |
| 256 | 16 | 16.1/15.5 | 75.1/67.1 |
| 1024 | 1 | 53.4/52.2 | 41.2/42.1 |
| 1024 | 4 | 54.4/53.5 | 161/157 |
| 1024 | 16 | 60.2/59.8 | 649/609 |
| 4096 | 1 | 212/211 | 852/743 |
| 4096 | 4 | 218/215 | 3,390/2,970 |
| 4096 | 16 | 234/248 | 6,960/6,080 |
| 16384 | 1 | 880/857 | 7,030/6,190 |
| 16384 | 4 | 897/886 | 28,000/24,300 |
| 16384 | 16 | 1,740/1,010 | 38,200/33,400 |
| 65536 | 1 | 2,050/1,980 | 38,700/34,200 |
| 65536 | 4 | 2,140/2,050 | 159,000/136,000 |
| 65536 | 16 | 5,650/3,430 | 620,000/545,000 |

### Mersenne31

| n | Lanes | Construction (µs) | Product (µs) |
| --- | --- | --- | --- |
| 64 | 1 | 11.8/11.3 | 0.95/0.969 |
| 64 | 4 | 11.9/11.4 | 3.98/3.84 |
| 64 | 16 | 13.4/12.7 | 15.6/15.4 |
| 256 | 1 | 31.5/29.6 | 11.2/12.4 |
| 256 | 4 | 32.8/30.9 | 44.5/49.6 |
| 256 | 16 | 39/36.4 | 181/206 |
| 1024 | 1 | 106/97.2 | 173/191 |
| 1024 | 4 | 112/103 | 691/771 |
| 1024 | 16 | 134/125 | 2,680/3,070 |
| 4096 | 1 | 408/366 | 770/725 |
| 4096 | 4 | 436/396 | 3,060/2,900 |
| 4096 | 16 | 559/523 | 13,500/11,700 |
| 16384 | 1 | 1,680/1,480 | 7,500/6,590 |
| 16384 | 4 | 1,810/1,570 | 28,900/26,400 |
| 16384 | 16 | 2,330/1,950 | 116,000/107,000 |
| 65536 | 1 | 6,910/6,070 | 67,000/59,400 |
| 65536 | 4 | 12,000/9,350 | 267,000/261,000 |
| 65536 | 16 | 11,100/11,100 | 1,160,000/958,000 |

### Goldilocks

| n | Lanes | Construction (µs) | Product (µs) |
| --- | --- | --- | --- |
| 64 | 1 | 8.84/8.39 | 3.35/3.83 |
| 64 | 4 | 8.98/8.53 | 13.4/15.7 |
| 64 | 16 | 10.3/9.93 | 54.4/61.2 |
| 256 | 1 | 20.5/19.7 | 51.4/59.7 |
| 256 | 4 | 21.5/21 | 204/234 |
| 256 | 16 | 27/26.3 | 826/941 |
| 1024 | 1 | 65.7/64.2 | 805/928 |
| 1024 | 4 | 71.1/70.2 | 3,240/3,800 |
| 1024 | 16 | 92.8/95.9 | 12,900/15,200 |
| 4096 | 1 | 262/259 | 2,190/2,510 |
| 4096 | 4 | 284/293 | 8,780/10,100 |
| 4096 | 16 | 389/387 | 35,500/40,400 |
| 16384 | 1 | 1,180/1,150 | 19,900/22,600 |
| 16384 | 4 | 1,330/1,270 | 80,200/91,700 |
| 16384 | 16 | 1,840/1,640 | 325,000/368,000 |
| 65536 | 1 | 5,300/5,190 | 185,000/207,000 |
| 65536 | 4 | 6,190/5,760 | 733,000/827,000 |
| 65536 | 16 | 14,700/13,700 | 2,980,000/3,320,000 |

### QuadMersenne31

| n | Lanes | Construction (µs) | Product (µs) |
| --- | --- | --- | --- |
| 64 | 1 | 12/11.5 | 2.86/3.3 |
| 64 | 4 | 12.2/11.6 | 11.4/13 |
| 64 | 16 | 13.6/12.8 | 45.5/51.8 |
| 256 | 1 | 33.2/30.7 | 42.5/48.5 |
| 256 | 4 | 34.3/32.2 | 169/194 |
| 256 | 16 | 39.6/37.7 | 685/801 |
| 1024 | 1 | 116/104 | 654/759 |
| 1024 | 4 | 122/110 | 2,630/3,070 |
| 1024 | 16 | 146/136 | 10,600/12,300 |
| 4096 | 1 | 472/406 | 1,890/2,080 |
| 4096 | 4 | 475/438 | 7,440/8,580 |
| 4096 | 16 | 588/532 | 30,000/34,500 |
| 16384 | 1 | 1,940/1,690 | 16,800/19,000 |
| 16384 | 4 | 2,060/1,820 | 67,400/75,500 |
| 16384 | 16 | 2,680/2,200 | 272,000/303,000 |
| 65536 | 1 | 8,300/7,110 | 154,000/170,000 |
| 65536 | 4 | 9,320/7,800 | 615,000/680,000 |
| 65536 | 16 | 17,900/15,800 | 2,490,000/2,740,000 |

Construction prepares the plans for the automatic route; a field without a
domain covering the requested size omits the transform plan instead of
faking one. Execution is the automatic public route: forced Karatsuba and
transform variants exist only under `internals` and are excluded here.

## Weighted evaluation

`MultiplicityPlan::uniform` construction, `plan.scratch(1)` construction,
and `plan.evaluate_into` execution are separate measurements. Each plan
declares capacity for 131072 input coefficients. Execution reuses the plan,
scratch, and output. `P` is the point count, `s` the uniform multiplicity,
and `W = P·s`. The input coefficient count is a multiple of `W`, shown as
its factor: `D=W/2`, `D=W`, `D=3W/2`.

### Gf8B

| P | s | Plan (µs) | Scratch (µs) | D=W/2 (µs) | D=W (µs) | D=3W/2 (µs) |
| 16 | 1 | 100,972/- | 706/- | 0.0605/- | 0.114/- | 0.161/- |
| 16 | 2 | 49,403/- | 723/- | 1.93/- | 2.26/- | 2.47/- |
| 16 | 4 | 25,432/- | 710/- | 3.11/- | 3.53/- | 3.80/- |
| 16 | 8 | 12,408/- | 712/- | 6.70/- | 7.21/- | 7.62/- |
| 16 | 16 | 5,780/- | 721/- | 19.9/- | 20.3/- | 20.7/- |
| 16 | 64 | 1,750/- | 718/- | 79.7/- | 82.0/- | 83.0/- |
| 64 | 1 | 80,439/- | 713/- | 0.247/- | 0.476/- | 0.703/- |
| 64 | 2 | 41,240/- | 732/- | 10.4/- | 11.4/- | 12.3/- |
| 64 | 4 | 20,201/- | 730/- | 15.7/- | 17.1/- | 18.0/- |
| 64 | 8 | 10,100/- | 712/- | 30.9/- | 33.4/- | 33.9/- |
| 64 | 16 | 5,013/- | 720/- | 83.3/- | 86.8/- | 88.7/- |
| 64 | 64 | 1,889/- | 714/- | 340/- | 444/- | 661/- |
| 256 | 1 | 586/- | 719/- | 1.40/- | 2.77/- | 4.16/- |
| 256 | 2 | 515/- | 732/- | 50.8/- | 56.0/- | 56.7/- |
| 256 | 4 | 511/- | 740/- | 74.1/- | 81.3/- | 82.7/- |
| 256 | 8 | 621/- | 734/- | 140/- | 152/- | 158/- |
| 256 | 16 | 841/- | 738/- | 362/- | 480/- | 590/- |
| 256 | 64 | 1,603/- | 1,475/- | 2,713/- | 4,333/- | 4,985/- |
| 1024 | 1 | - | - | - | - | - |
| 1024 | 2 | - | - | - | - | - |
| 1024 | 4 | - | - | - | - | - |
| 1024 | 8 | - | - | - | - | - |
| 1024 | 16 | - | - | - | - | - |
| 1024 | 64 | - | - | - | - | - |

### Gf16

| P | s | Plan (µs) | Scratch (µs) | D=W/2 (µs) | D=W (µs) | D=3W/2 (µs) |
| 16 | 1 | 236,104/- | 5,016/- | 0.0953/- | 0.182/- | 0.263/- |
| 16 | 2 | 115,117/- | 5,055/- | 2.91/- | 3.46/- | 3.85/- |
| 16 | 4 | 55,240/- | 5,058/- | 4.71/- | 5.41/- | 5.87/- |
| 16 | 8 | 25,723/- | 5,008/- | 10.7/- | 11.5/- | 12.1/- |
| 16 | 16 | 13,961/- | 5,017/- | 36.7/- | 37.8/- | 38.6/- |
| 16 | 64 | 3,140/- | 4,986/- | 156/- | 159/- | 161/- |
| 64 | 1 | 236,787/- | 5,228/- | 0.429/- | 0.833/- | 1.25/- |
| 64 | 2 | 115,809/- | 5,242/- | 16.1/- | 18.3/- | 20.2/- |
| 64 | 4 | 56,146/- | 5,194/- | 24.0/- | 26.7/- | 28.7/- |
| 64 | 8 | 26,463/- | 5,225/- | 49.5/- | 53.1/- | 55.7/- |
| 64 | 16 | 11,770/- | 5,222/- | 157/- | 162/- | 167/- |
| 64 | 64 | 3,890/- | 5,247/- | 656/- | 917/- | 1,261/- |
| 256 | 1 | 239,265/- | 5,432/- | 3.21/- | 6.32/- | 9.66/- |
| 256 | 2 | 120,553/- | 5,440/- | 83.8/- | 95.6/- | 105/- |
| 256 | 4 | 59,315/- | 5,379/- | 120/- | 138/- | 153/- |
| 256 | 8 | 28,794/- | 5,475/- | 232/- | 261/- | 285/- |
| 256 | 16 | 14,030/- | 5,404/- | 695/- | 1,043/- | 1,478/- |
| 256 | 64 | 7,740/- | 5,432/- | 5,261/- | 7,709/- | 10,607/- |
| 1024 | 1 | 262,450/- | 5,839/- | 37.3/- | 74.7/- | 111/- |
| 1024 | 2 | 128,442/- | 5,696/- | 450/- | 563/- | 654/- |
| 1024 | 4 | 70,946/- | 5,703/- | 667/- | 1,482/- | 2,406/- |
| 1024 | 8 | 38,343/- | 5,697/- | 2,091/- | 3,637/- | 5,496/- |
| 1024 | 16 | 25,395/- | 5,848/- | 6,483/- | 9,786/- | 13,590/- |
| 1024 | 64 | 13,083/- | 5,839/- | 45,302/- | 60,924/- | 94,255/- |

### Mersenne31

| P | s | Plan (µs) | Scratch (µs) | D=W/2 (µs) | D=W (µs) | D=3W/2 (µs) |
| 16 | 1 | 897,724/- | 24,859/- | 0.100/- | 0.183/- | 0.271/- |
| 16 | 2 | 895,335/- | 21,125/- | 3.88/- | 4.60/- | 5.17/- |
| 16 | 4 | 894,113/- | 24,730/- | 8.06/- | 9.75/- | 11.3/- |
| 16 | 8 | 898,067/- | 24,984/- | 21.1/- | 26.0/- | 30.1/- |
| 16 | 16 | 904,125/- | 25,022/- | 67.7/- | 84.8/- | 99.4/- |
| 16 | 64 | 937,463/- | 25,197/- | 434/- | 684/- | 900/- |
| 64 | 1 | 896,611/- | 25,544/- | 0.664/- | 1.30/- | 1.94/- |
| 64 | 2 | 899,423/- | 25,652/- | 21.7/- | 26.8/- | 31.0/- |
| 64 | 4 | 904,540/- | 25,512/- | 49.2/- | 66.4/- | 81.3/- |
| 64 | 8 | 915,090/- | 25,102/- | 142/- | 204/- | 260/- |
| 64 | 16 | 935,390/- | 25,520/- | 470/- | 721/- | 936/- |
| 64 | 64 | 1,054,614/- | 25,694/- | 4,851/- | 6,494/- | 8,271/- |
| 256 | 1 | 902,754/- | 26,040/- | 7.02/- | 13.9/- | 21.0/- |
| 256 | 2 | 914,604/- | 26,009/- | 144/- | 208/- | 262/- |
| 256 | 4 | 936,020/- | 25,808/- | 400/- | 648/- | 876/- |
| 256 | 8 | 976,695/- | 25,747/- | 1,326/- | 2,335/- | 3,159/- |
| 256 | 16 | 1,056,937/- | 26,065/- | 4,925/- | 6,634/- | 8,419/- |
| 256 | 64 | 1,467,952/- | 26,516/- | 36,577/- | 51,538/- | 66,208/- |
| 1024 | 1 | 939,852/- | 26,490/- | 107/- | 213/- | 322/- |
| 1024 | 2 | 982,763/- | 26,823/- | 1,334/- | 2,344/- | 3,194/- |
| 1024 | 4 | 1,066,245/- | 26,399/- | 4,628/- | 6,298/- | 8,084/- |
| 1024 | 8 | 1,215,128/- | 24,540/- | 12,932/- | 18,405/- | 23,335/- |
| 1024 | 16 | 1,476,919/- | 26,864/- | 37,113/- | 52,377/- | 67,159/- |
| 1024 | 64 | 1,732,045/- | 26,724/- | 299,788/- | 437,923/- | 567,114/- |

### Goldilocks

| P | s | Plan (µs) | Scratch (µs) | D=W/2 (µs) | D=W (µs) | D=3W/2 (µs) |
| 16 | 1 | 4,283,036/- | 25,748/- | 0.179/- | 0.339/- | 0.498/- |
| 16 | 2 | 4,244,689/- | 25,879/- | 4.31/- | 5.81/- | 7.04/- |
| 16 | 4 | 4,433,848/- | 26,500/- | 10.2/- | 15.6/- | 20.0/- |
| 16 | 8 | 4,382,344/- | 26,374/- | 30.2/- | 49.6/- | 66.8/- |
| 16 | 16 | 4,567,668/- | 26,721/- | 107/- | 184/- | 249/- |
| 16 | 64 | 4,596,789/- | 26,381/- | 1,068/- | 1,516/- | 1,825/- |
| 64 | 1 | 4,348,052/- | 25,264/- | 1.88/- | 3.64/- | 5.51/- |
| 64 | 2 | 4,293,403/- | 25,161/- | 34.4/- | 53.9/- | 70.3/- |
| 64 | 4 | 4,303,126/- | 25,279/- | 99.1/- | 174/- | 237/- |
| 64 | 8 | 4,398,679/- | 25,462/- | 353/- | 565/- | 722/- |
| 64 | 16 | 4,502,567/- | 25,242/- | 1,166/- | 1,616/- | 1,922/- |
| 64 | 64 | 5,027,158/- | 28,024/- | 7,765/- | 9,694/- | 11,000/- |
| 256 | 1 | 4,317,684/- | 30,966/- | 29.4/- | 60.9/- | 82.8/- |
| 256 | 2 | 4,337,049/- | 26,601/- | 365/- | 574/- | 720/- |
| 256 | 4 | 4,483,713/- | 26,250/- | 1,126/- | 1,571/- | 1,885/- |
| 256 | 8 | 4,710,662/- | 30,194/- | 3,234/- | 4,193/- | 4,860/- |
| 256 | 16 | 5,312,955/- | 30,012/- | 8,522/- | 10,450/- | 11,868/- |
| 256 | 64 | 7,174,367/- | 30,106/- | 49,428/- | 58,379/- | 64,555/- |
| 1024 | 1 | 4,654,601/- | 30,756/- | 443/- | 888/- | 1,326/- |
| 1024 | 2 | 4,948,887/- | 30,052/- | 3,296/- | 4,230/- | 4,877/- |
| 1024 | 4 | 5,227,888/- | 30,164/- | 8,379/- | 10,395/- | 11,807/- |
| 1024 | 8 | 5,932,262/- | 30,398/- | 21,648/- | 25,912/- | 29,157/- |
| 1024 | 16 | 7,408,318/- | 31,716/- | 51,069/- | 60,044/- | 66,161/- |
| 1024 | 64 | 8,478,714/- | 30,807/- | 272,420/- | 314,953/- | 341,535/- |

### QuadMersenne31

| P | s | Plan (µs) | Scratch (µs) | D=W/2 (µs) | D=W (µs) | D=3W/2 (µs) |
| 16 | 1 | 3,646,447/- | 31,718/- | 0.220/- | 0.422/- | 0.627/- |
| 16 | 2 | 3,685,950/- | 33,517/- | 4.55/- | 6.08/- | 7.18/- |
| 16 | 4 | 3,620,859/- | 31,934/- | 10.1/- | 14.7/- | 18.5/- |
| 16 | 8 | 3,606,755/- | 30,775/- | 29.5/- | 45.8/- | 65.7/- |
| 16 | 16 | 3,747,417/- | 32,682/- | 120/- | 174/- | 243/- |
| 16 | 64 | 3,893,373/- | 33,041/- | 1,126/- | 2,138/- | 2,951/- |
| 64 | 1 | 3,837,013/- | 33,495/- | 2.68/- | 5.33/- | 7.93/- |
| 64 | 2 | 3,672,208/- | 34,749/- | 32.8/- | 49.7/- | 64.0/- |
| 64 | 4 | 3,652,969/- | 34,742/- | 93.3/- | 158/- | 213/- |
| 64 | 8 | 3,717,630/- | 34,546/- | 318/- | 568/- | 787/- |
| 64 | 16 | 3,755,212/- | 34,666/- | 1,186/- | 2,175/- | 3,003/- |
| 64 | 64 | 4,256,574/- | 34,590/- | 16,487/- | 20,240/- | 23,894/- |
| 256 | 1 | 3,630,240/- | 35,192/- | 38.8/- | 77.5/- | 116/- |
| 256 | 2 | 3,720,791/- | 35,643/- | 334/- | 587/- | 803/- |
| 256 | 4 | 3,774,718/- | 35,703/- | 1,296/- | 2,121/- | 3,196/- |
| 256 | 8 | 3,936,490/- | 34,786/- | 4,260/- | 8,092/- | 11,389/- |
| 256 | 16 | 4,375,480/- | 35,310/- | 17,143/- | 20,886/- | 25,939/- |
| 256 | 64 | 5,909,106/- | 33,496/- | 106,491/- | 141,175/- | 174,384/- |
| 1024 | 1 | 3,774,796/- | 35,709/- | 614/- | 1,228/- | 1,860/- |
| 1024 | 2 | 3,981,397/- | 35,469/- | 4,348/- | 8,296/- | 11,651/- |
| 1024 | 4 | 4,294,626/- | 36,551/- | 16,434/- | 20,222/- | 24,059/- |
| 1024 | 8 | 4,815,558/- | 36,634/- | 41,408/- | 53,216/- | 63,440/- |
| 1024 | 16 | 5,885,403/- | 37,126/- | 105,600/- | 140,221/- | 170,333/- |
| 1024 | 64 | 6,785,149/- | 37,110/- | 785,587/- | 1,101,806/- | 1,372,948/- |

### Nonuniform weights

64 distinct points; input count `D = W`; capacity 4096.

| Field | Weights `[1, 2, 4, 8]` repeating, W = 240 (µs) | 63 unit weights and one weight 64, W = 127 (µs) |
| --- | --- | --- |
| `Gf8B` | 32.2/- | 15.2/- |
| `Gf16` | 58.5/- | 27.5/- |
| `Mersenne31` | 71.1/- | 32.3/- |
| `Goldilocks` | 167/- | 51.2/- |
| `QuadMersenne31` | 151/- | 51.1/- |

`Gf8B` cannot supply 1024 distinct points, so its 1024-point rows are
omitted rather than shrunk. The `scalar` harness suffix denotes one
polynomial lane, not a forced backend.

## Competitor panels

Dev-only harness (`benches/competitors.rs`). Every arm validates its output
before timing. Paired cells are Lunar Lake / Golden Cove.

### Ring operations, matched 8-byte element width

This crate runs over `fgf::Gf64`; `ark-poly` 0.6 and `lambdaworks-math` 0.13
have no binary-field coefficient type and run over the Goldilocks prime.
Numbers indicate algorithm cost at matched element width, not identical
domains.

| Operands | Operation | poly-ring (µs) | ark-poly (µs) | lambdaworks (µs) |
| --- | --- | --- | --- | --- |
| 64 × 64 | multiply, auto route | 4/4.08 | 4.29/3.76 | 7.42/6.99 |
| 64 × 64 | multiply, schoolbook | 4/4.08 | 6.1/4.83 | 7.42/6.99 |
| 256 × 256 | multiply, auto route | 35.2/38.1 | 20.9/16.4 | 115/110 |
| 256 × 256 | multiply, schoolbook | 35.2/38.2 | 456/381 | 115/110 |
| 1024 × 1024 | multiply, auto route | 427/497 | 251/223 | 1,820/1,750 |
| 1024 × 1024 | multiply, schoolbook | 429/497 | 7,530/6,090 | 1,820/1,750 |
| 4096 × 4096 | multiply, auto route | 3,530/3,530 | 1,310/1,150 | 29,200/27,900 |
| 4096 × 4096 | multiply, schoolbook | 6,550/7,490 | 115,000/92,800 | 29,200/27,900 |
| 512 × 128 | divide with remainder | 54.5/56.3 | 158/149 | 908/862 |
| 400 × 300 | extended gcd with Bézout cofactors | 559/581 | - | 2,830/2,690 |
| degree 4096, one point | Horner evaluation | 188/194 | 14.6/11.7 | 19.7/17.5 |
| 64 points | Lagrange interpolation, arbitrary points | 87.6/100 | - | 997/940 |
| 64 values | domain IFFT, radix-2 domain | 87.6/100 | 0.647/0.557 | - |

- `ark-poly` ships no polynomial gcd or EEA, so its gcd cell is empty.
- `ark-poly` has no arbitrary-point interpolation; its IFFT row runs over a
  fixed radix-2 domain of the same size, a different contract (points not
  caller-chosen). `lambdaworks` reports one schoolbook multiply, so its
  cells repeat across both multiply rows.

### Ring multiplication over the shared Goldilocks prime

All three libraries use the same Goldilocks field. `poly-ring before` uses the
previous one-shot NTT crossover; `poly-ring current` uses the corrected
crossover from the paired selector campaign.

| Operands | poly-ring before (µs) | poly-ring current (µs) | ark-poly (µs) | lambdaworks (µs) |
| --- | --- | --- | --- | --- |
| 64 × 64 | 3.31/- | 3.28/- | 4.13/- | 7.38/- |
| 256 × 256 | 51.7/- | 38.3/- | 18.3/- | 117/- |
| 1024 × 1024 | 165/- | 162/- | 246/- | 1,830/- |
| 4096 × 4096 | 712/- | 704/- | 1,270/- | 29,200/- |

Automatic Goldilocks routing uses the shorter operand:

| Public path | Batch | NTT crossover (coefficients) |
| --- | ---: | ---: |
| Prepared batch multiplication | 1–3 | 256 |
| Prepared batch multiplication | 4–15 | 128 |
| Prepared batch multiplication | 16 or more | 64 |
| Allocating `Polynomial::multiply` | 1 | 256 |

- Measured 2026-09-22 with one baseline/candidate session on Lunar Lake;
  Golden Cove was unavailable.
- The candidate source snapshot is SHA-256
  `ad799a71c1fe1780d9f3496d792ffa47d900e843c5a17b63a367def9f46ebcc4`
  over sorted relative-path, byte-length, and file-content records from
  `src/`, `benches/`, `tests/`, `Cargo.toml`, and `Cargo.lock`.
- Criterion used 20 samples, a 0.2 s warmup, and a 0.5 s requested measurement
  window. The aggregation is `median.point_estimate`.
- `RAYON_NUM_THREADS=1`; `/proc` reported core 3 for the Lunar Lake benchmark
  process.
- `Goldilocks` resolved to `v3`. Each library result was checked for exact
  equality before timing.
- The one-shot selector campaign compared the public prior route with the
  forced production NTT body in one binary. The NTT still lost at a shorter
  operand of 192 coefficients, won from 256 through 768, and the 1024 case
  served as an NTT-against-NTT control.
- Prepared thresholds come from forced Karatsuba, transform, and automatic
  routes over equal and asymmetric geometries. Internal route timings remain
  outside the published record.

### Hasse evaluation over the shared Goldilocks prime

Multiplicity one, steady-state evaluation, points × degree (µs):

| Points | Degree | MultiplicityPlan (µs) | ark-poly (µs) | lambdaworks (µs) | winter-math (µs) |
| --- | --- | --- | --- | --- | --- |
| 16 | 16 | 0.371/- | 0.797/- | 1.13/- | 0.634/- |
| 16 | 64 | 1.33/- | 3.54/- | 5.09/- | 3.22/- |
| 16 | 256 | 5.19/- | 14.5/- | 20.4/- | 13.3/- |
| 16 | 1024 | 20.7/- | 89.7/- | 81.5/- | 53.5/- |
| 64 | 16 | 0.997/- | 3.19/- | 4.75/- | 2.54/- |
| 64 | 64 | 3.82/- | 14.8/- | 19.9/- | 12.8/- |
| 64 | 256 | 14.6/- | 103/- | 87.4/- | 54.2/- |
| 64 | 1024 | 58.4/- | 461/- | 340/- | 214/- |
| 256 | 16 | 3.99/- | 12.9/- | 18.8/- | 10.1/- |
| 256 | 64 | 14.8/- | 91.2/- | 79.6/- | 50.3/- |
| 256 | 256 | 57.6/- | 448/- | 324/- | 222/- |
| 256 | 1024 | 251/- | 1,865/- | 1,329/- | 879/- |
| 1024 | 16 | 15.5/- | 96.9/- | 76.3/- | 41.1/- |
| 1024 | 64 | 58.6/- | 447/- | 318/- | 200/- |
| 1024 | 256 | 223/- | 1,849/- | 1,284/- | 844/- |
| 1024 | 1024 | 892/- | 7,292/- | 5,222/- | 3,414/- |

Plan construction at capacity = degree + 1, the setup a one-shot caller
pays once per request geometry (`hasse/plan-construction` cases):

| Points | Degree | Plan construction (µs) |
| --- | --- | --- |
| 16 | 16 | 19.5/- |
| 16 | 64 | 22.7/- |
| 16 | 256 | 44.1/- |
| 16 | 1024 | 327/- |
| 64 | 16 | 84.0/- |
| 64 | 64 | 86.8/- |
| 64 | 256 | 120/- |
| 64 | 1024 | 492/- |
| 256 | 16 | 411/- |
| 256 | 64 | 414/- |
| 256 | 256 | 411/- |
| 256 | 1024 | 1,021/- |
| 1024 | 16 | 2,713/- |
| 1024 | 64 | 2,777/- |
| 1024 | 256 | 2,706/- |
| 1024 | 1024 | 2,648/- |

Multiplicity above one has no direct competitor: none of the three libraries
exposes a weighted Hasse-jet API. The panel records an explicitly labelled
composed baseline — independent per-order derivative preparation, then each
library's per-point evaluation loop — asserted equal to this crate's jets
before timing. Points=64, degree=64 (µs):

| Multiplicity | MultiplicityPlan one-shot (µs) | ark-poly composed (µs) | lambdaworks composed (µs) | winter-math composed (µs) |
| 2 | 350/- | 59.1/- | 68.1/- | 52.9/- |
| 4 | 538/- | 242/- | 238/- | 209/- |
| 8 | 992/- | 890/- | 847/- | 784/- |
| 16 | 2,282/- | 2,907/- | 2,806/- | 2,760/- |

Prepared state, same geometry (µs):

| Multiplicity | MultiplicityPlan prepared (µs) | ark-poly composed (µs) | lambdaworks composed (µs) | winter-math composed (µs) |
| 2 | 36.4/- | 32.0/- | 41.2/- | 25.4/- |
| 4 | 64.3/- | 88.5/- | 79.3/- | 51.9/- |
| 8 | 126/- | 200/- | 157/- | 98.7/- |
| 16 | 283/- | 401/- | 293/- | 182/- |

- One-shot rows include each side's preparation; prepared rows reuse it.
- Competitor versions: `ark-poly`/`ark-ff` 0.6, `lambdaworks-math` 0.13,
  `winter-math` 0.13.1, all dev-only. Copyleft NTL, FLINT, and gf2x never
  enter the build.
- No GF(2^m) competitor exists among linkable non-copyleft libraries, so no
  binary-field competitor rows exist.
- The weighted-evaluation section and the three tables above were
  re-measured on 2026-09-22, on Lunar Lake only, after the remainder
  descent began sizing its work by the live dividend length; the Golden Cove
  column is `-` until that host runs the same targets. Aggregation and
  sampling are unchanged, and the competitor arms of the same run serve as
  the session control.

## Reproduction

From the crate root, with `FEC_GOLDEN_CORE` set to the host's pinned core
(3 locally, 8 on `suisei-cachy`):

```sh
export FEC_GOLDEN_CORE=<cpu> RAYON_NUM_THREADS=1
for target in poly competitors prepared hasse; do
    just _bench-run "$target" --noplot --sample-size 10 --warm-up-time 0.05 \
        --measurement-time 0.1 --save-baseline public-20260921
done
```

| Detail | Value |
| --- | --- |
| Source of each number | `median.point_estimate` in `target/criterion/**/public-20260921/estimates.json` |
| Filters | none; complete target runs |
| Cases | 912 per host |
| Affinity | one benchmark process per target, verified to equal the pinned core |
