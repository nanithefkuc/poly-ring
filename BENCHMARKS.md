# Benchmarks

Public API timings. Unless a subsection says otherwise, measurements are from
2026-09-24. Every paired cell is **Lunar Lake / Golden Cove**, in microseconds
unless the table header says otherwise. Each value is one pinned run's
Criterion median, rounded to three significant figures. `-` means the case was
not measured; the weighted-evaluation section, the nonuniform section, and the
Goldilocks Hasse panels carry paired 2026-09-24 runs from both hosts.

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
| 16 | 1 | 100,972/105,259 | 706/643 | 0.0605/0.0894 | 0.114/0.178 | 0.161/0.256 |
| 16 | 2 | 49,403/50,164 | 723/648 | 1.93/2.12 | 2.26/2.46 | 2.47/2.61 |
| 16 | 4 | 25,432/25,552 | 710/652 | 3.11/3.57 | 3.53/3.98 | 3.80/4.21 |
| 16 | 8 | 12,408/12,635 | 712/654 | 6.70/7.46 | 7.21/8.01 | 7.62/8.33 |
| 16 | 16 | 5,780/5,778 | 721/654 | 19.9/24.7 | 20.3/25.3 | 20.7/26.3 |
| 16 | 64 | 1,750/1,784 | 718/659 | 79.7/91.2 | 82.0/92.1 | 83.0/95.2 |
| 64 | 1 | 80,439/82,621 | 713/655 | 0.247/0.406 | 0.476/0.796 | 0.703/1.18 |
| 64 | 2 | 41,240/41,308 | 732/666 | 10.4/10.8 | 11.4/11.8 | 12.3/12.8 |
| 64 | 4 | 20,201/20,308 | 730/659 | 15.7/17.0 | 17.1/18.8 | 18.0/20.2 |
| 64 | 8 | 10,100/9,959 | 712/656 | 30.9/34.6 | 33.4/37.1 | 33.9/38.3 |
| 64 | 16 | 5,013/5,026 | 720/660 | 83.3/96.2 | 86.8/98.6 | 88.7/102 |
| 64 | 64 | 1,889/1,954 | 714/658 | 340/420 | 444/510 | 661/704 |
| 256 | 1 | 586/553 | 719/664 | 1.40/1.85 | 2.77/3.68 | 4.16/5.44 |
| 256 | 2 | 515/490 | 732/654 | 50.8/53.1 | 56.0/61.0 | 56.7/61.4 |
| 256 | 4 | 511/492 | 740/661 | 74.1/83.3 | 81.3/92.5 | 82.7/94.8 |
| 256 | 8 | 621/604 | 734/673 | 140/157 | 152/174 | 158/174 |
| 256 | 16 | 841/830 | 738/665 | 362/419 | 480/526 | 590/650 |
| 256 | 64 | 1,603/1,561 | 1,475/1,084 | 2,713/2,775 | 4,333/4,074 | 4,985/4,625 |

### Gf16

| P | s | Plan (µs) | Scratch (µs) | D=W/2 (µs) | D=W (µs) | D=3W/2 (µs) |
| 16 | 1 | 236,104/245,869 | 5,016/3,862 | 0.0953/0.129 | 0.182/0.253 | 0.263/0.369 |
| 16 | 2 | 115,117/121,183 | 5,055/3,859 | 2.91/2.62 | 3.46/3.11 | 3.85/3.47 |
| 16 | 4 | 55,240/62,577 | 5,058/3,909 | 4.71/4.42 | 5.41/5.04 | 5.87/5.44 |
| 16 | 8 | 25,723/29,699 | 5,008/3,915 | 10.7/10.4 | 11.5/11.1 | 12.1/11.7 |
| 16 | 16 | 13,961/15,677 | 5,017/3,917 | 36.7/35.1 | 37.8/36.1 | 38.6/37.0 |
| 16 | 64 | 3,140/4,242 | 4,986/3,915 | 156/149 | 159/153 | 161/154 |
| 64 | 1 | 236,787/259,252 | 5,228/4,020 | 0.429/0.573 | 0.833/1.13 | 1.25/1.68 |
| 64 | 2 | 115,809/127,609 | 5,242/4,012 | 16.1/14.4 | 18.3/16.6 | 20.2/18.1 |
| 64 | 4 | 56,146/62,639 | 5,194/4,006 | 24.0/22.4 | 26.7/24.9 | 28.7/26.9 |
| 64 | 8 | 26,463/29,984 | 5,225/4,007 | 49.5/47.8 | 53.1/51.3 | 55.7/54.0 |
| 64 | 16 | 11,770/16,873 | 5,222/4,025 | 157/149 | 162/155 | 167/160 |
| 64 | 64 | 3,890/5,162 | 5,247/4,058 | 656/622 | 917/884 | 1,261/1,186 |
| 256 | 1 | 239,265/265,272 | 5,432/4,170 | 3.21/3.78 | 6.32/7.50 | 9.66/11.3 |
| 256 | 2 | 120,553/130,138 | 5,440/4,172 | 83.8/74.9 | 95.6/85.7 | 105/95.6 |
| 256 | 4 | 59,315/64,892 | 5,379/4,173 | 120/111 | 138/129 | 153/143 |
| 256 | 8 | 28,794/32,405 | 5,475/4,182 | 232/224 | 261/255 | 285/283 |
| 256 | 16 | 14,030/19,408 | 5,404/4,174 | 695/659 | 1,043/975 | 1,478/1,350 |
| 256 | 64 | 7,740/7,824 | 5,432/4,191 | 5,261/5,079 | 7,709/7,402 | 10,607/10,244 |
| 1024 | 1 | 262,450/266,128 | 5,839/4,225 | 37.3/41.0 | 74.7/81.1 | 111/122 |
| 1024 | 2 | 128,442/131,174 | 5,696/4,262 | 450/403 | 563/515 | 654/605 |
| 1024 | 4 | 70,946/73,852 | 5,703/4,288 | 667/626 | 1,482/1,330 | 2,406/2,126 |
| 1024 | 8 | 38,343/40,046 | 5,697/4,258 | 2,091/1,899 | 3,637/3,230 | 5,496/4,764 |
| 1024 | 16 | 25,395/24,222 | 5,848/4,291 | 6,483/5,826 | 9,786/8,596 | 13,590/11,954 |
| 1024 | 64 | 13,083/12,989 | 5,839/4,372 | 45,302/44,224 | 60,924/58,574 | 94,255/88,877 |

### Mersenne31

| P | s | Plan (µs) | Scratch (µs) | D=W/2 (µs) | D=W (µs) | D=3W/2 (µs) |
| 16 | 1 | 897,724/1,009,261 | 24,859/16,544 | 0.100/0.0753 | 0.183/0.144 | 0.271/0.205 |
| 16 | 2 | 895,335/1,003,573 | 21,125/18,641 | 3.88/3.45 | 4.60/4.16 | 5.17/4.68 |
| 16 | 4 | 894,113/1,010,045 | 24,730/18,724 | 8.06/7.30 | 9.75/9.09 | 11.3/10.4 |
| 16 | 8 | 898,067/1,019,281 | 24,984/18,711 | 21.1/19.6 | 26.0/25.1 | 30.1/29.6 |
| 16 | 16 | 904,125/1,035,794 | 25,022/18,656 | 67.7/65.1 | 84.8/84.7 | 99.4/101 |
| 16 | 64 | 937,463/1,067,931 | 25,197/18,775 | 434/449 | 684/738 | 900/975 |
| 64 | 1 | 896,611/1,023,233 | 25,544/17,408 | 0.664/0.561 | 1.30/1.10 | 1.94/1.62 |
| 64 | 2 | 899,423/1,029,029 | 25,652/17,406 | 21.7/20.3 | 26.8/26.0 | 31.0/30.1 |
| 64 | 4 | 904,540/1,024,492 | 25,512/17,417 | 49.2/47.1 | 66.4/66.6 | 81.3/82.8 |
| 64 | 8 | 915,090/1,036,074 | 25,102/17,411 | 142/139 | 204/212 | 260/274 |
| 64 | 16 | 935,390/1,059,126 | 25,520/17,435 | 470/483 | 721/773 | 936/1,014 |
| 64 | 64 | 1,054,614/1,192,308 | 25,694/19,010 | 4,851/5,204 | 6,494/6,800 | 8,271/8,280 |
| 256 | 1 | 902,754/1,016,710 | 26,040/19,278 | 7.02/7.52 | 13.9/15.0 | 21.0/22.4 |
| 256 | 2 | 914,604/1,037,128 | 26,009/19,275 | 144/142 | 208/215 | 262/276 |
| 256 | 4 | 936,020/1,058,215 | 25,808/19,257 | 400/413 | 648/699 | 876/936 |
| 256 | 8 | 976,695/1,095,424 | 25,747/19,282 | 1,326/1,422 | 2,335/2,557 | 3,159/3,499 |
| 256 | 16 | 1,056,937/1,193,497 | 26,065/19,385 | 4,925/5,341 | 6,634/6,946 | 8,419/8,439 |
| 256 | 64 | 1,467,952/1,655,296 | 26,516/19,371 | 36,577/37,080 | 51,538/51,372 | 66,208/64,750 |
| 1024 | 1 | 939,852/1,063,821 | 26,490/19,480 | 107/118 | 213/240 | 322/354 |
| 1024 | 2 | 982,763/1,119,842 | 26,823/19,476 | 1,334/1,847 | 2,344/3,065 | 3,194/4,064 |
| 1024 | 4 | 1,066,245/1,206,509 | 26,399/19,488 | 4,628/5,116 | 6,298/6,718 | 8,084/8,197 |
| 1024 | 8 | 1,215,128/1,374,488 | 24,540/18,483 | 12,932/13,529 | 18,405/18,381 | 23,335/22,896 |
| 1024 | 16 | 1,476,919/1,646,651 | 26,864/19,599 | 37,113/37,467 | 52,377/52,119 | 67,159/65,734 |
| 1024 | 64 | 1,732,045/1,942,116 | 26,724/19,648 | 299,788/290,571 | 437,923/418,374 | 567,114/537,530 |

### Goldilocks

| P | s | Plan (µs) | Scratch (µs) | D=W/2 (µs) | D=W (µs) | D=3W/2 (µs) |
| 16 | 1 | 4,283,036/5,078,698 | 25,748/19,680 | 0.179/0.166 | 0.339/0.320 | 0.498/0.469 |
| 16 | 2 | 4,244,689/5,088,513 | 25,879/19,585 | 4.31/4.00 | 5.81/5.69 | 7.04/7.04 |
| 16 | 4 | 4,433,848/5,095,815 | 26,500/19,566 | 10.2/9.60 | 15.6/15.8 | 20.0/20.7 |
| 16 | 8 | 4,382,344/5,110,249 | 26,374/19,661 | 30.2/30.5 | 49.6/54.0 | 66.8/72.4 |
| 16 | 16 | 4,567,668/5,126,300 | 26,721/19,568 | 107/114 | 184/202 | 249/276 |
| 16 | 64 | 4,596,789/5,301,912 | 26,381/19,631 | 1,068/1,136 | 1,516/1,580 | 1,825/1,880 |
| 64 | 1 | 4,348,052/5,061,932 | 25,264/20,549 | 1.88/2.06 | 3.64/4.07 | 5.51/6.08 |
| 64 | 2 | 4,293,403/5,088,559 | 25,161/20,654 | 34.4/35.3 | 53.9/58.5 | 70.3/77.1 |
| 64 | 4 | 4,303,126/5,120,121 | 25,279/20,653 | 99.1/109 | 174/198 | 237/271 |
| 64 | 8 | 4,398,679/5,163,740 | 25,462/20,751 | 353/392 | 565/607 | 722/751 |
| 64 | 16 | 4,502,567/5,282,360 | 25,242/20,789 | 1,166/1,236 | 1,616/1,680 | 1,922/1,985 |
| 64 | 64 | 5,027,158/5,938,024 | 28,024/20,834 | 7,765/8,150 | 9,694/10,139 | 11,000/11,483 |
| 256 | 1 | 4,317,684/5,106,883 | 30,966/21,288 | 29.4/32.0 | 60.9/63.7 | 82.8/95.6 |
| 256 | 2 | 4,337,049/5,166,705 | 26,601/21,300 | 365/411 | 574/624 | 720/769 |
| 256 | 4 | 4,483,713/5,283,206 | 26,250/21,270 | 1,126/1,214 | 1,571/1,661 | 1,885/1,959 |
| 256 | 8 | 4,710,662/5,509,458 | 30,194/19,715 | 3,234/3,306 | 4,193/4,236 | 4,860/4,863 |
| 256 | 16 | 5,312,955/5,938,377 | 30,012/21,457 | 8,522/8,560 | 10,450/10,561 | 11,868/11,895 |
| 256 | 64 | 7,174,367/8,264,102 | 30,106/21,538 | 49,428/48,863 | 58,379/57,757 | 64,555/63,798 |
| 1024 | 1 | 4,654,601/5,386,663 | 30,756/21,363 | 443/515 | 888/1,031 | 1,326/1,529 |
| 1024 | 2 | 4,948,887/5,525,502 | 30,052/21,931 | 3,296/3,396 | 4,230/4,333 | 4,877/4,964 |
| 1024 | 4 | 5,227,888/5,962,673 | 30,164/22,083 | 8,379/8,490 | 10,395/10,479 | 11,807/11,826 |
| 1024 | 8 | 5,932,262/6,811,584 | 30,398/22,233 | 21,648/20,891 | 25,912/25,116 | 29,157/27,972 |
| 1024 | 16 | 7,408,318/8,301,668 | 31,716/22,306 | 51,069/50,318 | 60,044/59,083 | 66,161/65,021 |
| 1024 | 64 | 8,478,714/9,607,034 | 30,807/22,509 | 272,420/268,601 | 314,953/306,953 | 341,535/333,186 |

### QuadMersenne31

| P | s | Plan (µs) | Scratch (µs) | D=W/2 (µs) | D=W (µs) | D=3W/2 (µs) |
| 16 | 1 | 3,646,447/4,112,063 | 31,718/23,662 | 0.220/0.218 | 0.422/0.423 | 0.627/0.620 |
| 16 | 2 | 3,685,950/4,124,225 | 33,517/23,606 | 4.55/4.16 | 6.08/5.70 | 7.18/6.93 |
| 16 | 4 | 3,620,859/4,134,226 | 31,934/23,685 | 10.1/9.79 | 14.7/15.1 | 18.5/19.3 |
| 16 | 8 | 3,606,755/4,106,222 | 30,775/23,544 | 29.5/30.6 | 45.8/50.2 | 65.7/65.8 |
| 16 | 16 | 3,747,417/4,176,895 | 32,682/23,634 | 120/111 | 174/187 | 243/249 |
| 16 | 64 | 3,893,373/4,288,727 | 33,041/23,416 | 1,126/1,236 | 2,138/2,400 | 2,951/3,373 |
| 64 | 1 | 3,837,013/4,132,815 | 33,495/24,377 | 2.68/2.91 | 5.33/5.77 | 7.93/8.62 |
| 64 | 2 | 3,672,208/4,157,039 | 34,749/24,393 | 32.8/32.8 | 49.7/51.9 | 64.0/67.8 |
| 64 | 4 | 3,652,969/4,181,823 | 34,742/24,398 | 93.3/97.9 | 158/171 | 213/233 |
| 64 | 8 | 3,717,630/4,192,110 | 34,546/24,471 | 318/345 | 568/633 | 787/874 |
| 64 | 16 | 3,755,212/4,317,884 | 34,666/24,695 | 1,186/1,307 | 2,175/2,452 | 3,003/3,390 |
| 64 | 64 | 4,256,574/4,860,305 | 34,590/24,704 | 16,487/18,545 | 20,240/22,836 | 23,894/26,598 |
| 256 | 1 | 3,630,240/4,176,420 | 35,192/25,269 | 38.8/45.1 | 77.5/89.5 | 116/135 |
| 256 | 2 | 3,720,791/4,199,235 | 35,643/25,275 | 334/356 | 587/648 | 803/886 |
| 256 | 4 | 3,774,718/4,287,320 | 35,703/25,263 | 1,296/1,266 | 2,121/2,410 | 3,196/3,370 |
| 256 | 8 | 3,936,490/4,490,642 | 34,786/25,256 | 4,260/4,809 | 8,092/9,366 | 11,389/13,147 |
| 256 | 16 | 4,375,480/4,862,037 | 35,310/25,346 | 17,143/18,909 | 20,886/23,176 | 25,939/26,874 |
| 256 | 64 | 5,909,106/6,717,676 | 33,496/25,583 | 106,491/118,137 | 141,175/156,922 | 174,384/190,369 |
| 1024 | 1 | 3,774,796/4,344,059 | 35,709/25,860 | 614/723 | 1,228/1,454 | 1,860/2,165 |
| 1024 | 2 | 3,981,397/4,470,698 | 35,469/26,107 | 4,348/4,946 | 8,296/9,576 | 11,651/13,372 |
| 1024 | 4 | 4,294,626/4,833,330 | 36,551/26,234 | 16,434/19,034 | 20,222/23,390 | 24,059/27,133 |
| 1024 | 8 | 4,815,558/5,523,060 | 36,634/26,279 | 41,408/46,758 | 53,216/59,498 | 63,440/70,686 |
| 1024 | 16 | 5,885,403/6,711,478 | 37,126/26,240 | 105,600/118,255 | 140,221/156,888 | 170,333/190,532 |
| 1024 | 64 | 6,785,149/7,794,451 | 37,110/26,723 | 785,587/860,305 | 1,101,806/1,207,895 | 1,372,948/1,507,942 |

### Nonuniform weights

64 distinct points; input count `D = W`; capacity 4096.

| Field | Weights `[1, 2, 4, 8]` repeating, W = 240 (µs) | 63 unit weights and one weight 64, W = 127 (µs) |
| --- | --- | --- |
| `Gf8B` | 32.2/35.0 | 15.2/16.7 |
| `Gf16` | 58.5/53.6 | 27.5/25.6 |
| `Mersenne31` | 71.1/73.2 | 32.3/32.0 |
| `Goldilocks` | 167/185 | 51.2/53.7 |
| `QuadMersenne31` | 151/167 | 51.1/54.2 |

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
| 64 × 64 | multiply, auto route | 4.03/4.09 | 4.35/3.80 | 7.60/7.18 |
| 64 × 64 | multiply, schoolbook | 4.05/4.09 | 6.14/4.95 | 7.60/7.18 |
| 256 × 256 | multiply, auto route | 35.7/38.8 | 20.1/16.7 | 118/110 |
| 256 × 256 | multiply, schoolbook | 35.7/38.8 | 462/409 | 118/110 |
| 1024 × 1024 | multiply, auto route | 433/498 | 257/221 | 1,870/1,743 |
| 1024 × 1024 | multiply, schoolbook | 434/498 | 7,482/6,594 | 1,870/1,743 |
| 4096 × 4096 | multiply, auto route | 3,542/3,512 | 1,297/1,149 | 29,788/28,209 |
| 4096 × 4096 | multiply, schoolbook | 6,539/7,512 | 115,651/102,612 | 29,788/28,209 |
| 512 × 128 | divide with remainder | 55.1/56.5 | 158/148 | 918/859 |
| 400 × 300 | extended gcd with Bézout cofactors | 565/582 | - | 2,890/2,717 |
| degree 4096, one point | Horner evaluation | 201/194 | 14.8/11.6 | 20.4/17.5 |
| 64 points | Lagrange interpolation, arbitrary points | 104/108 | - | 1,024/945 |
| 64 values | domain IFFT, radix-2 domain | 104/108 | 0.684/0.553 | - |

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
| 64 × 64 | 4.03/4.09 | 3.38/3.88 | 4.33/3.76 | 7.68/7.11 |
| 256 × 256 | 35.7/38.8 | 40.0/38.9 | 20.1/17.1 | 120/112 |
| 1024 × 1024 | 433/498 | 170/168 | 252/220 | 1,905/1,777 |
| 4096 × 4096 | 3,542/3,512 | 741/739 | 1,314/1,165 | 30,371/28,411 |

Automatic routing uses the shorter operand:

| Public path | Batch | NTT crossover (coefficients) |
| --- | ---: | ---: |
| Prepared batch multiplication | 1–3 | 256 |
| Prepared batch multiplication | 4–15 | 128 |
| Prepared batch multiplication | 16 or more | 64 |
| Allocating `Polynomial::multiply`, Goldilocks | 1 | 512 |
| Allocating `Polynomial::multiply`, `QuadMersenne31` | 1 | 1024 |
| Allocating `Polynomial::multiply`, Mersenne31 (embedded) | 1 | 4096 |

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

QuadMersenne31 shares the prepared selector and its crossovers:

| Public path | Batch | NTT crossover (coefficients) |
| --- | ---: | ---: |
| Prepared batch multiplication | 1–3 | 256 |
| Prepared batch multiplication | 4–15 | 128 |
| Prepared batch multiplication | 16 or more | 64 |

Mersenne31 selects the embedded QuadMersenne31 route by its own crossovers:

| Public path | Batch | NTT crossover (coefficients) |
| --- | ---: | ---: |
| Prepared batch multiplication | 1–3 | 1024 |
| Prepared batch multiplication | 4–15 | 1024 |
| Prepared batch multiplication | 16 or more | 512 |

Decision cells measured on both hosts, medians in microseconds:

| Route | Crossover cell | Lunar Lake NTT / Karatsuba (µs) | Golden Cove NTT / Karatsuba (µs) |
| --- | --- | ---: | ---: |
| Prepared QM31, batch 1–3 | 256 × 256 | 35.9 / 42.7 | 36.1 / 48.8 |
| Prepared QM31, batch 4–15 | 128 × 128 | 38.8 / 43.4 | 38.8 / 51.9 |
| Prepared QM31, batch 16 or more | 64 × 64 | 37.4 / 45.7 | 36.0 / 53.1 |
| Prepared M31, batch 1–3 | 1024 × 1024 | 153.2 / 170.0 | 157.0 / 191.6 |
| Prepared M31, batch 4–15 | 1024 × 1024 | 397.4 / 683.4 | 392.2 / 756.7 |
| Prepared M31, batch 16 or more | 512 × 512 | 445.3 / 701.8 | 405.9 / 782.3 |
| One-shot Goldilocks | 512 × 512 | 68.4 / 78.5 | 69.2 / 95.2 |
| One-shot QM31 | 1024 × 1024 | 161.5 / 200.7 | 170.0 / 237.6 |
| One-shot M31 | 4096 × 4096 | 785.5 / 897.8 | 857.0 / 782.0 |

- Paired 2026-09-25 campaigns on both hosts; sampling and aggregation
  match the 2026-09-22 selector campaign above. `RAYON_NUM_THREADS=1`;
  affinity pinned to core 3 on Lunar Lake and core 8 on Golden Cove, both
  verified through `/proc`. `BackendClass` resolves identically on both
  hosts. Full records: `bench-records/routing-qm31-m31-20260925.md` and
  `bench-records/routing-oneshot-20260925.md`.
- Prepared crossovers reproduce on Golden Cove at the same cells. Two
  Mersenne31 boundary cells read slightly for the transform there against
  tie-or-Karatsuba on Lunar Lake; the thresholds stand.
- One-shot crossovers hold on both hosts, except Mersenne31 at 4096: a win
  on Lunar Lake and a boundary loss on Golden Cove. The threshold stays on
  the side that keeps the measured win. The Goldilocks one-shot crossover
  moves 256 → 512: the 2026-09-22 value reproduces on neither host against
  the current kernels.
- One-shot runs force the crossover constants to 1 so dispatch takes the
  NTT path at every size; adopted values are restored afterwards.

### Hasse evaluation over the shared Goldilocks prime

Multiplicity one, steady-state evaluation, points × degree (µs):

| Points | Degree | MultiplicityPlan (µs) | ark-poly (µs) | lambdaworks (µs) | winter-math (µs) |
| --- | --- | --- | --- | --- | --- |
| 16 | 16 | 0.371/0.340 | 0.797/0.618 | 1.13/0.953 | 0.634/0.532 |
| 16 | 64 | 1.33/1.23 | 3.54/2.80 | 5.09/4.24 | 3.22/2.76 |
| 16 | 256 | 5.19/4.83 | 14.5/11.6 | 20.4/17.4 | 13.3/11.8 |
| 16 | 1024 | 20.7/19.2 | 89.7/71.4 | 81.5/70.0 | 53.5/47.5 |
| 64 | 16 | 0.997/1.12 | 3.19/2.46 | 4.75/3.80 | 2.54/2.10 |
| 64 | 64 | 3.82/4.18 | 14.8/11.2 | 19.9/17.0 | 12.8/11.0 |
| 64 | 256 | 14.6/16.3 | 103/74.1 | 87.4/69.7 | 54.2/47.1 |
| 64 | 1024 | 58.4/63.6 | 461/389 | 340/280 | 214/190 |
| 256 | 16 | 3.99/4.36 | 12.9/9.88 | 18.8/15.2 | 10.1/8.36 |
| 256 | 64 | 14.8/16.4 | 91.2/71.1 | 79.6/67.8 | 50.3/44.0 |
| 256 | 256 | 57.6/63.8 | 448/393 | 324/279 | 222/187 |
| 256 | 1024 | 251/257 | 1,865/1,593 | 1,329/1,119 | 879/758 |
| 1024 | 16 | 15.5/17.5 | 96.9/74.7 | 76.3/60.7 | 41.1/33.4 |
| 1024 | 64 | 58.6/65.1 | 447/386 | 318/272 | 200/176 |
| 1024 | 256 | 223/255 | 1,849/1,622 | 1,284/1,115 | 844/751 |
| 1024 | 1024 | 892/1,009 | 7,292/6,492 | 5,222/4,480 | 3,414/3,062 |

Plan construction at capacity = degree + 1, the setup a one-shot caller
pays once per request geometry (`hasse/plan-construction` cases):

| Points | Degree | Plan construction (µs) |
| --- | --- | --- |
| 16 | 16 | 19.5/18.0 |
| 16 | 64 | 22.7/21.2 |
| 16 | 256 | 44.1/46.8 |
| 16 | 1024 | 327/367 |
| 64 | 16 | 84.0/80.0 |
| 64 | 64 | 86.8/81.0 |
| 64 | 256 | 120/121 |
| 64 | 1024 | 492/507 |
| 256 | 16 | 411/402 |
| 256 | 64 | 414/403 |
| 256 | 256 | 411/405 |
| 256 | 1024 | 1,021/986 |
| 1024 | 16 | 2,713/2,737 |
| 1024 | 64 | 2,777/2,737 |
| 1024 | 256 | 2,706/2,734 |
| 1024 | 1024 | 2,648/2,746 |

Multiplicity above one has no direct competitor: none of the three libraries
exposes a weighted Hasse-jet API. The panel records an explicitly labelled
composed baseline — independent per-order derivative preparation, then each
library's per-point evaluation loop — asserted equal to this crate's jets
before timing. Points=64, degree=64 (µs):

| Multiplicity | MultiplicityPlan one-shot (µs) | ark-poly composed (µs) | lambdaworks composed (µs) | winter-math composed (µs) |
| 2 | 350/328 | 59.1/62.9 | 68.1/62.5 | 52.9/50.0 |
| 4 | 538/533 | 242/244 | 238/228 | 209/203 |
| 8 | 992/1,008 | 890/879 | 847/827 | 784/780 |
| 16 | 2,282/2,444 | 2,907/2,950 | 2,806/2,834 | 2,760/2,748 |

Prepared state, same geometry (µs):

| Multiplicity | MultiplicityPlan prepared (µs) | ark-poly composed (µs) | lambdaworks composed (µs) | winter-math composed (µs) |
| 2 | 36.4/36.5 | 32.0/35.4 | 41.2/34.8 | 25.4/22.5 |
| 4 | 64.3/65.3 | 88.5/84.5 | 79.3/68.6 | 51.9/44.2 |
| 8 | 126/131 | 200/186 | 157/133 | 98.7/87.2 |
| 16 | 283/285 | 401/371 | 293/248 | 182/160 |

- One-shot rows include each side's preparation; prepared rows reuse it.
- Competitor versions: `ark-poly`/`ark-ff` 0.6, `lambdaworks-math` 0.13,
  `winter-math` 0.13.1, all dev-only. Copyleft NTL, FLINT, and gf2x never
  enter the build.
- No GF(2^m) competitor exists among linkable non-copyleft libraries, so no
  binary-field competitor rows exist.
- The weighted-evaluation section, the nonuniform section, and the
  competitor tables above were re-measured on 2026-09-24 on both hosts
  after the remainder descent began sizing its work by the live dividend
  length and the lane route moved onto `fgf` 1.2.1's in-place kernels.
  Aggregation and sampling are unchanged, and the competitor arms of the
  same run serve as the session control.

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
