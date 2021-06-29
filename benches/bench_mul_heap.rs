use criterion::{criterion_group, criterion_main};
use spam::make_bench_mul;

make_bench_mul!(bench_mul_heap, true, mul_heap);
criterion_group!(benches, bench_mul_heap);
criterion_main!(benches);
