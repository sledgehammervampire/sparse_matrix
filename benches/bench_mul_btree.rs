use criterion::{criterion_group, criterion_main};
use spam::make_bench_mul;

make_bench_mul!(bench_mul_btree, true, mul_btree);
criterion_group!(benches, bench_mul_btree);
criterion_main!(benches);
