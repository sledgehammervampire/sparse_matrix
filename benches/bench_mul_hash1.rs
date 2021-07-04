use criterion::{criterion_group, criterion_main};
use spam::make_bench_mul;

make_bench_mul!(bench_mul_hash1, false, mul_hash1::<false, false>);
criterion_group!(benches, bench_mul_hash1);
criterion_main!(benches);
