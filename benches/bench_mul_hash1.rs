use criterion::{criterion_main, Criterion};
use spam::gen_bench;

gen_bench!(mul_hash1, false);

criterion_main!(benches);
