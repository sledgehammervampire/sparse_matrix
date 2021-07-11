use criterion::{criterion_main, Criterion};
use spam::gen_bench;

gen_bench!(mul_hash, false);

criterion_main!(benches);
