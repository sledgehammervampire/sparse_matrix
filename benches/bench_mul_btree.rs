
use criterion::{criterion_main, Criterion};
use spam::gen_bench;

gen_bench!(mul_btree, true);

criterion_main!(benches);
