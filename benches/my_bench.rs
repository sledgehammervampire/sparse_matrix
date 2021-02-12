use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use proptest::{prelude::*, strategy::ValueTree, test_runner::TestRunner};
use sparse_matrix::dok_matrix::{arb_add_pair_with_rows_and_cols, AddPair};

pub fn bench_add(c: &mut Criterion) {
    c.bench_function("bench add", |b| {
        let mut runner = TestRunner::default();
        let p = arb_add_pair_with_rows_and_cols::<i32>(1000, 1000)
            .new_tree(&mut runner)
            .unwrap()
            .current();
        b.iter_batched(
            || p.clone(),
            |AddPair(mut m1, m2)| {
                m1 += &m2;
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_add);
criterion_main!(benches);
