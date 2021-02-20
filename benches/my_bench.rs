use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use proptest::{prelude::*, strategy::ValueTree, test_runner::TestRunner};
use sparse_matrix::{
    dok_matrix::{arb_add_pair_fixed_size, arb_mul_pair_fixed_size, AddPair, MulPair},
    CsrMatrix,
};

pub fn bench_add(c: &mut Criterion) {
    c.bench_function("bench add", |b| {
        let mut runner = TestRunner::default();
        let AddPair(m1, m2) = arb_add_pair_fixed_size::<i32>(1000, 1000)
            .new_tree(&mut runner)
            .unwrap()
            .current();
        let (m1, m2) = (CsrMatrix::from(m1), CsrMatrix::from(m2));
        b.iter_batched(
            || (m1.clone(), m2.clone()),
            |(m1, m2)| m1 + m2,
            BatchSize::SmallInput,
        )
    });
}

pub fn bench_mul(c: &mut Criterion) {
    c.bench_function("bench mul", |b| {
        let mut runner = TestRunner::default();
        let MulPair(m1, m2) = arb_mul_pair_fixed_size::<i32>(1000, 1000, 1000)
            .new_tree(&mut runner)
            .unwrap()
            .current();
        let (m1, m2) = (CsrMatrix::from(m1), CsrMatrix::from(m2));
        b.iter_batched(|| (&m1, &m2), |(m1, m2)| m1 * m2, BatchSize::SmallInput)
    });
}

criterion_group!(benches, bench_add, bench_mul);
criterion_main!(benches);
