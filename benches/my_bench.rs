use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use quickcheck::{Arbitrary, StdGen};
use rand::thread_rng;
use sparse_matrix::{AddPair, MulPair};

pub fn bench_add(c: &mut Criterion) {
    let mut g = StdGen::new(thread_rng(), 1 << 15);
    c.bench_function("bench add", |b| {
        b.iter_batched(
            || AddPair::<i32>::arbitrary(&mut g),
            |AddPair(mut m1, m2)| {
                m1 += &m2;
            },
            BatchSize::SmallInput,
        )
    });
}

pub fn bench_mul(c: &mut Criterion) {
    let mut g = StdGen::new(thread_rng(), 1 << 13);
    c.bench_function("bench mul", |b| {
        b.iter_batched(
            || MulPair::<i32>::arbitrary(&mut g),
            |MulPair(mut m1, m2)| {
                m1 *= &m2;
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_add, bench_mul);
criterion_main!(benches);
