use std::{fs::read_to_string, ops::Mul, str::FromStr};

use criterion::{criterion_group, criterion_main, Criterion};
use spam::{
    dok_matrix::{parse_matrix_market, DokMatrix},
    Matrix,
};

pub fn bench_mul<T: Clone + FromStr, M: From<DokMatrix<T>> + Clone + Matrix<T>>(c: &mut Criterion)
where
    for<'a> &'a M: Mul,
{
    const FILES: &[&str] = &["sc2010", "tube2", "big", "gr_30_30", "bcsstm01"];
    for f in FILES {
        let m1 = M::from(
            parse_matrix_market::<T>(&read_to_string(format!("matrices/{}.mtx", f)).unwrap())
                .unwrap()
                .1,
        );
        let m2 = m1.clone();
        c.bench_function(
            &format!(
                "bench csr mul {} ({}x{}, {} entries)",
                f,
                m1.rows(),
                m1.cols(),
                m1.len()
            ),
            |b| b.iter(|| &m1 * &m2),
        );
    }
}

criterion_group!(
    benches,
    bench_mul::<i32, spam::csr_matrix::CsrMatrix<_>>,
    bench_mul::<i32, spam::csr_matrix2::CsrMatrix<_>>
);
criterion_main!(benches);
