use std::{
    fs::{read_dir, read_to_string},
    ops::Mul,
    str::FromStr,
};

use criterion::{criterion_group, criterion_main, Criterion};
use num::Num;
use spam::{
    dok_matrix::{parse_matrix_market, DokMatrix},
    Matrix,
};

pub fn bench_mul<T: Clone + FromStr + Num, M: From<DokMatrix<T>> + Clone + Matrix<T>>(
    c: &mut Criterion,
) where
    for<'a> &'a M: Mul,
{
    for f in &["sc2010", "tube2"] {
        let m1 = M::from(
            parse_matrix_market::<T>(&read_to_string(format!("matrices/{}.mtx", f)).unwrap())
                .unwrap()
                .1,
        );
        let m2 = m1.clone();
        c.bench_function(
            &format!(
                "bench csr mul {:?} ({}x{}, {} entries)",
                f,
                m1.rows(),
                m1.cols(),
                m1.nnz()
            ),
            |b| b.iter(|| &m1 * &m2),
        );
    }
}

criterion_group!(benches, bench_mul::<i32, spam::csr_matrix::CsrMatrix<_>>,);
criterion_main!(benches);
