use std::fs::read_to_string;

use criterion::{criterion_group, criterion_main, Criterion};
use spam::{csr_matrix::CsrMatrix, dok_matrix::parse_matrix_market, Matrix};

pub fn bench_mul(c: &mut Criterion) {
    const FILES: &[&str] = &["big", "gr_30_30", "bcsstm01"];
    for f in FILES {
        let m1 = CsrMatrix::from(
            parse_matrix_market::<i32>(&read_to_string(format!("matrices/{}.mtx", f)).unwrap())
                .unwrap()
                .1,
        );
        let m2 = m1.clone();
        c.bench_function(
            &format!(
                "bench mul {} ({}x{}, {} entries)",
                f,
                m1.rows(),
                m1.cols(),
                m1.iter().count()
            ),
            |b| b.iter(|| &m1 * &m2),
        );
    }
}

criterion_group!(benches, bench_mul);
criterion_main!(benches);
