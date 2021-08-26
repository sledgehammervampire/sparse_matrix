#![no_main]
use libfuzzer_sys::{arbitrary::Unstructured, fuzz_target};
use sci::Sci;
use spam_csr::{
    mkl::{CMklSparseMatrix, MklCsrMatrix, RustMklSparseMatrix},
    CsrMatrix,
};
use spam_dok::DokMatrix;
use spam_matrix::{arbitrary::arb_mul_pair_fixed_size, Matrix, MulPair};
use std::convert::{TryFrom, TryInto};

fuzz_target!(|bytes| {
    const MAX_SIZE: usize = 100;
    let mut u = Unstructured::new(bytes);
    if let (Ok(l), Ok(m), Ok(n)) = (
        u.int_in_range(1..=MAX_SIZE),
        u.int_in_range(1..=MAX_SIZE),
        u.int_in_range(1..=MAX_SIZE),
    ) {
        if let Ok(MulPair(m1, m2)) = arb_mul_pair_fixed_size::<f64, CsrMatrix<f64, false>>(
            &mut u,
            l.try_into().unwrap(),
            m.try_into().unwrap(),
            n.try_into().unwrap(),
        ) {
            if let (Ok(mut m3), Ok(mut m4)) = (
                MklCsrMatrix::try_from(m1.clone()),
                MklCsrMatrix::try_from(m2.clone()),
            ) {
                let m3 = RustMklSparseMatrix::try_from(&mut m3).unwrap();
                let m3 = CMklSparseMatrix::from(m3);
                let m4 = RustMklSparseMatrix::try_from(&mut m4).unwrap();
                let m4 = CMklSparseMatrix::from(m4);
                let m5 = CsrMatrix::try_from((&m3 * &m4).unwrap()).unwrap();
                assert!(m5.invariants());
                let m5 = DokMatrix::from(m5);
                assert!(m5.invariants());
                let m6 = DokMatrix::from(&m1 * &m2);
                assert!(m6.invariants());
                let all_not_nan = |m: &DokMatrix<f64>| m.iter().all(|(_, t)| !t.is_nan());
                if all_not_nan(&m5) && all_not_nan(&m6) {
                    assert!(m5.approx_eq(&m6), "{:?}", {
                        let size = (m5.rows(), m5.cols());
                        (m5 - m6).into_iter().map(|(i, t)| (i, Sci(t))).fold(
                            DokMatrix::new(size),
                            |mut m, (i, t)| {
                                m.set_element(i, t).unwrap();
                                m
                            },
                        )
                    });
                }
            }
        }
    }
});
