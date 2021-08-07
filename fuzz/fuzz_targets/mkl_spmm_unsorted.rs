#![no_main]
use libfuzzer_sys::{arbitrary::Unstructured, fuzz_target};
use spam::{
    arbitrary::arb_mul_pair_fixed_size,
    csr_matrix::{
        mkl::{CMklSparseMatrix, MklCsrMatrix, RustMklSparseMatrix},
        CsrMatrix,
    },
    MulPair,
};
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
            let mut m1 = MklCsrMatrix::try_from(m1).unwrap();
            let m1 = RustMklSparseMatrix::try_from(&mut m1).unwrap();
            let m1 = CMklSparseMatrix::from(m1);
            let mut m2 = MklCsrMatrix::try_from(m2).unwrap();
            let m2 = RustMklSparseMatrix::try_from(&mut m2).unwrap();
            let m2 = CMklSparseMatrix::from(m2);
            let m3 = CsrMatrix::try_from((&m1 * &m2).unwrap()).unwrap();
            assert!(m3.invariants());
        }
    }
});
