#![no_main]
use std::convert::TryFrom;

use libfuzzer_sys::{arbitrary::Unstructured, fuzz_target};
use spam::{
    arbitrary::arb_mul_pair_fixed_size,
    csr_matrix::{
        ffi::{CMklSparseMatrix, MklCsrMatrix, RustMklSparseMatrix},
        CsrMatrix,
    },
    MulPair,
};

const MAX_SIZE: usize = 100;

fuzz_target!(|bytes: &[u8]| {
    let mut u = Unstructured::new(bytes);
    if let (Ok(l), Ok(m), Ok(n)) = (
        u.int_in_range(0..=MAX_SIZE),
        u.int_in_range(0..=MAX_SIZE),
        u.int_in_range(0..=MAX_SIZE),
    ) {
        if let Ok(Ok(MulPair(m1, m2))) =
            arb_mul_pair_fixed_size::<f64, CsrMatrix<f64>>(&mut u, l, m, n)
        {
            let mut m1 = MklCsrMatrix::try_from(m1).unwrap();
            let m1 = RustMklSparseMatrix::try_from(&mut m1).unwrap();
            let m1 = CMklSparseMatrix::from(m1);
            let mut m2 = MklCsrMatrix::try_from(m2).unwrap();
            let m2 = RustMklSparseMatrix::try_from(&mut m2).unwrap();
            let m2 = CMklSparseMatrix::from(m2);
            CsrMatrix::try_from((&m1 * &m2).unwrap()).unwrap();
        }
    }
});
