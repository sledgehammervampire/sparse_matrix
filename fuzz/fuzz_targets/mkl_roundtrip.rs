#![no_main]
use std::convert::{TryFrom, TryInto};

use libfuzzer_sys::{arbitrary::Unstructured, fuzz_target};
use spam::{
    arbitrary::arb_fixed_size_matrix,
    csr_matrix::{
        mkl::{CMklSparseMatrix, MklCsrMatrix, RustMklSparseMatrix},
        CsrMatrix,
    },
    Matrix,
};

const MAX_SIZE: usize = 100;

fuzz_target!(|bytes: &[u8]| {
    let mut u = Unstructured::new(bytes);
    if let (Ok(p), Ok(n)) = (u.int_in_range(1..=MAX_SIZE), u.int_in_range(1..=MAX_SIZE)) {
        if let Ok(m) = arb_fixed_size_matrix::<f64, CsrMatrix<f64, false>>(
            &mut u,
            p.try_into().unwrap(),
            n.try_into().unwrap(),
        ) {
            if let Ok(mut m) = MklCsrMatrix::try_from(m) {
                let m1 = RustMklSparseMatrix::try_from(&mut m).unwrap();
                let m1 = CMklSparseMatrix::from(m1);
                let m1 = CsrMatrix::try_from(m1).unwrap();
                assert!(m1.invariants());
            }
        }
    }
});
