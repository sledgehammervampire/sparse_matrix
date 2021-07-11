#![no_main]
use std::convert::TryFrom;

use libfuzzer_sys::{arbitrary::Unstructured, fuzz_target};
use spam::{
    arbitrary::arb_fixed_size_matrix,
    csr_matrix::{
        ffi::{CMklSparseMatrix, MklCsrMatrix, RustMklSparseMatrix},
        CsrMatrix,
    },
};

const MAX_SIZE: usize = 100;

fuzz_target!(|bytes: &[u8]| {
    let mut u = Unstructured::new(bytes);
    if let (Ok(m), Ok(n)) = (u.int_in_range(0..=MAX_SIZE), u.int_in_range(0..=MAX_SIZE)) {
        if let Ok(Ok(m)) = arb_fixed_size_matrix::<f64, CsrMatrix<f64, false>>(&mut u, m, n) {
            let mut m = MklCsrMatrix::try_from(m).unwrap();
            let m = RustMklSparseMatrix::try_from(&mut m).unwrap();
            let m = CMklSparseMatrix::from(m);
            CsrMatrix::try_from(m).unwrap();
        }
    }
});
