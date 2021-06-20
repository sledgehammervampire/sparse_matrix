#![no_main]
use std::convert::TryFrom;

use libfuzzer_sys::{arbitrary::Unstructured, fuzz_target};
use spam::{arbitrary::arb_fixed_size_matrix, csr_matrix::{CsrMatrix, ffi::{MklCsrMatrix, MklSparseMatrix}}};

const MAX_SIZE: usize = 100;

fuzz_target!(|bytes: &[u8]| {
    let mut u = Unstructured::new(bytes);
    let rows = if let Ok(n) = u.int_in_range(0..=MAX_SIZE) {
        n
    } else {
        return;
    };
    let cols = if let Ok(n) = u.int_in_range(0..=MAX_SIZE) {
        n
    } else {
        return;
    };

    if let Ok(Ok(m)) = arb_fixed_size_matrix::<f64, CsrMatrix<f64>>(&mut u, rows, cols) {
        if m.iter().all(|(_, t)| !t.is_nan()) {
            let m1 = MklCsrMatrix::try_from(m.clone()).unwrap();
            let m2 = MklSparseMatrix::try_from(m1).unwrap();
            // assert_eq!(m, CsrMatrix::try_from(m1).unwrap());
        }
    }
});
