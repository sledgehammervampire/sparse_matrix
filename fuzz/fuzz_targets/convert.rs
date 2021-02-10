#![no_main]
use libfuzzer_sys::fuzz_target;

use sparse_matrix::{CsrMatrix, dok_matrix::DokMatrix};

fuzz_target!(|m: DokMatrix<i32>| {
    assert_eq!(m, DokMatrix::from(CsrMatrix::from(m.clone())));
});
