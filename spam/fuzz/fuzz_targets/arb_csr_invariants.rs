#![no_main]
use libfuzzer_sys::fuzz_target;
use spam::{csr_matrix::CsrMatrix, dok_matrix::DokMatrix};

fuzz_target!(|m: DokMatrix<i8>| {
    let m = CsrMatrix::from(m);
    assert!(m.invariants());
});
