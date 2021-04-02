#![no_main]
use std::num::Wrapping;

use libfuzzer_sys::fuzz_target;
use spam::{csr_matrix::CsrMatrix, dok_matrix::DokMatrix, AddPair};

fuzz_target!(|p: AddPair<DokMatrix<Wrapping<i8>>>| {
    let AddPair(m1, m2) = p;
    let m = CsrMatrix::from(m1.clone()) + CsrMatrix::from(m2.clone());
    assert!(m.invariants());
    assert_eq!(m, CsrMatrix::from(m1 + m2));
});
