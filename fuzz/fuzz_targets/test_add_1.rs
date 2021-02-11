#![no_main]
use libfuzzer_sys::fuzz_target;

use sparse_matrix::{dok_matrix::AddPair, CsrMatrix};

fuzz_target!(|p: AddPair<i32>| {
    let AddPair(mut m1, m2) = p;
    let mut m3 = CsrMatrix::from(m1.clone());
    m3 += &CsrMatrix::from(m2.clone());
    m1 += &m2;
    assert_eq!(CsrMatrix::from(m1.clone()), m3);
});
