#![no_main]
use libfuzzer_sys::fuzz_target;
use spam::dok_matrix::DokMatrix;

fuzz_target!(|m: DokMatrix<i8>| {
    assert!(m.invariants());
});
