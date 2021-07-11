#![no_main]
use spam::fuzz_mkl_spmm;

fuzz_mkl_spmm!(false);
