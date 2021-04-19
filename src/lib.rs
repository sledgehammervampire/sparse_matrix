#![feature(allocator_api)]

use std::{borrow::Cow, ops::Range};

mod arbitrary;
pub mod csr_matrix;
pub mod dok_matrix;
#[cfg(test)]
pub mod proptest;

pub trait Matrix<T: ToOwned> {
    fn new(rows: usize, cols: usize) -> Self;
    fn new_square(n: usize) -> Self;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    // the number of nonzero entries in the matrix
    fn nnz(&self) -> usize;
    fn get_element(&self, pos: (usize, usize)) -> Cow<T>;
    fn set_element(&mut self, pos: (usize, usize), t: T) -> Option<T>;
    fn identity(n: usize) -> Self;
    fn transpose(self) -> Self;
}

// pair of matrices conformable for addition
#[derive(Clone, Debug)]
pub struct AddPair<M>(pub(crate) M, pub(crate) M);

// pair of matrices conformable for multiplication
#[derive(Clone, Debug)]
pub struct MulPair<M>(pub(crate) M, pub(crate) M);

fn is_sorted(s: &[usize]) -> bool {
    let mut max = None;
    for i in s {
        if Some(i) >= max {
            max = Some(i);
        } else {
            return false;
        }
    }
    true
}

fn is_increasing(s: &[usize]) -> bool {
    let mut max = None;
    for i in s {
        if Some(i) > max {
            max = Some(i);
        } else {
            return false;
        }
    }
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Slice {
    pub start: usize,
    pub len: usize,
}

impl From<Slice> for Range<usize> {
    fn from(s: Slice) -> Self {
        s.start..s.start + s.len
    }
}
