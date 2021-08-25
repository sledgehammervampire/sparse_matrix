#![cfg_attr(test, feature(no_coverage, type_alias_impl_trait))]
#![feature(allocator_api, is_sorted)]
#![deny(clippy::disallowed_method)]

use std::{fmt::Debug, num::NonZeroUsize};
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
#[cfg(feature = "proptest")]
pub mod proptest;

#[derive(Debug, PartialEq, Eq)]
pub struct IndexError;

pub trait Matrix<T>: Sized {
    fn invariants(&self) -> bool;
    fn new(size: (NonZeroUsize, NonZeroUsize)) -> Self;
    fn new_square(n: NonZeroUsize) -> Self;
    fn identity(n: NonZeroUsize) -> Self;
    fn rows(&self) -> NonZeroUsize;
    fn cols(&self) -> NonZeroUsize;
    // the number of explicit entries in the matrix
    fn nnz(&self) -> usize;
    fn get_element(&self, pos: (usize, usize)) -> Result<Option<&T>, IndexError>;
    fn set_element(&mut self, pos: (usize, usize), t: T) -> Result<Option<T>, IndexError>;
    fn transpose(self) -> Self;
}

// pair of matrices conformable for addition
#[derive(Clone, Debug)]
pub struct AddPair<M>(pub M, pub M);

// pair of matrices conformable for multiplication
#[derive(Clone, Debug)]
pub struct MulPair<M>(pub M, pub M);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Slice {
    pub start: usize,
    pub len: usize,
}

impl From<Slice> for std::ops::Range<usize> {
    fn from(s: Slice) -> Self {
        s.start..s.start + s.len
    }
}
