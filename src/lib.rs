use std::borrow::Cow;

#[cfg(test)]
pub mod proptest;
pub mod csr_matrix;
pub mod dok_matrix;
#[cfg(feature = "arbitrary_impl")]
mod arbitrary;

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
pub struct AddPair<M>(pub M, pub M);

// pair of matrices conformable for multiplication
#[derive(Clone, Debug)]
pub struct MulPair<M>(pub M, pub M);