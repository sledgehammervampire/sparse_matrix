use std::borrow::Cow;

pub mod arbitrary;
pub mod csr_matrix;
pub mod dok_matrix;

#[cfg(test)]
mod tests;

pub trait Matrix<T: ToOwned> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn get_element(&self, pos: (usize, usize)) -> Cow<T>;
    fn set_element(&mut self, pos: (usize, usize), t: T);
    fn identity(n: usize) -> Self;
    fn transpose(self) -> Self;
}
