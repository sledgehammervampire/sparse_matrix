use itertools::{iproduct, Itertools};
use num::Num;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{
    borrow::Cow,
    iter, mem,
    ops::{Add, Mul},
    sync::Mutex,
    vec,
};

pub mod dok_matrix;
#[cfg(test)]
mod test;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrMatrix<T> {
    rows: usize,
    cols: usize,
    // for v in vals, vals != 0
    vals: Vec<T>,
    cidx: Vec<usize>,
    // ridx.len() == rows+1
    // for i in 0..rows, cidx[ridx[i]..ridx[i+1]] are the column indices
    // containing nonzero entries on row i, and vals[ridx[i]..ridx[i+1]] are
    // the respective entries
    ridx: Vec<usize>,
}

impl<T> CsrMatrix<T> {
    pub fn new_square(n: usize) -> CsrMatrix<T> {
        CsrMatrix::new(n, n)
    }

    pub fn new(rows: usize, cols: usize) -> CsrMatrix<T> {
        assert!(
            rows > 0 && cols > 0,
            "number of rows and columns must be greater than 0"
        );

        let capacity = 1000.min(rows * cols / 5);
        CsrMatrix {
            rows,
            cols,
            vals: Vec::with_capacity(capacity),
            cidx: Vec::with_capacity(capacity),
            ridx: vec![0; rows + 1],
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    fn get_row_entries(&self, i: usize) -> (&[usize], &[T]) {
        let (j, k) = (self.ridx[i], self.ridx[i + 1]);
        (&self.cidx[j..k], &self.vals[j..k])
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = ((usize, usize), &T)> {
        (0..self.rows).flat_map(move |r| {
            let (cidx, vals) = self.get_row_entries(r);
            cidx.iter()
                .copied()
                .zip(vals.iter())
                .map(move |(c, t)| ((r, c), t))
        })
    }
}

impl<T: Num> CsrMatrix<T> {
    pub fn identity(n: usize) -> CsrMatrix<T> {
        let mut m = CsrMatrix::new_square(n);
        m.vals = iter::repeat_with(|| T::one()).take(n).collect();
        m.cidx = (0..n).map(|i| i).collect();
        m.ridx = (0..=n).map(|i| i).collect();
        m
    }

    pub fn set_element(&mut self, (i, j): (usize, usize), t: T) {
        assert!(
            (..self.rows).contains(&i) && (..self.cols).contains(&j),
            "values are not in bounds"
        );

        match self.get_row_entries(i).0.binary_search(&j) {
            Ok(k) => {
                let l = self.ridx[i] + k;
                if t.is_zero() {
                    self.vals.remove(l);
                    self.cidx.remove(l);
                    for m in i + 1..=self.rows {
                        self.ridx[m] -= 1;
                    }
                } else {
                    self.vals[l] = t;
                }
            }
            Err(k) => {
                if !t.is_zero() {
                    let l = self.ridx[i] + k;
                    self.vals.insert(l, t);
                    self.cidx.insert(l, j);
                    for m in i + 1..=self.rows {
                        self.ridx[m] += 1;
                    }
                }
            }
        }
    }
}

impl<T: Num + Clone> CsrMatrix<T> {
    pub fn get_element(&self, (i, j): (usize, usize)) -> Cow<T> {
        assert!(
            (..self.rows).contains(&i) && (..self.cols).contains(&j),
            "values are not in bounds"
        );

        let (cidx, vals) = self.get_row_entries(i);
        cidx.binary_search(&j)
            .map_or(Cow::Owned(T::zero()), |k| Cow::Borrowed(&vals[k]))
    }

    pub fn transpose(&self) -> CsrMatrix<T> {
        let mut new = CsrMatrix::new(self.cols, self.rows);
        for (j, i) in iproduct!(0..self.cols, 0..self.rows) {
            if let Cow::Borrowed(t) = self.get_element((i, j)) {
                new.set_element((j, i), t.clone());
            }
        }
        new
    }
}

impl<T: Num> Add for CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "matrices must have identical dimensions"
        );
        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], vec![0]);

        for ((a, b), (c, d)) in self
            .ridx
            .iter()
            .copied()
            .tuple_windows()
            .zip(rhs.ridx.iter().copied().tuple_windows())
        {
            let (mut rcidx, mut rvals) = self.cidx[a..b]
                .iter()
                .zip(self.vals[a..b].iter_mut())
                .merge_join_by(
                    rhs.cidx[c..d].iter().zip(rhs.vals[c..d].iter_mut()),
                    |c1, c2| c1.0.cmp(c2.0),
                )
                .map(|eob| match eob {
                    itertools::EitherOrBoth::Both((&c, t1), (_, t2)) => {
                        (c, mem::replace(t1, T::zero()) + mem::replace(t2, T::zero()))
                    }
                    itertools::EitherOrBoth::Left((&c, t))
                    | itertools::EitherOrBoth::Right((&c, t)) => (c, mem::replace(t, T::zero())),
                })
                .unzip();

            vals.append(&mut rvals);
            cidx.append(&mut rcidx);
            ridx.push(vals.len());
        }

        CsrMatrix {
            rows: self.rows,
            cols: self.cols,
            vals,
            cidx,
            ridx,
        }
    }
}

impl<T: Num + Clone + Send + Sync> Mul for &CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "LHS cols != RHS rows");

        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], vec![0]);
        for (i, (a, b)) in self.ridx.iter().copied().tuple_windows().enumerate() {
            let row = iter::repeat_with(|| Mutex::new(T::zero()))
                .take(rhs.cols)
                .collect::<Vec<_>>();
            self.cidx[a..b]
                .par_iter()
                .zip(&self.vals[a..b])
                .for_each(|(&k, t)| {
                    let (rcidx, rvals) = rhs.get_row_entries(k);
                    for (&j, t1) in rcidx.iter().zip(rvals.iter()) {
                        let mut entry = row[j].lock().unwrap();
                        *entry = mem::replace(&mut *entry, T::zero()) + t.clone() * t1.clone();
                    }
                });
            let (mut rcidx, mut rvals): (Vec<_>, Vec<_>) = row
                .into_iter()
                .map(|t| t.into_inner().unwrap())
                .enumerate()
                .filter(|(_, t)| !t.is_zero())
                .unzip();
            ridx.push(ridx[i] + rcidx.len());
            vals.append(&mut rvals);
            cidx.append(&mut rcidx);
        }
        CsrMatrix {
            rows: self.rows,
            cols: rhs.cols,
            vals,
            cidx,
            ridx,
        }
    }
}
