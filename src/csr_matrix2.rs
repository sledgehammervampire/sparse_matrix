use itertools::{iproduct, Itertools};
use num::Num;
use proptest::{arbitrary::Arbitrary, sample::subsequence, strategy::Strategy};
use rayon::{iter::ParallelIterator, prelude::*};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fmt::Debug,
    iter::repeat_with,
    mem,
    ops::{Add, Mul, Range},
    vec,
};

use crate::{dok_matrix::DokMatrix, is_increasing, is_sorted, Matrix};

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

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn len(&self) -> usize {
        self.cidx.len()
    }
}

impl<T: Num> CsrMatrix<T> {
    pub fn csr_invariants(&self) -> bool {
        self.csr_invariant_1()
            && self.csr_invariant_2()
            && self.csr_invariant_3()
            && self.csr_invariant_4()
            && self.csr_invariant_5()
            && self.csr_invariant_6()
    }

    fn csr_invariant_1(&self) -> bool {
        self.cidx.len() == self.vals.len()
    }

    fn csr_invariant_2(&self) -> bool {
        self.ridx.len() == self.rows + 1 && self.ridx[self.rows] == self.cidx.len()
    }

    fn csr_invariant_3(&self) -> bool {
        is_sorted(&self.ridx)
    }

    fn csr_invariant_4(&self) -> bool {
        self.vals.iter().all(|t| !t.is_zero())
    }

    fn csr_invariant_5(&self) -> bool {
        self.cidx.iter().all(|c| (0..self.cols).contains(c))
    }

    fn csr_invariant_6(&self) -> bool {
        self.ridx
            .iter()
            .copied()
            .zip(self.ridx.iter().skip(1).copied())
            .all(|(a, b)| is_increasing(&self.cidx[a..b]))
    }

    fn identity(n: usize) -> CsrMatrix<T> {
        CsrMatrix {
            rows: n,
            cols: n,
            vals: repeat_with(|| T::one()).take(n).collect(),
            cidx: (0..n).map(|i| i).collect(),
            ridx: (0..=n).map(|i| i).collect(),
        }
    }

    fn set_element(&mut self, (i, j): (usize, usize), t: T) {
        assert!(i < self.rows && j < self.cols, "position not in bounds");

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

impl<T: Arbitrary + Num> CsrMatrix<T> {
    pub fn arb_fixed_size_matrix(rows: usize, cols: usize) -> impl Strategy<Value = CsrMatrix<T>> {
        repeat_with(|| subsequence((0..cols).collect::<Vec<_>>(), 0..=cols))
            .take(rows)
            .collect::<Vec<_>>()
            .prop_flat_map(move |cidx| {
                let (mut cidx_flattened, mut ridx) = (vec![], vec![0]);
                for mut rcidx in cidx {
                    ridx.push(ridx.last().unwrap() + rcidx.len());
                    cidx_flattened.append(&mut rcidx);
                }
                repeat_with(|| T::arbitrary().prop_filter("T is 0", |t| !t.is_zero()))
                    .take(cidx_flattened.len())
                    .collect::<Vec<_>>()
                    .prop_map(move |vals| CsrMatrix {
                        rows,
                        cols,
                        vals,
                        cidx: cidx_flattened.clone(),
                        ridx: ridx.clone(),
                    })
            })
    }

    pub fn arb_matrix() -> impl Strategy<Value = Self> {
        crate::arbitrary::arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}

impl<T: Num + Clone> Matrix<T> for CsrMatrix<T> {
    fn get_element(&self, (i, j): (usize, usize)) -> Cow<T> {
        assert!(
            (..self.rows).contains(&i) && (..self.cols).contains(&j),
            "values are not in bounds"
        );

        let (cidx, vals) = self.get_row_entries(i);
        cidx.binary_search(&j)
            .map_or(Cow::Owned(T::zero()), |k| Cow::Borrowed(&vals[k]))
    }

    fn transpose(self) -> CsrMatrix<T> {
        let mut new = CsrMatrix::new(self.cols, self.rows);
        for (j, i) in iproduct!(0..self.cols, 0..self.rows) {
            if let Cow::Borrowed(t) = self.get_element((i, j)) {
                new.set_element((j, i), t.clone());
            }
        }
        new
    }

    fn rows(&self) -> usize {
        self.rows()
    }

    fn cols(&self) -> usize {
        self.cols()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn identity(n: usize) -> Self {
        Self::identity(n)
    }

    fn set_element(&mut self, pos: (usize, usize), t: T) {
        self.set_element(pos, t)
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
                .copied()
                .zip(
                    self.vals
                        .splice(a..b, repeat_with(|| T::zero()).take(b - a)),
                )
                .merge_join_by(
                    rhs.cidx[c..d]
                        .iter()
                        .copied()
                        .zip(rhs.vals.splice(c..d, repeat_with(|| T::zero()).take(d - c))),
                    |(c1, _), (c2, _)| c1.cmp(c2),
                )
                .filter_map(|eob| match eob {
                    itertools::EitherOrBoth::Both((c, t1), (_, t2)) => {
                        let t = t1 + t2;
                        (!t.is_zero()).then(|| (c, t))
                    }
                    itertools::EitherOrBoth::Left((c, t))
                    | itertools::EitherOrBoth::Right((c, t)) => Some((c, t)),
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

        let mut rows: Vec<(Vec<usize>, Vec<T>)> = vec![];
        self.ridx
            .par_iter()
            .zip(self.ridx.par_iter().skip(1))
            .map(|(&a, &b)| {
                let mut row = BTreeMap::new();

                for (&k, t) in self.cidx[a..b].iter().zip(self.vals[a..b].iter()) {
                    let (rcidx, rvals) = rhs.get_row_entries(k);
                    for (&j, t1) in rcidx.iter().zip(rvals.iter()) {
                        let entry = row.entry(j).or_insert(T::zero());
                        *entry = mem::replace(entry, T::zero()) + t.clone() * t1.clone();
                    }
                }

                row.into_iter().filter(|(_, t)| !t.is_zero()).unzip()
            })
            .collect_into_vec(&mut rows);
        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], vec![0]);
        for (rcidx, rvals) in rows {
            ridx.push(ridx.last().unwrap() + rcidx.len());
            vals.extend(rvals);
            cidx.extend(rcidx);
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

impl<T: Num + Clone> From<DokMatrix<T>> for CsrMatrix<T> {
    fn from(old: DokMatrix<T>) -> Self {
        let (rows, cols) = (old.rows(), old.cols());
        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], vec![0]);
        // note that (i, j) is iterated in lexicographic order
        for ((i, j), t) in old {
            if !t.is_zero() {
                vals.push(t);
                cidx.push(j);
                if let None = ridx.get(i + 1) {
                    let &k = ridx.last().unwrap();
                    for _ in ridx.len()..=i + 1 {
                        ridx.push(k);
                    }
                }
                ridx[i + 1] += 1;
            }
        }
        let &k = ridx.last().unwrap();
        for _ in ridx.len()..=rows {
            ridx.push(k);
        }
        CsrMatrix {
            rows,
            cols,
            vals,
            cidx,
            ridx,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::Wrapping;

    use itertools::iproduct;
    use proptest::{
        arbitrary::any,
        strategy::{Just, Strategy},
        test_runner::TestRunner,
    };

    use super::CsrMatrix;
    use crate::{
        arbitrary::{arb_add_pair, arb_mul_pair, AddPair, MulPair},
        dok_matrix::DokMatrix,
        Matrix,
    };

    const MAX_SIZE: usize = 10;

    // base cases
    #[test]
    fn new_invariants() {
        let mut runner = TestRunner::default();
        runner
            .run(&(1..MAX_SIZE, 1..MAX_SIZE), |(m, n)| {
                assert!(CsrMatrix::<i8>::new(m, n).csr_invariants());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn identity_invariants() {
        let mut runner = TestRunner::default();
        runner
            .run(&(1..MAX_SIZE), |n| {
                assert!(CsrMatrix::<i8>::identity(n).csr_invariants());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn arb_invariants() {
        let mut runner = TestRunner::default();
        runner
            .run(&CsrMatrix::<i8>::arb_matrix(), |m| {
                assert!(m.csr_invariants());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn from_arb_dok_invariants() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let m = CsrMatrix::from(m);
                assert!(m.csr_invariants());
                Ok(())
            })
            .unwrap();
    }

    // inductive cases
    #[test]
    fn add() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_add_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
                |AddPair(m1, m2)| {
                    let m = CsrMatrix::from(m1.clone()) + CsrMatrix::from(m2.clone());
                    assert!(m.csr_invariants());
                    assert_eq!(m, CsrMatrix::from(m1 + m2));
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn mul() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
                |MulPair(m1, m2)| {
                    let m = &CsrMatrix::from(m1.clone()) * &CsrMatrix::from(m2.clone());
                    assert!(m.csr_invariants());
                    assert_eq!(m, CsrMatrix::from(&m1 * &m2));
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn transpose() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let m1 = CsrMatrix::from(m.clone()).transpose();
                assert!(m1.csr_invariants());
                assert_eq!(m1, CsrMatrix::from(m.transpose()));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn set_element() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &DokMatrix::<i8>::arb_matrix()
                    .prop_flat_map(|m| (0..m.rows(), 0..m.cols(), Just(m), any::<i8>())),
                |(i, j, mut m, t)| {
                    let mut m1 = CsrMatrix::from(m.clone());
                    m1.set_element((i, j), t);
                    assert!(m1.csr_invariants());
                    m.set_element((i, j), t);
                    assert_eq!(m1, CsrMatrix::from(m));
                    Ok(())
                },
            )
            .unwrap();
    }

    // other
    #[test]
    fn iter() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let m1 = CsrMatrix::from(m.clone());
                assert!(m1.iter().eq(m.entries()));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn rows_and_cols() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let m1 = CsrMatrix::from(m.clone());
                assert_eq!((m.rows(), m.cols()), (m1.rows(), m1.cols()));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn get_element() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let m1 = CsrMatrix::from(m.clone());
                assert!(iproduct!(0..m.rows(), 0..m.cols())
                    .all(|pos| m.get_element(pos) == m1.get_element(pos)));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn convert() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                assert_eq!(m, DokMatrix::from(CsrMatrix::from(m.clone())));
                Ok(())
            })
            .unwrap();
    }
}