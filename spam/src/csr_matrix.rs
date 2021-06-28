use itertools::{iproduct, Itertools};
use num::{traits::NumAssign, Num};
use rayon::prelude::*;
use rustc_hash::FxHasher;
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    hash::BuildHasherDefault,
    iter::repeat_with,
    mem,
    ops::{Add, Mul, Sub},
};

use crate::{dok_matrix::DokMatrix, is_increasing, is_sorted, Matrix, MatrixError};

pub mod ffi;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrMatrix<T> {
    rows: usize,
    cols: usize,
    vals: Vec<T>,
    indices: Vec<usize>,
    offsets: Vec<usize>,
}

impl<T: Num> CsrMatrix<T> {
    pub fn row_nnz_freq(&self) -> BTreeMap<usize, usize> {
        let mut freq = BTreeMap::new();
        for (a, b) in self.offsets.iter().copied().tuple_windows() {
            *freq.entry(b - a).or_insert(0) += 1;
        }
        freq
    }

    fn get_row_entries(&self, i: usize) -> (&[usize], &[T]) {
        let (j, k) = (self.offsets[i], self.offsets[i + 1]);
        (&self.indices[j..k], &self.vals[j..k])
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

    pub fn invariants(&self) -> bool {
        self.invariant1()
            && self.invariant2()
            && self.invariant3()
            && self.invariant4()
            && self.invariant5()
            && self.invariant6()
    }

    fn invariant1(&self) -> bool {
        self.indices.len() == self.vals.len()
    }

    fn invariant2(&self) -> bool {
        self.offsets.len() == self.rows + 1 && self.offsets[self.rows] == self.indices.len()
    }

    fn invariant3(&self) -> bool {
        is_sorted(&self.offsets)
    }

    fn invariant4(&self) -> bool {
        self.rows > 0 && self.cols > 0
    }

    fn invariant5(&self) -> bool {
        self.indices.iter().all(|c| (0..self.cols).contains(c))
    }

    fn invariant6(&self) -> bool {
        self.offsets
            .iter()
            .copied()
            .tuple_windows()
            .all(|(a, b)| is_increasing(&self.indices[a..b]))
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn nnz(&self) -> usize {
        self.indices.len()
    }

    fn set_element(&mut self, (i, j): (usize, usize), t: T) -> Option<T> {
        assert!(i < self.rows && j < self.cols, "position not in bounds");

        match self.get_row_entries(i).0.binary_search(&j) {
            Ok(pos) => {
                let pos = self.offsets[i] + pos;
                Some(mem::replace(&mut self.vals[pos], t))
            }
            Err(pos) => {
                let pos = self.offsets[i] + pos;
                self.vals.insert(pos, t);
                self.indices.insert(pos, j);
                for m in i + 1..=self.rows {
                    self.offsets[m] += 1;
                }
                None
            }
        }
    }

    fn new(rows: usize, cols: usize) -> Result<CsrMatrix<T>, MatrixError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::HasZeroDimension);
        }
        let capacity = 1000.min(rows * cols / 5);
        Ok(CsrMatrix {
            rows,
            cols,
            vals: Vec::with_capacity(capacity),
            indices: Vec::with_capacity(capacity),
            offsets: vec![0; rows + 1],
        })
    }

    fn new_square(n: usize) -> Result<CsrMatrix<T>, MatrixError> {
        Self::new(n, n)
    }

    fn apply_elementwise<F>(mut self, mut rhs: Self, f: &F) -> Self
    where
        F: Fn(T, T) -> T,
    {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "matrices must have identical dimensions"
        );
        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], vec![0]);

        for ((a, b), (c, d)) in self
            .offsets
            .iter()
            .copied()
            .tuple_windows()
            .zip(rhs.offsets.iter().copied().tuple_windows())
        {
            let (mut rcidx, mut rvals) = self.indices[a..b]
                .iter()
                .copied()
                .zip(self.vals.splice(a..b, repeat_with(T::zero).take(b - a)))
                .merge_join_by(
                    rhs.indices[c..d]
                        .iter()
                        .copied()
                        .zip(rhs.vals.splice(c..d, repeat_with(T::zero).take(d - c))),
                    |(c1, _), (c2, _)| c1.cmp(c2),
                )
                .map(|eob| match eob {
                    itertools::EitherOrBoth::Both((c, t1), (_, t2)) => (c, f(t1, t2)),
                    itertools::EitherOrBoth::Left((c, t)) => (c, f(t, T::zero())),
                    itertools::EitherOrBoth::Right((c, t)) => (c, f(T::zero(), t)),
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
            indices: cidx,
            offsets: ridx,
        }
    }
}

impl <T:Num> CsrMatrix<T> {
    fn transpose(mut self) -> CsrMatrix<T> {
        let mut new = CsrMatrix::new(self.cols, self.rows).unwrap();
        for (j, i) in iproduct!(0..self.cols, 0..self.rows) {
            if let Some(t) = self.set_element((i, j), T::zero()) {
                new.set_element((j, i), t);
            }
        }
        new
    }

    fn identity(n: usize) -> Result<Self, MatrixError> {
        if n == 0 {
            return Err(MatrixError::HasZeroDimension);
        }
        Ok(CsrMatrix {
            rows: n,
            cols: n,
            vals: repeat_with(T::one).take(n).collect(),
            indices: (0..n).collect(),
            offsets: (0..=n).collect(),
        })
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

    fn new(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        Self::new(rows, cols)
    }

    fn new_square(n: usize) -> Result<Self, MatrixError> {
        Self::new_square(n)
    }

    fn rows(&self) -> usize {
        self.rows()
    }

    fn cols(&self) -> usize {
        self.cols()
    }

    fn nnz(&self) -> usize {
        self.nnz()
    }

    fn set_element(&mut self, pos: (usize, usize), t: T) -> Option<T> {
        self.set_element(pos, t)
    }

    fn identity(n: usize) -> Result<Self, MatrixError> {
        Self::identity(n)
    }

    fn transpose(self) -> Self {
        self.transpose()
    }
}

impl<T: NumAssign + Clone + Send + Sync> CsrMatrix<T> {
    pub fn mul_hash(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows, "LHS cols != RHS rows");

        let mut rows: Vec<(Vec<usize>, Vec<T>)> = Vec::new();
        self.offsets
            .par_iter()
            .zip(self.offsets.par_iter().skip(1))
            .map(|(&a, &b)| {
                let capacity = self.indices[a..b]
                    .iter()
                    .map(|&k| rhs.get_row_entries(k).0.len())
                    .sum();
                let mut row = HashMap::with_capacity_and_hasher(
                    capacity,
                    BuildHasherDefault::<FxHasher>::default(),
                );

                self.indices[a..b]
                    .iter()
                    .zip(self.vals[a..b].iter())
                    .flat_map(|(&k, t)| {
                        let (rcidx, rvals) = rhs.get_row_entries(k);
                        rcidx
                            .iter()
                            .zip(rvals.iter().map(move |t1| t.clone() * t1.clone()))
                    })
                    .for_each(|(&j, t)| {
                        let entry = row.entry(j).or_insert_with(T::zero);
                        *entry += t;
                    });

                // let mut row = row.into_iter().collect::<Vec<_>>();
                // row.par_sort_unstable_by_key(|(c, _)| *c);
                row.into_iter().unzip()
            })
            .collect_into_vec(&mut rows);
        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], vec![0]);
        for (rcidx, rvals) in rows {
            cidx.extend(rcidx);
            vals.extend(rvals);
            ridx.push(cidx.len());
        }
        CsrMatrix {
            rows: self.rows,
            cols: rhs.cols,
            vals,
            indices: cidx,
            offsets: ridx,
        }
    }

    pub fn mul_btree(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows, "LHS cols != RHS rows");

        let mut rows: Vec<(Vec<usize>, Vec<T>)> = Vec::new();
        self.offsets
            .par_iter()
            .zip(self.offsets.par_iter().skip(1))
            .map(|(&a, &b)| {
                let mut row = BTreeMap::new();

                self.indices[a..b]
                    .iter()
                    .zip(self.vals[a..b].iter())
                    .for_each(|(&k, t)| {
                        let (rcidx, rvals) = rhs.get_row_entries(k);
                        rcidx.iter().zip(rvals.iter()).for_each(|(&j, t1)| {
                            let entry = row.entry(j).or_insert_with(T::zero);
                            *entry += t.clone() * t1.clone();
                        });
                    });

                row.into_iter().unzip()
            })
            .collect_into_vec(&mut rows);
        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], vec![0]);
        for (rcidx, rvals) in rows {
            cidx.extend(rcidx);
            vals.extend(rvals);
            ridx.push(cidx.len());
        }
        CsrMatrix {
            rows: self.rows,
            cols: rhs.cols,
            vals,
            indices: cidx,
            offsets: ridx,
        }
    }

    pub fn mul_heap(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows, "LHS cols != RHS rows");

        let mut rows: Vec<(Vec<usize>, Vec<T>)> = Vec::new();
        self.offsets
            .par_iter()
            .zip(self.offsets.par_iter().skip(1))
            .map(|(&a, &b)| {
                self.indices[a..b]
                    .iter()
                    .zip(self.vals[a..b].iter())
                    .map(|(&k, t)| {
                        let (rcidx, rvals) = rhs.get_row_entries(k);
                        rcidx
                            .iter()
                            .copied()
                            .zip(rvals.iter().map(move |t1| t.clone() * t1.clone()))
                    })
                    .kmerge_by(|e1, e2| e1.0 < e2.0)
                    .fold((vec![], vec![]), |(mut rcidx, mut rvals), (c, t)| {
                        match (rcidx.last(), rvals.last_mut()) {
                            (Some(&c1), Some(t1)) if c == c1 => {
                                *t1 += t;
                            }
                            _ => {
                                rcidx.push(c);
                                rvals.push(t);
                            }
                        }
                        (rcidx, rvals)
                    })
            })
            .collect_into_vec(&mut rows);
        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], vec![0]);
        for (rcidx, rvals) in rows {
            cidx.extend(rcidx);
            vals.extend(rvals);
            ridx.push(cidx.len());
        }
        CsrMatrix {
            rows: self.rows,
            cols: rhs.cols,
            vals,
            indices: cidx,
            offsets: ridx,
        }
    }

    pub fn mul_hash1(&self, rhs: &Self) -> Self {
        let _offset = self.rows_to_threads(rhs);
        todo!()
    }

    fn rows_to_threads(&self, rhs: &Self) -> Vec<usize> {
        let flop: Vec<usize> = self
            .offsets
            .par_iter()
            .zip(self.offsets.par_iter().skip(1))
            .map(|(&a, &b)| {
                self.indices[a..b]
                    .iter()
                    .map(|&k| rhs.get_row_entries(k).0.len())
                    .sum()
            })
            .collect();
        // TODO: make prefix sum parallel
        let flop_ps: Vec<_> = flop
            .into_iter()
            .scan(0, |sum, x| {
                *sum += x;
                Some(*sum)
            })
            .collect();
        let sum_flop = flop_ps.last().copied().unwrap();
        // TODO: not sure what the equivalent of omp_get_max_threads is
        let tnum = num_cpus::get();
        let ave_flop = sum_flop / tnum;
        let mut v = vec![0];
        v.par_extend(
            (1..tnum)
                .into_par_iter()
                .map(|tid| flop_ps.partition_point(|&x| x >= ave_flop * tid)),
        );
        v.push(self.rows);
        v
    }
}

impl<T: Num> Add for CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.apply_elementwise(rhs, &T::add)
    }
}

impl<T: Num> Sub for CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.apply_elementwise(rhs, &T::sub)
    }
}

impl<T: NumAssign + Clone + Send + Sync> Mul for &CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_hash(rhs)
    }
}

impl<T: Num + Clone> From<DokMatrix<T>> for CsrMatrix<T> {
    fn from(old: DokMatrix<T>) -> Self {
        let (rows, cols) = (old.rows(), old.cols());
        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], vec![0]);
        // note that (i, j) is iterated in lexicographic order
        for ((i, j), t) in old {
            vals.push(t);
            cidx.push(j);
            if ridx.get(i + 1).is_none() {
                let &k = ridx.last().unwrap();
                for _ in ridx.len()..=i + 1 {
                    ridx.push(k);
                }
            }
            ridx[i + 1] += 1;
        }
        let &k = ridx.last().unwrap();
        for _ in ridx.len()..=rows {
            ridx.push(k);
        }
        CsrMatrix {
            rows,
            cols,
            vals,
            indices: cidx,
            offsets: ridx,
        }
    }
}

#[cfg(test)]
impl<T: proptest::arbitrary::Arbitrary + Num> CsrMatrix<T> {
    pub(crate) fn arb_fixed_size_matrix(
        rows: usize,
        cols: usize,
    ) -> impl proptest::strategy::Strategy<Value = CsrMatrix<T>> {
        use proptest::prelude::*;
        repeat_with(|| proptest::sample::subsequence((0..cols).collect::<Vec<_>>(), 0..=cols))
            .take(rows)
            .collect::<Vec<_>>()
            .prop_flat_map(move |cidx| {
                let (mut cidx_flattened, mut ridx) = (vec![], vec![0]);
                for mut rcidx in cidx {
                    ridx.push(ridx.last().unwrap() + rcidx.len());
                    cidx_flattened.append(&mut rcidx);
                }
                repeat_with(|| T::arbitrary())
                    .take(cidx_flattened.len())
                    .collect::<Vec<_>>()
                    .prop_map(move |vals| CsrMatrix {
                        rows,
                        cols,
                        vals,
                        indices: cidx_flattened.clone(),
                        offsets: ridx.clone(),
                    })
            })
    }

    pub fn arb_matrix() -> impl proptest::strategy::Strategy<Value = Self> {
        crate::proptest::arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}
