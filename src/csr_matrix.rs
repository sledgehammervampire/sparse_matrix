use itertools::{iproduct, Itertools};
use num::{traits::NumAssign, Num};
use rand::prelude::SliceRandom;
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
    vec,
};

use crate::{all_distinct, dok_matrix::DokMatrix, is_increasing, is_sorted, Matrix, MatrixError};

pub mod ffi;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrMatrix<T, const IS_SORTED: bool> {
    rows: usize,
    cols: usize,
    vals: Vec<T>,
    indices: Vec<usize>,
    offsets: Vec<usize>,
}

impl<T: Num, const IS_SORTED: bool> CsrMatrix<T, IS_SORTED> {
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
        self.offsets.iter().copied().tuple_windows().all(|(a, b)| {
            if IS_SORTED {
                is_increasing(&self.indices[a..b])
            } else {
                all_distinct(&self.indices[a..b])
            }
        })
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

        if IS_SORTED {
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
        } else {
            match self.get_row_entries(i).0.iter().position(|k| *k == j) {
                Some(pos) => {
                    let pos = self.offsets[i] + pos;
                    Some(mem::replace(&mut self.vals[pos], t))
                }
                None => {
                    let pos = self.offsets[i + 1];
                    self.vals.insert(pos, t);
                    self.indices.insert(pos, j);
                    for m in i + 1..=self.rows {
                        self.offsets[m] += 1;
                    }
                    None
                }
            }
        }
    }

    fn new(rows: usize, cols: usize) -> Result<Self, MatrixError> {
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

    fn new_square(n: usize) -> Result<Self, MatrixError> {
        Self::new(n, n)
    }

    fn transpose(mut self) -> Self {
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
            let (mut rcidx, mut rvals) = if IS_SORTED {
                self.indices[a..b]
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
                    .unzip()
            } else {
                let mut row: HashMap<_, _> = self.indices[a..b]
                    .iter()
                    .copied()
                    .zip(self.vals.splice(a..b, repeat_with(T::zero).take(b - a)))
                    .collect();
                for (c, t1) in rhs.indices[c..d]
                    .iter()
                    .copied()
                    .zip(rhs.vals.splice(c..d, repeat_with(T::zero).take(d - c)))
                {
                    let entry = row.entry(c).or_insert_with(T::zero);
                    *entry = f(mem::replace(entry, T::zero()), t1);
                }
                row.into_iter().unzip()
            };

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

impl<T: Num + Clone, const IS_SORTED: bool> Matrix<T> for CsrMatrix<T, IS_SORTED> {
    fn get_element(&self, (i, j): (usize, usize)) -> Cow<T> {
        assert!(
            (..self.rows).contains(&i) && (..self.cols).contains(&j),
            "values are not in bounds"
        );

        let (cidx, vals) = self.get_row_entries(i);
        if IS_SORTED {
            cidx.binary_search(&j)
                .map_or(Cow::Owned(T::zero()), |k| Cow::Borrowed(&vals[k]))
        } else {
            cidx.iter()
                .position(|k| *k == j)
                .map_or(Cow::Owned(T::zero()), |k| Cow::Borrowed(&vals[k]))
        }
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

impl<T: Num + Clone, const IS_SORTED: bool> From<DokMatrix<T>> for CsrMatrix<T, IS_SORTED> {
    fn from(old: DokMatrix<T>) -> Self {
        let (rows, cols) = (old.rows(), old.cols());
        let mut entries: Vec<_> = old.entries.into_iter().collect();
        if !IS_SORTED {
            let mut rng = rand::thread_rng();
            entries.shuffle(&mut rng);
            entries.sort_unstable_by_key(|((i, _), _)| *i);
        }
        let (mut vals, mut indices, mut offsets) = (vec![], vec![], vec![]);
        for ((i, j), t) in entries {
            offsets.extend(std::iter::repeat(vals.len()).take(i + 1 - offsets.len()));
            vals.push(t);
            indices.push(j);
        }
        offsets.extend(std::iter::repeat(vals.len()).take(rows + 1 - offsets.len()));
        CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        }
    }
}

impl<T: NumAssign + Clone + Send + Sync> CsrMatrix<T, true> {
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
}

impl<T: NumAssign + Clone + Send + Sync, const IS_SORTED: bool> CsrMatrix<T, IS_SORTED> {
    pub fn mul_hash<const B: bool, const B1: bool>(
        &self,
        rhs: &CsrMatrix<T, B>,
    ) -> CsrMatrix<T, B1> {
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

                if B1 {
                    let mut row = row.into_iter().collect::<Vec<_>>();
                    row.par_sort_unstable_by_key(|(c, _)| *c);
                    row.into_iter().unzip()
                } else {
                    row.into_iter().unzip()
                }
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

    pub fn mul_btree<const B: bool>(&self, rhs: &CsrMatrix<T, B>) -> CsrMatrix<T, true> {
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

    pub fn mul_hash1<const B: bool, const B1: bool>(
        &self,
        rhs: &CsrMatrix<T, B>,
    ) -> CsrMatrix<T, B1> {
        let (flop, offset) = self.rows_to_threads(rhs);
        crossbeam::scope(|s| {
            let mut handles = vec![];
            for (tlo, thi) in offset.iter().copied().tuple_windows() {
                let flop = &flop;
                handles.push(s.spawn(move |_| {
                    let (mut indices_part, mut vals_part, mut lens_part) = (vec![], vec![], vec![]);
                    let capacity = flop[tlo..thi]
                        .iter()
                        .copied()
                        .max()
                        .unwrap_or(0)
                        .min(rhs.cols);
                    let mut hm = HashMap::with_capacity_and_hasher(
                        capacity,
                        BuildHasherDefault::<FxHasher>::default(),
                    );
                    for (a, b) in self.offsets[tlo..=thi].iter().copied().tuple_windows() {
                        for (k, t) in self.indices[a..b].iter().copied().zip(&self.vals[a..b]) {
                            let (rcidx, rvals) = rhs.get_row_entries(k);
                            for (j, t1) in rcidx.iter().copied().zip(rvals) {
                                *hm.entry(j).or_insert_with(T::zero) += t.clone() * t1.clone();
                            }
                        }
                        lens_part.push(hm.len());
                        if B1 {
                            let mut row: Vec<_> = hm.drain().collect();
                            row.par_sort_unstable_by_key(|(c, _)| *c);
                            for (c, t) in row {
                                indices_part.push(c);
                                vals_part.push(t);
                            }
                        } else {
                            for (c, t) in hm.drain() {
                                indices_part.push(c);
                                vals_part.push(t);
                            }
                        }
                    }
                    (indices_part, vals_part, lens_part)
                }));
            }
            let (mut indices, mut vals, mut lens) = (vec![], vec![], vec![]);
            for h in handles {
                let (indices_part, vals_part, lens_part) = h.join().unwrap();
                indices.extend(indices_part);
                vals.extend(vals_part);
                lens.extend(lens_part);
            }
            let offsets = std::iter::once(0)
                .chain(lens.into_iter().scan(0, |sum, x| {
                    *sum += x;
                    Some(*sum)
                }))
                .collect();
            CsrMatrix {
                rows: self.rows,
                cols: rhs.cols,
                indices,
                vals,
                offsets,
            }
        })
        .unwrap()
    }

    fn rows_to_threads<const B: bool>(&self, rhs: &CsrMatrix<T, B>) -> (Vec<usize>, Vec<usize>) {
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
            .iter()
            .copied()
            .scan(0, |sum, x| {
                *sum += x;
                Some(*sum)
            })
            .collect();
        let sum_flop = flop_ps.last().copied().unwrap();
        // TODO: not sure what the equivalent of omp_get_max_threads is
        let tnum = num_cpus::get();
        let ave_flop = sum_flop / tnum;
        let mut offset = vec![0];
        offset.par_extend(
            (1..tnum)
                .into_par_iter()
                .map(|tid| flop_ps.partition_point(|&x| x < ave_flop * tid)),
        );
        offset.push(self.rows);
        (flop, offset)
    }
}

#[cfg(test)]
#[test]
fn test_rows_to_threads() {
    use proptest::{prelude::*, test_runner::TestRunner};
    let mut runner = TestRunner::default();
    runner
        .run(
            &crate::proptest::arb_mul_pair::<std::num::Wrapping<i8>, _, _>(
                CsrMatrix::<_, false>::arb_fixed_size_matrix,
            ),
            |crate::MulPair(m1, m2)| {
                let offsets = m1.rows_to_threads(&m2).1;
                prop_assert!(is_sorted(&offsets), "{:?}", offsets);
                prop_assert!(
                    offsets.iter().copied().all(|c| c <= m1.rows()),
                    "{:?}",
                    offsets
                );
                Ok(())
            },
        )
        .unwrap();
}

impl<T: Num, const IS_SORTED: bool> Add for CsrMatrix<T, IS_SORTED> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.apply_elementwise(rhs, &T::add)
    }
}

impl<T: Num, const IS_SORTED: bool> Sub for CsrMatrix<T, IS_SORTED> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.apply_elementwise(rhs, &T::sub)
    }
}

impl<T: NumAssign + Clone + Send + Sync, const IS_SORTED: bool> Mul for &CsrMatrix<T, IS_SORTED> {
    type Output = CsrMatrix<T, IS_SORTED>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_hash(rhs)
    }
}

#[cfg(test)]
impl<T: proptest::arbitrary::Arbitrary + Num> CsrMatrix<T, true> {
    pub(crate) fn arb_fixed_size_matrix(
        rows: usize,
        cols: usize,
    ) -> impl proptest::strategy::Strategy<Value = Self> {
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

#[cfg(test)]
impl<T: proptest::arbitrary::Arbitrary + Num> CsrMatrix<T, false> {
    pub(crate) fn arb_fixed_size_matrix(
        rows: usize,
        cols: usize,
    ) -> impl proptest::strategy::Strategy<Value = Self> {
        use proptest::prelude::*;
        repeat_with(|| proptest::collection::hash_set(0..cols, 0..=cols))
            .take(rows)
            .collect::<Vec<_>>()
            .prop_flat_map(move |cidx| {
                let (mut cidx_flattened, mut ridx) = (vec![], vec![0]);
                for rcidx in cidx {
                    ridx.push(ridx.last().unwrap() + rcidx.len());
                    cidx_flattened.extend(rcidx);
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
