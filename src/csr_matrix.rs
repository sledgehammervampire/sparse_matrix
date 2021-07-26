use cap_rand::prelude::*;
use core::slice;
use hashbrown::HashMap;
use itertools::{iproduct, Itertools};
use num::{traits::NumAssign, Num};
use rayon::prelude::*;
use rustc_hash::FxHasher;
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fmt::Debug,
    hash::BuildHasherDefault,
    iter::repeat_with,
    mem::{self, MaybeUninit},
    ops::{Add, Mul, Sub},
    vec,
};

use crate::{all_distinct, dok_matrix::DokMatrix, is_increasing, is_sorted, Matrix, MatrixError};

#[cfg(feature = "mkl")]
mod mkl;

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

impl<T: Num + Clone> From<DokMatrix<T>> for CsrMatrix<T, true> {
    fn from(old: DokMatrix<T>) -> Self {
        let (rows, cols) = (old.rows(), old.cols());
        let entries: Vec<_> = old.entries.into_iter().collect();
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

impl<T: Num + Clone> CsrMatrix<T, false> {
    pub fn from_dok(old: DokMatrix<T>, rng: &mut CapRng) -> Self {
        let (rows, cols) = (old.rows(), old.cols());
        let mut entries: Vec<_> = old.entries.into_iter().collect();
        entries.shuffle(rng);
        entries.sort_unstable_by_key(|((i, _), _)| *i);
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

impl<T: NumAssign + Copy + Send + Sync, const IS_SORTED: bool> CsrMatrix<T, IS_SORTED> {
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
                    .for_each(|(&k, &t)| {
                        let (rcidx, rvals) = rhs.get_row_entries(k);
                        rcidx
                            .iter()
                            .zip(rvals.iter().map(move |&t1| t * t1))
                            .for_each(|(&j, t1)| {
                                row.entry(j)
                                    .and_modify(|t| {
                                        *t += t1;
                                    })
                                    .or_insert(t1);
                            });
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
                    .for_each(|(&k, &t)| {
                        let (rcidx, rvals) = rhs.get_row_entries(k);
                        rcidx.iter().zip(rvals.iter()).for_each(|(&j, &t1)| {
                            let entry = row.entry(j).or_insert_with(T::zero);
                            *entry += t * t1;
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

    fn rows_to_threads<const B: bool>(
        &self,
        rhs: &CsrMatrix<T, B>,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let row_nz: Vec<usize> = self
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
        let ps_row_nz: Vec<_> = std::iter::once(0)
            .chain(row_nz.iter().copied().scan(0, |sum, x| {
                *sum += x;
                Some(*sum)
            }))
            .collect();
        let total_intprod = ps_row_nz.last().copied().unwrap();
        // TODO: not sure what the equivalent of omp_get_max_threads is
        let tnum = num_cpus::get();
        let average_intprod = total_intprod / tnum;
        let mut rows_offset = vec![0];
        rows_offset.par_extend(
            (1..tnum)
                .into_par_iter()
                .map(|tid| ps_row_nz.partition_point(|&x| x < average_intprod * tid)),
        );
        rows_offset.push(self.rows);
        (row_nz, ps_row_nz, rows_offset)
    }

    // based off pengdada/mtspgemmlib
    pub fn mul_hash2<const B: bool, const SORTED_OUTPUT: bool>(
        &self,
        rhs: &CsrMatrix<T, B>,
    ) -> CsrMatrix<T, SORTED_OUTPUT> {
        let (mut row_nz, _, rows_offset) = self.rows_to_threads(rhs);
        // mutate row_nz or alloc new vec for new row_nz?
        crossbeam::scope(|s| {
            let mut rest = &mut row_nz[..];
            for (tlo, thi) in rows_offset.iter().copied().tuple_windows() {
                let (s1, s2) = rest.split_at_mut(thi - tlo);
                rest = s2;
                s.spawn(move |_| {
                    let capacity = s1.iter().copied().max().unwrap_or(0);
                    let mut hs = crate::hash_set::HashSet::with_capacity(capacity);
                    for ((row_start, row_end), row_nz) in self.offsets[tlo..=thi]
                        .iter()
                        .copied()
                        .tuple_windows()
                        .zip(&mut *s1)
                    {
                        if *row_nz == 0 {
                            continue;
                        }
                        hs.shrink_to(*row_nz);
                        for &k in &self.indices[row_start..row_end] {
                            for &j in rhs.get_row_entries(k).0 {
                                hs.insert(j);
                            }
                        }
                        *row_nz = hs.drain().count();
                    }
                });
            }
        })
        .unwrap();
        let offsets: Vec<_> = std::iter::once(0)
            .chain(row_nz.iter().copied().scan(0, |sum, x| {
                *sum += x;
                Some(*sum)
            }))
            .collect();
        let nnz = *offsets.last().unwrap();
        let (mut indices, mut vals) = (Vec::with_capacity(nnz), Vec::with_capacity(nnz));
        crossbeam::scope(|s| {
            // SAFETY: indices_rest is the allocated capacity of indices
            // SAFETY: vals_rest is the allocated capacity of vals
            let (mut indices_rest, mut vals_rest) = unsafe {
                (
                    slice::from_raw_parts_mut(
                        indices.as_mut_ptr() as *mut MaybeUninit<usize>,
                        indices.capacity(),
                    ),
                    slice::from_raw_parts_mut(
                        vals.as_mut_ptr() as *mut MaybeUninit<T>,
                        vals.capacity(),
                    ),
                )
            };
            for (tlo, thi) in rows_offset.iter().copied().tuple_windows() {
                let trow_nz = &row_nz[tlo..thi];
                let (tindices, s2) = indices_rest.split_at_mut(offsets[thi] - offsets[tlo]);
                indices_rest = s2;
                let (tvals, s2) = vals_rest.split_at_mut(offsets[thi] - offsets[tlo]);
                vals_rest = s2;
                s.spawn(move |_| {
                    let capacity = trow_nz.iter().copied().max().unwrap_or(0);
                    let mut hm = crate::hash_map::HashMap::with_capacity(capacity);
                    let mut curr = 0;
                    for ((row_start, row_end), row_nz) in self.offsets[tlo..=thi]
                        .iter()
                        .copied()
                        .tuple_windows()
                        .zip(trow_nz)
                    {
                        if *row_nz == 0 {
                            continue;
                        }
                        hm.shrink_to(*row_nz);
                        for (k, &t) in self.indices[row_start..row_end]
                            .iter()
                            .copied()
                            .zip(&self.vals[row_start..row_end])
                        {
                            let (rcidx, rvals) = rhs.get_row_entries(k);
                            for (j, t1) in rcidx.iter().copied().zip(rvals.iter().map(|&t1| t * t1))
                            {
                                hm.entry(j)
                                    .and_modify(|t| {
                                        *t += t1;
                                    })
                                    .or_insert(t1);
                            }
                        }
                        if SORTED_OUTPUT {
                            let mut row: Vec<_> = hm.drain().collect();
                            row.sort_unstable_by_key(|(c, _)| *c);
                            for (c, t) in row {
                                // SAFETY: tindices of each thread are disjoint slices of indices
                                // SAFETY: tvals of each thread are disjoint slices of vals
                                unsafe {
                                    tindices[curr].as_mut_ptr().write(c);
                                    tvals[curr].as_mut_ptr().write(t);
                                }
                                curr += 1;
                            }
                        } else {
                            for (c, t) in hm.drain() {
                                // SAFETY: tindices of each thread are disjoint slices of indices
                                // SAFETY: tvals of each thread are disjoint slices of vals
                                unsafe {
                                    tindices[curr].as_mut_ptr().write(c);
                                    tvals[curr].as_mut_ptr().write(t);
                                }
                                curr += 1;
                            }
                        }
                    }
                });
            }
        })
        .unwrap();
        // SAFETY: rows_offsets induces a partition of the allocated capacity of indices and vals
        // given by (rows_start, rows_end)
        unsafe {
            indices.set_len(indices.capacity());
            vals.set_len(vals.capacity());
        }
        CsrMatrix {
            rows: self.rows,
            cols: rhs.cols,
            indices,
            vals,
            offsets,
        }
    }
}

// impl<T: NumAssign + Copy + Send + Sync, const IS_SORTED: bool> CsrMatrix<T, IS_SORTED> {
//     pub fn mul_esc<const B: bool>(&self, rhs: &CsrMatrix<T, B>) -> CsrMatrix<T, true> {
//         #[derive(Clone, Copy)]
//         struct Entry<T> {
//             col: usize,
//             t: T,
//         }
//         impl<T> PartialEq for Entry<T> {
//             fn eq(&self, other: &Self) -> bool {
//                 self.col == other.col
//             }
//         }
//         impl<T> PartialOrd for Entry<T> {
//             fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//                 self.col.partial_cmp(&other.col)
//             }
//         }
//         impl<T: Copy + Send + Sync> Radixable<usize> for Entry<T> {
//             type Key = usize;

//             fn key(&self) -> Self::Key {
//                 self.col
//             }
//         }

//         let (flop, flop_ps, offset) = self.rows_to_threads(rhs);
//         crossbeam::scope(|s| {
//             let mut handles = vec![];
//             for (tlo, thi) in offset.iter().copied().tuple_windows() {
//                 let flop = &flop;
//                 let flop_ps = &flop_ps;
//                 let offset = &offset;
//                 handles.push(s.spawn(move |_| {
//                     let tflop = flop_ps[thi] - flop_ps[tlo];
//                     let (mut indices_part, mut vals_part, mut lens_part) = (
//                         Vec::with_capacity(tflop),
//                         Vec::with_capacity(tflop),
//                         Vec::with_capacity(thi - tlo),
//                     );
//                     for (i, (row_start, row_end)) in
//                         (tlo..thi).zip(self.offsets[tlo..=thi].iter().copied().tuple_windows())
//                     {
//                         let mut entries = Vec::with_capacity(flop[i]);
//                         entries.extend(
//                             self.indices[row_start..row_end]
//                                 .iter()
//                                 .copied()
//                                 .zip(&self.vals[row_start..row_end])
//                                 .flat_map(|(k, &t)| {
//                                     let (rcidx, rvals) = rhs.get_row_entries(k);
//                                     rcidx
//                                         .iter()
//                                         .copied()
//                                         .zip(rvals.iter().map(move |&t1| t * t1))
//                                         .map(|(col, t)| Entry { col, t })
//                                 }),
//                         );
//                         entries.voracious_mt_sort(offset.len() - 1);
//                         let entries =
//                             entries
//                                 .into_iter()
//                                 .fold(vec![], |mut v, Entry { col, t: t1 }| {
//                                     match v.last_mut() {
//                                         Some((c, t)) if *c == col => {
//                                             *t += t1;
//                                         }
//                                         _ => {
//                                             v.push((col, t1));
//                                         }
//                                     }
//                                     v
//                                 });

//                         lens_part.push(entries.len());
//                         for (c, t) in entries {
//                             indices_part.push(c);
//                             vals_part.push(t);
//                         }
//                     }

//                     (indices_part, vals_part, lens_part)
//                 }));
//             }
//             let (mut indices, mut vals, mut lens) = (vec![], vec![], vec![]);
//             for h in handles {
//                 let (indices_part, vals_part, lens_part) = h.join().unwrap();
//                 indices.extend(indices_part);
//                 vals.extend(vals_part);
//                 lens.extend(lens_part);
//             }
//             let offsets = std::iter::once(0)
//                 .chain(lens.into_iter().scan(0, |sum, x| {
//                     *sum += x;
//                     Some(*sum)
//                 }))
//                 .collect();
//             CsrMatrix {
//                 rows: self.rows,
//                 cols: rhs.cols,
//                 indices,
//                 vals,
//                 offsets,
//             }
//         })
//         .unwrap()
//     }
// }

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
                let (flop, flop_ps, offsets) = m1.rows_to_threads(&m2);
                prop_assert!(flop.len() == m1.rows());
                prop_assert!(flop_ps.len() == m1.rows() + 1);
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

impl<T: NumAssign + Copy + Send + Sync, const IS_SORTED: bool> Mul for &CsrMatrix<T, IS_SORTED> {
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
