#![feature(is_sorted)]
#![feature(type_alias_impl_trait)]
#![feature(vec_spare_capacity)]
#![feature(int_roundings)]
#![deny(clippy::disallowed_method)]

#[cfg(feature = "test")]
use cap_rand::prelude::*;
use itertools::{iproduct, Itertools};
use num_traits::{Num, NumAssign};
use spam_dok::DokMatrix;
use spam_matrix::{IndexError, Matrix};
use std::{
    fmt::Debug,
    iter::repeat_with,
    mem,
    num::NonZeroUsize,
    ops::{Add, Mul, Sub},
    vec,
};

#[cfg(feature = "mkl")]
pub mod mkl;
mod mul_hash;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrMatrix<T, const IS_SORTED: bool> {
    rows: NonZeroUsize,
    cols: NonZeroUsize,
    vals: Vec<T>,
    indices: Vec<usize>,
    offsets: Vec<usize>,
}

impl<T: Num, const IS_SORTED: bool> CsrMatrix<T, IS_SORTED> {
    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        self.offsets
            .iter()
            .copied()
            .tuple_windows()
            .enumerate()
            .flat_map(move |(r, (rlo, rhi))| {
                let (indices, vals) = (&self.indices[rlo..rhi], &self.vals[rlo..rhi]);
                indices.iter().map(move |c| (r, *c)).zip(vals.iter())
            })
    }

    fn invariant1(&self) -> bool {
        self.indices.len() == self.vals.len()
    }

    fn invariant2(&self) -> bool {
        self.offsets.len() == self.rows.get().checked_add(1).unwrap()
    }

    fn invariant3(&self) -> bool {
        self.offsets.is_sorted()
    }

    fn invariant4(&self) -> bool {
        self.offsets[self.rows.get()] == self.indices.len()
    }

    fn invariant5(&self) -> bool {
        self.indices
            .iter()
            .all(|c| (0..self.cols.get()).contains(c))
    }

    fn invariant6(&self) -> bool {
        self.offsets.iter().copied().tuple_windows().all(|(a, b)| {
            if IS_SORTED {
                crate::is_increasing(&self.indices[a..b])
            } else {
                crate::all_distinct(&self.indices[a..b])
            }
        })
    }

    fn invariant7(&self) -> bool {
        self.offsets[0] == 0
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
                let mut row: std::collections::HashMap<_, _> = self.indices[a..b]
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

impl<T: Num, const IS_SORTED: bool> Matrix<T> for CsrMatrix<T, IS_SORTED> {
    fn invariants(&self) -> bool {
        self.invariant1()
            && self.invariant2()
            && self.invariant3()
            && self.invariant4()
            && self.invariant5()
            && self.invariant6()
            && self.invariant7()
    }

    fn new((rows, cols): (NonZeroUsize, NonZeroUsize)) -> Self {
        let capacity = 1000.min(rows.get() * cols.get() / 5);
        CsrMatrix {
            rows,
            cols,
            vals: Vec::with_capacity(capacity),
            indices: Vec::with_capacity(capacity),
            offsets: vec![0; rows.get() + 1],
        }
    }

    fn new_square(n: NonZeroUsize) -> Self {
        Self::new((n, n))
    }

    fn identity(n: NonZeroUsize) -> Self {
        CsrMatrix {
            rows: n,
            cols: n,
            vals: repeat_with(T::one).take(n.get()).collect(),
            indices: (0..n.get()).collect(),
            offsets: (0..=n.get()).collect(),
        }
    }

    fn rows(&self) -> NonZeroUsize {
        self.rows
    }

    fn cols(&self) -> NonZeroUsize {
        self.cols
    }

    fn nnz(&self) -> usize {
        self.indices.len()
    }

    fn get_element(&self, (i, j): (usize, usize)) -> Result<Option<&T>, IndexError> {
        if !(i < self.rows.get() && j < self.cols.get()) {
            return Err(IndexError);
        }

        let (cidx, vals) = {
            let (rlo, rhi) = (self.offsets[i], self.offsets[i + 1]);
            (&self.indices[rlo..rhi], &self.vals[rlo..rhi])
        };
        Ok(if IS_SORTED {
            cidx.binary_search(&j).ok().map(|k| &vals[k])
        } else {
            cidx.iter().position(|k| *k == j).map(|k| &vals[k])
        })
    }

    fn set_element(&mut self, (i, j): (usize, usize), t: T) -> Result<Option<T>, IndexError> {
        if !(i < self.rows.get() && j < self.cols.get()) {
            return Err(IndexError);
        }

        let (rlo, rhi) = (self.offsets[i], self.offsets[i + 1]);
        if IS_SORTED {
            match self.indices[rlo..rhi].binary_search(&j) {
                Ok(pos) => {
                    let pos = self.offsets[i] + pos;
                    Ok(Some(mem::replace(&mut self.vals[pos], t)))
                }
                Err(pos) => {
                    let pos = self.offsets[i] + pos;
                    self.vals.insert(pos, t);
                    self.indices.insert(pos, j);
                    for m in i + 1..=self.rows.get() {
                        self.offsets[m] += 1;
                    }
                    Ok(None)
                }
            }
        } else {
            match self.indices[rlo..rhi].iter().position(|k| *k == j) {
                Some(pos) => {
                    let pos = self.offsets[i] + pos;
                    Ok(Some(mem::replace(&mut self.vals[pos], t)))
                }
                None => {
                    let pos = self.offsets[i + 1];
                    self.vals.insert(pos, t);
                    self.indices.insert(pos, j);
                    for m in i + 1..=self.rows.get() {
                        self.offsets[m] += 1;
                    }
                    Ok(None)
                }
            }
        }
    }

    fn transpose(mut self) -> Self {
        let mut new = CsrMatrix::new((self.cols, self.rows));
        for (j, i) in iproduct!(0..self.cols.get(), 0..self.rows.get()) {
            if let Some(t) = self.set_element((i, j), T::zero()).unwrap() {
                new.set_element((j, i), t).unwrap();
            }
        }
        new
    }
}

fn checked_inclusive_scan(v: &[usize]) -> Vec<usize> {
    std::iter::once(0)
        .chain(v.iter().copied().scan(0usize, |sum, x| {
            *sum = sum.checked_add(x).unwrap();
            Some(*sum)
        }))
        .collect()
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

impl<T: NumAssign + Copy + Send + Sync, const B: bool> Mul for &CsrMatrix<T, B> {
    type Output = CsrMatrix<T, false>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_hash(rhs)
    }
}

fn is_increasing<T: Ord>(s: &[T]) -> bool {
    let mut max = None;
    for i in s {
        if max.map_or(true, |k| k < i) {
            max = Some(i);
        } else {
            return false;
        }
    }
    true
}

fn all_distinct<T: std::hash::Hash + Eq>(s: &[T]) -> bool {
    s.iter().collect::<std::collections::HashSet<_>>().len() == s.len()
}

impl<T: Num> From<DokMatrix<T>> for CsrMatrix<T, true> {
    fn from(old: DokMatrix<T>) -> Self {
        let (rows, cols) = (old.rows(), old.cols());
        let entries: Vec<_> = old.into_iter().collect();
        let (mut vals, mut indices, mut offsets) = (vec![], vec![], vec![]);
        for ((i, j), t) in entries {
            offsets.extend(std::iter::repeat(vals.len()).take(i + 1 - offsets.len()));
            vals.push(t);
            indices.push(j);
        }
        offsets.extend(std::iter::repeat(vals.len()).take(rows.get() + 1 - offsets.len()));
        CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        }
    }
}

#[cfg(feature = "test")]
impl<T: Num> CsrMatrix<T, false> {
    pub fn from_dok(old: DokMatrix<T>, rng: &mut CapRng) -> Self {
        let (rows, cols) = (old.rows(), old.cols());
        let mut entries: Vec<_> = old.into_iter().collect();
        entries.shuffle(rng);
        entries.sort_unstable_by_key(|((i, _), _)| *i);
        let (mut vals, mut indices, mut offsets) = (vec![], vec![], vec![]);
        for ((i, j), t) in entries {
            offsets.extend(std::iter::repeat(vals.len()).take(i + 1 - offsets.len()));
            vals.push(t);
            indices.push(j);
        }
        offsets.extend(std::iter::repeat(vals.len()).take(rows.get() + 1 - offsets.len()));
        CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        }
    }
}

impl<T, const B: bool> IntoIterator for CsrMatrix<T, B> {
    type Item = ((usize, usize), T);

    type IntoIter = impl Iterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let offsets = self.offsets;
        self.indices
            .into_iter()
            .zip(self.vals.into_iter())
            .enumerate()
            .map(move |(i, (c, t))| ((offsets.partition_point(|x| *x <= i) - 1, c), t))
    }
}

#[cfg(feature = "test")]
impl<T: Num, const IS_SORTED: bool> From<CsrMatrix<T, IS_SORTED>> for DokMatrix<T> {
    fn from(old: CsrMatrix<T, IS_SORTED>) -> Self {
        let mut m = DokMatrix::new((old.rows(), old.cols()));
        for (i, t) in old.into_iter() {
            m.set_element(i, t).unwrap();
        }
        m
    }
}

#[macro_export]
macro_rules! gen_bench_mul {
    ($f:ident) => {
        fn bench_mul<const OUTPUT_SORTED: bool>(dir: cap_std::fs::Dir) -> anyhow::Result<()> {
            use cap_std::fs::DirEntry;
            use criterion::Criterion;
            use num_traits::NumAssign;
            use spam_csr::CsrMatrix;
            use spam_dok::{parse_matrix_market, DokMatrix, MatrixType};
            use std::io::Read;

            fn inner<T: NumAssign + Send + Sync + Copy, const B: bool>(
                m: DokMatrix<T>,
                criterion: &mut Criterion,
                entry: DirEntry,
            ) {
                let m = CsrMatrix::from(m);
                criterion.bench_function(
                    &format!("bench {} {:?}", stringify!($f), entry.file_name()),
                    |b| {
                        b.iter(|| {
                            let _: CsrMatrix<_, B> = m.$f(&m);
                        });
                    },
                );
            }

            let mut criterion = Criterion::default().configure_from_args();
            for entry in dir.entries()? {
                let entry = entry?;
                let mut input = String::new();
                entry.open()?.read_to_string(&mut input)?;
                match parse_matrix_market::<i64, f64>(&input).unwrap() {
                    MatrixType::Integer(m) => {
                        inner::<_, OUTPUT_SORTED>(m, &mut criterion, entry);
                    }
                    MatrixType::Real(m) => {
                        inner::<_, OUTPUT_SORTED>(m, &mut criterion, entry);
                    }
                    MatrixType::Complex(m) => {
                        inner::<_, OUTPUT_SORTED>(m, &mut criterion, entry);
                    }
                }
            }
            criterion.final_summary();
            Ok(())
        }
    };
}

#[macro_export]
macro_rules! gen_mul_main {
    ($f:ident) => {
        fn mul_main(dir: cap_std::fs::Dir) -> anyhow::Result<()> {
            use spam_dok::{parse_matrix_market, MatrixType};
            use std::io::Read;

            for entry in dir.entries()? {
                let entry = entry?;
                dbg!(entry.file_name());
                let mut input = String::new();
                entry.open()?.read_to_string(&mut input)?;
                match parse_matrix_market::<i64, f64>(&input).unwrap() {
                    MatrixType::Integer(_) => {}
                    MatrixType::Real(m) => {
                        $f(m);
                    }
                    MatrixType::Complex(m) => {
                        $f(m);
                    }
                }
            }
            Ok(())
        }
    };
}
