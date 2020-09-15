#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

use num::{traits::NumAssignRef, Num};
use quickcheck::Arbitrary;
use rand::prelude::*;
use rayon::prelude::*;
use std::{
    borrow::Cow,
    collections::HashMap,
    collections::{BTreeMap, BTreeSet},
    iter::{repeat_with, Sum},
    ops::{AddAssign, MulAssign},
};

#[cfg(test)]
mod test;

// we use an extra indirection for vals and cidx to simplify the implementation
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CSRMatrix<T: Num + Clone> {
    rows: usize,
    cols: usize,
    // for v in vals, vals != 0
    vals: Vec<T>,
    cidx: Vec<usize>,
    // for s in ridx.values(), s.len > 0
    ridx: BTreeMap<usize, Slice>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Slice {
    start: usize,
    len: usize,
}
impl<T: Num + Clone> CSRMatrix<T> {
    pub fn new_square(n: usize) -> CSRMatrix<T> {
        CSRMatrix::new(n, n)
    }
    pub fn new(rows: usize, cols: usize) -> CSRMatrix<T> {
        assert!(
            rows > 0 && cols > 0,
            "number of rows and columns must be greater than 0"
        );

        let capacity = 1000.min(rows * cols / 5);
        CSRMatrix {
            rows,
            cols,
            vals: Vec::with_capacity(capacity),
            cidx: Vec::with_capacity(capacity),
            ridx: BTreeMap::new(),
        }
    }
    pub fn identity(n: usize) -> CSRMatrix<T> {
        let mut m = CSRMatrix::new_square(n);
        m.vals = vec![T::one(); n];
        m.cidx = (0..n).map(|i| i).collect();
        m.ridx = (0..n).map(|i| (i, Slice { start: i, len: 1 })).collect();
        m
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    fn get_row_start(&self, i: usize) -> usize {
        self.ridx
            .range(i..)
            .next()
            .map(|(_, &Slice { start, .. })| start)
            .unwrap_or_else(|| self.cidx.len())
    }
    fn get_row_entries(&self, i: usize) -> impl DoubleEndedIterator<Item = (usize, &T)> {
        let s = self.ridx.get(&i).copied().unwrap_or_else(|| Slice {
            start: self.get_row_start(i),
            len: 0,
        });
        let r = || s.start..s.start + s.len;
        self.cidx[r()].iter().copied().zip(self.vals[r()].iter())
    }
    pub fn get_element(&self, (i, j): (usize, usize)) -> Cow<T> {
        assert!(
            (..self.rows).contains(&i) && (..self.cols).contains(&j),
            "values are not in bounds"
        );

        self.get_row_entries(i)
            .find(|&(c, _)| c == j)
            .map_or(Cow::Owned(T::zero()), |(_, t)| Cow::Borrowed(t))
    }
    pub fn set_element(&mut self, (i, j): (usize, usize), t: T) {
        assert!(
            (..self.rows).contains(&i) && (..self.cols).contains(&j),
            "values are not in bounds"
        );

        if t.is_zero() {
            let start = self.get_row_start(i);
            if let Some(len) = self.ridx.get(&i).map(|s| s.len) {
                if let Ok(k) = self.cidx[start..start + len].binary_search(&j) {
                    self.vals.remove(start + k);
                    self.cidx.remove(start + k);
                    for (_, s) in self.ridx.range_mut(i + 1..) {
                        s.start -= 1;
                    }
                    let s = self.ridx.get_mut(&i).unwrap();
                    s.len -= 1;
                    if s.len == 0 {
                        self.ridx.remove(&i);
                    }
                }
            }
        } else {
            let start = self.get_row_start(i);
            let len = self.ridx.get(&i).map(|s| s.len).unwrap_or(0);
            match self.cidx[start..start + len].binary_search(&j) {
                Ok(k) => {
                    self.vals[start + k] = t;
                }
                Err(k) => {
                    self.vals.insert(start + k, t);
                    self.cidx.insert(start + k, j);
                    let entry = self.ridx.entry(i).or_insert(Slice { start, len: 0 });
                    entry.len += 1;
                    for (_, s) in self.ridx.range_mut(i + 1..) {
                        s.start += 1;
                    }
                }
            }
        }
    }
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = ((usize, usize), &T)> {
        self.ridx
            .keys()
            .flat_map(move |&r| self.get_row_entries(r).map(move |(c, t)| ((r, c), t)))
    }
    pub fn transpose(&self) -> CSRMatrix<T> {
        let mut new = CSRMatrix::new(self.cols, self.rows);
        let mut row_start = 0;
        for j in 0..new.rows {
            let mut row_len = 0;
            for (&r, &Slice { start, len }) in &self.ridx {
                if let Ok(k) = self.cidx[start..start + len].binary_search(&j) {
                    new.vals.push(self.vals[start + k].clone());
                    new.cidx.push(r);
                    row_len += 1;
                }
            }
            if row_len > 0 {
                new.ridx.insert(
                    j,
                    Slice {
                        start: row_start,
                        len: row_len,
                    },
                );
            }
            row_start += row_len;
        }
        new
    }
}

impl<T: NumAssignRef + Clone> AddAssign<&CSRMatrix<T>> for CSRMatrix<T> {
    fn add_assign(&mut self, rhs: &CSRMatrix<T>) {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "matrices must have identical dimensions"
        );

        let (mut vals, mut cidx, mut ridx, mut start) = (vec![], vec![], BTreeMap::new(), 0);
        // union of nonempty rows
        let rs = self
            .ridx
            .keys()
            .chain(rhs.ridx.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        // iterate in increasing order
        for r in rs {
            let mut row = self
                .get_row_entries(r)
                .map(|(c, t)| (c, t.clone()))
                .collect::<BTreeMap<_, _>>();
            for (col, val) in rhs.get_row_entries(r) {
                let entry = row.entry(col).or_insert(T::zero());
                *entry += val;
                if entry.is_zero() {
                    row.remove(&col);
                }
            }
            let len = row.len();
            let (mut ks, mut vs): (Vec<_>, Vec<_>) = row.into_iter().unzip();
            cidx.append(&mut ks);
            vals.append(&mut vs);
            if len > 0 {
                ridx.insert(r, Slice { start, len });
            }
            start += len;
        }
        self.vals = vals;
        self.cidx = cidx;
        self.ridx = ridx;
    }
}

impl<T: NumAssignRef + Sum + Clone> MulAssign<&CSRMatrix<T>> for CSRMatrix<T> {
    fn mul_assign(&mut self, rhs: &CSRMatrix<T>) {
        assert_eq!(self.cols, rhs.rows, "LHS cols != RHS rows");

        *self = self
            .ridx
            .iter()
            .fold(
                (CSRMatrix::new(self.rows, rhs.cols), 0),
                |(mut m, mut start), (&i, _)| {
                    let this: &CSRMatrix<_> = self;
                    let (mut cidx, mut vals): (Vec<_>, Vec<_>) = (0..rhs.cols)
                        .filter_map(move |j| {
                            let s = this
                                .get_row_entries(i)
                                .map(|(k, t)| {
                                    let mut t = t.clone();
                                    t *= rhs
                                        .ridx
                                        .get(&k)
                                        .and_then(|&Slice { start, len }| {
                                            rhs.cidx[start..start + len]
                                                .binary_search(&j)
                                                .ok()
                                                .map(|k| &rhs.vals[start + k])
                                        })
                                        .unwrap_or(&T::zero());
                                    // t *= rhs.get_element((k, j)).as_ref();
                                    t
                                })
                                .sum::<T>();
                            if s.is_zero() {
                                None
                            } else {
                                Some((j, s))
                            }
                        })
                        .unzip();

                    let len = vals.len();
                    if len > 0 {
                        m.vals.append(&mut vals);
                        m.cidx.append(&mut cidx);
                        m.ridx.insert(i, Slice { start, len });
                        start += len;
                    }
                    (m, start)
                },
            )
            .0;
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct DOKMatrix<T> {
    rows: usize,
    cols: usize,
    entries: HashMap<(usize, usize), T>,
}

impl<T: Num + Clone> From<DOKMatrix<T>> for CSRMatrix<T> {
    fn from(old: DOKMatrix<T>) -> Self {
        let mut new = CSRMatrix::new(old.rows, old.cols);
        for ((i, j), t) in old.entries {
            new.set_element((i, j), t);
        }
        new
    }
}
impl<T: Num + Clone> From<CSRMatrix<T>> for DOKMatrix<T> {
    fn from(old: CSRMatrix<T>) -> Self {
        let mut vals = old.vals.into_iter().map(Some).collect::<Vec<_>>();
        let mut entries = HashMap::new();
        for (r, Slice { start, len }) in old.ridx {
            for (&c, t) in old.cidx[start..start + len]
                .iter()
                .zip(vals[start..start + len].iter_mut())
            {
                entries.insert((r, c), t.take().unwrap());
            }
        }

        DOKMatrix {
            rows: old.rows,
            cols: old.cols,
            entries,
        }
    }
}

impl<T: NumAssignRef + Clone> AddAssign<&DOKMatrix<T>> for DOKMatrix<T> {
    fn add_assign(&mut self, rhs: &DOKMatrix<T>) {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "matrices must have identical dimensions"
        );

        for (&(i, j), t) in &rhs.entries {
            let entry = self.entries.entry((i, j)).or_insert(T::zero());
            *entry += t;
        }
    }
}

impl<T: NumAssignRef + Clone> MulAssign<&DOKMatrix<T>> for DOKMatrix<T> {
    fn mul_assign(&mut self, rhs: &DOKMatrix<T>) {
        assert_eq!(self.cols, rhs.rows, "LHS cols != RHS rows");

        let mut entries = HashMap::new();
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let t = (0..self.cols).fold(T::zero(), |mut s, k| {
                    let mut t = self.entries.get(&(i, k)).unwrap_or(&T::zero()).clone();
                    t *= rhs.entries.get(&(k, j)).unwrap_or(&T::zero());
                    s += t;
                    s
                });
                if !t.is_zero() {
                    entries.insert((i, j), t);
                }
            }
        }

        self.cols = rhs.cols;
        self.entries = entries;
    }
}

fn nonzero_arb<G: quickcheck::Gen, T: Arbitrary + Num>(g: &mut G) -> T {
    repeat_with(|| T::arbitrary(g))
        .skip_while(|t| t.is_zero())
        .next()
        .unwrap()
}

impl<T: Arbitrary + NumAssignRef> Arbitrary for DOKMatrix<T> {
    fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
        let rows = nonzero_arb(g);
        let cols = nonzero_arb(g);
        let mut entries = HashMap::new();
        // limit density of matrices to hopefully speed up computation
        for _ in 0..g.gen_range(0, 1.max(rows * cols / 10)) {
            entries.insert((g.gen_range(0, rows), g.gen_range(0, cols)), nonzero_arb(g));
        }
        DOKMatrix {
            rows,
            cols,
            entries,
        }
    }
}

// pair of matrices conformable for multiplication
#[derive(Clone, Debug)]
pub struct AddPair<T>(pub DOKMatrix<T>, pub DOKMatrix<T>);

impl<T: Arbitrary + NumAssignRef> Arbitrary for AddPair<T> {
    fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
        let m1 = DOKMatrix::arbitrary(g);
        let (rows, cols) = (m1.rows, m1.cols);
        let n = g.gen_range(0, 1.max(rows * cols / 10));
        let entries =
            repeat_with(|| ((g.gen_range(0, rows), g.gen_range(0, cols)), nonzero_arb(g)))
                .take(n)
                .collect();
        AddPair(
            m1,
            DOKMatrix {
                rows,
                cols,
                entries,
            },
        )
    }
}

// pair of matrices conformable for multiplication
#[derive(Clone, Debug)]
pub struct MulPair<T>(pub DOKMatrix<T>, pub DOKMatrix<T>);

impl<T: Arbitrary + NumAssignRef> Arbitrary for MulPair<T> {
    fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
        let m1 = DOKMatrix::arbitrary(g);
        let (rows, cols) = (m1.cols, nonzero_arb(g));
        let n = g.gen_range(0, 1.max(rows * cols / 10));
        let entries =
            repeat_with(|| ((g.gen_range(0, rows), g.gen_range(0, cols)), nonzero_arb(g)))
                .take(n)
                .collect();
        MulPair(
            m1,
            DOKMatrix {
                rows,
                cols,
                entries,
            },
        )
    }
}
