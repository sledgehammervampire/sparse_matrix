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

use crate::{Matrix, dok_matrix::DokMatrix, is_increasing};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrMatrix<T> {
    rows: usize,
    cols: usize,
    // for v in vals, vals != 0
    vals: Vec<T>,
    cidx: Vec<usize>,
    ridx: BTreeMap<usize, Slice>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Slice {
    pub start: usize,
    pub len: usize,
}

impl From<Slice> for Range<usize> {
    fn from(s: Slice) -> Self {
        s.start..s.start + s.len
    }
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
            ridx: BTreeMap::new(),
        }
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = ((usize, usize), &T)> {
        let Self {
            ridx, cidx, vals, ..
        } = self;
        ridx.iter().flat_map(move |(&r, &s)| {
            cidx[Range::from(s)]
                .iter()
                .copied()
                .zip(vals[Range::from(s)].iter())
                .map(move |(c, t)| ((r, c), t))
        })
    }
}

impl<T: Arbitrary + Num> CsrMatrix<T> {
    pub fn arb_fixed_size_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Self> {
        repeat_with(|| subsequence((0..cols).collect::<Vec<_>>(), 0..=cols))
            .take(rows)
            .collect::<Vec<_>>()
            .prop_flat_map(move |cidx| {
                let (mut cidx_flattened, mut ridx) = (vec![], BTreeMap::new());
                for (i, mut rcidx) in cidx.into_iter().enumerate() {
                    if rcidx.len() > 0 {
                        ridx.insert(
                            i,
                            Slice {
                                start: cidx_flattened.len(),
                                len: rcidx.len(),
                            },
                        );
                        cidx_flattened.append(&mut rcidx);
                    }
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

impl<T: Num> CsrMatrix<T> {
    pub fn csr_invariants(&self) -> bool {
        self.csr_invariant_1()
            && self.csr_invariant_2()
            && self.csr_invariant_3()
            && self.csr_invariant_4()
            && self.csr_invariant_5()
            && self.csr_invariant_6()
            && self.csr_invariant_7()
            && self.csr_invariant_8()
    }

    fn csr_invariant_1(&self) -> bool {
        self.ridx
            .iter()
            .all(|(i, s1)| self.ridx.range(..i).fold(0, |sum, (_, s2)| sum + s2.len) == s1.start)
    }

    fn csr_invariant_2(&self) -> bool {
        self.ridx.values().map(|s| s.len).sum::<usize>() == self.vals.len()
    }

    fn csr_invariant_3(&self) -> bool {
        self.cidx.len() == self.vals.len()
    }

    fn csr_invariant_4(&self) -> bool {
        self.ridx.values().all(|s| s.len > 0)
    }

    fn csr_invariant_5(&self) -> bool {
        self.ridx
            .values()
            .all(|s| is_increasing(&self.cidx[s.start..s.start + s.len]))
    }

    fn csr_invariant_6(&self) -> bool {
        self.vals.iter().all(|t| !t.is_zero())
    }

    fn csr_invariant_7(&self) -> bool {
        self.ridx.keys().all(|r| (0..self.rows).contains(r))
    }

    fn csr_invariant_8(&self) -> bool {
        self.cidx.iter().all(|c| (0..self.cols).contains(c))
    }
}

impl<T: Num + Clone> Matrix<T> for CsrMatrix<T> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn len(&self) -> usize {
        self.cidx.len()
    }

    fn get_element(&self, (i, j): (usize, usize)) -> Cow<T> {
        assert!(
            (..self.rows).contains(&i) && (..self.cols).contains(&j),
            "values are not in bounds"
        );

        self.ridx.get(&i).map_or_else(
            || Cow::Owned(T::zero()),
            |&s| {
                self.cidx[Range::from(s)].binary_search(&j).map_or_else(
                    |_| Cow::Owned(T::zero()),
                    |k| Cow::Borrowed(&self.vals[s.start + k]),
                )
            },
        )
    }

    fn set_element(&mut self, (i, j): (usize, usize), t: T) {
        assert!(
            (..self.rows).contains(&i) && (..self.cols).contains(&j),
            "values are not in bounds"
        );

        let mut iter = self.ridx.range_mut(i..);
        if let Some((&i1, Slice { start, len })) = iter.next() {
            if i == i1 {
                match self.cidx[*start..*start + *len].binary_search(&j) {
                    Ok(k) => {
                        let l = *start + k;
                        if t.is_zero() {
                            self.vals.remove(l);
                            self.cidx.remove(l);
                            *len -= 1;
                            for (_, Slice { start, .. }) in iter {
                                *start -= 1;
                            }
                            if *len == 0 {
                                self.ridx.remove(&i);
                            }
                        } else {
                            self.vals[l] = t;
                        }
                    }
                    Err(k) => {
                        if !t.is_zero() {
                            let l = *start + k;
                            self.vals.insert(l, t);
                            self.cidx.insert(l, j);
                            *len += 1;
                            for (_, Slice { start, .. }) in iter {
                                *start += 1;
                            }
                        }
                    }
                }
            } else if !t.is_zero() {
                self.vals.insert(*start, t);
                self.cidx.insert(*start, j);
                let prev_start = *start;
                *start += 1;
                for (_, Slice { start, .. }) in iter {
                    *start += 1;
                }
                // we insert after consuming iter to avoid overlapping mutable borrows
                self.ridx.insert(
                    i,
                    Slice {
                        start: prev_start,
                        len: 1,
                    },
                );
            }
        } else if !t.is_zero() {
            // note that push changes self.vals.len, so we must insert first
            self.ridx.insert(
                i,
                Slice {
                    start: self.vals.len(),
                    len: 1,
                },
            );
            self.vals.push(t);
            self.cidx.push(j);
        }
    }

    fn identity(n: usize) -> CsrMatrix<T> {
        CsrMatrix {
            rows: n,
            cols: n,
            vals: repeat_with(|| T::one()).take(n).collect(),
            cidx: (0..n).map(|i| i).collect(),
            ridx: (0..n).map(|i| (i, Slice { start: i, len: 1 })).collect(),
        }
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
}

impl<T: Num> Add for CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "matrices must have identical dimensions"
        );

        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], BTreeMap::new());

        for row in self.ridx.iter().map(|(r, s)| (*r, *s)).merge_join_by(
            rhs.ridx.iter().map(|(r, s)| (*r, *s)),
            |(r1, _), (r2, _)| r1.cmp(r2),
        ) {
            let prev_len = cidx.len();
            match row {
                itertools::EitherOrBoth::Both((r, s1), (_, s2)) => {
                    let mut len = 0;
                    for (c, t) in self.cidx[Range::from(s1)]
                        .iter()
                        .copied()
                        .zip(
                            self.vals
                                .splice(Range::from(s1), repeat_with(|| T::zero()).take(s1.len)),
                        )
                        .merge_join_by(
                            rhs.cidx[Range::from(s2)].iter().copied().zip(
                                rhs.vals.splice(
                                    Range::from(s2),
                                    repeat_with(|| T::zero()).take(s2.len),
                                ),
                            ),
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
                    {
                        vals.push(t);
                        cidx.push(c);
                        len += 1;
                    }
                    if len > 0 {
                        ridx.insert(
                            r,
                            Slice {
                                start: prev_len,
                                len,
                            },
                        );
                    }
                }
                itertools::EitherOrBoth::Left((r, s)) => {
                    ridx.insert(
                        r,
                        Slice {
                            start: prev_len,
                            len: s.len,
                        },
                    );
                    vals.extend(
                        self.vals
                            .splice(Range::from(s), repeat_with(|| T::zero()).take(s.len)),
                    );
                    cidx.extend(
                        self.cidx
                            .splice(Range::from(s), repeat_with(|| 0).take(s.len)),
                    );
                }
                itertools::EitherOrBoth::Right((r, s)) => {
                    ridx.insert(
                        r,
                        Slice {
                            start: prev_len,
                            len: s.len,
                        },
                    );
                    vals.extend(
                        rhs.vals
                            .splice(Range::from(s), repeat_with(|| T::zero()).take(s.len)),
                    );
                    cidx.extend(
                        rhs.cidx
                            .splice(Range::from(s), repeat_with(|| 0).take(s.len)),
                    );
                }
            }
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

        let rows: BTreeMap<_, _> = self
            .ridx
            .par_iter()
            .filter_map(|(&r, &s)| {
                let mut row = BTreeMap::new();

                for (&k, t) in self.cidx[Range::from(s)]
                    .iter()
                    .zip(self.vals[Range::from(s)].iter())
                {
                    if let Some(&s) = rhs.ridx.get(&k) {
                        for (&j, t1) in rhs.cidx[Range::from(s)]
                            .iter()
                            .zip(rhs.vals[Range::from(s)].iter())
                        {
                            let entry = row.entry(j).or_insert(T::zero());
                            *entry = mem::replace(entry, T::zero()) + t.clone() * t1.clone();
                        }
                    }
                }
                let (rcidx, rvals): (Vec<usize>, Vec<T>) =
                    row.into_iter().filter(|(_, t)| !t.is_zero()).unzip();
                (rcidx.len() > 0).then(|| (r, (rcidx, rvals)))
            })
            .collect();

        let (mut cidx, mut vals, mut ridx) = (vec![], vec![], BTreeMap::new());
        for (r, (mut rcidx, mut rvals)) in rows {
            ridx.insert(
                r,
                Slice {
                    start: cidx.len(),
                    len: rcidx.len(),
                },
            );
            cidx.append(&mut rcidx);
            vals.append(&mut rvals);
        }

        CsrMatrix {
            rows: self.rows,
            cols: rhs.cols,
            cidx,
            vals,
            ridx,
        }
    }
}

impl<T: Num + Clone> From<DokMatrix<T>> for CsrMatrix<T> {
    fn from(old: DokMatrix<T>) -> Self {
        let (rows, cols) = (old.rows(), old.cols());
        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], BTreeMap::new());
        // note that (i, j) is iterated in lexicographic order
        for ((i, j), t) in old {
            if !t.is_zero() {
                ridx.entry(i)
                    .or_insert(Slice {
                        start: vals.len(),
                        len: 0,
                    })
                    .len += 1;
                vals.push(t);
                cidx.push(j);
            }
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
