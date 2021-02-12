use itertools::{EitherOrBoth, Itertools};
use num::{traits::NumAssignRef, Num};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    iter,
    ops::{AddAssign, Bound, Range},
};

pub mod dok_matrix;
#[cfg(test)]
mod test;

// we use an extra indirection for vals and cidx to simplify the implementation
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrMatrix<T> {
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

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    fn get_row_entries(&self, i: usize) -> (&[usize], &[T]) {
        match self.ridx.range(i..).next() {
            Some((&j, &s)) if j == i => (&self.cidx[Range::from(s)], &self.vals[Range::from(s)]),
            _ => (&[], &[]),
        }
    }
}

impl<T: Num> CsrMatrix<T> {
    pub fn identity(n: usize) -> CsrMatrix<T> {
        let mut m = CsrMatrix::new_square(n);
        m.vals = iter::repeat_with(|| T::one()).take(n).collect();
        m.cidx = (0..n).map(|i| i).collect();
        m.ridx = (0..n).map(|i| (i, Slice { start: i, len: 1 })).collect();
        m
    }

    pub fn set_element(&mut self, (i, j): (usize, usize), t: T) {
        assert!(
            (..self.rows).contains(&i) && (..self.cols).contains(&j),
            "values are not in bounds"
        );

        // s.start indicates the position to insert an element if needed
        let mut s = self
            .ridx
            .range(i..)
            .next()
            .map(|(&k, &s)| {
                if k == i {
                    // s is in self.ridx, hence s.len > 0
                    s
                } else {
                    Slice { len: 0, ..s }
                }
            })
            .unwrap_or(Slice {
                start: self.cidx.len(),
                len: 0,
            });
        match self.cidx[Range::from(s)].binary_search(&j) {
            Ok(k) => {
                if t.is_zero() {
                    self.vals.remove(s.start + k);
                    self.cidx.remove(s.start + k);
                    // s.len > 0 since search succeeded
                    s.len -= 1;
                    if s.len == 0 {
                        self.ridx.remove(&i);
                    } else {
                        self.ridx.insert(i, s);
                    }
                    // Excluded(i) instead of i+1 for overflow
                    for (_, s) in self.ridx.range_mut((Bound::Excluded(i), Bound::Unbounded)) {
                        s.start -= 1;
                    }
                } else {
                    self.vals[s.start + k] = t;
                }
            }
            Err(k) => {
                if !t.is_zero() {
                    self.vals.insert(s.start + k, t);
                    self.cidx.insert(s.start + k, j);
                    s.len += 1;
                    self.ridx.insert(i, s);
                    // Excluded(i) instead of i+1 for overflow
                    for (_, s) in self.ridx.range_mut((Bound::Excluded(i), Bound::Unbounded)) {
                        s.start += 1;
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

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = ((usize, usize), &T)> {
        self.ridx.keys().flat_map(move |&r| {
            let (cidx, vals) = self.get_row_entries(r);
            cidx.iter().zip(vals.iter()).map(move |(&c, t)| ((r, c), t))
        })
    }

    pub fn transpose(&self) -> CsrMatrix<T> {
        let mut new = CsrMatrix::new(self.cols, self.rows);
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
impl<T: NumAssignRef + Clone> AddAssign<&CsrMatrix<T>> for CsrMatrix<T> {
    fn add_assign(&mut self, rhs: &CsrMatrix<T>) {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "matrices must have identical dimensions"
        );

        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], BTreeMap::new());
        // iterate keys merged in increasing order
        for eob in self
            .ridx
            .iter()
            .merge_join_by(rhs.ridx.iter(), |r1, r2| r1.0.cmp(r2.0))
        {
            match eob {
                EitherOrBoth::Both((&r, &s1), (_, &s2)) => {
                    // add row r of self and rhs
                    let (mut rcidx, mut rvals): (Vec<_>, Vec<_>) = self.cidx[Range::from(s1)]
                        .iter()
                        .zip(&self.vals[Range::from(s1)])
                        .merge_join_by(
                            rhs.cidx[Range::from(s2)]
                                .iter()
                                .zip(&rhs.vals[Range::from(s2)]),
                            |c1, c2| c1.0.cmp(c2.0),
                        )
                        .map(|eob| match eob {
                            EitherOrBoth::Both((&c, t1), (_, t2)) => {
                                let mut t = t1.clone();
                                t += t2;
                                (c, t)
                            }
                            EitherOrBoth::Left((&c, t)) | EitherOrBoth::Right((&c, t)) => {
                                (c, t.clone())
                            }
                        })
                        .filter(|(_, t)| !t.is_zero())
                        .unzip();

                    let len = rcidx.len();
                    if len > 0 {
                        ridx.insert(
                            r,
                            Slice {
                                start: cidx.len(),
                                len,
                            },
                        );
                        cidx.append(&mut rcidx);
                        vals.append(&mut rvals);
                    }
                }
                // all entries of row r of self are nonzero, row r of rhs is all 0
                EitherOrBoth::Left((&r, &s)) => {
                    ridx.insert(
                        r,
                        Slice {
                            start: cidx.len(),
                            len: s.len,
                        },
                    );
                    cidx.extend_from_slice(&self.cidx[Range::from(s)]);
                    vals.extend_from_slice(&self.vals[Range::from(s)]);
                }
                // all entries of row r of rhs are nonzero, row r of self is all 0
                EitherOrBoth::Right((&r, &s)) => {
                    ridx.insert(
                        r,
                        Slice {
                            start: cidx.len(),
                            len: s.len,
                        },
                    );
                    cidx.extend_from_slice(&rhs.cidx[Range::from(s)]);
                    vals.extend_from_slice(&rhs.vals[Range::from(s)]);
                }
            }
        }

        self.vals = vals;
        self.cidx = cidx;
        self.ridx = ridx;
    }
}
/*
impl<T: Num + Sum + Clone> Mul for &CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "LHS cols != RHS rows");

        let mut m = CsrMatrix::new(self.rows, rhs.cols);
        let mut start = 0;
        for (&i, _) in &self.ridx {
            let this: &CsrMatrix<_> = self;
            let (mut cidx, mut vals): (Vec<_>, Vec<_>) = (0..rhs.cols)
                .filter_map(move |j| {
                    let s = this
                        .get_row_entries(i)
                        .map(|(k, t)| {
                            // t * rhs.get_element((k, j)).as_ref();

                            t.clone()
                                * rhs
                                    .ridx
                                    .get(&k)
                                    .and_then(|&Slice { start, len }| {
                                        rhs.cidx[start..start + len]
                                            .binary_search(&j)
                                            .ok()
                                            .map(|k| &rhs.vals[start + k])
                                    })
                                    .unwrap_or(&T::zero())
                                    .clone()
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
        }
        m
    }
}
*/
