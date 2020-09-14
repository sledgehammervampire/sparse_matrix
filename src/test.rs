use crate::{AddPair, CSRMatrix, DOKMatrix, MulPair};
use quickcheck::QuickCheck;
use std::collections::HashMap;

#[quickcheck]
fn test_convert(m: DOKMatrix<i32>) -> bool {
    m == DOKMatrix::from(CSRMatrix::from(m.clone()))
}

#[test]
fn test_add() {
    fn prop_add_1(AddPair(mut m1, m2): AddPair<i32>) -> bool {
        let mut m3 = CSRMatrix::from(m1.clone());
        let m4 = CSRMatrix::from(m2.clone());
        m3 += &m4;
        m1 += &m2;
        CSRMatrix::from(m1) == m3
    }
    fn prop_add_2(AddPair(m1, m2): AddPair<i32>) -> bool {
        let (m1, m2) = (CSRMatrix::from(m1), CSRMatrix::from(m2));
        let mut m = m1.clone();
        m += &m2;
        (0..m.rows)
            .flat_map(|i| (0..m.cols).map(move |j| (i, j)))
            .all(|p| {
                m.get_element(p).into_owned()
                    == m1.get_element(p).into_owned() + m2.get_element(p).as_ref()
            })
    }
    QuickCheck::new()
        .tests(1 << 10)
        .quickcheck(prop_add_1 as fn(_) -> bool);
    QuickCheck::new()
        .tests(1 << 10)
        .quickcheck(prop_add_2 as fn(_) -> bool);
}

#[test]
fn test_mul() {
    fn prop_mul_1(MulPair(mut m1, m2): MulPair<i32>) -> bool {
        let mut m3 = CSRMatrix::from(m1.clone());
        let m4 = CSRMatrix::from(m2.clone());
        m3 *= &m4;
        m1 *= &m2;
        CSRMatrix::from(m1) == m3
    }
    fn prop_mul_2(MulPair(m1, m2): MulPair<i32>) -> bool {
        let (m1, m2) = (CSRMatrix::from(m1), CSRMatrix::from(m2));
        let mut m = m1.clone();
        m *= &m2;
        (0..m.rows)
            .flat_map(|i| (0..m.cols).map(move |j| (i, j)))
            .all(|(i, j)| {
                m.get_element((i, j)).into_owned()
                    == (0..m1.cols)
                        .map(|k| {
                            m1.get_element((i, k)).into_owned() * m2.get_element((k, j)).as_ref()
                        })
                        .sum()
            })
    }
    QuickCheck::new()
        .tests(1 << 10)
        .quickcheck(prop_mul_1 as fn(_) -> bool);
    QuickCheck::new()
        .tests(1 << 10)
        .quickcheck(prop_mul_2 as fn(_) -> bool);
}

#[quickcheck]
fn test_transpose(m: DOKMatrix<i32>) -> bool {
    let m = CSRMatrix::from(m);
    let m1 = m.transpose();
    (0..m.rows)
        .flat_map(|i| (0..m.cols).map(move |j| (i, j)))
        .all(|(i, j)| m.get_element((i, j)) == m1.get_element((j, i)))
}

#[quickcheck]
fn test_iter(m: DOKMatrix<i32>) -> bool {
    let m1 = CSRMatrix::from(m.clone());
    m1.iter().map(|(p, &t)| (p, t)).collect::<HashMap<_, _>>() == m.entries
}

#[test]
fn test_1() {
    assert_eq!(
        DOKMatrix::from(CSRMatrix::identity(2)),
        DOKMatrix {
            rows: 2,
            cols: 2,
            entries: vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect()
        }
    )
}
#[test]
fn test_2() {
    assert_eq!(
        CSRMatrix::from({
            let entries = vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect();
            DOKMatrix {
                rows: 2,
                cols: 2,
                entries,
            }
        }),
        CSRMatrix::identity(2)
    );
}
#[test]
fn test_3() {
    assert_eq!(
        {
            let mut m1 = CSRMatrix::identity(3);
            let m2 = CSRMatrix::identity(3);
            m1 += &m2;
            m1
        },
        CSRMatrix::from(DOKMatrix {
            rows: 3,
            cols: 3,
            entries: vec![((0, 0), 2), ((1, 1), 2), ((2, 2), 2)]
                .into_iter()
                .collect()
        })
    )
}
#[test]
fn test_4() {
    assert_eq!(
        {
            let mut m = CSRMatrix::identity(2);
            m.set_element((0, 1), 1);
            m
        },
        CSRMatrix::from(DOKMatrix {
            rows: 2,
            cols: 2,
            entries: vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
                .into_iter()
                .collect()
        })
    )
}
#[test]
fn test_5() {
    assert_eq!(
        {
            let mut m = CSRMatrix::from(DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
                    .into_iter()
                    .collect(),
            });
            m.set_element((0, 1), 0);
            m
        },
        CSRMatrix::identity(2)
    )
}
#[test]
fn test_6() {
    assert_eq!(
        {
            let mut m = CSRMatrix::from(DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
                    .into_iter()
                    .collect(),
            });
            m.set_element((0, 1), 2);
            m
        },
        CSRMatrix::from(DOKMatrix {
            rows: 2,
            cols: 2,
            entries: vec![((0, 0), 1), ((0, 1), 2), ((1, 1), 1)]
                .into_iter()
                .collect(),
        })
    )
}
#[test]
fn test_7() {
    assert_eq!(
        {
            let mut m = CSRMatrix::from(DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect(),
            });
            m.set_element((0, 1), 0);
            m
        },
        CSRMatrix::identity(2)
    )
}
#[test]
fn test_8() {
    assert_eq!(
        {
            let mut m = CSRMatrix::identity(2);
            m.set_element((0, 0), 0);
            m.set_element((1, 1), 0);
            m
        },
        CSRMatrix::from(DOKMatrix {
            rows: 2,
            cols: 2,
            entries: vec![].into_iter().collect()
        })
    )
}
#[test]
fn test_9() {
    assert_eq!(
        {
            let mut m = CSRMatrix::<i32>::identity(2);
            m *= &CSRMatrix::identity(2);
            m
        },
        CSRMatrix::identity(2)
    )
}
#[test]
fn test_10() {
    assert_eq!(
        {
            let mut m = CSRMatrix::from(DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 1), 1)].into_iter().collect(),
            });
            m *= &m.clone();
            m
        },
        {
            CSRMatrix::from(DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![].into_iter().collect(),
            })
        }
    )
}
#[test]
fn test_11() {
    assert_eq!(
        CSRMatrix::<i32>::identity(3),
        CSRMatrix::identity(3).transpose()
    )
}
#[test]
fn test_12() {
    assert_eq!(
        CSRMatrix::from(DOKMatrix {
            rows: 2,
            cols: 2,
            entries: vec![((0, 1), 1)].into_iter().collect()
        })
        .transpose(),
        CSRMatrix::from(DOKMatrix {
            rows: 2,
            cols: 2,
            entries: vec![((1, 0), 1)].into_iter().collect()
        })
    )
}
