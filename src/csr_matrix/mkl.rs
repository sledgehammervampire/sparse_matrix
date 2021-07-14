use std::{marker::PhantomData, mem::MaybeUninit, ops::Mul, slice};

use mkl_sys::{
    mkl_sparse_d_export_csr, mkl_sparse_spmm, mkl_sparse_z_export_csr,
    sparse_operation_t::SPARSE_OPERATION_NON_TRANSPOSE, sparse_status_t::SPARSE_STATUS_SUCCESS,
};

use crate::mkl::MklError;

use {
    super::CsrMatrix,
    crate::{
        mkl::{CMklSparseMatrix, MklCsrMatrix},
        ComplexNewtype, MKL_Complex16,
    },
    mkl_sys::MKL_INT,
    std::{
        convert::{TryFrom, TryInto},
        num::TryFromIntError,
    },
};

impl<const IS_SORTED: bool> TryFrom<CsrMatrix<f64, IS_SORTED>> for MklCsrMatrix<f64, IS_SORTED> {
    type Error = TryFromIntError;

    fn try_from(m: CsrMatrix<f64, IS_SORTED>) -> Result<Self, Self::Error> {
        let rows = m.rows.try_into()?;
        let cols = m.cols.try_into()?;
        let offsets: Vec<_> = m
            .offsets
            .into_iter()
            .map(MKL_INT::try_from)
            .collect::<Result<_, _>>()?;
        let indices: Vec<_> = m
            .indices
            .into_iter()
            .map(MKL_INT::try_from)
            .collect::<Result<_, _>>()?;
        Ok(MklCsrMatrix {
            rows,
            cols,
            vals: m.vals,
            indices,
            offsets,
        })
    }
}

impl<const IS_SORTED: bool> TryFrom<MklCsrMatrix<f64, IS_SORTED>> for CsrMatrix<f64, IS_SORTED> {
    type Error = TryFromIntError;

    fn try_from(m: MklCsrMatrix<f64, IS_SORTED>) -> Result<Self, Self::Error> {
        let rows = m.rows.try_into()?;
        let cols = m.cols.try_into()?;
        let offsets: Vec<_> = m
            .offsets
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<_, _>>()?;
        let indices: Vec<_> = m
            .indices
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<_, _>>()?;
        Ok(CsrMatrix {
            rows,
            cols,
            vals: m.vals,
            indices,
            offsets,
        })
    }
}

impl<const IS_SORTED: bool> TryFrom<CMklSparseMatrix<f64, IS_SORTED>>
    for CsrMatrix<f64, IS_SORTED>
{
    type Error = crate::mkl::Error;

    fn try_from(m: CMklSparseMatrix<f64, IS_SORTED>) -> Result<Self, Self::Error> {
        let mut indexing = MaybeUninit::uninit();
        let mut rows = MaybeUninit::uninit();
        let mut cols = MaybeUninit::uninit();
        let mut rows_start = MaybeUninit::uninit();
        let mut rows_end = MaybeUninit::uninit();
        let mut col_indx = MaybeUninit::uninit();
        let mut values = MaybeUninit::uninit();
        let status = unsafe {
            mkl_sparse_d_export_csr(
                m.handle,
                indexing.as_mut_ptr(),
                rows.as_mut_ptr(),
                cols.as_mut_ptr(),
                rows_start.as_mut_ptr(),
                rows_end.as_mut_ptr(),
                col_indx.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if status != SPARSE_STATUS_SUCCESS {
            return Err(crate::mkl::Error::Mkl(MklError::try_from(status).unwrap()));
        }
        let indexing = usize::try_from(unsafe { indexing.assume_init() })?;
        let rows = usize::try_from(unsafe { rows.assume_init() })?;
        let cols = usize::try_from(unsafe { cols.assume_init() })?;
        let mut offsets: Vec<_> = unsafe { slice::from_raw_parts(rows_start.assume_init(), rows) }
            .iter()
            .map(|i| Ok(usize::try_from(*i)? - indexing))
            .collect::<Result<_, TryFromIntError>>()?;
        let nnz = unsafe { *rows_end.assume_init().wrapping_add(rows - 1) }.try_into()?;
        offsets.push(nnz);
        let indices: Vec<_> = unsafe { slice::from_raw_parts(col_indx.assume_init(), nnz) }
            .iter()
            .map(|i| Ok(usize::try_from(*i)? - indexing))
            .collect::<Result<_, TryFromIntError>>()?;
        let vals = unsafe { slice::from_raw_parts(values.assume_init(), nnz) }.to_vec();
        Ok(CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        })
    }
}

impl<const IS_SORTED: bool> Mul for &CMklSparseMatrix<f64, IS_SORTED> {
    type Output = Result<CMklSparseMatrix<f64, false>, crate::mkl::Error>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = MaybeUninit::uninit();
        let status = unsafe {
            mkl_sparse_spmm(
                SPARSE_OPERATION_NON_TRANSPOSE,
                self.handle,
                rhs.handle,
                res.as_mut_ptr(),
            )
        };
        if status != SPARSE_STATUS_SUCCESS {
            return Err(crate::mkl::Error::Mkl(MklError::try_from(status).unwrap()));
        }
        Ok(CMklSparseMatrix {
            handle: unsafe { res.assume_init() },
            _phantom: PhantomData,
        })
    }
}

impl<const IS_SORTED: bool> TryFrom<CsrMatrix<ComplexNewtype<f64>, IS_SORTED>>
    for MklCsrMatrix<MKL_Complex16, IS_SORTED>
{
    type Error = TryFromIntError;

    fn try_from(m: CsrMatrix<ComplexNewtype<f64>, IS_SORTED>) -> Result<Self, Self::Error> {
        let rows = m.rows.try_into()?;
        let cols = m.cols.try_into()?;
        let offsets: Vec<_> = m
            .offsets
            .into_iter()
            .map(MKL_INT::try_from)
            .collect::<Result<_, _>>()?;
        let indices: Vec<_> = m
            .indices
            .into_iter()
            .map(MKL_INT::try_from)
            .collect::<Result<_, _>>()?;
        let vals = m.vals.into_iter().map(MKL_Complex16::from).collect();
        Ok(MklCsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        })
    }
}

impl<const IS_SORTED: bool> TryFrom<MklCsrMatrix<MKL_Complex16, IS_SORTED>>
    for CsrMatrix<ComplexNewtype<f64>, IS_SORTED>
{
    type Error = TryFromIntError;

    fn try_from(m: MklCsrMatrix<MKL_Complex16, IS_SORTED>) -> Result<Self, Self::Error> {
        let rows = m.rows.try_into()?;
        let cols = m.cols.try_into()?;
        let offsets: Vec<_> = m
            .offsets
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<_, _>>()?;
        let indices: Vec<_> = m
            .indices
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<_, _>>()?;
        let vals = m.vals.into_iter().map(ComplexNewtype::from).collect();
        Ok(CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        })
    }
}

impl<const IS_SORTED: bool> TryFrom<CMklSparseMatrix<MKL_Complex16, IS_SORTED>>
    for CsrMatrix<ComplexNewtype<f64>, IS_SORTED>
{
    type Error = crate::mkl::Error;

    fn try_from(m: CMklSparseMatrix<MKL_Complex16, IS_SORTED>) -> Result<Self, Self::Error> {
        let mut indexing = MaybeUninit::uninit();
        let mut rows = MaybeUninit::uninit();
        let mut cols = MaybeUninit::uninit();
        let mut rows_start = MaybeUninit::uninit();
        let mut rows_end = MaybeUninit::uninit();
        let mut col_indx = MaybeUninit::uninit();
        let mut values = MaybeUninit::uninit();
        let status = unsafe {
            mkl_sparse_z_export_csr(
                m.handle,
                indexing.as_mut_ptr(),
                rows.as_mut_ptr(),
                cols.as_mut_ptr(),
                rows_start.as_mut_ptr(),
                rows_end.as_mut_ptr(),
                col_indx.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if status != SPARSE_STATUS_SUCCESS {
            return Err(crate::mkl::Error::Mkl(MklError::try_from(status).unwrap()));
        }
        let indexing = usize::try_from(unsafe { indexing.assume_init() })?;
        let rows = usize::try_from(unsafe { rows.assume_init() })?;
        let cols = usize::try_from(unsafe { cols.assume_init() })?;
        let mut offsets: Vec<_> = unsafe { slice::from_raw_parts(rows_start.assume_init(), rows) }
            .iter()
            .map(|i| Ok(usize::try_from(*i)? - indexing))
            .collect::<Result<_, TryFromIntError>>()?;
        let nnz = unsafe { *rows_end.assume_init().wrapping_add(rows - 1) }.try_into()?;
        offsets.push(nnz);
        let indices: Vec<_> = unsafe { slice::from_raw_parts(col_indx.assume_init(), nnz) }
            .iter()
            .map(|i| Ok(usize::try_from(*i)? - indexing))
            .collect::<Result<_, TryFromIntError>>()?;
        let vals = unsafe { slice::from_raw_parts(values.assume_init().cast(), nnz) }.to_vec();
        Ok(CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        })
    }
}
