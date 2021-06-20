use std::{
    convert::{TryFrom, TryInto},
    mem::{self, MaybeUninit},
    num::TryFromIntError,
    ptr,
};

use super::CsrMatrix;
use intel_mkl_spblas_sys::{
    mkl_sparse_d_create_csr, mkl_sparse_destroy, sparse_index_base_t_SPARSE_INDEX_BASE_ZERO,
    sparse_matrix_t, sparse_status_t_SPARSE_STATUS_ALLOC_FAILED,
    sparse_status_t_SPARSE_STATUS_EXECUTION_FAILED, sparse_status_t_SPARSE_STATUS_INTERNAL_ERROR,
    sparse_status_t_SPARSE_STATUS_INVALID_VALUE, sparse_status_t_SPARSE_STATUS_NOT_INITIALIZED,
    sparse_status_t_SPARSE_STATUS_NOT_SUPPORTED, sparse_status_t_SPARSE_STATUS_SUCCESS,
};
use libc::c_double;
use num_enum::TryFromPrimitive;
use thiserror::Error;

// CsrMatrix into MklCsrMatrix
// &MklCsrMatrix into MklSparseMatrix

#[derive(Debug, PartialEq, Eq)]
pub struct MklCsrMatrix<T> {
    rows: i32,
    cols: i32,
    indices: Vec<i32>,
    vals: Vec<T>,
    offsets: Vec<i32>,
}

impl TryFrom<CsrMatrix<f64>> for MklCsrMatrix<c_double> {
    type Error = TryFromIntError;

    fn try_from(
        CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        }: CsrMatrix<f64>,
    ) -> Result<Self, Self::Error> {
        let rows = rows.try_into()?;
        let cols = cols.try_into()?;
        let offsets = offsets
            .into_iter()
            .map(|i| i32::try_from(i))
            .collect::<Result<Vec<_>, _>>()?;
        let indices = indices
            .into_iter()
            .map(|i| i32::try_from(i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MklCsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        })
    }
}

impl TryFrom<MklCsrMatrix<c_double>> for CsrMatrix<f64> {
    type Error = TryFromIntError;

    fn try_from(
        MklCsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        }: MklCsrMatrix<c_double>,
    ) -> Result<Self, Self::Error> {
        let rows = rows.try_into()?;
        let cols = cols.try_into()?;
        let offsets = offsets
            .into_iter()
            .map(|i| usize::try_from(i))
            .collect::<Result<Vec<_>, _>>()?;
        let indices = indices
            .into_iter()
            .map(|i| usize::try_from(i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        })
    }
}

pub struct MklSparseMatrix<T> {
    handle: sparse_matrix_t,
    rows_start: (*mut i32, usize, usize),
    col_indx: (*mut i32, usize, usize),
    values: (*mut T, usize, usize),
}

impl<T> Drop for MklSparseMatrix<T> {
    fn drop(&mut self) {
        unsafe {
            mkl_sparse_destroy(self.handle);
            let (ptr, length, capacity) = self.rows_start;
            drop(Vec::from_raw_parts(ptr, length, capacity));
            let (ptr, length, capacity) = self.col_indx;
            drop(Vec::from_raw_parts(ptr, length, capacity));
            let (ptr, length, capacity) = self.values;
            drop(Vec::from_raw_parts(ptr, length, capacity));
        }
    }
}

#[derive(TryFromPrimitive, Error, Debug)]
#[repr(u32)]
pub enum MklCreateError {
    #[error("The routine encountered an empty handle or matrix array.")]
    NotInitialized = sparse_status_t_SPARSE_STATUS_NOT_INITIALIZED,
    #[error("Internal memory allocation failed.")]
    AllocFailed = sparse_status_t_SPARSE_STATUS_ALLOC_FAILED,
    #[error("The input parameters contain an invalid value.")]
    InvalidValue = sparse_status_t_SPARSE_STATUS_INVALID_VALUE,
    #[error("Execution failed.")]
    ExecutionFailed = sparse_status_t_SPARSE_STATUS_EXECUTION_FAILED,
    #[error("An error in algorithm implementation occurred.")]
    InternalError = sparse_status_t_SPARSE_STATUS_INTERNAL_ERROR,
    #[error("The requested operation is not supported.")]
    NotSupported = sparse_status_t_SPARSE_STATUS_NOT_SUPPORTED,
}

impl TryFrom<MklCsrMatrix<c_double>> for MklSparseMatrix<c_double> {
    type Error = MklCreateError;

    fn try_from(
        MklCsrMatrix {
            rows,
            cols,
            mut vals,
            mut indices,
            mut offsets,
        }: MklCsrMatrix<f64>,
    ) -> Result<Self, Self::Error> {
        let mut handle = MaybeUninit::uninit();
        indices.shrink_to_fit();
        let col_indx: (*mut i32, _, _) = (indices.as_mut_ptr(), indices.len(), indices.capacity());
        mem::forget(indices);
        vals.shrink_to_fit();
        let values: (*mut c_double, _, _) = (vals.as_mut_ptr(), vals.len(), vals.capacity());
        mem::forget(vals);
        offsets.shrink_to_fit();
        let rows_start: (*mut i32, _, _) =
            (offsets.as_mut_ptr(), offsets.len(), offsets.capacity());
        mem::forget(offsets);
        let status = unsafe {
            mkl_sparse_d_create_csr(
                handle.as_mut_ptr(),
                sparse_index_base_t_SPARSE_INDEX_BASE_ZERO,
                rows,
                cols,
                rows_start.0,
                rows_start.0.wrapping_add(1),
                col_indx.0,
                values.0,
            )
        };
        if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
            return Err(MklCreateError::try_from(status).unwrap());
        }
        Ok(MklSparseMatrix {
            handle: ptr::null_mut(),
            rows_start,
            col_indx,
            values,
        })
    }
}

#[derive(TryFromPrimitive, Error, Debug)]
#[repr(u32)]
pub enum MklExportError {
    #[error("The routine encountered an empty handle or matrix array.")]
    NotInitialized = sparse_status_t_SPARSE_STATUS_NOT_INITIALIZED,
    #[error("Internal memory allocation failed.")]
    AllocFailed = sparse_status_t_SPARSE_STATUS_ALLOC_FAILED,
    #[error("The input parameters contain an invalid value.")]
    InvalidValue = sparse_status_t_SPARSE_STATUS_INVALID_VALUE,
    #[error("Execution failed.")]
    ExecutionFailed = sparse_status_t_SPARSE_STATUS_EXECUTION_FAILED,
    #[error("An error in algorithm implementation occurred.")]
    InternalError = sparse_status_t_SPARSE_STATUS_INTERNAL_ERROR,
    #[error("The requested operation is not supported.")]
    NotSupported = sparse_status_t_SPARSE_STATUS_NOT_SUPPORTED,
}
