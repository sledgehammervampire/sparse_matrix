#![feature(type_alias_impl_trait)]
use derive_more::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
#[cfg(feature = "mkl")]
use mkl_sys::MKL_Complex16;
use num::{traits::NumAssign, Complex, Num, One, Zero};
#[cfg(feature = "proptest")]
use proptest::prelude::*;

#[repr(C)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Neg,
    Mul,
    MulAssign,
    Div,
    DivAssign,
    Rem,
    RemAssign,
)]
#[mul(forward)]
#[mul_assign(forward)]
#[div(forward)]
#[div_assign(forward)]
#[rem(forward)]
#[rem_assign(forward)]
pub struct ComplexNewtype<T: NumAssign + Clone>(Complex<T>);

impl<T: NumAssign + Clone> Num for ComplexNewtype<T> {
    type FromStrRadixErr = <Complex<T> as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Complex::from_str_radix(str, radix).map(ComplexNewtype)
    }
}

impl<T: NumAssign + Clone> One for ComplexNewtype<T> {
    fn one() -> Self {
        ComplexNewtype(Complex::one())
    }
}

impl<T: NumAssign + Clone> Zero for ComplexNewtype<T> {
    fn zero() -> Self {
        ComplexNewtype(Complex::one())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

#[cfg(feature = "proptest")]
impl<T: Arbitrary + NumAssign + Clone> Arbitrary for ComplexNewtype<T> {
    type Parameters = ();
    type Strategy = impl Strategy<Value = ComplexNewtype<T>>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (any::<T>(), any::<T>()).prop_map(|(re, im)| ComplexNewtype(Complex { re, im }))
    }
}

#[cfg(feature = "mkl")]
impl From<MKL_Complex16> for ComplexNewtype<f64> {
    fn from(z: MKL_Complex16) -> Self {
        ComplexNewtype(Complex {
            re: z.real,
            im: z.imag,
        })
    }
}

#[cfg(feature = "mkl")]
impl From<ComplexNewtype<f64>> for MKL_Complex16 {
    fn from(z: ComplexNewtype<f64>) -> Self {
        MKL_Complex16 {
            real: z.0.re,
            imag: z.0.im,
        }
    }
}
