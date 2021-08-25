use std::fmt::{Debug, LowerExp};

use derive_more::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use num::{Num, NumCast, One, ToPrimitive, Zero};

// f64 with Debug impl in scientific notation
#[derive(
    Clone,
    Copy,
    PartialEq,
    AddAssign,
    MulAssign,
    RemAssign,
    DivAssign,
    SubAssign,
    Add,
    Div,
    Sub,
    Rem,
    Mul,
    Neg,
)]
#[mul(forward)]
#[mul_assign(forward)]
#[div(forward)]
#[div_assign(forward)]
#[rem(forward)]
#[rem_assign(forward)]
pub struct Sci<F>(F);

impl<F: LowerExp> Debug for Sci<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:e}", self.0)
    }
}
impl<'a, F: ::arbitrary::Arbitrary<'a>> ::arbitrary::Arbitrary<'a> for Sci<F> {
    fn arbitrary(u: &mut ::arbitrary::Unstructured<'a>) -> ::arbitrary::Result<Self> {
        Ok(Sci(u.arbitrary()?))
    }
}
impl<F: num::Float> Num for Sci<F> {
    type FromStrRadixErr = <F as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(Sci)
    }
}
impl<F: num::Float> Zero for Sci<F> {
    fn zero() -> Self {
        Sci(F::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}
impl<F: num::Float> One for Sci<F> {
    fn one() -> Self {
        Sci(F::one())
    }
}
impl<F: num::Float> PartialOrd for Sci<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl<F: num::Float> ToPrimitive for Sci<F> {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }
}
impl<F: num::Float> NumCast for Sci<F> {
    fn from<T: num::ToPrimitive>(n: T) -> Option<Self> {
        <F as NumCast>::from(n).map(Sci)
    }
}
impl<F: num::Float> num::Float for Sci<F> {
    fn nan() -> Self {
        Sci(F::nan())
    }

    fn infinity() -> Self {
        Sci(F::infinity())
    }

    fn neg_infinity() -> Self {
        Sci(F::neg_infinity())
    }

    fn neg_zero() -> Self {
        Sci(F::neg_zero())
    }

    fn min_value() -> Self {
        Sci(F::min_value())
    }

    fn min_positive_value() -> Self {
        Sci(F::min_positive_value())
    }

    fn max_value() -> Self {
        Sci(F::max_value())
    }

    fn is_nan(self) -> bool {
        self.0.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    fn is_normal(self) -> bool {
        self.0.is_normal()
    }

    fn classify(self) -> std::num::FpCategory {
        self.0.classify()
    }

    fn floor(self) -> Self {
        Sci(self.0.floor())
    }

    fn ceil(self) -> Self {
        Sci(self.0.ceil())
    }

    fn round(self) -> Self {
        Sci(self.0.round())
    }

    fn trunc(self) -> Self {
        Sci(self.0.trunc())
    }

    fn fract(self) -> Self {
        Sci(self.0.fract())
    }

    fn abs(self) -> Self {
        Sci(self.0.abs())
    }

    fn signum(self) -> Self {
        Sci(self.0.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Sci(F::mul_add(self.0, a.0, b.0))
    }

    fn recip(self) -> Self {
        Sci(self.0.recip())
    }

    fn powi(self, n: i32) -> Self {
        Sci(self.0.powi(n))
    }

    fn powf(self, n: Self) -> Self {
        Sci(self.0.powf(n.0))
    }

    fn sqrt(self) -> Self {
        Sci(self.0.sqrt())
    }

    fn exp(self) -> Self {
        Sci(self.0.exp())
    }

    fn exp2(self) -> Self {
        Sci(self.0.exp2())
    }

    fn ln(self) -> Self {
        Sci(self.0.ln())
    }

    fn log(self, base: Self) -> Self {
        Sci(self.0.log(base.0))
    }

    fn log2(self) -> Self {
        Sci(self.0.log2())
    }

    fn log10(self) -> Self {
        Sci(self.0.log10())
    }

    fn max(self, other: Self) -> Self {
        Sci(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        Sci(self.0.min(other.0))
    }

    fn abs_sub(self, other: Self) -> Self {
        Sci(self.0.abs_sub(other.0))
    }

    fn cbrt(self) -> Self {
        Sci(self.0.cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        Sci(self.0.hypot(other.0))
    }

    fn sin(self) -> Self {
        Sci(self.0.sin())
    }

    fn cos(self) -> Self {
        Sci(self.0.cos())
    }

    fn tan(self) -> Self {
        Sci(self.0.tan())
    }

    fn asin(self) -> Self {
        Sci(self.0.asin())
    }

    fn acos(self) -> Self {
        Sci(self.0.acos())
    }

    fn atan(self) -> Self {
        Sci(self.0.atan())
    }

    fn atan2(self, other: Self) -> Self {
        Sci(self.0.atan2(other.0))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.0.sin_cos();
        (Sci(sin), Sci(cos))
    }

    fn exp_m1(self) -> Self {
        Sci(self.0.exp_m1())
    }

    fn ln_1p(self) -> Self {
        Sci(self.0.ln_1p())
    }

    fn sinh(self) -> Self {
        Sci(self.0.sinh())
    }

    fn cosh(self) -> Self {
        Sci(self.0.cosh())
    }

    fn tanh(self) -> Self {
        Sci(self.0.tanh())
    }

    fn asinh(self) -> Self {
        Sci(self.0.asinh())
    }

    fn acosh(self) -> Self {
        Sci(self.0.acosh())
    }

    fn atanh(self) -> Self {
        Sci(self.0.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.0.integer_decode()
    }
}
