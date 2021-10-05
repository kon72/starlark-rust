/*
 * Copyright 2018 The Starlark in Rust Authors.
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! The floating point number type (3.14, 4e2).

use crate::values::{
    num::Num, AllocFrozenValue, AllocValue, FrozenHeap, FrozenValue, Heap, SimpleValue,
    StarlarkValue, Value, ValueError,
};
use gazebo::{any::AnyLifetime, prelude::*};
use std::{
    cmp::Ordering,
    fmt::{self, Display, Write},
};

#[derive(Clone, Dupe, Copy, Debug, AnyLifetime)]
pub struct StarlarkFloat(pub f64);

impl StarlarkFloat {
    /// The result of calling `type()` on floats.
    pub const TYPE: &'static str = "float";
}

impl<'v> AllocValue<'v> for f64 {
    fn alloc_value(self, heap: &'v Heap) -> Value<'v> {
        heap.alloc_simple(StarlarkFloat(self))
    }
}

impl AllocFrozenValue for f64 {
    fn alloc_frozen_value(self, heap: &FrozenHeap) -> FrozenValue {
        heap.alloc_simple(StarlarkFloat(self))
    }
}

impl SimpleValue for StarlarkFloat {}

fn f64_arith_bin_op<'v, F>(
    left: f64,
    right: Value,
    heap: &'v Heap,
    op: &'static str,
    f: F,
) -> anyhow::Result<Value<'v>>
where
    F: FnOnce(f64, f64) -> anyhow::Result<f64>,
{
    if let Some(right) = right.unpack_num().map(|n| n.as_float()) {
        Ok(heap.alloc_simple(StarlarkFloat(f(left, right)?)))
    } else {
        ValueError::unsupported_with(&StarlarkFloat(left), op, right)
    }
}

impl Display for StarlarkFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_nan() {
            write!(f, "nan")
        } else if self.0.is_infinite() {
            if self.0.is_sign_positive() {
                write!(f, "+inf")
            } else {
                write!(f, "-inf")
            }
        } else if self.0.fract() == 0.0 {
            write!(f, "{:.1}", self.0)
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl<'v> StarlarkValue<'v> for StarlarkFloat {
    starlark_type!(StarlarkFloat::TYPE);

    fn equals(&self, other: Value) -> anyhow::Result<bool> {
        if other.unpack_num().is_some() {
            Ok(self.compare(other)? == Ordering::Equal)
        } else {
            Ok(false)
        }
    }

    fn collect_repr(&self, s: &mut String) {
        write!(s, "{}", self).unwrap()
    }

    fn to_json(&self) -> anyhow::Result<String> {
        // NaN/Infinity are not part of the JSON spec,
        // but it's unclear what should go here.
        // Perhaps strings with these values? null?
        // Leave it with these values for now.
        Ok(if self.0.is_nan() {
            "NaN".to_owned()
        } else if self.0.is_infinite() {
            if self.0.is_sign_positive() {
                "Infinity"
            } else {
                "-Infinity"
            }
            .to_owned()
        } else {
            self.to_string()
        })
    }

    fn to_bool(&self) -> bool {
        self.0 != 0.0
    }

    fn get_hash(&self) -> anyhow::Result<u64> {
        Ok(Num::from(self.0).get_hash())
    }

    fn plus(&self, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        Ok(heap.alloc_simple(*self))
    }

    fn minus(&self, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        Ok(heap.alloc_simple(StarlarkFloat(-self.0)))
    }

    fn add(&self, other: Value, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        f64_arith_bin_op(self.0, other, heap, "+", |l, r| Ok(l + r))
    }

    fn sub(&self, other: Value, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        f64_arith_bin_op(self.0, other, heap, "-", |l, r| Ok(l - r))
    }

    fn mul(&self, other: Value<'v>, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        f64_arith_bin_op(self.0, other, heap, "*", |l, r| Ok(l * r))
    }

    fn div(&self, other: Value, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        f64_arith_bin_op(self.0, other, heap, "/", |l, r| {
            if r == 0.0 {
                Err(ValueError::DivisionByZero.into())
            } else {
                Ok(l / r)
            }
        })
    }

    fn percent(&self, other: Value, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        f64_arith_bin_op(self.0, other, heap, "%", |a, b| {
            if b == 0.0 {
                Err(ValueError::DivisionByZero.into())
            } else {
                let r = a % b;
                if r == 0.0 {
                    Ok(0.0)
                } else {
                    Ok(if b.signum() != r.signum() { r + b } else { r })
                }
            }
        })
    }

    fn floor_div(&self, other: Value, heap: &'v Heap) -> anyhow::Result<Value<'v>> {
        f64_arith_bin_op(self.0, other, heap, "//", |l, r| {
            if r == 0.0 {
                Err(ValueError::DivisionByZero.into())
            } else {
                Ok((l / r).floor())
            }
        })
    }

    fn compare(&self, other: Value) -> anyhow::Result<Ordering> {
        if let Some(other_float) = other.unpack_num().map(|n| n.as_float()) {
            // According to the spec (https://github.com/bazelbuild/starlark/blob/689f54426951638ef5b7c41a14d8fc48e65c5f77/spec.md#floating-point-numbers)
            // All NaN values compare equal to each other, but greater than any non-NaN float value.
            match (self.0.is_nan(), other_float.is_nan()) {
                (true, true) => Ok(Ordering::Equal),
                (true, false) => Ok(Ordering::Greater),
                (false, true) => Ok(Ordering::Less),
                (false, false) => {
                    if let Some(ordering) = self.0.partial_cmp(&other_float) {
                        Ok(ordering)
                    } else {
                        // This shouldn't happen as we handle potential NaNs above
                        ValueError::unsupported_with(self, "==", other)
                    }
                }
            }
        } else {
            ValueError::unsupported_with(self, "==", other)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::assert;

    #[test]
    fn test_arithmetic_operators() {
        assert::all_true(
            r#"
+1.0 == 1.0
-1.0 == 0. - 1.
1.0 + 2.0 == 3.0
1.0 - 2.0 == -1.0
2.0 * 3.0 == 6.0
5.0 / 2.0 == 2.5
5.0 % 3.0 == 2.0
5.0 // 2.0 == 2.0
"#,
        );
    }

    #[test]
    fn test_dictionary_key() {
        assert::pass(
            r#"
x = {0: 123}
assert_eq(x[0], 123)
assert_eq(x[0.0], 123)
assert_eq(x[-0.0], 123)
assert_eq(1 in x, False)
        "#,
        );
    }

    #[test]
    fn test_comparisons() {
        assert::all_true(
            r#"
+0.0 == -0.0
0.0 == 0
0 == 0.0
0 < 1.0
0.0 < 1
1 > 0.0
1.0 > 0
0.0 < float("nan")
float("+inf") < float("nan")
"#,
        );
    }

    #[test]
    fn test_comparisons_by_sorting() {
        assert::all_true(
            r#"
sorted([float('inf'), float('-inf'), float('nan'), 1e300, -1e300, 1.0, -1.0, 1, -1, 1e-300, -1e-300, 0, 0.0, float('-0.0'), 1e-300, -1e-300]) == [float('-inf'), -1e+300, -1.0, -1, -1e-300, -1e-300, 0, 0.0, -0.0, 1e-300, 1e-300, 1.0, 1, 1e+300, float('+inf'), float('nan')]
"#,
        );
    }
}
