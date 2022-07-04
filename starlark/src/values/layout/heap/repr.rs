/*
 * Copyright 2019 The Starlark in Rust Authors.
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

use std::hash::Hash;
use std::mem;
use std::mem::ManuallyDrop;
use std::ptr;

use either::Either;
use gazebo::any::AnyLifetime;
use gazebo::dupe::Dupe;

use crate::values::layout::avalue::AValue;
use crate::values::layout::vtable::AValueDyn;
use crate::values::layout::vtable::AValueVTable;
use crate::values::StarlarkValue;

#[derive(Clone)]
#[repr(transparent)]
pub(crate) struct AValueHeader(pub(crate) &'static AValueVTable);

impl Hash for AValueHeader {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        ptr::hash(self.0, state);
    }
}

impl PartialEq for AValueHeader {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0, other.0)
    }
}

impl Eq for AValueHeader {}

// Implements Copy so this is fine
impl Dupe for AValueHeader {}

/// How object is represented in arena.
#[repr(C)]
pub(crate) struct AValueRepr<T> {
    pub(crate) header: AValueHeader,
    pub(crate) payload: T,
}

/// This is object written over [`AValueRepr`] during GC.
#[repr(C)]
pub(crate) struct AValueForward {
    /// Moved object pointer with lowest bit set.
    pub(crate) forward_ptr: usize,
    /// Size of `<T>`. Does not include [`AValueHeader`].
    pub(crate) object_size: usize,
}

impl AValueForward {
    /// Unpack forward pointer.
    fn forward_ptr(&self) -> usize {
        debug_assert!((self.forward_ptr & 1) != 0);
        self.forward_ptr & !1
    }
}

/// Object on the heap, either a real object or a forward.
#[repr(C)]
pub(crate) union AValueOrForward {
    // We intentionally do not implement `Copy` for these types
    // to avoid accidentally copying them.
    pub(crate) header: ManuallyDrop<AValueHeader>,
    pub(crate) forward: ManuallyDrop<AValueForward>,
    pub(crate) flags: usize,
}

impl AValueOrForward {
    /// Is this pointer a value or forward?
    fn is_forward(&self) -> bool {
        unsafe { (self.flags & 1) != 0 }
    }

    pub(crate) fn unpack(&self) -> Either<&AValueHeader, &AValueForward> {
        if self.is_forward() {
            Either::Right(unsafe { &self.forward })
        } else {
            Either::Left(unsafe { &self.header })
        }
    }
}

impl AValueForward {
    pub(crate) fn assert_does_not_overwrite_extra<'v, T: AValue<'v>>() {
        assert!(mem::size_of::<AValueForward>() <= AValueRepr::<T>::offset_of_extra());
    }
}

impl AValueHeader {
    pub(crate) fn new<'v, T: AValue<'v>>() -> AValueHeader {
        let header = AValueHeader::new_const::<T>();

        let vtable_ptr = header.0 as *const AValueVTable as usize;
        // Check that the LSB is not set, as we reuse that for overwrite
        debug_assert!(vtable_ptr & 1 == 0);

        header
    }

    pub(crate) const fn new_const<'v, T: AValue<'v>>() -> AValueHeader {
        AValueHeader(AValueVTable::new::<T>())
    }

    #[inline]
    pub(crate) fn payload_ptr(&self) -> *const () {
        let self_repr = self as *const AValueHeader as *const AValueRepr<()>;
        unsafe { &(*self_repr).payload }
    }

    pub(crate) unsafe fn payload<'v, T: StarlarkValue<'v>>(&self) -> &T {
        debug_assert_eq!(self.unpack().static_type_of_value(), T::static_type_id());
        &*(self.payload_ptr() as *const T)
    }

    pub(crate) fn unpack<'v>(&'v self) -> AValueDyn<'v> {
        unsafe {
            // TODO: this assertion does not belong here.
            //   Instead, `Value` should be a `Pointer<AValueOrForward>`
            //   instead of `Pointer<AValueHeader>`,
            //   and assertion should be where we unpack the pointer.
            debug_assert!(
                !(*(self as *const AValueHeader as *const AValueOrForward)).is_forward(),
                "value is a forward pointer; value cannot be unpacked during GC or freeze"
            );
        }
        unsafe {
            AValueDyn {
                value: &*self.payload_ptr(),
                vtable: self.0,
            }
        }
    }

    /// Unpack something that might have been overwritten.
    // TODO(nga): this function does not belong here, it should accept `AValueOrForward`.
    pub(crate) fn unpack_overwrite<'v>(&'v self) -> Either<usize, AValueDyn<'v>> {
        let x = unsafe { &*(self as *const AValueHeader as *const AValueOrForward) };
        match x.unpack() {
            Either::Left(header) => Either::Right(header.unpack()),
            Either::Right(forward) => Either::Left(forward.forward_ptr()),
        }
    }

    /// After performing the overwrite any existing pointers to this value
    /// are corrupted.
    pub unsafe fn overwrite_with_forward<'v, T: AValue<'v>>(
        me: *mut AValueRepr<T>,
        forward_ptr: usize,
    ) -> T {
        assert!(forward_ptr & 1 == 0, "Can't have the lowest bit set");

        // TODO(nga): we don't need to do virtual call to obtain memory size
        let sz = (*me).header.unpack().memory_size();
        let p = me as *const AValueRepr<T>;
        let res = ptr::read(p).payload;
        let p = me as *mut AValueForward;
        *p = AValueForward {
            forward_ptr: forward_ptr | 1,
            object_size: sz,
        };
        res
    }

    /// Cast header pointer to repr pointer.
    #[inline]
    pub(crate) unsafe fn as_repr<'v, A: AValue<'v>>(&self) -> &AValueRepr<A> {
        debug_assert_eq!(
            A::StarlarkValue::static_type_id(),
            self.unpack().static_type_of_value()
        );
        &*(self as *const AValueHeader as *const AValueRepr<A>)
    }
}

impl<T> AValueRepr<T> {
    pub(crate) const fn with_metadata(
        metadata: &'static AValueVTable,
        payload: T,
    ) -> AValueRepr<T> {
        AValueRepr {
            header: AValueHeader(metadata),
            payload,
        }
    }

    fn assert_no_padding_between_header_and_payload() {
        // We can make it work when there's padding, but we don't need to,
        // and for now just make an assertion.
        assert!(memoffset::offset_of!(Self, payload) == mem::size_of::<AValueHeader>());
    }

    /// Offset of value extra content relative to `AValueRepr` start.
    pub(crate) fn offset_of_extra<'v>() -> usize
    where
        T: AValue<'v>,
    {
        Self::assert_no_padding_between_header_and_payload();

        mem::size_of::<AValueHeader>() + T::offset_of_extra()
    }

    pub(crate) fn from_payload_ptr_mut(payload_ptr: *mut T) -> *mut AValueRepr<T> {
        let payload_ptr = payload_ptr as usize;
        let header_ptr = payload_ptr - mem::size_of::<AValueHeader>();
        header_ptr as *mut AValueRepr<T>
    }
}