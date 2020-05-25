
// MIT License
//
// Copyright (c) 2020 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#if defined( _MSC_VER )

#    ifndef NOMINMAX
#        define NOMINMAX
#    endif

#    ifndef _AMD64_
#        define _AMD64_
#    endif

#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN_DEFINED
#        define WIN32_LEAN_AND_MEAN
#    endif

#    include <windef.h>
#    include <WinBase.h>

#    include <xmmintrin.h>
#    include <emmintrin.h>
#    include <immintrin.h>

#    ifdef WIN32_LEAN_AND_MEAN_DEFINED
#        undef WIN32_LEAN_AND_MEAN_DEFINED
#        undef WIN32_LEAN_AND_MEAN
#    endif

#    define _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING
#    define _ENABLE_EXTENDED_ALIGNED_STORAGE

#else

#    include <xmmintrin.h>
#    include <emmintrin.h>
#    include <immintrin.h>

#endif

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <atomic>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>

#include <sax/iostream.hpp>

#include "../../../this_local/hedley.h"

#include "../../../this_local/plf_colony.h"
#include "../../../this_local/plf_list.h"
#include "../../../this_local/plf_stack.h"

namespace sax {

inline bool constexpr SAX_ENABLE_OSTREAMS = true;
inline bool constexpr SAX_SYNCED_OSTREAMS = true;

[[nodiscard]] bool has_thread_exited ( std::thread::native_handle_type handle_ ) noexcept {
#if defined( _MSC_VER )
    FILETIME creationTime;
    FILETIME exitTime = { 0, 0 };
    FILETIME kernelTime;
    FILETIME userTime;
    GetThreadTimes ( handle_, &creationTime, &exitTime, &kernelTime, &userTime );
    return exitTime.dwLowDateTime;
#else

#endif
}

namespace detail {
template<typename ValueType>
[[nodiscard]] inline constexpr std::enable_if_t<SAX_ENABLE_OSTREAMS, std::uint16_t>
abbreviate_pointer_implementation ( std::uintptr_t pointer_ ) noexcept {
    if ( pointer_ ) {
        //
        // Byte 8 is empty, byte 7 used to be empty but is now 'reserved' by Microsoft The 'old' 48-bit pointer corresponds to the
        // addressable space of Intel hardware (we'll ignore this one). The below implemented mapping from 2^48 space to 2^16 space,
        // results 2^32 theroretical collisions per pointer. I think in practice (because the space is not cut up randomly, but
        // split into user/kernel space and probably many more partitions internally) the chance of collissions is fairly small.
        //
        pointer_ ^= pointer_ >> 16;
        pointer_ ^= pointer_ >> 32;
        pointer_ ^= pointer_ >> 16;
        return ( std::uint16_t ) pointer_;
    }
    else {
        return 0xFF'FF;
    }
}
} // namespace detail
template<typename ValueType>
[[nodiscard]] inline constexpr std::enable_if_t<SAX_ENABLE_OSTREAMS, std::uint16_t>
abbreviate_pointer ( ValueType const * pointer_ ) noexcept {
    return detail::abbreviate_pointer_implementation<ValueType> ( std::forward<std::uintptr_t> ( ( std::uintptr_t ) pointer_ ) );
}

namespace lockless {

// With a big Thanks to Google Benchmark (the people).
//
// The do_not_optimize (...) function can be used to prevent a value or expression from being optimized away by the compiler. This
// function is intended to add little to no overhead. See: https://youtu.be/nXaxk27zwlk?t=2441
namespace detail {
inline void use_char_pointer ( char const volatile * ) noexcept {}
} // namespace detail
template<typename Anything>
HEDLEY_ALWAYS_INLINE void do_not_optimize ( Anything * value_ ) noexcept {
#if defined( _MSC_VER )
    detail::use_char_pointer ( &reinterpret_cast<char const volatile &> ( value_ ) );
    _ReadWriteBarrier ( );
#elif defined( __clang__ )
    asm volatile( "" : "+r,m"( value_ ) : : "memory" );
#else
    asm volatile( "" : "+m,r"( value_ ) : : "memory" );
#endif
}

// With a big Thanks to Google Benchmark (the people).
//
// Force the compiler to flush pending writes to global memory. Acts as an effective read/write barrier
HEDLEY_ALWAYS_INLINE void clobber_memory ( ) noexcept {
#if defined( _MSC_VER )
    _ReadWriteBarrier ( );
#else
    asm volatile( "" : : : "memory" );
#endif
}

} // namespace lockless

template<typename ValueType>
inline constexpr int hi_index ( ) noexcept {
    short v = 1;
    return *reinterpret_cast<char *> ( &v ) ? sizeof ( ValueType ) - 1 : 0;
}

template<typename ValueType>
inline constexpr int lo_index ( ) noexcept {
    short v = 1;
    return not *reinterpret_cast<char *> ( &v ) ? sizeof ( ValueType ) - 1 : 0;
}

// https://godbolt.org/z/efTuAz https://godbolt.org/z/XQYNzT

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool equal_m64 ( void const * const a_, void const * const b_ ) noexcept {
    __int64 a;
    memcpy ( &a, a_, sizeof ( __int64 ) );
    __int64 b;
    memcpy ( &b, b_, sizeof ( __int64 ) );
    return a == b;
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool equal_m128 ( void const * const a_, void const * const b_ ) noexcept {
    return not _mm_movemask_pd ( _mm_cmpneq_pd ( _mm_load_pd ( ( double const * ) a_ ), _mm_load_pd ( ( double const * ) b_ ) ) );
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool equal_m192 ( void const * const a_, void const * const b_ ) noexcept {
    return equal_m64 ( ( __m64 const * ) a_ + 2, ( __m64 const * ) b_ + 2 ) ? equal_m128 ( a_, b_ ) : false;
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool equal_m256 ( void const * const a_, void const * const b_ ) noexcept {
    return not _mm256_movemask_pd (
        _mm256_cmp_pd ( _mm256_load_pd ( ( double const * ) a_ ), _mm256_load_pd ( ( double const * ) b_ ), _CMP_NEQ_UQ ) );
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool unequal_m64 ( void const * const a_, void const * const b_ ) noexcept {
    return not equal_m64 ( a_, b_ );
}
[[nodiscard]] HEDLEY_ALWAYS_INLINE bool unequal_m128 ( void const * const a_, void const * const b_ ) noexcept {
    return not equal_m128 ( a_, b_ );
}
[[nodiscard]] HEDLEY_ALWAYS_INLINE bool unequal_m192 ( void const * const a_, void const * const b_ ) noexcept {
    return not equal_m192 ( a_, b_ );
}
[[nodiscard]] HEDLEY_ALWAYS_INLINE bool unequal_m256 ( void const * const a_, void const * const b_ ) noexcept {
    return not equal_m256 ( a_, b_ );
}

union _m64 {

    __m64 m64_m64;
    __int32 m64_m32[ 4 ];

    _m64 ( ) = default;

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m64 )>>
    _m64 ( ValueType const & v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m64 ) );
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m64 )>>
    _m64 ( ValueType && v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m64 ) );
    }

    template<typename HalfSizeValueType, typename = std::enable_if_t<sizeof ( HalfSizeValueType ) >= sizeof ( __m64 )>>
    _m64 ( HalfSizeValueType const & o0_, HalfSizeValueType const & o1_ ) noexcept {
        memcpy ( m64_m32 + 0, &o0_, sizeof ( __int32 ) );
        memcpy ( m64_m32 + 1, &o1_, sizeof ( __int32 ) );
    };

    ~_m64 ( ) = default;

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m64 )>>
    std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m64 ), _m64> operator= ( ValueType const & v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m64 ) );
        return *this;
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m64 )>>
    std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m64 ), _m64> operator= ( ValueType && v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m64 ) );
        return *this;
    }

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m64 )>>
    [[nodiscard]] bool operator== ( ValueType const & r_ ) const noexcept {
        return equal_m64 ( this, &r_ );
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m64 )>>
    [[nodiscard]] bool operator!= ( ValueType const & r_ ) const noexcept {
        return unequal_m64 ( this, &r_ );
    }
};
union _m128 {

    __m128 m128_m128;
    __m64 m128_m64[ 2 ];
    __int32 m128_m32[ 4 ];
#if defined( _MSC_VER )
    LONG64 m128_long64[ 2 ];
#endif

    _m128 ( ) = default;

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    _m128 ( ValueType const & v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m128 ) );
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    _m128 ( ValueType && v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m128 ) );
    }

    template<typename HalfSizeValueType, typename = std::enable_if_t<sizeof ( HalfSizeValueType ) >= sizeof ( __m64 )>>
    _m128 ( HalfSizeValueType const & o0_, HalfSizeValueType const & o1_ ) noexcept {
        memcpy ( m128_m64 + 0, &o0_, sizeof ( __m64 ) );
        memcpy ( m128_m64 + 1, &o1_, sizeof ( __m64 ) );
    };

    ~_m128 ( ) = default;

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 ), _m128> operator= ( ValueType const & v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m128 ) );
        return *this;
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 ), _m128> operator= ( ValueType && v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m128 ) );
        return *this;
    }

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    [[nodiscard]] bool operator== ( ValueType const & r_ ) const noexcept {
        return equal_m128 ( this, &r_ );
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    [[nodiscard]] bool operator!= ( ValueType const & r_ ) const noexcept {
        return unequal_m128 ( this, &r_ );
    }
};

union _m256 {

    __m256 m256_m256;
    __m128 m256_m128[ 2 ];
    __m64 m256_m64[ 4 ];
    __int32 m256_m32[ 8 ];

    _m256 ( ) = default;

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m256 )>>
    _m256 ( ValueType const & v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m256 ) );
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m256 )>>
    _m256 ( ValueType && v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m256 ) );
    }

    template<typename HalfSizeValueType, typename = std::enable_if_t<sizeof ( HalfSizeValueType ) >= sizeof ( __m256 )>>
    _m256 ( HalfSizeValueType const & o0_, HalfSizeValueType const & o1_ ) noexcept {
        memcpy ( m256_m128 + 0, &o0_, sizeof ( __m128 ) );
        memcpy ( m256_m128 + 1, &o1_, sizeof ( __m128 ) );
    };

    ~_m256 ( ) = default;

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m256 )>>
    std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m256 ), _m256> operator= ( ValueType const & v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m256 ) );
        return *this;
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m256 )>>
    std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m256 ), _m256> operator= ( ValueType && v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m256 ) );
        return *this;
    }

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m256 )>>
    [[nodiscard]] bool operator== ( ValueType const & r_ ) const noexcept {
        return equal_m256 ( this, &r_ );
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m256 )>>
    [[nodiscard]] bool operator!= ( ValueType const & r_ ) const noexcept {
        return unequal_m256 ( this, &r_ );
    }
};

template<typename ValueType>
using pun_type = std::conditional_t<
    sizeof ( ValueType ) == 32, _m256,
    std::conditional_t<sizeof ( ValueType ) == 16, _m128,
                       std::conditional_t<sizeof ( ValueType ) == 8, _m64,
                                          std::conditional_t<sizeof ( ValueType ) == 4, __int32,
                                                             std::conditional_t<sizeof ( ValueType ) == 2, __int16, __int8>>>>>;

template<typename ValueType>
[[nodiscard]] inline pun_type<ValueType> type_pun ( void const * const value_ ) noexcept {
    pun_type<ValueType> value;
    std::memcpy ( &value, value_, sizeof ( pun_type<ValueType> ) );
    return value;
}

namespace lockless {

// Algorithms built around CAS typically read some key memory location and remember the old value. Based on that old value, they
// compute some new value. Then they try to swap in the new value using CAS, where the comparison checks for the location still
// being equal to the old value. If CAS indicates that the attempt has failed, it has to be repeated from the beginning: the
// location is re-read, a new value is re-computed and the CAS is tried again. Instead of immediately retrying after a CAS
// operation fails, researchers have found that total system performance can be improved in multiprocessor systems—where many
// threads constantly update some particular shared variable if threads that see their CAS fail use exponential backoff, in
// other words, wait a little before retrying the CAS.

template<typename MutexType>
[[nodiscard]] HEDLEY_ALWAYS_INLINE bool soft_dwcas ( _m128 * dest_, _m128 ex_new_, _m128 * cr_old_ ) noexcept {
    alignas ( 64 ) static MutexType cas_mutex;
    std::scoped_lock lock ( cas_mutex );
    bool check = not equal_m128 ( dest_, cr_old_ );
    std::memcpy ( cr_old_, dest_, 16 );
    if ( check )
        return false;
    std::memcpy ( dest_, &ex_new_, 16 );
    return true;
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool dwcas_implementation ( _m128 volatile * dest_, _m128 ex_new_, _m128 * cr_old_ ) noexcept {
#if ( defined( __clang__ ) or defined( __GNUC__ ) )
    bool value;
    __asm__ __volatile__( "lock cmpxchg16b %1\n\t"
                          "setz %0"
                          : "=q"( value ), "+m"( dest_ ), "+d"( cr_old_->m128_m64[ hi_index<short> ( ) ] ),
                            "+a"( cr_old_->m128_m64[ lo_index<short> ( ) ] )
                          : "c"( ex_new_.m128_m64[ hi_index<short> ( ) ] ), "b"( ex_new_.m128_m64[ lo_index<short> ( ) ] )
                          : "cc" );
    return value;
#else
    return _InterlockedCompareExchange128 ( ( long long volatile * ) dest_, ex_new_.m128_long64[ hi_index<short> ( ) ],
                                            ex_new_.m128_long64[ lo_index<short> ( ) ], ( long long * ) cr_old_ );
#endif
}

template<typename T1, typename T2, typename T3>
[[nodiscard]] HEDLEY_ALWAYS_INLINE bool dwcas ( T1 volatile * dest_, T2 && ex_new_, T3 * cr_old_ ) noexcept {
    return dwcas_implementation ( ( _m128 volatile * ) std::forward<T1 volatile *> ( dest_ ), _m128{ std::forward<T2> ( ex_new_ ) },
                                  ( _m128 * ) std::forward<T3 *> ( cr_old_ ) );
}

HEDLEY_ALWAYS_INLINE void yield ( ) noexcept {
#if ( defined( __clang__ ) or defined( __GNUC__ ) )
    asm( "pause" );
#elif defined( _MSC_VER )
    _mm_pause ( );
#else
#    error support for the babbage engine has ended
#endif
}

// Never NULL ttas rw spin-lock.
template<typename FlagType = int>
struct spin_rw_lock final { // test-test-and-set

    spin_rw_lock ( ) noexcept                 = default;
    spin_rw_lock ( spin_rw_lock const & )     = delete;
    spin_rw_lock ( spin_rw_lock && ) noexcept = delete;
    ~spin_rw_lock ( ) noexcept                = default;

    spin_rw_lock & operator= ( spin_rw_lock const & ) = delete;
    spin_rw_lock & operator= ( spin_rw_lock && ) noexcept = delete;

    static constexpr int uninitialized               = 0;
    static constexpr int unlocked                    = 1;
    static constexpr int locked_reader               = 2;
    static constexpr int unlocked_locked_reader_mask = 3;
    static constexpr int locked_writer               = 4;

    HEDLEY_ALWAYS_INLINE void lock ( ) noexcept {
        do {
            while ( unlocked != flag )
                yield ( );
        } while ( not try_lock ( ) );
    }
    [[nodiscard]] HEDLEY_ALWAYS_INLINE bool try_lock ( ) noexcept {
        return unlocked == flag.exchange ( locked_writer, std::memory_order_acquire );
    }
    HEDLEY_ALWAYS_INLINE void unlock ( ) noexcept { flag.store ( unlocked, std::memory_order_release ); }

    HEDLEY_ALWAYS_INLINE void lock ( ) const noexcept {
        do {
            while ( not( unlocked_locked_reader_mask & flag ) )
                yield ( );
        } while ( not try_lock ( ) );
    }
    [[nodiscard]] HEDLEY_ALWAYS_INLINE bool try_lock ( ) const noexcept {
        return unlocked_locked_reader_mask &
               const_cast<std::atomic<int> *> ( &flag )->exchange ( locked_reader, std::memory_order_acquire );
    }
    HEDLEY_ALWAYS_INLINE void unlock ( ) const noexcept {
        const_cast<std::atomic<FlagType> *> ( &flag )->store ( unlocked, std::memory_order_release );
    }

    private:
    std::atomic<int> flag = { unlocked };
};

// Win32 - Slim Reader Writer Lock wrapper.
struct slim_rw_lock final {

    slim_rw_lock ( ) noexcept                 = default;
    slim_rw_lock ( slim_rw_lock const & )     = delete;
    slim_rw_lock ( slim_rw_lock && ) noexcept = delete;
    ~slim_rw_lock ( ) noexcept                = default;

    slim_rw_lock & operator= ( slim_rw_lock const & ) = delete;
    slim_rw_lock & operator= ( slim_rw_lock && ) noexcept = delete;

    // read and write.

    HEDLEY_ALWAYS_INLINE void lock ( ) noexcept { AcquireSRWLockExclusive ( &handle ); }
    [[nodiscard]] HEDLEY_ALWAYS_INLINE bool try_lock ( ) noexcept { return TryAcquireSRWLockExclusive ( &handle ); }
    HEDLEY_ALWAYS_INLINE void unlock ( ) noexcept { ReleaseSRWLockExclusive ( &handle ); }

    // read.

    HEDLEY_ALWAYS_INLINE void lock ( ) const noexcept { AcquireSRWLockShared ( const_cast<PSRWLOCK> ( &handle ) ); }
    [[nodiscard]] HEDLEY_ALWAYS_INLINE bool try_lock ( ) const noexcept {
        return TryAcquireSRWLockShared ( const_cast<PSRWLOCK> ( &handle ) );
    }
    HEDLEY_ALWAYS_INLINE void unlock ( ) const noexcept { ReleaseSRWLockShared ( const_cast<PSRWLOCK> ( &handle ) ); }

    private:
    SRWLOCK handle = SRWLOCK_INIT;
};

} // namespace lockless

#define ever                                                                                                                       \
    ;                                                                                                                              \
    ;

alignas ( 64 ) inline static lockless::spin_rw_lock<long long> ostream_mutex;

struct front_insertion {};
struct back_insertion {};

namespace lockless {
template<typename ValueType, template<typename> typename Allocator = std::allocator, typename DefaultInsertionMode = back_insertion>
class unbounded_circular_list final {

    public:
    using value_type = ValueType;
    template<typename Type>
    using allocator_type         = Allocator<Type>;
    using default_insertion_mode = DefaultInsertionMode;

    struct front_insertion {};
    struct back_insertion {};

    using pointer       = ValueType *;
    using const_pointer = ValueType const *;

    using reference       = ValueType &;
    using const_reference = ValueType const &;

    struct link_type {
        alignas ( 16 ) link_type * prev = nullptr, *next = nullptr;

        [[nodiscard]] bool operator== ( link_type const & r_ ) const noexcept { return equal_m128 ( this, &r_ ); }
        [[nodiscard]] bool operator!= ( link_type const & r_ ) const noexcept { return unequal_m128 ( this, &r_ ); }

        template<typename Stream>
        [[maybe_unused]] friend std::enable_if_t<SAX_ENABLE_OSTREAMS, Stream &> operator<< ( Stream & out_,
                                                                                             link_type const & link_ ) noexcept {
            auto a = [] ( auto p ) { return abbreviate_pointer ( p ); };
            if constexpr ( SAX_SYNCED_OSTREAMS )
                std::scoped_lock lock ( ostream_mutex );
            out_ << "<l " << a ( link_.prev ) << ' ' << a ( link_.next ) << '>';
            return out_;
        }
    };

    struct counted_link_type {
        link_type link;
        unsigned long external_count = 0;

        void set_aba_id ( unsigned char aba_id_ ) noexcept {
            std::memcpy ( reinterpret_cast<unsigned char *> ( &link.prev ) + hi_index<void *> ( ), &aba_id_, 1 );
        }
        // clears the current id
        unsigned char get_aba_id ( ) noexcept {
            return std::exchange ( reinterpret_cast<unsigned char *> ( &link.prev ) + hi_index<void *> ( ), ( unsigned char ) 0 );
        }

        template<typename Stream>
        [[maybe_unused]] friend std::enable_if_t<SAX_ENABLE_OSTREAMS, Stream &>
        operator<< ( Stream & out_, counted_link_type const & counted_ ) noexcept {
            auto a = [] ( auto p ) { return abbreviate_pointer ( p ); };
            if constexpr ( SAX_SYNCED_OSTREAMS )
                std::scoped_lock lock ( ostream_mutex );
            out_ << "<c " << a ( counted_.link.prev ) << ' ' << a ( counted_.link.next ) << '.' << counted_.external_count << '>';
            return out_;
        }
    };

    using counted_link_type_ptr       = counted_link_type *;
    using const_counted_link_type_ptr = counted_link_type const *;

    struct end_node_type final {
        std::atomic<counted_link_type> counted;
        std::atomic<unsigned long> internal_count = { 0 };

        template<typename Stream>
        [[maybe_unused]] friend std::enable_if_t<SAX_ENABLE_OSTREAMS, Stream &>
        operator<< ( Stream & out_, end_node_type const * node_ ) noexcept {
            auto a = [] ( auto p ) { return abbreviate_pointer ( p ); };
            if constexpr ( SAX_SYNCED_OSTREAMS )
                std::scoped_lock lock ( ostream_mutex );
            out_ << "<n " << a ( node_ ) << ' ' << a ( node_->counted.link.prev ) << ' ' << a ( node_->counted.link.next ) << '.'
                 << node_->internal_count << '-' << node_->counted.link.external_count << '>';
            return out_;
        }
        template<typename Stream>
        [[maybe_unused]] friend std::enable_if_t<SAX_ENABLE_OSTREAMS, Stream &>
        operator<< ( Stream & out_, end_node_type const & node_ ) noexcept {
            return operator<< ( out_, &node_ );
        }
    };

    struct node_type final {
        counted_link_type counted;
        std::atomic<unsigned long> internal_count = { 0 };

        template<typename... Args>
        node_type ( Args &&... args_ ) : data{ std::forward<Args> ( args_ )... } {}

        template<typename Stream>
        [[maybe_unused]] friend std::enable_if_t<SAX_ENABLE_OSTREAMS, Stream &> operator<< ( Stream & out_,
                                                                                             node_type const * node_ ) noexcept {
            auto a = [] ( auto p ) { return abbreviate_pointer ( p ); };
            if constexpr ( SAX_SYNCED_OSTREAMS )
                std::scoped_lock lock ( ostream_mutex );
            out_ << "<n " << a ( node_ ) << ' ' << a ( node_->counted.link.prev ) << ' ' << a ( node_->counted.link.next ) << '.'
                 << node_->internal_count << '-' << node_->counted.link.external_count << '>';
            return out_;
        }
        template<typename Stream>
        [[maybe_unused]] friend std::enable_if_t<SAX_ENABLE_OSTREAMS, Stream &> operator<< ( Stream & out_,
                                                                                             node_type const & node_ ) noexcept {
            return operator<< ( out_, &node_ );
        }

        value_type data;
    };

    using node_type_ptr       = node_type *;
    using const_node_type_ptr = node_type const *;

    using storage_type    = plf::colony<node_type, allocator_type<node_type>>;
    using size_type       = typename storage_type::size_type;
    using difference_type = typename storage_type::difference_type;

    using storage_iterator       = typename storage_type::iterator;
    using const_storage_iterator = typename storage_type::const_iterator;

    [[nodiscard]] const_storage_iterator storage_begin ( ) const noexcept { return nodes.begin ( ); }
    [[nodiscard]] const_storage_iterator storage_cbegin ( ) const noexcept { return nodes.cbegin ( ); }
    [[nodiscard]] storage_iterator storage_begin ( ) noexcept { return nodes.begin ( ); }
    [[nodiscard]] const_storage_iterator storage_end ( ) const noexcept { return nodes.end ( ); }
    [[nodiscard]] const_storage_iterator storage_cend ( ) const noexcept { return nodes.cend ( ); }
    [[nodiscard]] storage_iterator storage_end ( ) noexcept { return nodes.end ( ); }

    // class variables

    alignas ( 64 ) spin_rw_lock<long long> instance_mutex;

    private:
    end_node_type end_node;
    storage_type nodes;

    storage_iterator ( unbounded_circular_list::*insert_front_implementation ) ( storage_iterator && ) noexcept;
    storage_iterator ( unbounded_circular_list::*insert_back_implementation ) ( storage_iterator && ) noexcept;

    // constructors

    public:
    unbounded_circular_list ( ) :
        insert_front_implementation{ &unbounded_circular_list::insert_initial_implementation<front_insertion> },
        insert_back_implementation{ &unbounded_circular_list::insert_initial_implementation<back_insertion> } {}

    unbounded_circular_list ( value_type const & data_ ) {
        insert_initial_implementation<DefaultInsertionMode> ( nodes.emplace ( data_ ) );
    }
    unbounded_circular_list ( value_type && data_ ) {
        insert_initial_implementation<DefaultInsertionMode> ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    unbounded_circular_list ( Args &&... args_ ) {
        insert_initial_implementation<DefaultInsertionMode> ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    private:
    [[nodiscard]] HEDLEY_ALWAYS_INLINE counted_link_type load_end_link ( ) const noexcept {
        return end_node.link.load ( std::memory_order_relaxed );
    }

    [[nodiscard]] HEDLEY_ALWAYS_INLINE unsigned char load_end_link ( counted_link_type & l_ ) const noexcept {
        l_ = end_node.link.load ( std::memory_order_relaxed );
        return l_.get_aba_id ( );
    }

    HEDLEY_ALWAYS_INLINE void store_end_link ( counted_link_type l_, unsigned char aba_id_ = 0 ) noexcept {
        l_.set_aba_id ( aba_id_ );
        end_node.store ( { std::forward<counted_link_type> ( l_ ) }, std::memory_order_relaxed );
    }

    template<typename At>
    [[maybe_unused]] HEDLEY_NEVER_INLINE storage_iterator insert_regular_implementation ( storage_iterator && it_ ) noexcept {
        node_type_ptr new_node = &*it_;
        // the body of the cas loop un-rolled once (same as below)
        counted_sentinel old             = sentinel.load ( std::memory_order_relaxed );
        unsigned char new_aba_id         = std::exchange ( *( ( ( char * ) &old.node ) + hi_index<node_type_ptr> ( ) ), 0 ) + 1;
        *( ( counted_link * ) new_node ) = counted_link{ link{ ( link * ) old.node, old.node->next }, 1 };
        if constexpr ( std::is_same<front_insertion, At>::value )
            store_sentinel ( old.node, counted_link{ link{ old.node->prev, ( link * ) new_node }, 1 }, new_aba_id );
        else
            store_sentinel ( new_node, counted_link{ link{ old.node->prev, ( link * ) new_node }, 1 }, new_aba_id );
        // end of un-rolled loop
        while ( not dwcas ( old.node, sentinel.load ( std::memory_order_relaxed ), new_node ) ) {
            old = sentinel.load ( std::memory_order_relaxed );
            std::memset ( ( ( ( char * ) &old.node ) + hi_index<node_type_ptr> ( ) ), 0, 1 );
            *( ( counted_link * ) new_node ) = counted_link{ link{ ( link * ) old.node, old.node->next }, 1 };
            if constexpr ( std::is_same<front_insertion, At>::value )
                store_sentinel ( old.node, counted_link{ link{ old.node->prev, ( link * ) new_node }, 1 }, new_aba_id );
            else
                store_sentinel ( new_node, counted_link{ link{ old.node->prev, ( link * ) new_node }, 1 }, new_aba_id );
        }
        new_node->next->prev = new_node;
        return std::forward<storage_iterator &&> ( it_ );
    }

    template<typename At>
    [[maybe_unused]] HEDLEY_NEVER_INLINE storage_iterator insert_initial_implementation ( storage_iterator && it_ ) noexcept {
        std::scoped_lock lock ( instance_mutex );
        node_type_ptr new_node           = &*it_;
        *( ( counted_link * ) new_node ) = counted_link{ link{ &end_link, &end_link }, 1 };
        end_link                         = link{ ( link * ) new_node, ( link * ) new_node };
        if constexpr ( std::is_same<At, front_insertion>::value ) {
            store_sentinel ( ( node_type_ptr ) &end_link, counted_link{ link{ &end_link, &end_link }, 1 } );
            insert_front_implementation = &unbounded_circular_list::insert_regular_implementation<front_insertion>;
        }
        else {
            store_sentinel ( new_node, counted_link{ link{ &end_link, &end_link }, 1 } );
            insert_back_implementation = &unbounded_circular_list::insert_regular_implementation<back_insertion>;
        }
        return std::forward<storage_iterator> ( it_ );
    }

    public:
    [[maybe_unused]] storage_iterator push_back ( value_type const & data_ ) {
        return ( this->*insert_back_implementation ) ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] storage_iterator push_back ( value_type && data_ ) {
        return ( this->*insert_back_implementation ) ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] storage_iterator emplace_back ( Args &&... args_ ) {
        return ( this->*insert_back_implementation ) ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    [[maybe_unused]] storage_iterator push ( value_type const & data_ ) {
        if constexpr ( std::is_same<DefaultInsertionMode, front_insertion>::value )
            return push_front ( data_ );
        else
            return push_back ( data_ );
    }
    [[maybe_unused]] storage_iterator push ( value_type && data_ ) {
        if constexpr ( std::is_same<DefaultInsertionMode, front_insertion>::value )
            return push_front ( std::forward<value_type> ( data_ ) );
        else
            return push_back ( std::forward<value_type> ( data_ ) );
    }
    template<typename... Args>
    [[maybe_unused]] storage_iterator emplace ( Args &&... args_ ) {
        if constexpr ( std::is_same<DefaultInsertionMode, front_insertion>::value )
            return push_front ( std::forward<Args> ( args_ )... );
        else
            return emplace_back ( std::forward<Args> ( args_ )... );
    }

    [[maybe_unused]] storage_iterator push_front ( value_type const & data_ ) {
        return ( this->*insert_front_implementation ) ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] storage_iterator push_front ( value_type && data_ ) {
        return ( this->*insert_front_implementation ) ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] storage_iterator emplace_front ( Args &&... args_ ) {
        return ( this->*insert_front_implementation ) ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    private:
    template<bool MemoryOrderAcquire>
    void delete_implementation ( node_type_ptr node_ ) noexcept {
        if constexpr ( MemoryOrderAcquire )
            auto const _ = node_->internal_count.load ( std::memory_order_acquire );
        node_->prev = node_->next = nullptr; // !
        nodes.erase ( nodes.get_iterator_from_pointer ( node_ ) );
    }

    void delete_order_relaxed ( node_type_ptr node_ ) noexcept {
        delete_implementation<false> ( std::forward<node_type_ptr> ( node_ ) );
    }
    void delete_order_acquire ( node_type_ptr node_ ) noexcept {
        delete_implementation<true> ( std::forward<node_type_ptr> ( node_ ) );
    }

    public:
    void pop ( ) noexcept {
        counted_sentinel volatile old_sentinel = sentinel.load ( std::memory_order_relaxed );
        for ( ever ) {
            // increase external count
            counted_link new_counter;
            do {
                new_counter = *( ( counted_link * ) &old_sentinel );
                new_counter.external_count += 1;
            } while ( not dwcas ( &old_sentinel, sentinel.load ( std::memory_order_relaxed ), &new_counter ) );
            old_sentinel.external_count = new_counter.external_count;
            // we're poppin', go get the box
            if ( node_type_ptr node = old_sentinel.node; node ) {
                if ( not dwcas ( &old_sentinel, sentinel.load ( std::memory_order_relaxed ), &node->next ) ) {
                    unsigned long count_increase = old_sentinel.external_count - 2;
                    if ( node->internal_count.fetch_add ( count_increase, std::memory_order_release ) == -count_increase )
                        delete_order_relaxed ( node );
                    return;
                }
                else {
                    if ( node->internal_count.fetch_add ( -1, std::memory_order_relaxed ) == 1 )
                        delete_order_acquire ( node );
                }
            }
            else {
                return;
            }
        }
    }

    class concurrent_iterator final {
        friend class unbounded_circular_list;
        friend class iterator;

        alignas ( 16 ) node_type_ptr node, end_node;
        long long skip_end = 0; // will throw on (negative-) overflow, not handled

        concurrent_iterator ( node_type_ptr node_, node_type_ptr end_node_, long long end_passes_ ) noexcept :
            node{ std::forward<node_type_ptr> ( node_ ) }, end_node{ std::forward<node_type_ptr> ( end_node_ ) }, skip_end{
                std::forward<long long> ( end_passes_ )
            } {}

        public:
        using iterator_category = std::bidirectional_iterator_tag;

        concurrent_iterator ( concurrent_iterator && ) noexcept      = default;
        concurrent_iterator ( concurrent_iterator const & ) noexcept = default;

        [[maybe_unused]] concurrent_iterator & operator= ( concurrent_iterator && ) noexcept = default;
        [[maybe_unused]] concurrent_iterator & operator= ( concurrent_iterator const & ) noexcept = default;

        ~concurrent_iterator ( ) = default;

        [[maybe_unused]] concurrent_iterator & operator++ ( ) noexcept {
            node = ( node_type_ptr ) node->next;
            if ( HEDLEY_UNLIKELY ( node == end_node and skip_end-- ) )
                node = ( node_type_ptr ) node->next;
            return *this;
        }
        [[maybe_unused]] concurrent_iterator & operator-- ( ) noexcept {
            if ( not node->prev )
                unbounded_circular_list::repair_back_links ( node );
            node = ( node_type_ptr ) node->prev;
            if ( HEDLEY_UNLIKELY ( node == end_node and skip_end-- ) )
                node = node->prev;
            return *this;
        }

        [[nodiscard]] bool operator== ( concurrent_iterator const & r_ ) const noexcept { return node == r_.node; }
        [[nodiscard]] bool operator!= ( concurrent_iterator const & r_ ) const noexcept { return not operator== ( r_ ); }
        [[nodiscard]] reference operator* ( ) const noexcept { return node->data; }
        [[nodiscard]] pointer operator-> ( ) const noexcept { return &node->data; }
    };

    class const_concurrent_iterator final {
        friend class unbounded_circular_list;
        friend class const_iterator;

        alignas ( 16 ) const_node_type_ptr node, end_node;
        long long skip_end = 0; // will throw on (negative-) overflow, not handled

        const_concurrent_iterator ( const_node_type_ptr node_, const_node_type_ptr end_node_, long long end_passes_ ) noexcept :
            node{ std::forward<const_node_type_ptr> ( node_ ) }, end_node{ std::forward<const_node_type_ptr> ( end_node_ ) },
            skip_end{ std::forward<long long> ( end_passes_ ) } {}

        public:
        using iterator_category = std::bidirectional_iterator_tag;

        const_concurrent_iterator ( const_concurrent_iterator && ) noexcept      = default;
        const_concurrent_iterator ( const_concurrent_iterator const & ) noexcept = default;

        const_concurrent_iterator ( concurrent_iterator const & o_ ) noexcept :
            node{ o_.node }, end_node{ o_.end_node }, skip_end{ o_.skip_end } {}

        [[maybe_unused]] const_concurrent_iterator & operator= ( const_concurrent_iterator && ) noexcept = default;
        [[maybe_unused]] const_concurrent_iterator & operator= ( const_concurrent_iterator const & ) noexcept = default;

        [[maybe_unused]] const_concurrent_iterator & operator= ( concurrent_iterator const & o_ ) noexcept {
            node     = o_.node;
            end_node = o_.end_node;
            skip_end = o_.skip_end;
        };

        ~const_concurrent_iterator ( ) = default;

        [[maybe_unused]] const_concurrent_iterator & operator++ ( ) noexcept {
            node = node->next;
            if ( HEDLEY_UNLIKELY ( node == end_node and skip_end-- ) )
                node = ( const_node_type_ptr ) node->next;
            return *this;
        }
        [[maybe_unused]] const_concurrent_iterator & operator-- ( ) noexcept {
            if ( not node->prev )
                unbounded_circular_list::repair_back_links ( node );
            node = node->prev;
            if ( HEDLEY_UNLIKELY ( node == end_node and skip_end-- ) )
                node = ( const_node_type_ptr ) node->prev;
            return *this;
        }

        [[nodiscard]] bool operator== ( const_concurrent_iterator const & r_ ) const noexcept { return node == r_.node; }
        [[nodiscard]] bool operator!= ( const_concurrent_iterator const & r_ ) const noexcept { return not operator== ( r_ ); }
        [[nodiscard]] const_reference operator* ( ) const noexcept { return node->data; }
        [[nodiscard]] const_pointer operator-> ( ) const noexcept { return &node->data; }
    };

    private:
    friend class concurrent_iterator;
    friend class const_concurrent_iterator;
    friend class iterator;
    friend class const_iterator;

    [[nodiscard]] const_concurrent_iterator end_implementation ( long long end_passes_ ) const noexcept {
        return const_concurrent_iterator{ reinterpret_cast<const_node_type_ptr> ( &end_link ),
                                          reinterpret_cast<const_node_type_ptr> ( &end_link ),
                                          std::forward<long long> ( end_passes_ ) };
    }
    [[nodiscard]] const_concurrent_iterator cend_implementation ( long long end_passes_ ) const noexcept {
        return end_implementation ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] concurrent_iterator end_implementation ( long long end_passes_ ) noexcept {
        return concurrent_iterator{ reinterpret_cast<node_type_ptr> ( &end_link ), reinterpret_cast<node_type_ptr> ( &end_link ),
                                    std::forward<long long> ( end_passes_ ) };
    }

    public:
    [[nodiscard]] const_concurrent_iterator concurrent_begin ( long long end_passes_ = 0 ) const noexcept {
        return ++end_implementation ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] const_concurrent_iterator concurrent_cbegin ( long long end_passes_ = 0 ) const noexcept {
        return begin ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] concurrent_iterator concurrent_begin ( long long end_passes_ = 0 ) noexcept {
        return ++end_implementation ( std::forward<long long> ( end_passes_ ) );
    }

    [[nodiscard]] const_concurrent_iterator concurrent_rbegin ( long long end_passes_ = 0 ) const noexcept {
        return --end_implementation ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] const_concurrent_iterator concurrent_crbegin ( long long end_passes_ = 0 ) const noexcept {
        return rbegin ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] concurrent_iterator concurrent_rbegin ( long long end_passes_ = 0 ) noexcept {
        return --end_implementation ( std::forward<long long> ( end_passes_ ) );
    }

    [[nodiscard]] const_concurrent_iterator concurrent_end ( long long end_passes_ = 0 ) const noexcept {
        return end_implementation ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] const_concurrent_iterator concurrent_cend ( long long end_passes_ = 0 ) const noexcept {
        return end ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] concurrent_iterator concurrent_end ( long long end_passes_ = 0 ) noexcept {
        return end_implementation ( std::forward<long long> ( end_passes_ ) );
    }

    [[nodiscard]] const_concurrent_iterator concurrent_rend ( long long end_passes_ = 0 ) const noexcept {
        return end ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] const_concurrent_iterator concurrent_crend ( long long end_passes_ = 0 ) const noexcept {
        return rend ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] concurrent_iterator concurrent_rend ( long long end_passes_ = 0 ) noexcept {
        return end ( std::forward<long long> ( end_passes_ ) );
    }

    class iterator final {
        friend class unbounded_circular_list;

        alignas ( 16 ) node_type_ptr node, end_node;

        iterator ( node_type_ptr node_, node_type_ptr end_node_ ) noexcept :
            node{ std::forward<node_type_ptr> ( node_ ) }, end_node{ std::forward<node_type_ptr> ( end_node_ ) } {}
        iterator ( concurrent_iterator const & o_ ) noexcept {
            node     = o_.node;
            end_node = o_.end_node;
        }

        public:
        using iterator_category = std::bidirectional_iterator_tag;

        iterator ( iterator && ) noexcept      = default;
        iterator ( iterator const & ) noexcept = default;

        [[maybe_unused]] iterator & operator= ( iterator && ) noexcept = default;
        [[maybe_unused]] iterator & operator= ( iterator const & ) noexcept = default;

        ~iterator ( ) = default;

        [[maybe_unused]] iterator & operator++ ( ) noexcept {
            node = ( node_type_ptr ) node->next;
            return *this;
        }
        [[maybe_unused]] iterator & operator-- ( ) noexcept {
            node = ( node_type_ptr ) node->prev;
            return *this;
        }

        [[nodiscard]] bool operator== ( iterator const & r_ ) const noexcept { return node == r_.node; }
        [[nodiscard]] bool operator!= ( iterator const & r_ ) const noexcept { return not operator== ( r_ ); }
        [[nodiscard]] reference operator* ( ) const noexcept { return node->data; }
        [[nodiscard]] pointer operator-> ( ) const noexcept { return &node->data; }
    };

    class const_iterator final {
        friend class unbounded_circular_list;

        alignas ( 16 ) const_node_type_ptr node, end_node;

        const_iterator ( const_node_type_ptr node_, const_node_type_ptr end_node_ ) noexcept :
            node{ std::forward<const_node_type_ptr> ( node_ ) }, end_node{ std::forward<const_node_type_ptr> ( end_node_ ) } {}
        const_iterator ( const_concurrent_iterator const & o_ ) noexcept {
            node     = o_.node;
            end_node = o_.end_node;
        }

        public:
        using iterator_category = std::bidirectional_iterator_tag;

        const_iterator ( const_iterator && ) noexcept      = default;
        const_iterator ( const_iterator const & ) noexcept = default;

        [[maybe_unused]] const_iterator & operator= ( const_iterator && ) noexcept = default;
        [[maybe_unused]] const_iterator & operator= ( const_iterator const & ) noexcept = default;

        ~const_iterator ( ) = default;

        [[maybe_unused]] const_iterator & operator++ ( ) noexcept {
            node = ( const_node_type_ptr ) node->next;
            return *this;
        }
        [[maybe_unused]] const_iterator & operator-- ( ) noexcept {
            node = ( const_node_type_ptr ) node->prev;
            return *this;
        }

        [[nodiscard]] bool operator== ( const_iterator const & r_ ) const noexcept { return node == r_.node; }
        [[nodiscard]] bool operator!= ( const_iterator const & r_ ) const noexcept { return not operator== ( r_ ); }
        [[nodiscard]] const_reference operator* ( ) const noexcept { return node->data; }
        [[nodiscard]] const_pointer operator-> ( ) const noexcept { return &node->data; }
    };

    [[nodiscard]] const_iterator begin ( ) const noexcept { return ++end_implementation ( 0 ); }
    [[nodiscard]] const_iterator cbegin ( ) const noexcept { return ++cend_implementation ( 0 ); }
    [[nodiscard]] iterator begin ( ) noexcept { return ++end_implementation ( 0 ); }
    [[nodiscard]] const_iterator end ( ) const noexcept { return end_implementation ( 0 ); }
    [[nodiscard]] const_iterator cend ( ) const noexcept { return cend_implementation ( 0 ); }
    [[nodiscard]] iterator end ( ) noexcept { return end_implementation ( 0 ); }

    [[nodiscard]] const_iterator rbegin ( ) const noexcept { return --end_implementation ( 0 ); }
    [[nodiscard]] const_iterator crbegin ( ) const noexcept { return --cend_implementation ( 0 ); }
    [[nodiscard]] iterator rbegin ( ) noexcept { return --end_implementation ( 0 ); }
    [[nodiscard]] const_iterator rend ( ) const noexcept { return end_implementation ( 0 ); }
    [[nodiscard]] const_iterator crend ( ) const noexcept { return cend_implementation ( 0 ); }
    [[nodiscard]] iterator rend ( ) noexcept { return end_implementation ( 0 ); }

    // Atomically returns the new end link.
    link_type reverse ( ) noexcept {
        counted_link_type old_end_link = load_end_link ( ), new_end_link;
        do {
            new_end_link = { old_end_link.next, old_end_link.prev };
        } while ( not dwcas ( &old_end_link, load_end_link ( ), &new_end_link ) );
        return load_end_link ( ).link;
    }

    private:
    static void repair_back_links ( node_type_ptr node_ ) noexcept {
        if ( HEDLEY_LIKELY ( node_ ) ) {
            counted_link * node = ( counted_link * ) node_->next_;
            while ( HEDLEY_LIKELY ( node != ( ( counted_link * ) node_ ) ) ) {
                node->prev = ( counted_link * ) node;
                node       = node->next;
            }
        }
    }

    public:
    void repair_back_links ( ) noexcept { repair_back_links ( ( node_type_ptr ) end_link ); }

    template<typename Stream>
    std::enable_if_t<SAX_ENABLE_OSTREAMS, Stream &> ostream ( Stream & out_ ) noexcept {
        for ( auto & n : nodes )
            out_ << n;
        out_ << std::endl;
        return out_;
    }

    template<typename Stream>
    [[maybe_unused]] friend std::enable_if_t<SAX_ENABLE_OSTREAMS, Stream &>
    operator<< ( Stream & out_, unbounded_circular_list const & list_ ) noexcept {
        return list_.ostream ( out_ );
    }
};

} // namespace lockless

#undef ever

#if defined( _MSC_VER )
#    undef _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING
#endif

} // namespace sax

template<typename T>
class lock_free_stack {
    private:
    struct node;
    struct counted_node_ptr {
        int external_count;
        node * ptr;
    };
    struct node {
        std::shared_ptr<T> data;
        std::atomic<int> internal_count;
        counted_node_ptr next;
        node ( T const & data_ ) : data ( std::make_shared<T> ( data_ ) ), internal_count ( 0 ) {}
    };
    std::atomic<counted_node_ptr> head;
    void increase_head_count ( counted_node_ptr & old_counter ) {
        counted_node_ptr new_counter;
        do {
            new_counter = old_counter;
            ++new_counter.external_count;
        } while (
            !head.compare_exchange_strong ( old_counter, new_counter, std::memory_order_acquire, std::memory_order_relaxed ) );
        old_counter.external_count = new_counter.external_count;
    }

    public:
    ~lock_free_stack ( ) {
        while ( pop ( ) )
            ;
    }
    void push ( T const & data ) {
        counted_node_ptr new_node;
        new_node.ptr            = new node ( data );
        new_node.external_count = 1;
        new_node.ptr->next      = head.load ( std::memory_order_relaxed );
        while ( !head.compare_exchange_weak ( new_node.ptr->next, new_node, std::memory_order_release, std::memory_order_relaxed ) )
            ;
    }
    std::shared_ptr<T> pop ( ) {
        counted_node_ptr old_head = head.load ( std::memory_order_relaxed );
        for ( ;; ) {
            increase_head_count ( old_head );
            node * const ptr = old_head.ptr;
            if ( !ptr ) {
                return std::shared_ptr<T> ( );
            }
            if ( head.compare_exchange_strong ( old_head, ptr->next, std::memory_order_relaxed ) ) {
                std::shared_ptr<T> res;
                res.swap ( ptr->data );
                int const count_increase = old_head.external_count - 2;
                if ( ptr->internal_count.fetch_add ( count_increase, std::memory_order_release ) == -count_increase ) {
                    delete ptr;
                }
                return res;
            }
            else if ( ptr->internal_count.fetch_add ( -1, std::memory_order_relaxed ) == 1 ) {
                ptr->internal_count.load ( std::memory_order_acquire );
                delete ptr;
            }
        }
    }
};
