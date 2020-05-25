
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

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool is_equal_m128 ( void const * const a_, void const * const b_ ) noexcept {
    return not _mm_movemask_pd ( _mm_cmpneq_pd ( _mm_load_pd ( ( double const * ) a_ ), _mm_load_pd ( ( double const * ) b_ ) ) );
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool equal_m192 ( void const * const a_, void const * const b_ ) noexcept {
    return equal_m64 ( ( __m64 const * ) a_ + 2, ( __m64 const * ) b_ + 2 ) ? is_equal_m128 ( a_, b_ ) : false;
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool equal_m256 ( void const * const a_, void const * const b_ ) noexcept {
    return not _mm256_movemask_pd (
        _mm256_cmp_pd ( _mm256_load_pd ( ( double const * ) a_ ), _mm256_load_pd ( ( double const * ) b_ ), _CMP_NEQ_UQ ) );
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
        return is_equal_m128 ( this, &r_ );
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
[[nodiscard]] HEDLEY_ALWAYS_INLINE bool soft_compare_and_swap_m128 ( _m128 * dest_, _m128 ex_new_, _m128 * cr_old_ ) noexcept {
    alignas ( 64 ) static MutexType cas_mutex;
    std::scoped_lock lock ( cas_mutex );
    bool check = not is_equal_m128 ( dest_, cr_old_ );
    std::memcpy ( cr_old_, dest_, 16 );
    if ( check )
        return false;
    std::memcpy ( dest_, &ex_new_, 16 );
    return true;
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool compare_and_swap_m128_implementation ( _m128 volatile * dest_, _m128 ex_new_,
                                                                               _m128 * cr_old_ ) noexcept {
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

template<typename T>
[[nodiscard]] HEDLEY_ALWAYS_INLINE bool compare_and_swap_m128 ( T volatile * dest_, T && ex_new_, T * cr_old_ ) noexcept {
    return compare_and_swap_m128_implementation ( ( _m128 volatile * ) std::forward<T volatile *> ( dest_ ),
                                                  _m128{ std::forward<T> ( ex_new_ ) }, ( _m128 * ) std::forward<T *> ( cr_old_ ) );
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
        return unlocked == flag.link_exchange ( locked_writer, std::memory_order_acquire );
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
               const_cast<std::atomic<int> *> ( &flag )->link_exchange ( locked_reader, std::memory_order_acquire );
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

struct insert_before {};
struct insert_after {};

namespace lockless {
template<typename ValueType, template<typename> typename Allocator = std::allocator, typename DefaultInsertionMode = insert_after>
class unbounded_circular_list final {

    public:
    using value_type = ValueType;
    template<typename Type>
    using allocator_type         = Allocator<Type>;
    using default_insertion_mode = DefaultInsertionMode;

    struct insert_before {};
    struct insert_after {};

    using pointer       = ValueType *;
    using const_pointer = ValueType const *;

    using reference       = ValueType &;
    using const_reference = ValueType const &;

    using counter_type = unsigned char;

    struct link_type {
        alignas ( 16 ) link_type * prev = nullptr, *next = nullptr;

        [[maybe_unused]] counter_type fetch_and_add ( int incr_ = 1 ) noexcept {
            counter_type & ctr = *( reinterpret_cast<counter_type *> ( &prev ) + hi_index<void *> ( ) );
            return std::exchange ( ctr, char ( ctr + incr_ ) );
        }

        [[nodiscard]] bool operator== ( link_type const & r_ ) const noexcept { return is_equal_m128 ( this, &r_ ); }
        [[nodiscard]] bool operator!= ( link_type const & r_ ) const noexcept { return not operator== ( this, &r_ ); }

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

    struct pointer_type {
        link_type * value = nullptr;

        [[maybe_unused]] counter_type fetch_and_add ( int incr_ = 1 ) noexcept {
            counter_type & ctr = *( reinterpret_cast<counter_type *> ( &value ) + hi_index<void *> ( ) );
            return std::exchange ( ctr, char ( ctr + incr_ ) );
        }

        [[nodiscard]] bool operator== ( pointer_type const & r_ ) const noexcept { return is_equal_m128 ( this, &r_ ); }
        [[nodiscard]] bool operator!= ( pointer_type const & r_ ) const noexcept { return not operator== ( this, &r_ ); }

        template<typename Stream>
        [[maybe_unused]] friend std::enable_if_t<SAX_ENABLE_OSTREAMS, Stream &>
        operator<< ( Stream & out_, pointer_type const & pointer_ ) noexcept {
            auto a = [] ( auto p ) { return abbreviate_pointer ( p ); };
            if constexpr ( SAX_SYNCED_OSTREAMS )
                std::scoped_lock lock ( ostream_mutex );
            out_ << "<p " << a ( pointer_.value ) << '.' << pointer_.external_count << '>';
            return out_;
        }
    };

    struct end_node_type final {
        link_type link;
        std::atomic<link_type> link_exchange;
        std::atomic<pointer_type> pointer_exchange;
    };

    struct node_type final {
        link_type link;
        value_type data;

        template<typename... Args>
        node_type ( Args &&... args_ ) : data{ std::forward<Args> ( args_ )... } {}
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

    storage_iterator ( unbounded_circular_list::*insert_before_implementation ) ( node_type_ptr, storage_iterator && ) noexcept;
    storage_iterator ( unbounded_circular_list::*insert_after_implementation ) ( node_type_ptr, storage_iterator && ) noexcept;

    // constructors

    public:
    unbounded_circular_list ( ) :
        insert_before_implementation{ &unbounded_circular_list::insert_initial_implementation<insert_before> },
        insert_after_implementation{ &unbounded_circular_list::insert_initial_implementation<insert_after> } {}

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
    [[nodiscard]] HEDLEY_ALWAYS_INLINE link_type get_link_exchange ( ) const noexcept {
        return end_node.link_exchange.load ( std::memory_order_relaxed );
    }
    [[maybe_unused]] HEDLEY_ALWAYS_INLINE counter_type load_link_exchange ( link_type & l_ ) const noexcept {
        l_ = end_node.link_exchange.load ( std::memory_order_relaxed );
        return l_.get_counter ( );
    }
    [[maybe_unused]] HEDLEY_ALWAYS_INLINE link_type store_link_exchange ( link_type l_, counter_type value_ = 0 ) noexcept {
        if ( value_ )
            l_.set_counter ( value_ );
        end_node.link_exchange.store ( l_, std::memory_order_relaxed );
        return std::forward<link_type> ( l_ );
    }

    HEDLEY_ALWAYS_INLINE void make_links_implementation ( node_type_ptr node_a_, node_type_ptr node_b_,
                                                          counter_type value_ ) noexcept {
        node_b_->counted = { &node_a_->counted.link, node_a_->counted.link.next, 1 };
        store_link_exchange ( { node_a_->counted.link.prev, &node_b_->counted.link, 1 }, value_ );
    }

    template<typename Order>
    HEDLEY_ALWAYS_INLINE void make_links ( node_type_ptr node_a_, node_type_ptr node_b_, counter_type value_ ) noexcept {
        if constexpr ( std::is_same<Order, insert_before>::value )
            make_links_implementation ( node_b_, node_a_, value_ );
        else
            make_links_implementation ( node_a_, node_b_, value_ );
    }

    template<typename Order>
    [[maybe_unused]] HEDLEY_NEVER_INLINE storage_iterator insert_regular_implementation ( node_type_ptr old_node_,
                                                                                          storage_iterator && it_ ) noexcept {
        node_type_ptr new_node = &*it_;
        link_pointer_type old;
        counter_type const new_aba_id = load_link_exchange ( old ) + 1;
        make_links<Order> ( old_node_, new_node, new_aba_id );
        while ( not compare_and_swap_m128 ( &old.link, get_link_exchange ( ).link, &new_node->counted.link ) ) {
            load_link_exchange ( old ); // removes and returns the id of ols and clears the id
            make_links<Order> ( old_node_, new_node, new_aba_id );
        }
        new_node->next->prev = new_node;
        return std::forward<storage_iterator &&> ( it_ );
    }

    template<typename Order>
    [[maybe_unused]] HEDLEY_NEVER_INLINE storage_iterator insert_initial_implementation ( node_type_ptr,
                                                                                          storage_iterator && it_ ) noexcept {
        static bool not_yet_created = true;

        std::scoped_lock lock ( instance_mutex );
        if ( not_yet_created ) {
            node_type_ptr new_node = &*it_;
            make_links<Order> ( &end_node.link, new_node, 0 );
            insert_before_implementation = &unbounded_circular_list::insert_regular_implementation<insert_before>;
            insert_after_implementation  = &unbounded_circular_list::insert_regular_implementation<insert_after>;
            not_yet_created              = false;
            return std::forward<storage_iterator> ( it_ );
        }
        else {
            if constexpr ( std::is_same<Order, insert_before>::value )
                return insert_regular_implementation ( reinterpret_cast<node_type_ptr> ( &end_node ),
                                                       std::forward<storage_iterator> ( it_ ) ); // maybe wrong order 50/50
            else
                return insert_regular_implementation ( new_node,
                                                       std::forward<storage_iterator> ( it_ ) ); // maybe wrong order 50/50
        }
    }

    public:
    [[maybe_unused]] storage_iterator push_back ( value_type const & data_ ) {
        return ( this->*insert_after_implementation ) ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] storage_iterator push_back ( value_type && data_ ) {
        return ( this->*insert_after_implementation ) ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] storage_iterator emplace_back ( Args &&... args_ ) {
        return ( this->*insert_after_implementation ) ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    [[maybe_unused]] storage_iterator push ( value_type const & data_ ) {
        if constexpr ( std::is_same<DefaultInsertionMode, insert_before>::value )
            return push_front ( data_ );
        else
            return push_back ( data_ );
    }
    [[maybe_unused]] storage_iterator push ( value_type && data_ ) {
        if constexpr ( std::is_same<DefaultInsertionMode, insert_before>::value )
            return push_front ( std::forward<value_type> ( data_ ) );
        else
            return push_back ( std::forward<value_type> ( data_ ) );
    }
    template<typename... Args>
    [[maybe_unused]] storage_iterator emplace ( Args &&... args_ ) {
        if constexpr ( std::is_same<DefaultInsertionMode, insert_before>::value )
            return push_front ( std::forward<Args> ( args_ )... );
        else
            return emplace_back ( std::forward<Args> ( args_ )... );
    }

    [[maybe_unused]] storage_iterator push_front ( value_type const & data_ ) {
        return ( this->*insert_before_implementation ) ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] storage_iterator push_front ( value_type && data_ ) {
        return ( this->*insert_before_implementation ) ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] storage_iterator emplace_front ( Args &&... args_ ) {
        return ( this->*insert_before_implementation ) ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    private:
    template<bool MemoryOrderAcquire>
    void ordered_delete_implementation ( node_type_ptr node_ ) noexcept {
        if constexpr ( MemoryOrderAcquire )
            auto const _ = node_->internal_count.load ( std::memory_order_acquire );
        node_->prev = node_->next = nullptr; // !
        nodes.erase ( nodes.get_iterator_from_pointer ( node_ ) );
    }

    [[nodiscard]] HEDLEY_ALWAYS_INLINE pointer_type get_pointer_exchange ( ) const noexcept {
        return end_node.pointer_exchange.load ( std::memory_order_relaxed );
    }
    [[maybe_unused]] HEDLEY_ALWAYS_INLINE counter_type load_pointer_exchange ( pointer_type & p_ ) const noexcept {
        p_ = end_node.pointer_exchange.load ( std::memory_order_relaxed );
        return p_.get_counter ( );
    }
    HEDLEY_ALWAYS_INLINE pointer_type store_pointer_exchange ( pointer_type p_, counter_type value_ = 0 ) noexcept {
        if ( value_ )
            p_.set_counter ( value_ );
        end_node.pointer_exchange.store ( p_, std::memory_order_relaxed );
        return std::forward<pointer_type> ( p_ );
    }

    void delete_node_implementation ( node_type_ptr old_node_ ) noexcept {
        pointer_type volatile old_pointer = store_pointer_exchange ( old_node_ );
        for ( ever ) {
            // increase external count
            pointer_type new_pointer;
            do {
                new_pointer = old_pointer;
                new_pointer.fetch_and_add ( 1 );
            } while ( not compare_and_swap_m128 ( &old_pointer, get_pointer_exchange ( ), &new_pointer ) );
            old_pointer = new_pointer;
            // we're poppin', go get the box
            if ( not compare_and_swap_m128 ( &old_pointer, get_pointer_exchange ( ), &old_node_->link.next ) ) {
                counter_type count_increase = old_pointer.fetch_and_add ( -2 );
                if ( old_node_->internal_count.fetch_add ( count_increase, std::memory_order_release ) == -count_increase )
                    ordered_delete_implementation<false> ( std::forward<node_type_ptr> ( old_node_ ) );
                return;
            }
            else {
                if ( old_node_->internal_count.fetch_add ( -1, std::memory_order_relaxed ) == 1 )
                    ordered_delete_implementation<true> ( std::forward<node_type_ptr> ( old_node_ ) );
            }
        }
    }

    public:
    // Atomically returns the new end link.
    link_type reverse ( ) noexcept {
        pointer_type old_end_link = load_link_exchange ( ), new_end_link;
        do {
            new_end_link = { old_end_link.next, old_end_link.prev };
        } while ( not compare_and_swap_m128 ( &old_end_link, load_link_exchange ( ), &new_end_link ) );
        return load_link_exchange ( ).link;
    }

#include "iterators.inl"

    private:
    static void repair_after_links ( node_type_ptr node_ ) noexcept {
        if ( HEDLEY_LIKELY ( node_ ) ) {
            counted_link * node = ( counted_link * ) node_->next_;
            while ( HEDLEY_LIKELY ( node != ( ( counted_link * ) node_ ) ) ) {
                node->prev = ( counted_link * ) node;
                node       = node->next;
            }
        }
    }

    public:
    void repair_after_links ( ) noexcept { repair_after_links ( ( node_type_ptr ) end_link ); }

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
}; // namespace lockless

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
        while (
            not head.compare_exchange_weak ( new_node.ptr->next, new_node, std::memory_order_release, std::memory_order_relaxed ) )
            continue;
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

/* Given a reference (pointer to pointer) to the head of a list
   and a position, deletes the node at the given position */
void deleteNode ( struct Node ** head_ref, int position ) {
    // If linked list is empty
    if ( *head_ref == NULL )
        return;

    // Store head node
    struct Node * temp = *head_ref;

    // If head needs to be removed
    if ( position == 0 ) {
        *head_ref = temp->next; // Change head
        free ( temp );          // free old head
        return;
    }

    // Find previous node of the node to be deleted
    for ( int i = 0; temp != NULL && i < position - 1; i++ )
        temp = temp->next;

    // If position is more than number of ndoes
    if ( temp == NULL || temp->next == NULL )
        return;

    // Node temp->next is the node to be deleted
    // Store pointer to the next of node to be deleted
    struct Node * next = temp->next->next;

    // Unlink the node from linked list
    free ( temp->next ); // Free memory

    temp->next = next; // Unlink the deleted node from list
}
