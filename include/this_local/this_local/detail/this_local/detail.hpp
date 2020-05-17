
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
#    include <emmintrin.h>

#    ifdef WIN32_LEAN_AND_MEAN_DEFINED
#        undef WIN32_LEAN_AND_MEAN_DEFINED
#        undef WIN32_LEAN_AND_MEAN
#    endif

#    define _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING

#else

#    include <emmintrin.h>

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

#include "../../../this_local/hedley.h"

#include "../../../this_local/plf_colony.h"
#include "../../../this_local/plf_list.h"  // a queue
#include "../../../this_local/plf_stack.h" // a vector

namespace sax {

HEDLEY_ALWAYS_INLINE void yield ( ) noexcept {
#if ( defined( __clang__ ) or defined( __GNUC__ ) )
    asm( "pause" );
#elif defined( _MSC_VER )
    _mm_pause ( );
#else
#    error support for the babbage engine has ended
#endif
}

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

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool equal_m64 ( void const * const a_, void const * const b_ ) noexcept {
    std::uint64_t a;
    std::memcpy ( &a, a_, sizeof ( a ) );
    std::uint64_t b;
    std::memcpy ( &b, b_, sizeof ( b ) );
    return a == b;
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool equal_m128 ( void const * const a_, void const * const b_ ) noexcept {
    return 0xF == _mm_movemask_ps ( _mm_cmpeq_ps ( _mm_load_ps ( ( float * ) a_ ), _mm_load_ps ( ( float * ) b_ ) ) );
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

#define ever                                                                                                                       \
    ;                                                                                                                              \
    ;

template<typename T, typename Allocator = std::allocator<T>>
class lock_free_plf_stack { // straigth from: C++ Concurrency In Action, 2nd Ed., Listing 7.13 - Anthony Williams

    public:
    using value_type      = T;
    using pointer         = T *;
    using reference       = T &;
    using const_pointer   = T const *;
    using const_reference = T const &;

    struct lf_node;

    struct counted_lf_node_ptr {

        long long external_count = 0;
        lf_node * ptr            = nullptr;
    };

    struct lf_node {

        std::atomic<long long> internal_count = { 0 };
        counted_lf_node_ptr prev              = { 0, this };
        value_type data;

        template<typename... Args>
        lf_node ( Args &&... args_ ) : data{ std::forward<Args> ( args_ )... } {}
    };

    private:
    using lf_nodes = plf::colony<lf_node, typename Allocator::template rebind<lf_node>::other>;

    using lf_nodes_iterator       = typename lf_nodes::iterator;
    using lf_nodes_const_iterator = typename lf_nodes::const_iterator;

    lf_nodes nodes;
    std::atomic<counted_lf_node_ptr> tail;

    void increase_tail_count ( counted_lf_node_ptr & old_counter_ ) noexcept {
        counted_lf_node_ptr new_counter;
        do {
            new_counter = old_counter_;
            new_counter.external_count += 1;
        } while (
            not tail.compare_exchange_strong ( old_counter_, new_counter, std::memory_order_acquire, std::memory_order_relaxed ) );
        old_counter_.external_count = new_counter.external_count;
    }

    lf_nodes_iterator insert_implementation ( lf_nodes_iterator && it_ ) noexcept {
        counted_lf_node_ptr node = { 1, &*it_ };
        node.ptr->prev           = tail.load ( std::memory_order_relaxed );
        while ( not tail.compare_exchange_weak ( node.ptr->prev, node, std::memory_order_release, std::memory_order_relaxed ) )
            yield ( );
        return std::forward<lf_nodes_iterator> ( it_ );
    }

    public:
    [[maybe_unused]] lf_nodes_iterator push ( value_type const & data_ ) {
        return insert_implementation ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] lf_nodes_iterator push ( value_type && data_ ) {
        return insert_implementation ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] lf_nodes_iterator emplace ( Args &&... args_ ) {
        return insert_implementation ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    void pop ( ) noexcept {
        counted_lf_node_ptr old_tail = tail.load ( std::memory_order_relaxed );
        for ( ever ) {
            increase_tail_count ( old_tail );
            if ( lf_node * const ptr = old_tail.ptr; ptr ) {
                if ( tail.compare_exchange_strong ( old_tail, ptr->prev, std::memory_order_relaxed ) ) {
                    int const count_increase = old_tail.external_count - 2;
                    if ( ptr->internal_count.fetch_add ( count_increase, std::memory_order_release ) == -count_increase )
                        nodes.erase ( get_iterator_from_pointer ( ptr ) );
                    return;
                }
                else {
                    if ( ptr->internal_count.fetch_add ( -1, std::memory_order_relaxed ) == 1 ) {
                        ptr->internal_count.load ( std::memory_order_acquire );
                        nodes.erase ( get_iterator_from_pointer ( ptr ) );
                    }
                }
            }
            else {
                return;
            }
        }
    }

    /*

    class iterator {
        friend class lock_free_plf_stack;

        lf_node * p;

        iterator ( lf_node * p_ ) noexcept : p{ std::forward<lf_node *> ( p_ ) } {}

        public:
        using iterator_category = std::forward_iterator_tag;

        iterator ( iterator && it_ ) noexcept : p{ std::forward<lf_node *> ( it_.p ) } {}
        iterator ( iterator const & it_ ) noexcept : p{ it_.p } {}
        [[maybe_unused]] iterator & operator= ( iterator && r_ ) noexcept { p = std::forward<lf_node *> ( r_.p ); }
        [[maybe_unused]] iterator & operator= ( iterator const & r_ ) noexcept { p = r_.p; }

        ~iterator ( ) = default;

        [[maybe_unused]] iterator & operator++ ( ) noexcept {
            p = p->prev.ptr;
            return *this;
        }
        [[nodiscard]] bool operator== ( iterator const & r_ ) const noexcept { return p == r_.p; }
        [[nodiscard]] bool operator!= ( iterator const & r_ ) const noexcept { return p != r_.p; }
        [[nodiscard]] reference operator* ( ) const noexcept { return p->data; }
        [[nodiscard]] pointer operator-> ( ) const noexcept { return &p->data; }
    };

    class const_iterator {
        friend class lock_free_plf_stack;

        lf_node const * p;

        const_iterator ( lf_node const * p_ ) noexcept : p{ std::forward<lf_node const *> ( p_ ) } {}

        public:
        using iterator_category = std::forward_iterator_tag;

        const_iterator ( const_iterator && it_ ) noexcept : p{ std::forward<lf_node *> ( it_.p ) } {}
        const_iterator ( const_iterator const & it_ ) noexcept : p{ it_.p } {}
        [[maybe_unused]] const_iterator & operator= ( const_iterator && r_ ) noexcept {
            p = std::forward<lf_node const *> ( r_.p );
        }
        [[maybe_unused]] const_iterator & operator= ( const_iterator const & r_ ) noexcept { p = r_.p; }

        ~const_iterator ( ) = default;

        [[maybe_unused]] const_iterator & operator++ ( ) noexcept {
            p = p->prev.ptr;
            return *this;
        }
        [[nodiscard]] bool operator== ( const_iterator const & r_ ) const noexcept { return p == r_.p; }
        [[nodiscard]] bool operator!= ( const_iterator const & r_ ) const noexcept { return p != r_.p; }
        [[nodiscard]] const_reference operator* ( ) const noexcept { return p->data; }
        [[nodiscard]] const_pointer operator-> ( ) const noexcept { return &p->data; }
    };

    [[nodiscard]] const_iterator begin ( ) const noexcept { return { tail.load ( std::memory_order_relaxed ).ptr }; }
    [[nodiscard]] const_iterator cbegin ( ) const noexcept { return { tail.load ( std::memory_order_relaxed ).ptr }; }
    [[nodiscard]] iterator begin ( ) noexcept { return { tail.load ( std::memory_order_relaxed ).ptr }; }

    [[nodiscard]] const_iterator end ( ) const noexcept { return { tail.load ( std::memory_order_relaxed ).ptr }; }
    [[nodiscard]] const_iterator cend ( ) const noexcept { return { tail.load ( std::memory_order_relaxed ).ptr }; }
    [[nodiscard]] iterator end ( ) noexcept { return { tail.load ( std::memory_order_relaxed ).ptr }; }

    */

    [[nodiscard]] lf_nodes_const_iterator begin ( ) const noexcept { return nodes.begin ( ); }
    [[nodiscard]] lf_nodes_const_iterator cbegin ( ) const noexcept { return nodes.cbegin ( ); }
    [[nodiscard]] lf_nodes_iterator begin ( ) noexcept { return nodes.begin ( ); }
    [[nodiscard]] lf_nodes_const_iterator end ( ) const noexcept { return nodes.end ( ); }
    [[nodiscard]] lf_nodes_const_iterator cend ( ) const noexcept { return nodes.cend ( ); }
    [[nodiscard]] lf_nodes_iterator end ( ) noexcept { return nodes.end ( ); }

}; // namespace sax

#undef ever

#if defined( _MSC_VER )
#    undef _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING
#endif

} // namespace sax
