
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
#    define _ENABLE_EXTENDED_ALIGNED_STORAGE

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
#include "../../../this_local/plf_list.h" // a queue
#include "../../../this_local/plf_stack.h"

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

struct alignas ( 16 ) uint128_t {
    long long lo;
    long long hi;
};

template<class T>
inline bool double_compare_and_swap ( volatile T *, T, T ) noexcept;

template<>
[[nodiscard]] inline bool double_compare_and_swap ( volatile sax::uint128_t * destination_, sax::uint128_t result_,
                                                    sax::uint128_t exchange_ ) noexcept {
#if ( defined( __clang__ ) or defined( __GNUC__ ) )
    bool result;
    __asm__ __volatile__( "lock cmpxchg16b %1\n\t"
                          "setz %0"
                          : "=q"( result ), "+m"( *destination_ ), "+d"( result_.hi ), "+a"( result_.lo )
                          : "c"( exchange_.hi ), "b"( exchange_.lo )
                          : "cc" );
    return result;
#else
    return _InterlockedCompareExchange128 ( &destination_->lo, exchange_.hi, exchange_.lo, &result_.lo );
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

    struct node;
    using node_ptr       = node *;
    using const_node_ptr = node const *;

    struct counted_link {

        long long external_count = 0;
        node_ptr ptr;
    };

    struct alignas ( 16 ) node {

        std::atomic<long long> internal_count;
        counted_link next;
        value_type data;

        template<typename... Args>
        node ( Args &&... args_ ) : internal_count{ 0 }, data{ std::forward<Args> ( args_ )... } {}

        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, const_node_ptr n_ ) noexcept {
            std::scoped_lock lock ( lock_free_plf_stack::output_mutex );
            out_ << '<' << lock_free_plf_stack::abbreviate_pointer ( n_ ) << ' '
                 << lock_free_plf_stack::abbreviate_pointer ( n_->next.ptr ) << '>';
            return out_;
        }
        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, node const & n_ ) noexcept {
            return operator<< ( out_, &n_ );
        }
    };

    private:
    using nodes_type = plf::colony<node, typename Allocator::template rebind<node>::other>;

    using nodes_iterator       = typename nodes_type::iterator;
    using nodes_const_iterator = typename nodes_type::const_iterator;

    nodes_type nodes;
    std::atomic<counted_link> head;

    public:
    alignas ( 64 ) static spin_rw_lock<long long> output_mutex;

    private:
    HEDLEY_ALWAYS_INLINE void increase_head_count ( counted_link & old_counter_ ) noexcept {
        counted_link new_counter;
        do {
            new_counter = old_counter_;
            new_counter.external_count += 1;
        } while (
            not head.compare_exchange_strong ( old_counter_, new_counter, std::memory_order_acquire, std::memory_order_relaxed ) );
        old_counter_.external_count = new_counter.external_count;
    }

    HEDLEY_ALWAYS_INLINE nodes_iterator insert_implementation ( nodes_iterator && it_ ) noexcept {
        counted_link node = { 1, &*it_ };
        node.ptr->next    = head.load ( std::memory_order_relaxed );
        while ( not head.compare_exchange_weak ( node.ptr->next, node, std::memory_order_release, std::memory_order_relaxed ) )
            yield ( );
        return std::forward<nodes_iterator> ( it_ );
    }

    public:
    lock_free_plf_stack ( ) : head{ init_head ( ) } {}

    [[maybe_unused]] nodes_iterator push ( value_type const & data_ ) { return insert_implementation ( nodes.emplace ( data_ ) ); }
    [[maybe_unused]] nodes_iterator push ( value_type && data_ ) {
        return insert_implementation ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] nodes_iterator emplace ( Args &&... args_ ) {
        return insert_implementation ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    void pop ( ) noexcept {
        counted_link old_head = head.load ( std::memory_order_relaxed );
        for ( ever ) {
            increase_head_count ( old_head );
            if ( node_ptr const ptr = old_head.ptr; ptr ) {
                if ( head.compare_exchange_strong ( old_head, ptr->next, std::memory_order_relaxed ) ) {
                    int const count_increase = old_head.external_count - 2;
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

    class const_iterator {
        friend class lock_free_plf_stack;

        const_node_ptr p;

        public:
        using iterator_category = std::forward_iterator_tag;

        const_iterator ( const_node_ptr p_ ) noexcept : p{ std::forward<const_node_ptr> ( p_ ) } {}
        const_iterator ( const_iterator && it_ ) noexcept : p{ std::forward<const_node_ptr> ( it_.p ) } {}
        const_iterator ( const_iterator const & it_ ) noexcept : p{ it_.p } {}
        [[maybe_unused]] const_iterator & operator= ( const_iterator && r_ ) noexcept { p = std::forward<const_node_ptr> ( r_.p ); }
        [[maybe_unused]] const_iterator & operator= ( const_iterator const & r_ ) noexcept { p = r_.p; }

        ~const_iterator ( ) = default;

        [[maybe_unused]] const_iterator & operator++ ( ) noexcept {
            p = p->next.ptr;
            return *this;
        }
        [[nodiscard]] bool operator== ( const_iterator const & r_ ) const noexcept { return p == r_.p; }
        [[nodiscard]] bool operator!= ( const_iterator const & r_ ) const noexcept { return p != r_.p; }
        [[nodiscard]] const_reference operator* ( ) const noexcept { return p->data; }
        [[nodiscard]] const_pointer operator-> ( ) const noexcept { return &p->data; }
    };

    class iterator {
        friend class lock_free_plf_stack;

        node_ptr p;

        public:
        using iterator_category = std::forward_iterator_tag;

        iterator ( const_iterator it_ ) noexcept : p{ const_cast<node_ptr> ( std::forward<const_node_ptr> ( it_.p ) ) } {}
        iterator ( node_ptr p_ ) noexcept : p{ std::forward<node_ptr> ( p_ ) } {}
        iterator ( iterator && it_ ) noexcept : p{ std::forward<node_ptr> ( it_.p ) } {}
        iterator ( iterator const & it_ ) noexcept : p{ it_.p } {}
        [[maybe_unused]] iterator & operator= ( iterator && r_ ) noexcept { p = std::forward<node_ptr> ( r_.p ); }
        [[maybe_unused]] iterator & operator= ( iterator const & r_ ) noexcept { p = r_.p; }

        ~iterator ( ) = default;

        [[maybe_unused]] iterator & operator++ ( ) noexcept {
            p = p->next.ptr;
            return *this;
        }
        [[nodiscard]] bool operator== ( iterator const & r_ ) const noexcept { return p == r_.p; }
        [[nodiscard]] bool operator!= ( iterator const & r_ ) const noexcept { return p != r_.p; }
        [[nodiscard]] reference operator* ( ) const noexcept { return p->data; }
        [[nodiscard]] pointer operator-> ( ) const noexcept { return &p->data; }
    };

    [[nodiscard]] const_iterator begin ( ) const noexcept {
        const_iterator it = head.load ( std::memory_order_relaxed ).ptr;
        ++it;
        return it;
    }
    [[nodiscard]] const_iterator end ( ) const noexcept { return head.load ( std::memory_order_relaxed ).ptr; }

    [[nodiscard]] const_iterator cbegin ( ) const noexcept { return begin ( ); }
    [[nodiscard]] const_iterator cend ( ) const noexcept { return end ( ); }
    [[nodiscard]] iterator begin ( ) noexcept { return iterator{ const_cast<lock_free_plf_stack const *> ( this )->begin ( ) }; }
    [[nodiscard]] iterator end ( ) noexcept { return iterator{ const_cast<lock_free_plf_stack const *> ( this )->end ( ) }; }

    */

    [[nodiscard]] nodes_const_iterator begin ( ) const noexcept { return nodes.begin ( ); }
    [[nodiscard]] nodes_const_iterator cbegin ( ) const noexcept { return nodes.cbegin ( ); }
    [[nodiscard]] nodes_iterator begin ( ) noexcept { return nodes.begin ( ); }
    [[nodiscard]] nodes_const_iterator end ( ) const noexcept { return nodes.end ( ); }
    [[nodiscard]] nodes_const_iterator cend ( ) const noexcept { return nodes.cend ( ); }
    [[nodiscard]] nodes_iterator end ( ) noexcept { return nodes.end ( ); }

    template<typename U>
    [[nodiscard]] static constexpr std::uint16_t abbreviate_pointer ( U const * pointer_ ) noexcept {
        std::uintptr_t a = ( std::uintptr_t ) pointer_;
        a >>= log2 ( alignof ( U ) ); // strip lower bits
        a ^= a >> 32;                 // fold high over low
        a ^= a >> 16;                 // fold high over low
        return ( std::uint16_t ) a;
    }

    private:
    [[nodiscard]] static constexpr int log2 ( std::uint64_t v_ ) noexcept {
        int c = !!v_;
        while ( v_ >>= 1 )
            c += 1;
        return c;
    }

    counted_link init_head ( ) {
        nodes_iterator it = nodes.emplace ( );
        counted_link p;
        p.ptr = &*it;
        nodes.erase ( it );
        return p;
    }
};

template<typename T, typename Allocator>
alignas ( 64 ) spin_rw_lock<long long> lock_free_plf_stack<T, Allocator>::output_mutex;

template<typename T, typename Allocator = std::allocator<T>>
class lock_free_plf_list {

    public:
    using value_type      = T;
    using pointer         = T *;
    using reference       = T &;
    using const_pointer   = T const *;
    using const_reference = T const &;

    struct node;
    using node_ptr       = node *;
    using const_node_ptr = node const *;

    struct alignas ( 16 ) counted_link {

        node_ptr prev, next;
        long long external_count = 0, _;
    };

    struct alignas ( 16 ) node {

        counted_link ptr;
        std::atomic<long long> internal_count = 0;

        value_type data;

        template<typename... Args>
        node ( Args &&... args_ ) : internal_count{ 0 }, data{ std::forward<Args> ( args_ )... } {}

        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, const_node_ptr n_ ) noexcept {
            std::scoped_lock lock ( lock_free_plf_list::output_mutex );
            out_ << '<' << lock_free_plf_list::abbreviate_pointer ( n_ ) << ' '
                 << lock_free_plf_list::abbreviate_pointer ( n_->next.ptr ) << '>';
            return out_;
        }
        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, node const & n_ ) noexcept {
            return operator<< ( out_, &n_ );
        }
    };

    private:
    using nodes_type = plf::colony<node, typename Allocator::template rebind<node>::other>;

    using nodes_iterator       = typename nodes_type::iterator;
    using nodes_const_iterator = typename nodes_type::const_iterator;

    nodes_type nodes;
    node_ptr anchor = nullptr;

    public:
    alignas ( 64 ) static spin_rw_lock<long long> output_mutex;

    private:
    HEDLEY_ALWAYS_INLINE void increase_external_count ( counted_link * old_counter_ ) noexcept {
        counted_link new_counter;
        do {
            new_counter = *old_counter_;
            new_counter.external_count += 1;
        } while ( not double_compare_and_swap ( old_counter_, head,
                                                new_counter ) ); // not head.compare_exchange_strong ( old_counter_, new_counter,
                                                                 // std::memory_order_acquire, std::memory_order_relaxed )
        old_counter_->external_count = new_counter.external_count;
    }

    HEDLEY_ALWAYS_INLINE nodes_iterator insert_implementation ( nodes_iterator && it_ ) noexcept {
        counted_link new_node = { 1, &*it_ };
        new_node.ptr->next    = anchor->load ( std::memory_order_relaxed );
        while ( not double_compare_and_swap ( new_node.ptr->next, head,
                                              new_node ) ) // not head.compare_exchange_weak ( new_node.ptr->next, new_node,
                                                           // std::memory_order_release, std::memory_order_relaxed )
            yield ( );
        return std::forward<nodes_iterator> ( it_ );
    }

    HEDLEY_ALWAYS_INLINE nodes_iterator insert_anchor_implementation ( nodes_iterator && it_ ) noexcept {
        anchor                     = &*it_;
        anchor->ptr.prev           = anchor;
        anchor->ptr.next           = anchor;
        anchor->ptr.external_count = 1;
        return std::forward<nodes_iterator> ( it_ );
    }

    public:
    lock_free_plf_list ( ) : head{ init_head ( ) } {}

    [[maybe_unused]] nodes_iterator push ( value_type const & data_ ) {
        if ( HEDLEY_LIKELY ( anchor ) )
            return insert_implementation ( nodes.emplace ( data_ ) );
        return insert_anchor_implementation ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] nodes_iterator push ( value_type && data_ ) {
        if ( HEDLEY_LIKELY ( anchor ) )
            return insert_implementation ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
        return insert_anchor_implementation ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] nodes_iterator emplace ( Args &&... args_ ) {
        if ( HEDLEY_LIKELY ( anchor ) )
            return insert_implementation ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
        return insert_anchor_implementation ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    [[maybe_unused]] nodes_iterator push_anchor ( value_type const & data_ ) {
        return insert_anchor_implementation ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] nodes_iterator push_anchor ( value_type && data_ ) {
        return insert_anchor_implementation ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] nodes_iterator emplace_anchor ( Args &&... args_ ) {
        return insert_anchor_implementation ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    [[maybe_unused]] nodes_iterator push_anchored ( value_type const & data_ ) {
        return insert_implementation ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] nodes_iterator push_anchored ( value_type && data_ ) {
        return insert_implementation ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] nodes_iterator emplace_anchored ( Args &&... args_ ) {
        return insert_implementation ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    /*
    void pop ( ) noexcept {
        counted_link old_head = head.load ( std::memory_order_relaxed );
        for ( ever ) {
            increase_external_count ( &old_head );
            if ( node_ptr const ptr = old_head.ptr; ptr ) {
                if ( head.compare_exchange_strong ( old_head, ptr->next, std::memory_order_relaxed ) ) {
                    int const count_increase = old_head.external_count - 2;
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
    */
    /*

    class const_iterator {
        friend class lock_free_plf_list;

        const_node_ptr p;

        public:
        using iterator_category = std::forward_iterator_tag;

        const_iterator ( const_node_ptr p_ ) noexcept : p{ std::forward<const_node_ptr> ( p_ ) } {}
        const_iterator ( const_iterator && it_ ) noexcept : p{ std::forward<const_node_ptr> ( it_.p ) } {}
        const_iterator ( const_iterator const & it_ ) noexcept : p{ it_.p } {}
        [[maybe_unused]] const_iterator & operator= ( const_iterator && r_ ) noexcept { p = std::forward<const_node_ptr> ( r_.p ); }
        [[maybe_unused]] const_iterator & operator= ( const_iterator const & r_ ) noexcept { p = r_.p; }

        ~const_iterator ( ) = default;

        [[maybe_unused]] const_iterator & operator++ ( ) noexcept {
            p = p->next.ptr;
            return *this;
        }
        [[nodiscard]] bool operator== ( const_iterator const & r_ ) const noexcept { return p == r_.p; }
        [[nodiscard]] bool operator!= ( const_iterator const & r_ ) const noexcept { return p != r_.p; }
        [[nodiscard]] const_reference operator* ( ) const noexcept { return p->data; }
        [[nodiscard]] const_pointer operator-> ( ) const noexcept { return &p->data; }
    };

    class iterator {
        friend class lock_free_plf_list;

        node_ptr p;

        public:
        using iterator_category = std::forward_iterator_tag;

        iterator ( const_iterator it_ ) noexcept : p{ const_cast<node_ptr> ( std::forward<const_node_ptr> ( it_.p ) ) } {}
        iterator ( node_ptr p_ ) noexcept : p{ std::forward<node_ptr> ( p_ ) } {}
        iterator ( iterator && it_ ) noexcept : p{ std::forward<node_ptr> ( it_.p ) } {}
        iterator ( iterator const & it_ ) noexcept : p{ it_.p } {}
        [[maybe_unused]] iterator & operator= ( iterator && r_ ) noexcept { p = std::forward<node_ptr> ( r_.p ); }
        [[maybe_unused]] iterator & operator= ( iterator const & r_ ) noexcept { p = r_.p; }

        ~iterator ( ) = default;

        [[maybe_unused]] iterator & operator++ ( ) noexcept {
            p = p->next.ptr;
            return *this;
        }
        [[nodiscard]] bool operator== ( iterator const & r_ ) const noexcept { return p == r_.p; }
        [[nodiscard]] bool operator!= ( iterator const & r_ ) const noexcept { return p != r_.p; }
        [[nodiscard]] reference operator* ( ) const noexcept { return p->data; }
        [[nodiscard]] pointer operator-> ( ) const noexcept { return &p->data; }
    };

    [[nodiscard]] const_iterator begin ( ) const noexcept {
        const_iterator it = head.load ( std::memory_order_relaxed ).ptr;
        ++it;
        return it;
    }
    [[nodiscard]] const_iterator end ( ) const noexcept { return head.load ( std::memory_order_relaxed ).ptr; }

    [[nodiscard]] const_iterator cbegin ( ) const noexcept { return begin ( ); }
    [[nodiscard]] const_iterator cend ( ) const noexcept { return end ( ); }
    [[nodiscard]] iterator begin ( ) noexcept { return iterator{ const_cast<lock_free_plf_list const *> ( this )->begin ( ) }; }
    [[nodiscard]] iterator end ( ) noexcept { return iterator{ const_cast<lock_free_plf_list const *> ( this )->end ( ) }; }

    */

    [[nodiscard]] nodes_const_iterator begin ( ) const noexcept { return nodes.begin ( ); }
    [[nodiscard]] nodes_const_iterator cbegin ( ) const noexcept { return nodes.cbegin ( ); }
    [[nodiscard]] nodes_iterator begin ( ) noexcept { return nodes.begin ( ); }
    [[nodiscard]] nodes_const_iterator end ( ) const noexcept { return nodes.end ( ); }
    [[nodiscard]] nodes_const_iterator cend ( ) const noexcept { return nodes.cend ( ); }
    [[nodiscard]] nodes_iterator end ( ) noexcept { return nodes.end ( ); }

    template<typename U>
    [[nodiscard]] static constexpr std::uint16_t abbreviate_pointer ( U const * pointer_ ) noexcept {
        std::uintptr_t a = ( std::uintptr_t ) pointer_;
        a >>= log2 ( alignof ( U ) ); // strip lower bits
        a ^= a >> 32;                 // fold high over low
        a ^= a >> 16;                 // fold high over low
        return a;
    }

    private:
    [[nodiscard]] static constexpr int log2 ( std::uint64_t v_ ) noexcept {
        int c = !!v_;
        while ( v_ >>= 1 )
            c += 1;
        return c;
    }
    node_ptr init ( ) {
        nodes_iterator it = nodes.emplace ( );
        node_ptr p        = &*it;
        nodes.erase ( it );
        return p;
    }
};

template<typename T, typename Allocator>
alignas ( 64 ) spin_rw_lock<long long> lock_free_plf_list<T, Allocator>::output_mutex;

#undef ever

#if defined( _MSC_VER )
#    undef _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING
#endif

} // namespace sax
