
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

#include <sax/iostream.hpp>

#include "../../../this_local/hedley.h"

#include "../../../this_local/plf_colony.h"
#include "../../../this_local/plf_list.h"
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

struct alignas ( 16 ) uint128_t {
    long long lo;
    long long hi;
};

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool dwcas ( volatile sax::uint128_t & dest_, sax::uint128_t ex_new_,
                                                sax::uint128_t & cr_old_ ) noexcept {
#if ( defined( __clang__ ) or defined( __GNUC__ ) )
    bool value;
    __asm__ __volatile__( "lock cmpxchg16b %1\n\t"
                          "setz %0"
                          : "=q"( value ), "+m"( dest_ ), "+d"( cr_old_.hi ), "+a"( cr_old_.lo )
                          : "c"( ex_new_.hi ), "b"( ex_new_.lo )
                          : "cc" );
    return value;
#else
    return _InterlockedCompareExchange128 ( ( volatile long long * ) &dest_.lo, ex_new_.hi, ex_new_.lo,
                                            ( long long * ) &cr_old_.lo );
#endif
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool equal_m64 ( void const * const a_, void const * const b_ ) noexcept {
    std::uint64_t a;
    std::memcpy ( &a, a_, 8 );
    std::uint64_t b;
    std::memcpy ( &b, b_, 8 );
    return a == b;
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool equal_m128 ( void const * const a_, void const * const b_ ) noexcept {
    return 0xF == _mm_movemask_ps ( _mm_cmpeq_ps ( _mm_load_ps ( ( float * ) a_ ), _mm_load_ps ( ( float * ) b_ ) ) );
}

// Algorithms built around CAS typically read some key memory location and remember the old value. Based on that old value, they
// compute some new value. Then they try to swap in the new value using CAS, where the comparison checks for the location still
// being equal to the old value. If CAS indicates that the attempt has failed, it has to be repeated from the beginning: the
// location is re-read, a new value is re-computed and the CAS is tried again. Instead of immediately retrying after a CAS
// operation fails, researchers have found that total system performance can be improved in multiprocessor systems—where many
// threads constantly update some particular shared variable if threads that see their CAS fail use exponential backoff, in
// other words, wait a little before retrying the CAS.

template<typename MutexType>
[[nodiscard]] HEDLEY_ALWAYS_INLINE bool soft_dwcas ( sax::uint128_t & dest_, sax::uint128_t ex_new_,
                                                     sax::uint128_t & cr_old_ ) noexcept {
    alignas ( 64 ) static MutexType cas_mutex;
    std::scoped_lock lock ( cas_mutex );
    bool check = not equal_m128 ( &dest_, &cr_old_ );
    std::memcpy ( &cr_old_, &dest_, 16 );
    if ( check )
        return false;
    std::memcpy ( &dest_, &ex_new_, 16 );
    return true;
}

[[nodiscard]] static constexpr int ilog2 ( std::uint64_t v_ ) noexcept {
    int c = !!v_;
    while ( v_ >>= 1 )
        c += 1;
    return c;
}

template<typename T>
[[nodiscard]] static constexpr std::uint16_t abbreviate_pointer ( T const * pointer_ ) noexcept {
    std::uintptr_t a = ( std::uintptr_t ) pointer_;
    a >>= ilog2 ( alignof ( T ) ); // strip lower bits
    a ^= a >> 32;                  // fold high over low
    a ^= a >> 16;                  // fold high over low
    return ( std::uint16_t ) a;
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

        node_ptr next;
        long long external_count = 0;
    };

    struct node {

        counted_link link;
        std::atomic<long long> internal_count;

        template<typename... Args>
        node ( Args &&... args_ ) : internal_count{ 0 }, data{ std::forward<Args> ( args_ )... } {}

        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, const_node_ptr link_ ) noexcept {
            auto ap = [] ( auto p ) { return abbreviate_pointer ( p ); };
            std::scoped_lock lock ( lock_free_plf_stack::global );
            out_ << '<' << ap ( link_ ) << ' ' << ap ( link_->link.next ) << '>';
            return out_;
        }
        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, node const & link_ ) noexcept {
            return operator<< ( out_, &link_ );
        }

        value_type data;
    };

    private:
    using nodes_type = plf::colony<node, typename Allocator::template rebind<node>::other>;

    using nodes_iterator       = typename nodes_type::iterator;
    using nodes_const_iterator = typename nodes_type::const_iterator;

    nodes_type nodes;
    std::atomic<counted_link> head;

    public:
    alignas ( 64 ) static spin_rw_lock<long long> global;

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

    [[maybe_unused]] HEDLEY_ALWAYS_INLINE nodes_iterator insert_regular_implementation ( nodes_iterator && it_ ) noexcept {
        counted_link link = { &*it_, 1 };
        link.next->link   = head.load ( std::memory_order_relaxed );
        while ( not head.compare_exchange_weak ( link.next->link, link, std::memory_order_release, std::memory_order_relaxed ) )
            yield ( );
        return std::forward<nodes_iterator> ( it_ );
    }

    public:
    lock_free_plf_stack ( ) : head{ init_head ( ) } {}

    [[maybe_unused]] nodes_iterator push ( value_type const & data_ ) {
        return insert_regular_implementation ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] nodes_iterator push ( value_type && data_ ) {
        return insert_regular_implementation ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] nodes_iterator emplace ( Args &&... args_ ) {
        return insert_regular_implementation ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    void pop ( ) noexcept {
        counted_link old_head = head.load ( std::memory_order_relaxed );
        for ( ever ) {
            increase_head_count ( old_head );
            if ( node_ptr const next = old_head.next; next ) {
                if ( head.compare_exchange_strong ( old_head, next->link, std::memory_order_relaxed ) ) {
                    int const count_increase = old_head.external_count - 2;
                    if ( next->internal_count.fetch_add ( count_increase, std::memory_order_release ) == -count_increase )
                        nodes.erase ( get_iterator_from_pointer ( next ) );
                    return;
                }
                else {
                    if ( next->internal_count.fetch_add ( -1, std::memory_order_relaxed ) == 1 ) {
                        next->internal_count.load ( std::memory_order_acquire );
                        nodes.erase ( get_iterator_from_pointer ( next ) );
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
            p = p->link.next;
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
            p = p->link.next;
            return *this;
        }
        [[nodiscard]] bool operator== ( iterator const & r_ ) const noexcept { return p == r_.p; }
        [[nodiscard]] bool operator!= ( iterator const & r_ ) const noexcept { return p != r_.p; }
        [[nodiscard]] reference operator* ( ) const noexcept { return p->data; }
        [[nodiscard]] pointer operator-> ( ) const noexcept { return &p->data; }
    };

    [[nodiscard]] const_iterator begin ( ) const noexcept {
        const_iterator it = head.load ( std::memory_order_relaxed ).next;
        ++it;
        return it;
    }
    [[nodiscard]] const_iterator end ( ) const noexcept { return head.load ( std::memory_order_relaxed ).next; }

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

    counted_link init_head ( ) {
        nodes_iterator it = nodes.emplace ( );
        counted_link p;
        p.next = &*it;
        nodes.erase ( it );
        return p;
    }
};

template<typename T, typename Allocator>
alignas ( 64 ) spin_rw_lock<long long> lock_free_plf_stack<T, Allocator>::global;

alignas ( 64 ) inline static spin_rw_lock<long long> global_mutex;

template<typename T, typename Allocator = std::allocator<T>>
class lock_free_plf_list {

    public:
    using value_type      = T;
    using pointer         = T *;
    using reference       = T &;
    using const_pointer   = T const *;
    using const_reference = T const &;

    struct counted_link {

        alignas ( 16 ) counted_link * prev, *next;
        unsigned long external_count;

        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, counted_link const & link_ ) noexcept {
            auto ap = [] ( auto p ) { return abbreviate_pointer ( p ); };
            std::scoped_lock lock ( global_mutex );
            out_ << '<' << ap ( link_.prev ) << ' ' << ap ( link_.next ) << '.' << link_.external_count << '>';
            return out_;
        }
    };

    using counted_link_ptr       = counted_link *;
    using const_counted_link_ptr = counted_link const *;

    struct node : counted_link {

        std::atomic<unsigned long> internal_count = { 0 };

        template<typename... Args>
        node ( Args &&... args_ ) : counted_link{ }, data{ std::forward<Args> ( args_ )... } {}

        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, node const * link_ ) noexcept {
            auto ap = [] ( auto p ) { return abbreviate_pointer ( p ); };
            std::scoped_lock lock ( global_mutex );
            out_ << '<' << ap ( &*link_ ) << ' ' << ap ( link_->prev ) << ' ' << ap ( link_->next ) << '.' << link_->external_count
                 << '>';
            return out_;
        }
        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, node const & link_ ) noexcept {
            return operator<< ( out_, &link_ );
        }

        value_type data;
    };

    using node_ptr       = node *;
    using const_node_ptr = node const *;

    struct alignas ( node ) {
        char _[ sizeof ( node ) ];
    };

    struct counted_node_link : public counted_link {

        node_ptr node = nullptr;

        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, counted_node_link const & link_ ) noexcept {
            auto ap = [] ( auto p ) { return abbreviate_pointer ( p ); };
            std::scoped_lock lock ( global_mutex );
            out_ << '<' << ap ( &link_ ) << ' ' << ap ( link_->prev ) << ' ' << ap ( link_->next ) << '.' << link_->external_count
                 << '>';
            return out_;
        }

        [[nodiscard]] bool operator== ( counted_link const & r_ ) const noexcept {
            return equal_m128 ( this, &r_ );
            // counted_link::prev == r_.prev and counted_link::next == r_.next;
        }
    };

    using counted_node_link_ptr       = counted_node_link *;
    using const_counted_node_link_ptr = counted_node_link const *;

    private:
    using nodes_type = plf::colony<node, typename Allocator::template rebind<node>::other>;

    using nodes_iterator       = typename nodes_type::iterator;
    using nodes_const_iterator = typename nodes_type::const_iterator;

    nodes_type nodes;
    std::atomic<counted_node_link> back;
    nodes_iterator ( lock_free_plf_list::*insert_implementation ) ( nodes_iterator && ) noexcept;

    public:
    spin_rw_lock<char> instance;

    lock_free_plf_list ( ) : insert_implementation{ &lock_free_plf_list::insert_first_implementation } {}

    lock_free_plf_list ( value_type const & data_ ) { insert_first_implementation ( nodes.emplace ( data_ ) ); }
    lock_free_plf_list ( value_type && data_ ) {
        insert_first_implementation ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    lock_free_plf_list ( Args &&... args_ ) {
        insert_first_implementation ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    private:
    HEDLEY_ALWAYS_INLINE void increase_external_count ( counted_link * old_counter_ ) noexcept {
        counted_link new_counter;
        do {
            new_counter = *old_counter_;
            new_counter.external_count += 1;
        } while ( not dwcas ( old_counter_, back.load ( std::memory_order_relaxed ),
                              new_counter ) ); // not head.compare_exchange_strong ( old_counter_, new_counter,
                                               // std::memory_order_acquire, std::memory_order_relaxed )
        old_counter_->external_count = new_counter.external_count;
    }

    HEDLEY_ALWAYS_INLINE void counted_back_store ( node_ptr p_, counted_link l_, char aba_id_ = 0 ) noexcept {
        std::memcpy ( reinterpret_cast<char *> ( &p_ ) + 7, &aba_id_, 1 ); // little-endian?
        back.store ( { std::forward<counted_link> ( l_ ), std::forward<node_ptr> ( p_ ) }, std::memory_order_relaxed );
    }

    [[maybe_unused]] HEDLEY_NEVER_INLINE nodes_iterator insert_regular_implementation ( nodes_iterator && it_ ) noexcept {

        node_ptr new_node = &*it_;

        counted_node_link old = back.load ( std::memory_order_relaxed );
        char new_aba_id       = std::exchange ( *( reinterpret_cast<char> ( &old.node ) + 7 ), 0 )++;

        *( ( counted_link * ) new_node ) = { old.node, old.node->next };
        counted_back_store ( new_node, { old.node->prev, new_node }, new_aba_id );

        while ( not dwcas ( *( ( counted_link * ) old_node ), back.load ( std::memory_order_relaxed ),
                            *( ( counted_link * ) new_node ) ) ) {
            old = back.load ( std::memory_order_relaxed );
            std::memset ( reinterpret_cast<char> ( &old.node ) + 7 ) , 0, 1 )
            *( ( counted_link * ) new_node ) = { old.node, old.node->next };
            counted_back_store ( new_node, { old.node->prev, new_node }, new_aba_id );
        }

        back_store ( regular );
        new_node->next->prev = regular;
        return std::forward<nodes_iterator> ( it_ );
    }

    [[maybe_unused]] HEDLEY_NEVER_INLINE nodes_iterator insert_second_implementation ( nodes_iterator && it_ ) noexcept {
        auto ap = [] ( auto p ) { return abbreviate_pointer ( p ); };
        std::scoped_lock lock ( instance );
        node_ptr second                         = &*it_;
        counted_node_link first                 = back.load ( std::memory_order_relaxed );
        ( ( counted_link & ) *second )          = { ( counted_link_ptr ) first.node, ( counted_link_ptr ) first.node, 1 };
        ( ( counted_link & ) *first.node ).prev = ( ( counted_link & ) *first.node ).next = ( counted_link_ptr ) second;
        back_store ( second );
        insert_implementation = &lock_free_plf_list::insert_regular_implementation;
        return std::forward<nodes_iterator> ( it_ );
    }

    [[maybe_unused]] HEDLEY_NEVER_INLINE nodes_iterator insert_first_implementation ( nodes_iterator && it_ ) noexcept {
        std::scoped_lock lock ( instance );
        node_ptr first                = &*it_;
        ( ( counted_link & ) *first ) = { ( counted_link_ptr ) first, ( counted_link_ptr ) first, 1 };
        back_store ( first );
        insert_implementation = &lock_free_plf_list::insert_second_implementation;
        return std::forward<nodes_iterator> ( it_ );
    }

    public:
    [[maybe_unused]] nodes_iterator push ( value_type const & data_ ) {
        return ( this->*insert_implementation ) ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] nodes_iterator push ( value_type && data_ ) {
        return ( this->*insert_implementation ) ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] nodes_iterator emplace ( Args &&... args_ ) {
        return ( this->*insert_implementation ) ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    /*
    void pop ( ) noexcept {
        counted_link old_head = head.load ( std::memory_order_relaxed );
        for ( ever ) {
            increase_external_count ( &old_head );
            if ( node_ptr const link = old_head.link; link ) {
                if ( head.compare_exchange_strong ( old_head, link->next, std::memory_order_relaxed ) ) {
                    int const count_increase = old_head.external_count - 2;
                    if ( link->internal_count.fetch_add ( count_increase, std::memory_order_release ) == -count_increase )
                        nodes.erase ( get_iterator_from_pointer ( link ) );
                    return;
                }
                else {
                    if ( link->internal_count.fetch_add ( -1, std::memory_order_relaxed ) == 1 ) {
                        link->internal_count.load ( std::memory_order_acquire );
                        nodes.erase ( get_iterator_from_pointer ( link ) );
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
            p = p->next.link;
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
            p = p->next.link;
            return *this;
        }
        [[nodiscard]] bool operator== ( iterator const & r_ ) const noexcept { return p == r_.p; }
        [[nodiscard]] bool operator!= ( iterator const & r_ ) const noexcept { return p != r_.p; }
        [[nodiscard]] reference operator* ( ) const noexcept { return p->data; }
        [[nodiscard]] pointer operator-> ( ) const noexcept { return &p->data; }
    };

    [[nodiscard]] const_iterator begin ( ) const noexcept {
        const_iterator it = head.load ( std::memory_order_relaxed ).link;
        ++it;
        return it;
    }
    [[nodiscard]] const_iterator end ( ) const noexcept { return head.load ( std::memory_order_relaxed ).link; }

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

    template<typename Stream>
    Stream & ostream ( Stream & out_ ) noexcept {
        for ( auto & n : nodes )
            out_ << n;
        out_ << std::endl;
        return out_;
    }

    template<typename Stream>
    [[maybe_unused]] friend Stream & operator<< ( Stream & out_, lock_free_plf_list const & list_ ) noexcept {
        return list_.ostream ( out_ );
    }

    static constexpr int offset_data = static_cast<int> ( offsetof ( node, data ) );
}; // namespace sax

template<typename Stream>
[[maybe_unused]] Stream & operator<< ( Stream & out_, uint128_t const & i_ ) noexcept {
    std::scoped_lock lock ( global_mutex );
    out_ << '<' << i_.lo << ' ' << i_.hi << '>';
    return out_;
}

#undef ever

#if defined( _MSC_VER )
#    undef _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING
#endif

} // namespace sax
