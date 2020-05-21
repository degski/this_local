
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

namespace at {
struct front {};
struct back {};
}; // namespace at

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
class lock_free_plf_list final {

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

    struct node final : counted_link {

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

    struct counted_node_link final : public counted_link {

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

    // class variables

    public:
    alignas ( 64 ) spin_rw_lock<long long> instance;

    private:
    alignas ( 64 ) std::atomic<counted_node_link> sentinel; // the work-horse
    counted_link end_link;

    nodes_type nodes;
    nodes_iterator ( lock_free_plf_list::*insert_front_implementation ) ( nodes_iterator && ) noexcept;
    nodes_iterator ( lock_free_plf_list::*insert_back_implementation ) ( nodes_iterator && ) noexcept;

    // constructors

    public:
    lock_free_plf_list ( ) :
        insert_front_implementation{ &lock_free_plf_list::insert_init_implementation }, insert_back_implementation{
            &lock_free_plf_list::insert_init_implementation
        } {}

    lock_free_plf_list ( value_type const & data_ ) { insert_init_implementation ( nodes.emplace ( data_ ) ); }
    lock_free_plf_list ( value_type && data_ ) {
        insert_init_implementation ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    lock_free_plf_list ( Args &&... args_ ) {
        insert_init_implementation ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    private:
    HEDLEY_ALWAYS_INLINE void increase_external_count ( counted_link * old_counter_ ) noexcept {
        counted_link new_counter;
        do {
            new_counter = *old_counter_;
            new_counter.external_count += 1;
        } while ( not dwcas ( old_counter_, sentinel.load ( std::memory_order_relaxed ),
                              new_counter ) ); // not head.compare_exchange_strong ( old_counter_, new_counter,
                                               // std::memory_order_acquire, std::memory_order_relaxed )
        old_counter_->external_count = new_counter.external_count;
    }

    HEDLEY_ALWAYS_INLINE void store_sentinel ( node_ptr p_, counted_link l_, unsigned char aba_id_ = 0 ) noexcept {
        std::memcpy ( reinterpret_cast<char *> ( &p_ ) + 7, &aba_id_, 1 ); // little-endian?
        sentinel.store ( { std::forward<counted_link> ( l_ ), std::forward<node_ptr> ( p_ ) }, std::memory_order_relaxed );
    }

    template<typename At>
    [[maybe_unused]] HEDLEY_NEVER_INLINE nodes_iterator insert_regular_implementation ( nodes_iterator && it_ ) noexcept {
        node_ptr new_node       = &*it_;
        counted_node_link old   = sentinel.load ( std::memory_order_relaxed );
        unsigned char new_aba   = std::exchange ( *( reinterpret_cast<char *> ( &old.node ) + 7 ), 0 )++;
        *counted_link::new_node = { old.node, old.node->next };
        store_sentinel ( new_node, { old.node->prev, new_node }, new_aba );
        while ( not dwcas ( *counted_link::old.node, sentinel.load ( std::memory_order_relaxed ), *counted_link::new_node ) ) {
            old = sentinel.load ( std::memory_order_relaxed );
            std::memset ( reinterpret_cast<char *> ( &old.node ) + 7, 0, 1 );
            *counted_link::new_node = { old.node, old.node->next };
            if constexpr ( std::is_same<at::front, At>::value )
                store_sentinel ( old.node, { old.node->prev, new_node }, new_aba );
            else
                store_sentinel ( new_node, { old.node->prev, new_node }, new_aba );
        }
        new_node->next->prev = new_node;
        return std::forward<nodes_iterator> ( it_ );
    }

    template<typename At>
    [[maybe_unused]] HEDLEY_NEVER_INLINE nodes_iterator insert_init_implementation ( nodes_iterator && it_ ) noexcept {
        std::scoped_lock lock ( instance );
        node_ptr new_node       = &*it_;
        *counted_link::new_node = { &end_link, &end_link, 1 };
        end_link.prev = end_link.next = counted_link::new_node;
        if constexpr ( std::is_same<at::front, At>::value ) {
            store_sentinel ( &end_link, { &end_link, &end_link }, 1 );
            insert_front_implementation = &lock_free_plf_list::insert_regular_implementation<at::front>;
        }
        else {
            store_sentinel ( new_node, { &end_link, &end_link }, 1 );
            insert_back_implementation = &lock_free_plf_list::insert_regular_implementation<at::back>;
        }
        return std::forward<nodes_iterator> ( it_ );
    }

    public:
    [[maybe_unused]] nodes_iterator push_back ( value_type const & data_ ) {
        return ( this->*insert_back_implementation ) ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] nodes_iterator push_back ( value_type && data_ ) {
        return ( this->*insert_back_implementation ) ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] nodes_iterator emplace_back ( Args &&... args_ ) {
        return ( this->*insert_back_implementation ) ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    [[maybe_unused]] nodes_iterator push ( value_type const & data_ ) { return push_back ( data_ ); }
    [[maybe_unused]] nodes_iterator push ( value_type && data_ ) { return push_back ( std::forward<value_type> ( data_ ) ); }
    template<typename... Args>
    [[maybe_unused]] nodes_iterator emplace ( Args &&... args_ ) {
        return emplace_back ( std::forward<Args> ( args_ )... );
    }

    [[maybe_unused]] nodes_iterator push_front ( value_type const & data_ ) {
        return ( this->*insert_front_implementation ) ( nodes.emplace ( data_ ) );
    }
    [[maybe_unused]] nodes_iterator push_front ( value_type && data_ ) {
        return ( this->*insert_front_implementation ) ( nodes.emplace ( std::forward<value_type> ( data_ ) ) );
    }
    template<typename... Args>
    [[maybe_unused]] nodes_iterator emplace_front ( Args &&... args_ ) {
        return ( this->*insert_front_implementation ) ( nodes.emplace ( std::forward<Args> ( args_ )... ) );
    }

    class alignas ( 16 ) iterator final {

        friend class lock_free_plf_list;

        node_ptr node, end_node;
        long long skip_end; // will throw on (negative-) overflow, not handled

        iterator ( node_ptr node_, node_ptr end_node_, long long end_passes_ ) noexcept :
            node{ std::forward<node_ptr> ( node_ ) }, end_node{ std::forward<node_ptr> ( end_node_ ) }, skip_end{
                std::forward<long long> ( end_passes_ )
            } {}

        public:
        using iterator_category = std::bidirectional_iterator_tag;

        iterator ( iterator && ) noexcept      = default;
        iterator ( iterator const & ) noexcept = default;

        [[maybe_unused]] iterator & operator= ( iterator && ) noexcept = default;
        [[maybe_unused]] iterator & operator= ( iterator const & ) noexcept = default;

        ~iterator ( ) = default;

        [[maybe_unused]] iterator & operator++ ( ) noexcept {
            node = node->next;
            if ( node == end_node and skip_end ) {
                node = node->next;
                skip_end -= 1;
            }
            return *this;
        }
        [[maybe_unused]] iterator & operator-- ( ) noexcept {
            node = node->prev;
            if ( node == end_node and skip_end ) {
                node = node->prev;
                skip_end -= 1;
            }
            return *this;
        }

        [[nodiscard]] bool operator== ( iterator const & r_ ) const noexcept { return node == r_.node; }
        [[nodiscard]] bool operator!= ( iterator const & r_ ) const noexcept { return node != r_.node; }
        [[nodiscard]] reference operator* ( ) const noexcept { return node->data; }
        [[nodiscard]] pointer operator-> ( ) const noexcept { return &node->data; }
    };

    class alignas ( 16 ) const_iterator final {

        friend class lock_free_plf_list;

        const_node_ptr node, end_node;
        long long skip_end; // will throw on (negative-) overflow, not handled

        const_iterator ( const_node_ptr node_, const_node_ptr end_node_, long long end_passes_ ) noexcept :
            node{ std::forward<const_node_ptr> ( node_ ) }, end_node{ std::forward<const_node_ptr> ( end_node_ ) }, skip_end{
                std::forward<long long> ( end_passes_ )
            } {}

        public:
        using iterator_category = std::bidirectional_iterator_tag;

        const_iterator ( const_iterator && ) noexcept      = default;
        const_iterator ( const_iterator const & ) noexcept = default;
        const_iterator ( iterator const & o_ ) noexcept : node{ o_.node }, end_node{ o_.end_node }, skip_end{ o_.skip_end } {}

        [[maybe_unused]] const_iterator & operator= ( const_iterator && ) noexcept = default;
        [[maybe_unused]] const_iterator & operator= ( const_iterator const & ) noexcept = default;

        [[maybe_unused]] const_iterator & operator= ( iterator const & o_ ) noexcept {
            node     = o_.node;
            end_node = o_.end_node;
            skip_end = o_.skip_end;
        };

        ~const_iterator ( ) = default;

        [[maybe_unused]] const_iterator & operator++ ( ) noexcept {
            node = node->next;
            if ( node == end_node and skip_end ) {
                node = node->next;
                skip_end -= 1;
            }
            return *this;
        }
        [[maybe_unused]] const_iterator & operator-- ( ) noexcept {
            node = node->prev;
            if ( node == end_node and skip_end ) {
                node = node->prev;
                skip_end -= 1;
            }
            return *this;
        }

        [[nodiscard]] bool operator== ( const_iterator const & r_ ) const noexcept { return node == r_.node; }
        [[nodiscard]] bool operator!= ( const_iterator const & r_ ) const noexcept { return node != r_.node; }
        [[nodiscard]] reference operator* ( ) const noexcept { return node->data; }
        [[nodiscard]] pointer operator-> ( ) const noexcept { return &node->data; }
    };

    private:
    [[nodiscard]] const_iterator end_implementation ( long long end_passes_ ) const noexcept {
        return const_iterator{ reinterpret_cast<const_node_ptr> ( &end_link ), reinterpret_cast<const_node_ptr> ( &end_link ),
                               std::forward<long long> ( end_passes_ ) };
    }
    [[nodiscard]] const_iterator cend_implementation ( long long end_passes_ ) const noexcept {
        return end_implementation ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] iterator end_implementation ( long long end_passes_ ) noexcept {
        return const_cast<iterator> ( std::as_const ( this ) )->end_implementation ( std::forward<long long> ( end_passes_ ) );
    }

    public:
    [[nodiscard]] const_iterator begin ( long long end_passes_ = 0 ) const noexcept {
        return ++end_implementation ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] const_iterator cbegin ( long long end_passes_ = 0 ) const noexcept {
        return begin ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] iterator begin ( long long end_passes_ = 0 ) noexcept {
        return const_cast<iterator> ( std::as_const ( this ) )->begin ( std::forward<long long> ( end_passes_ ) );
    }

    [[nodiscard]] const_iterator rbegin ( long long end_passes_ = 0 ) const noexcept {
        return --end_implementation ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] const_iterator crbegin ( long long end_passes_ = 0 ) const noexcept {
        return rbegin ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] iterator rbegin ( long long end_passes_ = 0 ) noexcept {
        return const_cast<iterator> ( std::as_const ( this ) )->rbegin ( std::forward<long long> ( end_passes_ ) );
    }

    [[nodiscard]] const_iterator end ( long long end_passes_ = 0 ) const noexcept {
        return end_implementation ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] const_iterator cend ( long long end_passes_ = 0 ) const noexcept {
        return end ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] iterator end ( long long end_passes_ = 0 ) noexcept {
        return const_cast<iterator> ( std::as_const ( this ) )->end ( std::forward<long long> ( end_passes_ ) );
    }

    [[nodiscard]] const_iterator rend ( long long end_passes_ = 0 ) const noexcept {
        return end ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] const_iterator crend ( long long end_passes_ = 0 ) const noexcept {
        return rend ( std::forward<long long> ( end_passes_ ) );
    }
    [[nodiscard]] iterator rend ( long long end_passes_ = 0 ) noexcept {
        return const_cast<iterator> ( std::as_const ( this ) )->rend ( std::forward<long long> ( end_passes_ ) );
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
