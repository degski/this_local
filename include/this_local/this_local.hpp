
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

#include "this_local/detail/this_local/detail.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sax {

template<typename Type, typename ValueType, template<typename> typename Allocator = std::allocator>
class this_local {

    using type_pointer_type             = Type *;
    using uninitialized_value_type_type = std::array<char, sizeof ( ValueType )>;

    class this_id { // folds this-pointer to bytes

        using void_pointer_type = void *;

        std::uint32_t const val;

        public:
        this_id ( void_pointer_type && pointer_ ) noexcept :
            val{ ( std::uint32_t ) ( ( ( std::uintptr_t ) std::forward<void_pointer_type> ( pointer_ ) >> 32 ) ^
                                     ( ( std::uintptr_t ) pointer_ ) ) } {}

        [[nodiscard]] bool operator== ( this_id const & r_ ) const noexcept { return val == r_.val; }

        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, this_id const & id_ ) noexcept {
            out_ << id_.val;
            return out_;
        }
    };

    using thread_id = std::thread::id;

    class alignas ( 8 ) key { // 8 bytes

        friend class this_local;

        this_id const self;
        thread_id const thread;

        public:
        key ( this_id && t_, thread_id && i_ ) noexcept :
            self ( std::forward<this_id> ( t_ ) ), thread ( std::forward<thread_id> ( i_ ) ) {}

        [[nodiscard]] bool operator== ( key r_ ) const noexcept { return equal_m64 ( this, &r_ ); }
    };

    struct node {

        key const id;

        alignas ( alignof ( ValueType ) ) uninitialized_value_type_type storage;

        ValueType & operator( ) ( ) noexcept { return *reinterpret_cast<ValueType *> ( &storage ); }
        ValueType const & operator( ) ( ) const noexcept { return *reinterpret_cast<ValueType *> ( &storage ); }

        node ( key && k_ ) noexcept : id ( std::forward<key> ( k_ ) ) {}

        [[nodiscard]] bool operator== ( node const & r_ ) const noexcept { return id == r_.id; }
    };

    template<typename MessageType>
    class alignas ( 16 ) message {

        friend class this_local;

        key const sender;
        key const receiver;
        MessageType contents;

        // re-purpose input-output-operators
        // if ( message << sender ) // if message was sent by sender.
        //     do something ...
        // receiver << message; // send message to

        public:
        template<typename... Args>
        message ( key && s_, key && r_, Args &&... args_ ) noexcept :
            sender{ std::forward<key> ( s_ ) }, receiver{ std::forward<key> ( r_ ) }, contents{ std::forward<Args> ( args_ )... } {}

        [[nodiscard]] bool operator<< ( key const & key_ ) const noexcept { return key_ == sender; }
        [[nodiscard]] bool operator>> ( key const & key_ ) const noexcept { return key_ == receiver; }

        [[nodiscard]] bool operator== ( message const & r_ ) const noexcept { return equal_m128 ( this, std::addressof ( r_ ) ); }
    };

    [[nodiscard]] auto find_implementation ( type_pointer_type && self_ ) const noexcept;
    template<typename... Args>
    [[nodiscard]] ValueType & get_implementation ( type_pointer_type && self_, Args &&... args_ );

    mutable spin_rw_lock<int> rw_mutex;
    plf::list<node, Allocator<node>> nodes;

    public:
    using type           = Type;
    using type_pointer   = type_pointer_type;
    using size_type      = typename plf::list<node, Allocator<node>>::size_type;
    using iterator       = typename plf::list<node, Allocator<node>>::iterator;
    using const_iterator = typename plf::list<node, Allocator<node>>::const_iterator;
    using value_type     = ValueType;
    using allocator      = Allocator<node>;
    using mutex          = spin_rw_lock<int>;

    template<typename... Args>
    [[nodiscard]] value_type & operator( ) ( type_pointer && self_, Args &&... args_ ) noexcept {
        return get ( std::forward<type_pointer> ( self_ ), std::forward<Args> ( args_ )... );
    }
    template<typename... Args>
    [[nodiscard]] value_type const & operator( ) ( type_pointer && self_, Args &&... args_ ) const noexcept {
        return get ( std::forward<type_pointer> ( self_ ), std::forward<Args> ( args_ )... );
    }

    template<typename... Args>
    [[nodiscard]] ValueType & get ( type_pointer_type && self_, Args &&... args_ );

    // Call `this_local::destroy ( this )` in the destructor of type.
    void destroy ( type_pointer self_ ) noexcept;
    // Call `this_local::destroy ( std::this_thread::get_id ( ) )` before return of thread.
    void destroy ( thread_id thread_ ) noexcept;

    [[nodiscard]] const_iterator begin ( ) const noexcept { return nodes.begin ( ); }
    [[nodiscard]] const_iterator cbegin ( ) const noexcept { return nodes.cbegin ( ); }
    [[nodiscard]] iterator begin ( ) noexcept { return nodes.begin ( ); }
    [[nodiscard]] const_iterator end ( ) const noexcept { return nodes.end ( ); }
    [[nodiscard]] const_iterator cend ( ) const noexcept { return nodes.cend ( ); }
    [[nodiscard]] iterator end ( ) noexcept { return nodes.end ( ); }
};

/*
.
.
.
.
.
.
.
.
.
.           '// clang-format off' does not work on empty space
.
.
.
.
.
.
.
.
.
*/

template<typename Type, typename ValueType, template<typename> typename Allocator>
[[nodiscard]] auto this_local<Type, ValueType, Allocator>::find_implementation ( type_pointer_type && self_ ) const noexcept {
    node key_node = { { std::forward<type_pointer_type> ( self_ ), std::this_thread::get_id ( ) } };
    std::scoped_lock lock ( rw_mutex );
    return nodes.unordered_find_single ( key_node );
}

template<typename Type, typename ValueType, template<typename> typename Allocator>
template<typename... Args>
[[nodiscard]] ValueType & this_local<Type, ValueType, Allocator>::get_implementation ( type_pointer_type && self_,
                                                                                       Args &&... args_ ) {
    auto it = find_implementation ( std::forward<type_pointer_type> ( self_ ) );
    if ( nodes.end ( ) != it ) {
        return *reinterpret_cast<ValueType *> ( &it->storage );
    }
    else {
        uninitialized_value_type_type * storage = nullptr;
        {
            std::scoped_lock lock ( rw_mutex );
            storage = &nodes.emplace_back ( key{ type_pointer_type{ self_ }, std::this_thread::get_id ( ) } ).storage;
        }
        return *new ( storage ) ValueType{ std::forward<Args> ( args_ )... };
    }
}

template<typename Type, typename ValueType, template<typename> typename Allocator>
template<typename... Args>
[[nodiscard]] ValueType & this_local<Type, ValueType, Allocator>::get ( type_pointer_type && self_, Args &&... args_ ) {
    static thread_local value_type & this_local_storage =
        get_implementation ( std::forward<type_pointer_type> ( self_ ), std::forward<Args> ( args_ )... );
    return this_local_storage;
}

// Call `this_local::destroy ( this )` in the destructor of type.
template<typename Type, typename ValueType, template<typename> typename Allocator>
void this_local<Type, ValueType, Allocator>::destroy ( type_pointer self_ ) noexcept {
    this_id self = { std::forward<type_pointer> ( self_ ) };
    std::scoped_lock lock ( rw_mutex );
    iterator it = nodes.begin ( ), end = nodes.end ( );
    while ( it != end ) {
        if ( it->id.self == self ) {
            if constexpr ( std::is_class<value_type>::value )
                it->storage.~value_type ( );
            it = nodes.erase ( it );
        }
        else {
            ++it;
        }
    }
}

// Call `this_local::destroy ( std::this_thread::get_id ( ) )` before return of thread.
template<typename Type, typename ValueType, template<typename> typename Allocator>
void this_local<Type, ValueType, Allocator>::destroy ( thread_id thread_ ) noexcept {
    std::scoped_lock lock ( rw_mutex );
    iterator it = nodes.begin ( ), end = nodes.end ( );
    while ( it != end ) {
        if ( it->id.thread == thread_ ) {
            if constexpr ( std::is_class<value_type>::value )
                it->storage.~value_type ( );
            it = nodes.erase ( it );
        }
        else {
            ++it;
        }
    }
}

} // namespace sax
