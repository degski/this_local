
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

#include "./include/this_local/this_local.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <jthread>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <random>
#include <sax/iostream.hpp>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <plf/plf_nanotimer.h>

#include <sax/prng_sfc.hpp>
#include <sax/uniform_int_distribution.hpp>

#ifdef NDEBUG
#    define RANDOM true
#else
#    define RANDOM false
#endif

namespace test {
// Creates a new ID.
[[nodiscard]] inline int get_id ( bool ) noexcept {
    static std::atomic<int> global_id = 1;
    return global_id++;
}
// Returns ID of this thread.
[[nodiscard]] inline int get_id ( ) noexcept {
    static thread_local int thread_local_id = get_id ( false );
    return thread_local_id;
}
} // namespace test

namespace sfc {

// sfc64 - Chris Doty-Humphrey’s Small Fast Chaotic PRNG (cycle ~2^255, state 256).
//
// http://pracrand.sourceforge.net/RNG_engines.txt
// https://numpy.org/devdocs/reference/random/bit_generators/sfc64.html (noted as fastest prng)
// https://gist.github.com/imneme/f1f7821f07cf76504a97f6537c818083 (the (this repo's) reference implementation)

using the_babbage_engine_is_no_longer_supported = void;
using sfc                                       = std::conditional_t<
    UINTPTR_MAX == 0xFFFF'FFFF, sax::sfc32,
    std::conditional_t<UINTPTR_MAX == 0xFFFF'FFFF'FFFF'FFFF, sax::sfc64, the_babbage_engine_is_no_longer_supported>>;

[[nodiscard]] inline auto & generator ( ) noexcept {
    if constexpr ( RANDOM ) {
        static thread_local sfc generator ( sax::os_seed ( ), sax::os_seed ( ), sax::os_seed ( ), sax::os_seed ( ) );
        return generator;
    }
    else {
        static thread_local sfc generator ( sax::fixed_seed ( ) + test::get_id ( ) );
        return generator;
    }
}
} // namespace sfc

auto & rng = sfc::generator ( ); // always a reference, avoids checking for the tls-object's creation on every access.

namespace test {
inline void micro_sleep ( ) noexcept {
    std::this_thread::sleep_for ( std::chrono::microseconds ( sax::uniform_int_distribution<int> ( 0, 16 ) ( rng ) ) );
}
[[nodiscard]] inline constexpr int log2 ( std::uint64_t v_ ) noexcept {
    int c = !!v_;
    while ( v_ >>= 1 )
        c += 1;
    return c;
}
template<typename U>
[[nodiscard]] inline constexpr std::uint16_t abbreviate_pointer ( U const * pointer_ ) noexcept {
    std::uintptr_t a = ( std::uintptr_t ) pointer_;
    a >>= log2 ( alignof ( U ) ); // strip lower bits
    a ^= a >> 32;                 // fold high over low
    a ^= a >> 16;                 // fold high over low
    return a;
}
} // namespace test

template<typename ValueType>
struct concurrent_vector {

    std::mutex vector_mutex;
    std::vector<ValueType> vector;

    void push_back ( ValueType const & v_ ) {
        std::scoped_lock lock ( vector_mutex );
        vector.push_back ( v_ );
    }
};

template<typename ValueType, typename ThisLocalType>
struct this_concurrent_vector {

    using this_local_type = sax::this_local<this_concurrent_vector, ThisLocalType>;

    static this_local_type this_local_storage;

    std::mutex vector_mutex;
    std::vector<ValueType> vector;

    ~this_concurrent_vector ( ) { this_local_storage.destroy ( this ); } // call the instance-destructor

    void push_back ( ValueType const & v_ ) {

        ThisLocalType & this_local_object_new = this_local_storage ( this /* , variadic constructor parameters */ );

        // do something with the newly (on first access, optionally with constructor parameters) constructed this_local_object.

        ThisLocalType & tlo = this_local_storage ( this );

        // do something with the this_local_object created earlier, through a new reference.

        std::scoped_lock lock ( vector_mutex );
        vector.push_back ( v_ );
    }
};

template<typename ValueType, typename ThisLocalType>
typename this_concurrent_vector<ValueType, ThisLocalType>::this_local_type
    this_concurrent_vector<ValueType, ThisLocalType>::this_local_storage;

template<typename Type>
std::tuple<std::thread::id, int> work ( Type & t_ ) {
    test::micro_sleep ( );
    t_.emplace ( test::get_id ( ) );
    test::micro_sleep ( );
    // t_.this_local_storage.destroy ( std::this_thread::get_id ( ) ); // call the thread-destructor
    return { std::this_thread::get_id ( ), test::get_id ( ) };
}

/*

int main5686780 ( ) {

    this_concurrent_vector<std::thread::id, int> ids;

    std::uint64_t duration;
    plf::nanotimer timer;
    timer.start ( );

    for ( int n = 0; n < 32; ++n )
        std::jthread{ work<this_concurrent_vector<std::thread::id, int>>, std::ref ( ids ) };

    duration = static_cast<std::uint64_t> ( timer.get_elapsed_ms ( ) );
    std::cout << nl << duration << "ms" << nl;

    for ( auto id : ids.vector )
        std::cout << id << sp;
    std::cout << nl;

    return EXIT_SUCCESS;
}

*/

int main56867567 ( ) {

    /*
    auto a = _mm_set_ps ( 0.0, 0.0, 0.0, 0.0 );
    auto b = _mm_set_ps ( 0.0, 0.0, 0.0, 0.0 );

    std::cout << sax::equal_m128 ( &a, &b ) << ' ';

    auto c = _mm_set_ps ( 0.0, 0.0, 0.0, 1.0 );

    std::cout << sax::equal_m128 ( &a, &c ) << nl;
    */

    {
        std::uint64_t a[ 3 ] = { 1, 1, 1 };
        std::uint64_t b[ 3 ] = { 1, 1, 1 };

        std::cout << sax::equal_m192 ( &a, &b ) << nl;
    }

    {
        std::uint64_t a[ 3 ] = { 0, 1, 1 };
        std::uint64_t b[ 3 ] = { 1, 1, 1 };

        std::cout << sax::equal_m192 ( &a, &b ) << nl;
    }
    {
        std::uint64_t a[ 3 ] = { 1, 0, 1 };
        std::uint64_t b[ 3 ] = { 1, 1, 1 };

        std::cout << sax::equal_m192 ( &a, &b ) << nl;
    }
    {
        std::uint64_t a[ 3 ] = { 1, 1, 0 };
        std::uint64_t b[ 3 ] = { 1, 1, 1 };

        std::cout << sax::equal_m192 ( &a, &b ) << nl;
    }

    {
        std::uint64_t a[ 3 ] = { 0, 0, 1 };
        std::uint64_t b[ 3 ] = { 1, 1, 1 };

        std::cout << sax::equal_m192 ( &a, &b ) << nl;
    }

    {
        std::uint64_t a[ 3 ] = { 1, 0, 0 };
        std::uint64_t b[ 3 ] = { 1, 1, 1 };

        std::cout << sax::equal_m192 ( &a, &b ) << nl;
    }

    //  std::cout << sax::equal_m192 ( a.data ( ), b.data ( ) ) << nl;

    // for ( std::size_t i = 0; i < 256; ++i ) {

    //     std::cout << sax::equal_m128 ( a.data ( ), b.data ( ) ) << nl;
    // }

    /*

    sax::uint128_t de = { 1, 2 };
    sax::uint128_t ex = { 3, 4 };
    sax::uint128_t cr = { 0, 2 };

    std::cout << de << cr << nl;
    std::cout << sax::lockless::soft_dwcas<sax::lockless::spin_rw_lock<long long>> ( &de, ex, &cr ) << nl;
    std::cout << de << cr << nl;

    sax::lockless::unbounded_circular_list<int> list;

    std::ios::sync_with_stdio ( false );

    constexpr int N = 4;

    sax::lockless::unbounded_circular_list<int> list;

    list.emplace ( 0 );
    list.ostream ( std::cout );

    list.emplace ( 1 );
    list.ostream ( std::cout );

    list.emplace ( 2 );
    list.ostream ( std::cout );

    // list.emplace ( 3 );
    // list.ostream ( std::cout );
    */
    /*

    std::uint64_t duration;
    plf::nanotimer timer;
    timer.start ( );

    for ( int n = 0; n < N; ++n )
        std::jthread{ work<sax::lockless::unbounded_circular_list<int>>, std::ref ( stk ) };

    duration = static_cast<std::uint64_t> ( timer.get_elapsed_ms ( ) );

    int sum = std::accumulate ( stk.begin ( ), stk.end ( ), 0, [] ( auto & l, auto & r ) { return l + r.data; } );

    std::cout << std::boolalpha << ( ( ( N * ( N + 1 ) ) / 2 ) == sum ) << sp << std::dec << ( ( ( duration * 10 ) / N ) / 10.0 )
              << " ms/thread" << nl;

    for ( auto & n : stk )
        std::cout << n;
    std::cout << nl;

    */

    return EXIT_SUCCESS;
}

std::vector<std::uint64_t> random_vector ( std::size_t length_ ) {
    std::vector<std::uint64_t> v;
    for ( std::size_t i = 0; i < length_; ++i )
        v.emplace_back ( rng ( ) );
    return v;
}

int main ( ) {

    std::array<std::uint64_t, 12> pr1 = { 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    std::array<std::uint64_t, 12> pr2 = { 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 99 };

    std::array<std::uint64_t, 12> p1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    std::array<std::uint64_t, 12> p2 = { pr1 };

    {
        std::array<std::uint64_t, 12> a{ p1 }, b = { p2 };
        bool result = true;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < 12; i += 1 )
            result = result and a[ i ] == b[ i ];

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " ms " << result << nl;
    }

    {
        std::array<std::uint64_t, 12> a{ p1 }, b = { p2 };
        bool result = true;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < 12; i += 2 )
            result = result and sax::equal_m128 ( a.data ( ) + i, b.data ( ) + i );

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " ms " << result << nl;
    }

    {
        std::array<std::uint64_t, 12> a{ p1 }, b = { p2 };
        bool result = true;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < 12; i += 3 )
            result = result and sax::equal_m192 ( a.data ( ) + i, b.data ( ) + i );

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " ms " << result << nl;
    }

    {
        std::array<std::uint64_t, 12> a{ p1 }, b = { p2 };
        bool result = true;

        std::uint64_t duration;
        plf::nanotimer timer;
        timer.start ( );

        for ( std::size_t i = 0; i < 12; i += 4 )
            result = result and sax::equal_m256 ( a.data ( ) + i, b.data ( ) + i );

        duration = static_cast<std::uint64_t> ( timer.get_elapsed_us ( ) );
        std::cout << std::dec << duration << " ms " << result << nl;
    }

    return EXIT_SUCCESS;
}
