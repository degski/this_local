
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

#include <array>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <sax/iostream.hpp>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <jthread>
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

namespace thread {
// Creates a new ID.
[[nodiscard]] inline int get_id ( bool ) noexcept {
    static std::atomic<int> global_id = 0;
    return global_id++;
}
// Returns ID of this thread.
[[nodiscard]] inline int get_id ( ) noexcept {
    static thread_local int thread_local_id = get_id ( false );
    return thread_local_id;
}
} // namespace thread

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
        static thread_local sfc generator ( sax::fixed_seed ( ) + thread::get_id ( ) );
        return generator;
    }
}
} // namespace sfc

auto & rng = sfc::generator ( ); // always a reference, avoids checking for the tls-object's creation on every access.

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
std::tuple<std::thread::id, int> work ( Type & vec_ ) {

    static int ctr = 0;

    std::this_thread::sleep_for ( std::chrono::microseconds ( sax::uniform_int_distribution<int> ( 1, 20 ) ( rng ) ) );

    vec_.emplace ( thread::get_id ( ) ); // do something concurrently (f.e. push_back ())
    // vec_.this_local_storage.destroy ( std::this_thread::get_id ( ) ); // call the thread-destructor

    return { std::this_thread::get_id ( ), ctr++ };
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

int main ( ) {

    sax::lock_free_plf_stack<int> stk;

    std::uint64_t duration;
    plf::nanotimer timer;
    timer.start ( );

    for ( int n = 0; n < 128; ++n )
        std::jthread{ work<sax::lock_free_plf_stack<int>>, std::ref ( stk ) };

    duration = static_cast<std::uint64_t> ( timer.get_elapsed_ms ( ) );
    std::cout << nl << duration << "ms" << nl;

    std::set<void *> p;
    for ( auto & node : stk )
        p.emplace ( ( void * ) std::addressof ( node ) );

    std::cout << p.size ( ) << nl;

    return EXIT_SUCCESS;
}
