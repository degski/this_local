
# this_local


The `this_local`-library is header-only and requires `C++17` with external dependencies on [`hedley`](https://github.com/nemequ/hedley) (optional) and [`plf/list`](https://github.com/mattreecebentley/plf_list) (required). The required external header-files are locally included and kept up to date.



## tl;dr


The library has been named `this_local` in opposition to `thread_local`, it is both a PoC and and a WIP. It provides a facility to easily add **instance-thread_local storage** to a class. This library is only useful in a context of multithreading.

The main and only class has the following signature:


    template<typename Type, typename ValueType, template<typename> typename Allocator = std::allocator>
    class this_local


`Type` is the type to which to add the `this_local` storage, ValueType is here the type of the storage created. There are **no restrictions** on the ValueType. The this-thread-local-object (the ValueType) is (not necessarilly) default-constructed on demand and will never be moved during its lifetime.



## example


In this example it is demonstrated how to add `this_local` storage to a concurrent vector, this is the brilliant design:

    template<typename ValueType>
    struct concurrent_vector {

        std::mutex vector_mutex;
        std::vector<ValueType> vector;

        void push_back ( ValueType const & v_ ) {
            std::scoped_lock lock ( vector_mutex );
            vector.push_back ( v_ );
        }
    };


The instrumented vector looks like this:


    template<typename ValueType, typename ThisLocalType>
    struct this_concurrent_vector {

        using this_local_type = sax::this_local<this_concurrent_vector, ThisLocalType>;

        static this_local_type this_local_storage;

        std::mutex vector_mutex;
        std::vector<ValueType> vector;

        ~this_concurrent_vector ( ) { this_local_storage.destroy ( this ); } // call the instance-destructor with
                                                                             // the `this`-pointer as argument.

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


We add `this_local` as a static member to our class. In de the destructor of the Type-class, the `this_local`-instance-destructor has to be called. In the thread function we call the `this_local`-thread-destructor before returning, like shown here in the function below:


    template<typename ValueType, typename ThisLocalType>
    std::tuple<std::thread::id, int, int> work ( this_concurrent_vector<ValueType, ThisLocalType> & vec_ ) {

        ...

        vec_.push_back ( some Type-argument to push ); // do something concurrently (f.e. push_back ( something ))

        vec_.this_local_storage.destroy ( std::this_thread::get_id ( ) ); // call the this_local-thread-destructor

        return { this_thread_id, sleep_duration, ctr++ };
    }


That's all.

Storage is created on demand, there can be concurrently instances of the same class that have no storage allocated whatsover, together with those who do.  Over the object's life-time, storage can be (per object) created-destructed and recreated later. On default construction, no memory allocation takes place.

There is no absolute need to call the thread-destructor (depending on your use pattern), eventually all the storage will be torn down by the instance of Type (in its destructor), also the storage of the threads that already ceased to exist, the cleanest is of coarse to call the `this_local`-thread-destructor.
