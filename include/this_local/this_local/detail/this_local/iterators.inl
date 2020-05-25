
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
            unbounded_circular_list::repair_after_links ( node );
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
        node{ std::forward<const_node_type_ptr> ( node_ ) }, end_node{ std::forward<const_node_type_ptr> ( end_node_ ) }, skip_end{
            std::forward<long long> ( end_passes_ )
        } {}

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
            unbounded_circular_list::repair_after_links ( node );
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
