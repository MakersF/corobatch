#ifndef COROBATCH_UTILITY_ALLOCATOR_HPP
#define COROBATCH_UTILITY_ALLOCATOR_HPP

#include <cassert>
#include <corobatch/private_/log.hpp>
#include <memory>

namespace corobatch {

namespace private_ {

template<typename UnderlyingAlloc, typename T>
class AllocatorWrapper
{
private:
    UnderlyingAlloc& d_underlyingAlloc;

    template<typename, typename>
    friend class AllocatorWrapper;

public:
    explicit AllocatorWrapper(UnderlyingAlloc& underlyingAlloc) : d_underlyingAlloc(underlyingAlloc) {}

    template<typename Q>
    explicit AllocatorWrapper(const AllocatorWrapper<UnderlyingAlloc, Q> o) : AllocatorWrapper(o.d_underlyingAlloc)
    {
    }

    template<typename Q>
    struct rebind
    {
        using other = AllocatorWrapper<UnderlyingAlloc, Q>;
    };

    using value_type = T;

    T* allocate(std::size_t num) { return static_cast<T*>(d_underlyingAlloc.allocate(alignof(T), sizeof(T) * num)); }

    void deallocate(T* ptr, std::size_t num) { d_underlyingAlloc.deallocate(static_cast<void*>(ptr), sizeof(T) * num); }

    bool operator==(const AllocatorWrapper& o) { return &d_underlyingAlloc == &o.d_underlyingAlloc; }
};

} // namespace private_

// Can be used to allocate only one size and one alignment.
// Reuse deallocated blocks for new allocations;
class PoolAlloc
{
private:
    struct header
    {
        header* d_next;
    };

    header* d_root = nullptr;
    size_t d_supported_size = 0;
    size_t d_supported_alignment = 0;
    size_t d_allocations_count = 0;

public:
    PoolAlloc() = default;
    PoolAlloc(const PoolAlloc&) = delete;
    PoolAlloc(PoolAlloc&& other)
    : d_root(other.d_root)
    , d_supported_size(other.d_supported_size)
    , d_supported_alignment(other.d_supported_alignment)
    , d_allocations_count(other.d_allocations_count)
    {
        other.d_root = nullptr;
        other.d_supported_size = 0;
        other.d_supported_alignment = 0;
        other.d_allocations_count = 0;
    }

    ~PoolAlloc()
    {
        std::size_t available_blocks = 0;
        auto current = d_root;
        while (current)
        {
            auto next = current->d_next;
            std::free(current);
            available_blocks++;
            current = next;
        }
        COROBATCH_LOG_INFO << "Total allocations: " << d_allocations_count << ". Supported size: " << d_supported_size
                           << ". Supported alignment: " << d_supported_alignment;
        assert(available_blocks == d_allocations_count &&
               "The allocator has been destructed and some memory allocated through it was not freed yet");
    }

    void* allocate(std::size_t align, std::size_t sz)
    {
        COROBATCH_LOG_TRACE << "Allocating " << sz << " with alignment " << align;
        assert(sz >= sizeof(header) && "The allocated requires the size to be bigger");
        if (d_supported_size == 0)
        {
            d_supported_size = sz;
            d_supported_alignment = align;
        }
        assert(d_supported_size == sz &&
               "The allocator supports allocating only a single size. Asked a size different from a previous one");
        assert(d_supported_alignment == align && "The allocator supports allocating only with a single alignment. "
                                                 "Asked an alignment differnt from a previous one");
        if (d_root)
        {
            header* mem = d_root;
            d_root = d_root->d_next;
            mem->~header();
            return static_cast<void*>(mem);
        }
        ++d_allocations_count;
        return std::aligned_alloc(align, sz);
    }

    void deallocate(void* p, std::size_t sz)
    {
        COROBATCH_LOG_TRACE << "Deallocating " << sz;
        assert(sz >= sizeof(header));
        auto new_entry = new (p) header;
        new_entry->d_next = d_root;
        d_root = new_entry;
    }

    template<typename T>
    using Allocator = private_::AllocatorWrapper<PoolAlloc, T>;

    template<typename T>
    Allocator<T> allocator()
    {
        return Allocator<T>(*this);
    }
};

// Can be used to allocate only one size and one alignment.
// On the first allocation it allocates the memory for the expected number of items
class UniformSizeLazyBulkAlloc
{
private:
    char* d_memory = nullptr;
    std::size_t d_capacity = 0; // How much memory is allocated
    std::size_t d_supported_size = 0; // the size of the items this allocator is used to allocate
    std::size_t d_currentoffset = 0; // where the free memory begins
    std::size_t d_num_allocs = 0; // The number of active allocations
    std::size_t d_max_num_items = 0; // How many items at most this allocator can contain
    std::size_t d_max_num_allocs = 0; // The maximum number of items allocated on this allocator at the same time
public:
    UniformSizeLazyBulkAlloc(std::size_t max_num_items) : d_max_num_items(max_num_items) {}
    UniformSizeLazyBulkAlloc(const UniformSizeLazyBulkAlloc&) = delete;
    UniformSizeLazyBulkAlloc(UniformSizeLazyBulkAlloc&& other)
    : d_memory(other.d_memory)
    , d_capacity(other.d_capacity)
    , d_supported_size(other.d_supported_size)
    , d_currentoffset(other.d_currentoffset)
    , d_num_allocs(other.d_num_allocs)
    , d_max_num_items(other.d_max_num_items)
    {
        other.d_memory = nullptr;
        other.d_capacity = 0;
        other.d_supported_size = 0;
        other.d_currentoffset = 0;
        other.d_num_allocs = 0;
        other.d_max_num_items = 0;
    }

    ~UniformSizeLazyBulkAlloc()
    {
        assert(d_num_allocs == 0);
        std::free(d_memory);
        COROBATCH_LOG_INFO << "Total allocations: " << d_max_num_allocs << " out of " << d_max_num_items
                           << "(max) . Supported size: " << d_supported_size
                           << ". Total memory allocated: " << d_capacity;
    }

    void* allocate(std::size_t align, std::size_t sz)
    {
        COROBATCH_LOG_TRACE << "Allocating " << sz << " with alignment " << align;
        if (d_memory == nullptr)
        {
            d_capacity = sz * d_max_num_items;
            COROBATCH_LOG_TRACE << "Allocating " << d_capacity << " for " << d_max_num_items << " items of size " << sz;
            d_memory = static_cast<char*>(std::aligned_alloc(align, d_capacity));
            d_supported_size = sz;
        }
        assert(sz == d_supported_size && "The allocator can only allocate elements with the same size");
        assert(d_currentoffset + sz <= d_capacity && "Allocated more memory than the allocator can support");
        void* ptr = d_memory + d_currentoffset;
        std::size_t left = d_capacity - d_currentoffset;
        ptr = std::align(align, sz, ptr, left);
        assert(ptr != nullptr);
        d_currentoffset = (d_capacity - left) + sz;
        d_num_allocs++;
        d_max_num_allocs = std::max(d_max_num_allocs, d_num_allocs);
        return ptr;
    }

    void deallocate(void*, std::size_t sz)
    {
        COROBATCH_LOG_TRACE << "Deallocating " << sz;
        d_num_allocs--;
        if (d_num_allocs == 0)
        {
            d_currentoffset = 0;
        }
    }

    template<typename T>
    using Allocator = private_::AllocatorWrapper<UniformSizeLazyBulkAlloc, T>;

    template<typename T>
    Allocator<T> allocator()
    {
        return Allocator<T>(*this);
    }
};

} // namespace corobatch

#endif
