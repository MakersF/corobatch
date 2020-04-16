#ifndef COROBATCH_UTILITY_ALLOCATOR_HPP
#define COROBATCH_UTILITY_ALLOCATOR_HPP

#include <cassert>
#include <corobatch/private_/log.hpp>
#include <memory>

namespace corobatch {

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
        auto current = d_root;
        while (current)
        {
            auto next = current->d_next;
            std::free(current);
            current = next;
        }
        COROBATCH_LOG_INFO << "Total allocations: " << d_allocations_count << ". Supported size: " << d_supported_size
                           << ". Supported alignmen: " << d_supported_alignment;
    }

    void* allocate(size_t align, size_t sz)
    {
        COROBATCH_LOG_TRACE << "Allocationg " << sz << " with alignment " << align;
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

    void deallocate(void* p, size_t sz)
    {
        COROBATCH_LOG_TRACE << "Deallocating " << sz;
        assert(sz >= sizeof(header));
        auto new_entry = new (p) header;
        new_entry->d_next = d_root;
        d_root = new_entry;
    }

    template<typename T>
    class Allocator
    {
    private:
        PoolAlloc& d_poolAlloc;

        template<typename>
        friend class Allocator;

    public:
        Allocator(PoolAlloc& PoolAlloc) : d_poolAlloc(PoolAlloc) {}

        template<typename Q>
        Allocator(const Allocator<Q> o) : Allocator(o.d_poolAlloc)
        {
        }

        using value_type = T;

        T* allocate(std::size_t num) { return static_cast<T*>(d_poolAlloc.allocate(alignof(T), sizeof(T) * num)); }

        void deallocate(T* ptr, std::size_t num) { d_poolAlloc.deallocate(static_cast<void*>(ptr), sizeof(T) * num); }

        bool operator==(const Allocator& o) { return &d_poolAlloc == &o.d_poolAlloc; }
    };
};

template<typename T, PoolAlloc* StaticAllocator>
struct StaticPoolAllocator : PoolAlloc::Allocator<T>
{
    StaticPoolAllocator() : PoolAlloc::Allocator<T>(*StaticAllocator) {}

    template<typename Q>
    struct rebind
    {
        using other = StaticPoolAllocator<Q, StaticAllocator>;
    };
};

} // namespace corobatch

#endif
