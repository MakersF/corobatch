#pragma once

#include <cassert>
#include <experimental/coroutine>
#include <functional>

#define MY_FWD(...) ::std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)
namespace corobatch {

template<typename T>
class task
{
private:
    template<typename Q>
    struct promise_return
    {
        void return_value(Q val)
        {
            assert(this->d_cb);
            this->d_cb(MY_FWD(val));
        }

        using OnReturnValueCb = std::function<void(T)>;
        void set_on_return_value_cb(OnReturnValueCb cb)
        {
            assert(not d_cb);
            d_cb = cb;
        }

    private:
        OnReturnValueCb d_cb;
    };

    template<>
    struct promise_return<void>
    {
        void return_void()
        {
            assert(this->d_cb);
            this->d_cb();
        }

        using OnReturnValueCb = std::function<void()>;
        void set_on_return_value_cb(OnReturnValueCb cb)
        {
            assert(not d_cb);
            d_cb = cb;
        }

    private:
        OnReturnValueCb d_cb;
    };

public:
    struct promise_type : promise_return<T>
    {
        task get_return_object() { return task{*this}; }

        std::experimental::suspend_always initial_suspend() { return {}; }
        void unhandled_exception() noexcept { assert(false && "Unhandled exception in coroutine"); }
        std::experimental::suspend_never final_suspend() { return {}; }
    };

private:
public:
    using Handle = std::experimental::coroutine_handle<promise_type>;
    using ReturnType = T;

    Handle handle() &&
    {
        Handle other = d_handle;
        d_handle = nullptr;
        return other;
    }

    task(const task&) = delete;
    task(task&& other) : d_handle(other.d_handle) { other.d_handle = nullptr; }

    ~task()
    {
        if (d_handle)
        {
            d_handle.destroy();
        }
    }

private:
    explicit task(promise_type& promise) : d_handle(Handle::from_promise(promise)) {}

    Handle d_handle;
};

template<typename OnDone, typename ReturnType>
void submit(OnDone&& onDone, task<ReturnType> taskObj)
{
    typename task<ReturnType>::Handle coro_handle = std::move(taskObj).handle();
    coro_handle.promise().set_on_return_value_cb(MY_FWD(onDone));
    coro_handle.resume();
}

inline void submit(task<void> task)
{
    submit([]() {}, std::move(task));
}

} // namespace corobatch

#undef MY_FWD