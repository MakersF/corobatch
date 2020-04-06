# Corobatch

`corobatch` makes batching operations easy by transforming your single-item code to perform batched operations thanks to coroutines.

Batch operations allow to manipulate multiple items at once.
In many situation they tend to be more efficient, but require the code to be changed to issue the operations together.
A typical example is writing/reading data to/from the network/files.

`corobatch` aims to allow you to write your logic as if you were manipulating items one by one, and it transforms it into a batch operation for you.

# Sync example

Imagine you need to fetch some preferences for a user.


```c++

// 1. Define the operations you want to do.
//    `get_user_preferences` is a callable object which takes care of the batching
auto do_action = [](UserId user_id, auto&& get_user_preferences) -> corobatch::task<void> {
    // [ code using user_id ]
    UserPreference user_preferences = co_await get_user_preferences(user_id);
    // [ other code using user_id or user_preferences]
    co_return;
};

// 2. Specify how the batch operation is performed
auto get_user_preferences_accumulator = corobatch::syncVectorAccumulator<UserPreference, UserId>(
    [](const std::vector<UserId>& user_ids) -> std::vector<UserPreference> {
        return blocking_call_to_fetch_user_preferences_from_the_network_in_bulk(user_ids);
};
auto get_user_preferences_batcher = corobatch::Batcher(get_user_preferences_accumulator);

// 3. Create an executor
corobatch::Executor executor;

// 4. For each user submit the task
for(UserId user_id : user_ids) {
    corobatch::submit(executor, do_action(user_id, get_user_preferences_batcher));
}

// 5. Keep executing until all the tasks have completed
while(corobatch::has_any_pending(get_user_preferences_batcher)) {
    corobatch::force_execution(get_user_preferences_batcher);
    executor.run();
}
```

## What is going on

**TL;DR**: `corobatch` intercepts the calls you make to your functions in the task, stores the parameters provided to them and suspend the execution of the task. When requested, it invokes the batch operation you provided with all the parameters stored earlier, and resumes the suspended tasks with the result of the call.

Here a more in depth explanation

### 1. The task

Define a task that performs the logic you want.
The task gets 2 kinds of parameters:
1. The data you want to pass to it (like a regular function)
2. A callable for each function you want to batch.
    This callable is used to intercept the call to your function.
    When you call the callable, the parameters you used are stored. They are later going to be passed to the batch operation you'll provide.
    When you `co_await` the result of the call, the task is suspended until the batch call is performed.
    Once the batch operation finishes, the task can be resumed with the result of the batch operation.

There is no restriction on the number of callables or the order in which they need to be `co_await`ed.

### 2. The accumulator

When the task calls the callable, we need to store the parameters provided to it.
Also, at one point we will have to do a batch call.
How can it be done?
The `Accumulator` specifies how that happens.
In this case we are using a class provided by `corobatch`: a vector accumulator.
A vector accumulator stores the parameters in a vector, and when it needs to perform a batch operation it passes the vector to the function provided when constructing it. It expects the result to be a vector, and associates `result[i]` to the task which recorded the `parameter[i]`.

Once we have the accumulator, we can construct the `Batcher` from it, which will provide the machinery to suspend the task and reschedule it once the batch operation has terminated.

You can implement your own accumulators, as long as they satisfy the required concept.

### 3. The executor

Tasks need to be executed by someone. That is the role of the executor.
A task can be submitted to an executor, which takes care of executing it from it's `run()` method.
When a task reaches a batch operation it is suspended.
When a batch operation concludes, the tasks that were blocked waiting for it are scheduled again on the executor to which they were submitted, so that you can call its `run()` method when you want to resume the tasks.

### 4. Start the task

Submit the task to be executed for each of the items you want to process.
If you want to know when the task terminates, or the task returns a value, you can use the `submit()` overload that takes a callable which is invoked when the task terminates with the result of `co_return`.

### 5. Execute until done

`corobatch` does not know how many tasks you will submit, so it does not start executing a batch operation automatically, as you might want to batch more.

You can call `has_any_pending` passing the `Batcher` created at step 2 to check if there is any task that is blocked waiting for a batch operation to be performed, and the batch operation has not be started yet. When used with multiple wrappers, it returns true if any of them has a waiting task.

You can also use `force_execution` to force the execution of the batch operation (and unblock the tasks waiting for it). When used with multiple wrappers, it forces the execution of one of them.

After the batch has been forced to execute, since it is synchronous, the tasks will be unblocked and added to the executor, so you need to call it's run method if you want to schedule them again.

When all the tasks run to completion and they don't stop again waiting for any other batch operations, the loop will terminate.

## Async example

You can see an example of integrating `corobatch` with `boost::asio` in `corobatch/examples/asio`

## Accumulators

An accumulator is a stateless object which specifies how to store parameters and how to later execute the batch operation.
`corobatch` comes with an handful of accumulators:

- `SyncVectorAccumulator`: accumulates parameters in a vector and calls a function to synchronously compute the result of the operation.
- `VectorAccumulator`: accumulates parameters in a vector and calls a function to asynchronously compute the result of the operation. When calling the function it provides a callback to be called when the result is ready.
- `SizedAccumulator`: wraps another accumulator and makes sure to execute it when the number of parameters reaches a threshold. Useful to enforce a maximum number of items.
- `WaitableAccumulator`: wraps another accumulator and provides a way to wait for the async operation is concluded. Useful when executing operations on multiple thread and there is a need to synchronize on the execution.

### Implementing an accumulator

To implement an accumulator follow this format

```c++
struct MyAccumulator {
    using AccumulationStorage = /* the type where to store the parameters. Example: a vector */;
    using ExecutedResults = /* the type which contains the result of the batch operation. Example: a vector */;
    using Handle = /* the type to retrieve the element for a single task from the result. Example: an index into the vector */;
    using Args = ArgTypeList<Arg1, Arg2, Arg3 /*the list of parameters that the function is going to provide when calling the batch */>;
    using ResultType = /* the single type that is returned to the task.*/;

    // Create a new object to store arguments into
    AccumulationStorage get_accumulation_storage();
    // Store the arguments inside the accumulation storage and return the handle that will be used to
    // look up the element in the result
    Handle record_arguments(AccumulationStorage& , Arg1&&, Arg2&&, Arg3&& /* the same as inside the ArgTypeList */);
    // Start the execution of the batch operation with the parameters stored in the storage.
    // Call the provided function with the result when done
    void execute(AccumulationStorage&&, std::function<void(ExecutedResults)>);
    // Extract the single item to return to the task from the executed result
    ResultType get_result(Handle, ExecutedResults&);
    // Tell whether the accumulator must execute (for example when a maximum number of items has been addded to the storage)
    bool must_execute(const AccumulationStorage&);
};
```

## (Almost) Header only

`corobatch` is almost header only: it has a small amount of code that needs to be compiled and linked for it to work.

In a single translation unit in your project, defined the macro `COROBATCH_TRANSLATION_UNIT` before including corobatach, like this

```c++
#define COROBATCH_TRANSLATION_UNIT
#include <corobatch/corobatch.hpp>
```

This will make sure to compile the code required in that translation unit only.
