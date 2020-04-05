# Corobatch

Make batching operations easy by transforming your single-item code with coroutines.

Batch operations allow to manipulate multiple items at once.
In many situation they tend to be more efficient, but require the code to be changed to issue the operations together.
A typical example is writing/reading data to/from the network/files.

`corobatch` aims to allow you to write your logic as if you were manipulating items one by one, and it transforms it into a batch operation for you.

# Sync example

Imagine you need to fetch some preferences for a user in a way t
```c++

// 1. Define the operations you want to do.
auto do_action = [](UserId user_id, auto&& get_user_preferences) -> corobatch::task<void> {
    // [ code using user_id ]
    UserPreference user_preferences = co_await get_user_preferences(user_id);
    // [ other code using user_id or user_preferences]
    co_return;
};

// 2. Specify how the batch operation is performed
auto get_user_preferences = corobatch::syncVectorBatcher<UserPreference, UserId>(
    [](const std::vector<UserId>& user_ids) -> std::vector<UserPreference> {
        return blocking_call_to_fetch_user_preferences_from_the_network(user_ids);
};

// 3. Create an executor
corobatch::Executor executor;
auto get_user_preferences_callable = corobatch::make_batcher(executor, get_user_preferences);

// 4. For each user submit the task
for(UserId user_id : user_ids) {
    corobatch::submit(do_action(user_id, get_user_preferences_callable));
}

// 5. Keep executing until all the tasks have completed
while(corobatch::has_any_pending(get_user_preferences_callable)) {
    corobatch::force_execution(get_user_preferences_callable);
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
    When you call the function, the parameters you used are stored. They are later going to be passed to the batch operation you'll provide.
    When you `co_await` the result of the call, the task is suspended until the batch call is performed.

There is no restriction on the number of functions or the order in which they need to be `co_await`ed.

### 2. The batch operation

When the task calls the callable, we need to store the parameters provided to it.
Also, at one point we will have to do a batch call.
How can it be done?
The `Batcher` specifies how that happens.
In this case we are using a class that's provided by `corobatch`: a vector batcher.
A vector batcher stores the parameters in a vector, and when it needs to perform a batch operation it passes the vector to the function provided when constructing it. It expects the result to be a vector, and associates `result[i]` to the task which recorded the `parameter[i]`.

You can implement your own batchers, as long as they satisfy the required concept.

### 3. The executor

When a batch operation concludes, the tasks that were blocked waiting for it need to be scheduled.
They are added to the executor, so that you can call its `run()` method when you want to resume the tasks.
The `make_batcher` call is used to create a wrapper that implements the call operator and invokes the appropriate methods of the `Batcher`.

### 4. Start the task

Submit the task to be executed for each of the items you want to process.
If you want to know when the task terminates, or the task returns a value, you can use the `submit()` overload that takes a callable which is invoked when the task terminates with the result of `co_return`.

### 5. Execute until done

`corobatch` does not know how many tasks you will submit, so it does not start executing a batch operation automatically, as you might want to batch more.

You can call `has_any_pending` passing the wrapper created at step 3 to check if there is any task that is blocked waiting for a batch operation to be performed, and the batch operation has not be started yet. When used with multiple wrappers, it returns true if any of them has a waiting task.

You can also use `force_execution` to force the execution of the batch operation (and unblock the tasks waiting for it). When used with multiple wrappers, it forces the execution of one of them.

After the batch has been forced to execute, since it is synchronous, the tasks will be unblocked and added to the executor, so you need to call it's run method if you want to schedule them again.

When the tasks all run to completion and don't wait for any other batch operation, the loop will terminate.

## Async example

You can see an example of integrating `corobatch` with `boost::asio` in `corobatch/examples/asio`

## (Almost) Header only

`corobatch` is almost header only: it has a small amount of code that needs to be compiled and linked for it to work.

In a single translation unit in your project, defined the macro `COROBATCH_TRANSLATION_UNIT` before including corobatach, like this

```c++
#define COROBATCH_TRANSLATION_UNIT
#include <corobatch/corobatch.hpp>
```

This will make sure to compile the code required in that translation unit only.
