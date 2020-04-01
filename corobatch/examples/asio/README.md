# Integrating with `boost::asio`

This example creates a client and a server to compute some operation (the power of 2 of a number).

## Run the example

1. First run the server: `server <port>`
2. Run the client: `client localhost <port>`

The client will print out that it sent a request and it received a response, then each task will terminate and it will print its result.

## Explanation

The client creates a batcher which calls in an async way the server.
When the batcher is executed, a request is sent to the server, and when the response is received the execution on the tasks resumes.

The server does not use `corobatch`. It should be seen as a black box.
