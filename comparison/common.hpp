#include <string>
#include <vector>

struct Bar6 {
    std::string baz6;
};

struct Bar5 {
    Bar6 baz5;
};

struct Bar4 {
    Bar5 baz4;
};

struct Bar3 {
    Bar4 baz3;
};

struct Bar2 {
    Bar3 baz2;
};

struct Bar1 {
    Bar2 baz1;
};

std::vector<Bar3> foo1(const std::vector<Bar2>&);
std::vector<Bar5> foo2(const std::vector<Bar4>&);
