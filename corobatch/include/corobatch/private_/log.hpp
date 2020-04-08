#ifndef COROBATCH_PRIVATE_LOG_HPP
#define COROBATCH_PRIVATE_LOG_HPP

#include <corobatch/logging.hpp>
#include <iomanip>
#include <iostream>

#define COROBATCH_LOGLEVEL_NAME loglevel##__LINE__
#define COROBATCH_STREAM_NAME stream##__LINE__

#define COROBATCH_PRINT_LOGLINE_PREFIX(stream) \
    stream << __FILE__ << ":" << __LINE__ << " " << (COROBATCH_LOGLEVEL_NAME) << " "

#ifndef COROBATCH_DISABLE_LOGGING
// Get the log stream ptr, check if it's valid, if it is, the body of the loop will be executed.
// In the increment statement, use the comma operator to put a new line at the end of the stream
// and then reset the pointer so that we don't enter the loop anymore
#define COROBATCH_LOG_BLOCK(level, levelname)                                                                    \
    for (const char* COROBATCH_LOGLEVEL_NAME = levelname; COROBATCH_LOGLEVEL_NAME != nullptr;                    \
         COROBATCH_LOGLEVEL_NAME = nullptr)                                                                      \
        for (::std::ostream* COROBATCH_STREAM_NAME = ::corobatch::private_::getLogStream((level));               \
             COROBATCH_STREAM_NAME != nullptr && (COROBATCH_PRINT_LOGLINE_PREFIX(*COROBATCH_STREAM_NAME), true); \
             COROBATCH_STREAM_NAME = (*(COROBATCH_STREAM_NAME) << '\n', nullptr))
#else
#define COROBATCH_LOG_BLOCK(level, levelname)                                                                       \
    for (const char* COROBATCH_LOGLEVEL_NAME = nullptr; COROBATCH_LOGLEVEL_NAME; COROBATCH_LOGLEVEL_NAME = nullptr) \
        for (::std::ostream* COROBATCH_STREAM_NAME = nullptr; COROBATCH_STREAM_NAME; COROBATCH_STREAM_NAME = nullptr)
#endif

#define COROBATCH_LOG_STREAM (*(COROBATCH_STREAM_NAME))

// Block macros

#define COROBATCH_LOG_DEBUG_BLOCK COROBATCH_LOG_BLOCK(::corobatch::LogLevel::DEBUG, "DEBUG")

#define COROBATCH_LOG_INFO_BLOCK COROBATCH_LOG_BLOCK(::corobatch::LogLevel::INFO, "INFO")

#define COROBATCH_LOG_ERROR_BLOCK COROBATCH_LOG_BLOCK(::corobatch::LogLevel::ERROR, "ERROR")

// Line macros

#define COROBATCH_LOG_DEBUG COROBATCH_LOG_DEBUG_BLOCK COROBATCH_LOG_STREAM
#define COROBATCH_LOG_INFO COROBATCH_LOG_INFO_BLOCK COROBATCH_LOG_STREAM
#define COROBATCH_LOG_ERROR COROBATCH_LOG_ERROR_BLOCK COROBATCH_LOG_STREAM

namespace corobatch {
namespace private_ {

template<typename T>
struct PrintIfPossible
{
    PrintIfPossible(const T& value) : d_value(value) {}

    friend std::ostream& operator<<(std::ostream& os, PrintIfPossible obj)
    {
        if constexpr (requires { os << d_value; })
        {
            os << obj.d_value;
        }
        else
        {
            // Print as bytes
            const char* begin = reinterpret_cast<const char*>(&obj.d_value);
            const char* end = begin + sizeof(obj.d_value);
            std::ios_base::fmtflags previous_flags = os.flags();
            os << "[ ";
            for (; begin != end; begin++)
            {
                os << std::showbase << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*begin);
            }
            os << " ]";
            os.flags(previous_flags);
        }
        return os;
    }

    const T& d_value;
};

} // namespace private_
} // namespace corobatch

#endif
