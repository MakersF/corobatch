#ifndef COROBATCH_LOGGING_HPP
#define COROBATCH_LOGGING_HPP

#include <functional>
#include <iosfwd>

namespace corobatch {

enum class LogLevel
{
    TRACE = 32,
    DEBUG = 64,
    INFO = 128,
    ERROR = 256
};

// Register logger
using LoggerCb = std::function<std::ostream*(LogLevel)>;

extern LoggerCb disabled_logger;
extern LoggerCb trace_logger;
extern LoggerCb debug_logger;
extern LoggerCb info_logger;
extern LoggerCb error_logger;

void registerLoggerCb(LoggerCb);

namespace private_ {

std::ostream* getLogStream(LogLevel);

}

} // namespace corobatch

#endif
