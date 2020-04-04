#pragma once

#include <functional>
#include <iosfwd>

namespace corobatch {

enum class LogLevel
{
    DEBUG = 64,
    INFO = 128,
    ERROR = 256
};

// Register logger
using LoggerCb = std::function<std::ostream*(LogLevel)>;

extern LoggerCb disabled_logger;
extern LoggerCb debug_logger;
extern LoggerCb error_logger;

void registerLoggerCb(LoggerCb);

namespace private_ {

std::ostream* getLogStream(LogLevel);

}

} // namespace corobatch

#ifdef COROBATCH_TRANSLATION_UNIT
#include <corobatch/private_/logging.cpp>
#endif
