#ifndef COROBATCH_PRIVATE_LOGGING_CPP
#define COROBATCH_PRIVATE_LOGGING_CPP

#ifdef COROBATCH_TRANSLATION_UNIT

#include <iostream>

#include <corobatch/logging.hpp>

namespace corobatch {

LoggerCb disabled_logger = [](LogLevel) -> std::ostream* { return nullptr; };

LoggerCb debug_logger = [](LogLevel level) -> std::ostream* {
    if (level >= LogLevel::DEBUG)
    {
        return &std::cerr;
    }
    return nullptr;
};

LoggerCb error_logger = [](LogLevel level) -> std::ostream* {
    if (level >= LogLevel::ERROR)
    {
        return &std::cerr;
    }
    return nullptr;
};

namespace {

LoggerCb logger_cb = error_logger;

}

void registerLoggerCb(LoggerCb cb) { logger_cb = cb; }

namespace private_ {

std::ostream* getLogStream(LogLevel level) { return logger_cb(level); }

} // namespace private_

} // namespace corobatch

#endif
#endif
