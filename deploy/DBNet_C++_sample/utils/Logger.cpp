#include "Logger.h"

#include <atomic>

#include "spdlog/sinks/stdout_sinks.h" // use stdout logger
#include "spdlog/sinks/basic_file_sink.h" // use basic file logger


namespace LOGGING
{

class Logger::LoggerImpl
{
public:
    LoggerImpl();
    ~LoggerImpl();

    bool init();
    bool init(const char* file, bool truncate = false);
    void release();

    void set_level(LogLevel lv);
    LogLevel get_level() const;

    void log_string(LogLevel lv, std::string& str);

private:
    //bool filter_level(LogLevel lv) const;

    std::shared_ptr<spdlog::logger> logger_;
    bool init_;
    std::atomic<LogLevel> lv_;
};

Logger::LoggerImpl::LoggerImpl()
    : logger_(nullptr)
    , init_(false)
    , lv_(LL_Off)
{
}

Logger::LoggerImpl::~LoggerImpl()
{
    release();
}

bool Logger::LoggerImpl::init()
{
    if (init_)
        return false;

    try {
        logger_ = spdlog::stdout_logger_mt("main_logger");
        //spdlog::register_logger(_logger);
        spdlog::set_level(spdlog::level::trace);

        spdlog::set_pattern("[%C%m%d %T.%e][%L][%t] %v");

        logger_->flush_on(spdlog::level::warn);
    }
    catch (const spdlog::spdlog_ex& ex) {
        //std::cout << "Log init failed: " << ex.what() << std::endl;
        return false;
    }

    init_ = true;
    lv_ = LL_Info; // default info
    return true;
}

bool Logger::LoggerImpl::init(const char* file, bool truncate)
{
    if (init_ || nullptr == file)
        return false;

    try {
        logger_ = spdlog::basic_logger_mt("main_logger", file, truncate);
        //spdlog::register_logger(_logger);
        spdlog::set_level(spdlog::level::trace);

        spdlog::set_pattern("[%C%m%d %T.%e][%L][%t] %v");

        logger_->flush_on(spdlog::level::warn);
        spdlog::flush_every(std::chrono::seconds(5));
    }
    catch (const spdlog::spdlog_ex& ex) {
        //std::cout << "Log init failed: " << ex.what() << std::endl;
        return false;
    }

    init_ = true;
    lv_ = LL_Info; // default info
    return true;
}

void Logger::LoggerImpl::release()
{
    if (!init_)
        return;

    logger_->flush();
    spdlog::drop("main_logger");
    init_ = false;
    lv_ = LL_Off;
    //spdlog::drop_all() // release and close all loggers
}

void Logger::LoggerImpl::set_level(LogLevel lv)
{
    if (!init_)
        return;

    lv_.store(lv);
}

LogLevel Logger::LoggerImpl::get_level() const
{
    return lv_.load(std::memory_order::memory_order_relaxed);
}

void Logger::LoggerImpl::log_string(LogLevel lv, std::string& str)
{
    if (!init_)
        return;

    spdlog::level::level_enum lv_spd = spdlog::level::off; // map spdlog level
    switch (lv)
    {
    case LL_Trace:
        lv_spd = spdlog::level::trace;
        break;
    case LL_Debug:
        lv_spd = spdlog::level::debug;
        break;
    case LL_Info:
        lv_spd = spdlog::level::info;
        break;
    case LL_Warn:
        lv_spd = spdlog::level::warn;
        break;
    case LL_Error:
        lv_spd = spdlog::level::err;
        break;
    case LL_Critical:
        lv_spd = spdlog::level::critical;
        break;
    default:
        break;
    }

    logger_->log(lv_spd, str.c_str());
    //logger_->flush();

// format variable parameter for ansi c
// char buffer[1024] = {0};
// va_list arglist;
// va_start(arglist, msg);
// vsnprintf(buffer, sizeof(buffer), msg, arglist);
// va_end(arglist);
}

//bool Logger::LoggerImpl::filter_level(LogLevel lv) const
//{
//    if (lv == LL_Off || lv < get_level())
//        return true;
//
//    return false;
//}


Logger Logger::instance_;

bool Logger::init()
{
    return impl_->init();
}

bool Logger::init(const char* file, bool truncate)
{
    return impl_->init(file, truncate);
}

void Logger::release()
{
    impl_->release();
}

void Logger::set_level(LogLevel lv)
{
    impl_->set_level(lv);
}
    
LogLevel Logger::get_level() const
{
    return impl_->get_level();
}

void Logger::log_string(LogLevel lv, std::string& str)
{
    impl_->log_string(lv, str);
}

Logger::Logger()
{
    impl_ = std::make_unique<LoggerImpl>();
}

Logger::~Logger()
{
}

}