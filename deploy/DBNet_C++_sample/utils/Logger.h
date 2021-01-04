#ifndef __LOGGER_H__
#define __LOGGER_H__

#include <memory>
#include <string>

#include "spdlog/fmt/fmt.h"

#define LOGGER LOGGING::Logger::instance_

namespace LOGGING
{
    enum LogLevel
    {
        LL_Trace = 0,
        LL_Debug,
        LL_Info,
        LL_Warn,
        LL_Error,
        LL_Critical,
        LL_Off
    };

    class Logger
    {
    public:
        static Logger instance_;

        bool init();
        bool init(const char* file, bool truncate = false);
        void release();

        template<typename... Tn>
        void write_log(LogLevel lv, const char* fmt, const Tn&... tn)
        {
            if ((lv == LL_Off || lv < get_level()) || nullptr == fmt)
                return;
    
            std::string msg = fmt::format(fmt, tn...);
            log_string(lv, msg);
        }

        void set_level(LogLevel lv);
        LogLevel get_level() const;

    private:
        Logger();
        ~Logger();

        Logger(const Logger&) = delete;
        const Logger& operator=(const Logger&) = delete;

        //bool filter_level(LogLevel lv) const;
        void log_string(LogLevel lv, std::string& str);

        class LoggerImpl;
        std::unique_ptr<LoggerImpl> impl_;
    };
}

#endif // __LOGGER_H__