//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "os_common.h"
#include "assetManager.hpp"
#include <cstdarg>
#include <sys/stat.h>

#if defined (OS_ANDROID) || defined(OS_LINUX)
#include <unistd.h>
#include <sys/time.h>
#endif // defined (OS_ANDROID) || defined (OS_LINUX)

#if defined (OS_WINDOWS)
#define NOMINMAX
#include <windows.h>
#endif // defined (OS_WINDOWS)


#if defined (OS_ANDROID)
const char* gpAndroidAppName = nullptr;     // must be initialized with OS_SetApplicationName befoe use
const char* gpAndroidAppShortName = nullptr;// set by OS_SetApplicationName
#endif // defined (OS_ANDROID)

//-----------------------------------------------------------------------------
void OS_SetApplicationName(const char* applicationName)
//-----------------------------------------------------------------------------
{
#if defined (OS_ANDROID)
    if (gpAndroidAppName)
        free((void*)gpAndroidAppName);
    gpAndroidAppName = strdup(applicationName);
    gpAndroidAppShortName = strrchr(gpAndroidAppName, '.');
    if (gpAndroidAppShortName == nullptr)
        gpAndroidAppShortName = gpAndroidAppName;
    else
        ++gpAndroidAppShortName;
#endif // defined (OS_ANDROID)
}

//-----------------------------------------------------------------------------
uint32_t OS_GetNumCores()
//-----------------------------------------------------------------------------
{
#if defined(OS_WINDOWS)

    SYSTEM_INFO SysInfo;
    GetSystemInfo( &SysInfo );

    return SysInfo.dwNumberOfProcessors;

#elif defined(OS_ANDROID)

    // sysconf can return a negative number!
    int iNumCores = sysconf( _SC_NPROCESSORS_ONLN );  // Number of processors online
    if( iNumCores <= 0 )
        return 0;
    else
        return (uint32_t) iNumCores;

#elif defined(OS_LINUX)

    // sysconf can return a negative number!
    int iNumCores = sysconf( _SC_NPROCESSORS_ONLN );  // Number of processors online
    if( iNumCores <= 0 )
        return 0;
    else
        return (uint32_t) iNumCores;

#else

#error Must define an OS!

#endif  // defined(OS_XXX)
}

//-----------------------------------------------------------------------------
uint32_t OS_GetTimeMS()
//-----------------------------------------------------------------------------
{
#if defined (OS_WINDOWS)

    LARGE_INTEGER   nFrequency;
    LARGE_INTEGER   nTime;

    if (!QueryPerformanceFrequency(&nFrequency) || !QueryPerformanceCounter(&nTime))
    {
        return 0;
    }

    return (uint32_t)(((double)nTime.QuadPart / (double)nFrequency.QuadPart) * 1000.0f);

#elif defined (OS_ANDROID) || defined (OS_LINUX)

    struct timeval t;
    t.tv_sec = t.tv_usec = 0;
    if (gettimeofday(&t, NULL) == -1)
    {
        return 0;
    }

    return (uint32_t)(t.tv_sec * 1000LL + t.tv_usec / 1000LL);

#endif // defined (OS_WINDOWS|OS_ANDROID|OS_LINUX)
}

//-----------------------------------------------------------------------------
uint64_t OS_GetTimeUS()
//-----------------------------------------------------------------------------
{
#if defined (OS_WINDOWS)

    LARGE_INTEGER   nFrequency;
    LARGE_INTEGER   nTime;

    if (!QueryPerformanceFrequency(&nFrequency) || !QueryPerformanceCounter(&nTime))
    {
        return 0;
    }

    return (uint64_t)((double)nTime.QuadPart / (((double)nFrequency.QuadPart) / 1000000.0));

#elif defined (OS_ANDROID) || defined(OS_LINUX)

    struct timeval t;
    t.tv_sec = t.tv_usec = 0;

    if (gettimeofday(&t, NULL) == -1)
    {
        return 0;
    }

    return (uint64_t)(t.tv_sec * 1000000LL + t.tv_usec);

#endif // defined (OS_WINDOWS|OS_ANDROID|OS_LINUX)
}

//-----------------------------------------------------------------------------
void OS_SleepMs(uint32_t ms)
//-----------------------------------------------------------------------------
{
#if defined (OS_WINDOWS)

    Sleep(ms);

#elif defined (OS_ANDROID) || defined (OS_LINUX)

    usleep(ms*1000);

#endif // defined (OS_WINDOWS|OS_ANDROID|OS_LINUX)
}

#if defined (OS_WINDOWS)
//-----------------------------------------------------------------------------
static void LOG_(WORD ConsoleColor, const char* pszFormat, va_list args)
//-----------------------------------------------------------------------------
{
    char szBuffer[2048];    // Bumped to 2048 based on Vulkan validation errors
    szBuffer[0] = 0;

    vsnprintf_s(szBuffer, sizeof(szBuffer), _TRUNCATE, pszFormat, args);

    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), ConsoleColor);

    //print error message to stdout
    fprintf(stdout, "\n");
    fprintf(stdout, szBuffer);

    // Also send message to Visual Studio output (but not to the system dbg so we don't spam that)
    if( IsDebuggerPresent() )
    {
        OutputDebugString("\n");
        OutputDebugString(szBuffer);
    }
}
//-----------------------------------------------------------------------------
void LOGE(const char* pszFormat, ...)
//-----------------------------------------------------------------------------
{
    va_list args;
    va_start(args, pszFormat);
    LOG_(FOREGROUND_RED | FOREGROUND_INTENSITY, pszFormat, args);
    va_end(args);
}
//-----------------------------------------------------------------------------
void LOGW(const char* pszFormat, ...)
//-----------------------------------------------------------------------------
{
    va_list args;
    va_start(args, pszFormat);
    LOG_(FOREGROUND_RED | FOREGROUND_GREEN, pszFormat, args);
    va_end(args);
}
//-----------------------------------------------------------------------------
void LOGI(const char* pszFormat, ...)
//-----------------------------------------------------------------------------
{
    va_list args;
    va_start(args, pszFormat);
    LOG_(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE, pszFormat, args);
    va_end(args);
}
#endif // defined (OS_WINDOWS)

#if defined (OS_LINUX)
//-----------------------------------------------------------------------------
void LOGE(const char* pszFormat, ...)
//-----------------------------------------------------------------------------
{
    va_list args;
    va_start(args, pszFormat);
    vfprintf(stderr, pszFormat, args);
    va_end(args);
    fprintf(stderr, "\n");
}
//-----------------------------------------------------------------------------
void LOGW(const char* pszFormat, ...)
//-----------------------------------------------------------------------------
{
    va_list args;
    va_start(args, pszFormat);
    vfprintf(stdout, pszFormat, args);
    va_end(args);
    fprintf(stdout, "\n");
}
//-----------------------------------------------------------------------------
void LOGI(const char* pszFormat, ...)
//-----------------------------------------------------------------------------
{
    va_list args;
    va_start(args, pszFormat);
    vfprintf(stdout, pszFormat, args);
    va_end(args);
    fprintf(stdout, "\n");
}
#endif // defined (OS_LINUX)

