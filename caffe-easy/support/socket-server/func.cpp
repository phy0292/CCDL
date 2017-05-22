#include "func.h"
#include <time.h>
#include <stdio.h>
#include  <dbghelp.h> 
#pragma comment(lib,  "dbghelp.lib")

CRITICAL_SECTION g_log_cs;

LONG WINAPI UnhandledExceptionFiltera(struct _EXCEPTION_POINTERS* ExceptionInfo)
{
	char   strDumpFile[260];
	sprintf(strDumpFile, "debug_%d.dmp",GetTickCount());
	HANDLE   hFile   =   CreateFileA(strDumpFile,   GENERIC_WRITE,   FILE_SHARE_WRITE,   NULL,   CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,   NULL   );

	if   (hFile!=INVALID_HANDLE_VALUE)
	{ 
		MINIDUMP_EXCEPTION_INFORMATION   ExInfo; 

		ExInfo.ThreadId   =   ::GetCurrentThreadId();
		ExInfo.ExceptionPointers   =   ExceptionInfo;
		ExInfo.ClientPointers   =   NULL;

		//   write   the   dump
		BOOL   bOK   =   MiniDumpWriteDump(GetCurrentProcess(),   GetCurrentProcessId(),   hFile,   MiniDumpNormal,  &ExInfo,   NULL,   NULL   );
		CloseHandle(hFile); 
	} 

	fcLog("UnhandledExceptionFiltera::%s", "非常抱歉，程序运行中发生了异常，软件正准备退出。请联系开发人员来处理此类异常，"
		"届时请将软件目录下的debug_0123XXX.dmp文件发送给我们以排查问题所在，谢谢。");
	return EXCEPTION_EXECUTE_HANDLER;
}

void DisableSetUnhandledExceptionFilter()
{
	void *addr = (void*)GetProcAddress(LoadLibraryA("kernel32.dll"),
		"SetUnhandledExceptionFilter");
	if (addr)
	{
		unsigned char code[16];
		int size = 0;
		code[size++] = 0x33;
		code[size++] = 0xC0;
		code[size++] = 0xC2;
		code[size++] = 0x04;
		code[size++] = 0x00;

		DWORD dwOldFlag, dwTempFlag;
		VirtualProtect(addr, size, PAGE_READWRITE, &dwOldFlag);
		WriteProcessMemory(GetCurrentProcess(), addr, code, size, NULL);
		VirtualProtect(addr, size, dwOldFlag, &dwTempFlag);
	}
}

unsigned int fcGetTickTime()
{
	return GetTickCount();
}

DWORD fcGetNumberOfProcessors()
{
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	return sysinfo.dwNumberOfProcessors;
}

const char* fcCurrentTimeString()
{
	static char time_string[20];
	tm t;
	_getsystime(&t);
	sprintf(time_string, "%04d-%02d-%02d %02d:%02d:%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
	return time_string;
}

void fcInitialize()
{
	InitializeCriticalSection(&g_log_cs);
	
#if 0
#ifndef _WIN64
	SetUnhandledExceptionFilter(UnhandledExceptionFiltera);
	DisableSetUnhandledExceptionFilter();
#endif
#endif
}

void fcLog(const char* fmt, ...)
{
	EnterCriticalSection(&g_log_cs);
	printf("[%s]:", fcCurrentTimeString());

	va_list vl;
	va_start(vl, fmt);
	vprintf(fmt, vl);
	printf("\n");
	LeaveCriticalSection(&g_log_cs);
}

void fcUninitialize()
{
	DeleteCriticalSection(&g_log_cs);
}

char* fcCopyString(char* dest, const char* src, unsigned int destLength)
{
	char* sv = dest;
	int remain = (int)destLength - 1;
	while(*src && remain > 0){
		*dest++ = *src++;
		remain--;
	}

	sv[destLength - remain - 1] = 0;
	return sv;
}