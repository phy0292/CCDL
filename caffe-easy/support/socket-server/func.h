
#pragma once
#include <windows.h>

unsigned int fcGetTickTime();
DWORD fcGetNumberOfProcessors();

void fcInitialize();
void fcLog(const char* fmt, ...);
void fcUninitialize();
char* fcCopyString(char* dest, const char* src, unsigned int destLength);
#define fcStrcpyFixed(dest, src)		fcCopyString(dest, src, sizeof(dest))