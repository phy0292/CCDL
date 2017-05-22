#pragma once
#include "def.h"
bool ljInitialize();
bool ljSetup(unsigned short server_port, unsigned int client_idle_timeout, int num_threads);
const char* ljErrorMessage();
void ljLoop();
void ljStop();
void ljUninitialize();