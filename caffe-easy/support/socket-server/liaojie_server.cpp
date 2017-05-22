#include "liaojie_server.h"
#include "server_iocp.h"
#include "client_context.h"

IOCP g_iocp;
char g_errorMessage[1024] = {0};


bool ljInitialize()
{
	IOCP::Initialize();
	return true;
}

bool ljSetup(unsigned short server_port, unsigned int client_idle_timeout, int num_threads)
{
	if (!g_iocp.Setup(server_port, client_idle_timeout, num_threads))
	{
		sprintf(g_errorMessage, "IOCP setup error.");
		return false;
	}
	printf("listen: %d\n", server_port);
	return true;
}

const char* ljErrorMessage()
{
	return g_errorMessage;
}

void ljStop()
{
	g_iocp.Stop();
}

void ljLoop()
{
	g_iocp.Loop();
}

void ljUninitialize()
{
	IOCP::Uninitialize();
}
