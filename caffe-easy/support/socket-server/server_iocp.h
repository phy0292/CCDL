
#pragma once
#include<stdio.h>
#include<winsock2.h> 
#include "def.h"
#include "func.h"
#include "context_pool.h"

struct workerDetail{
	int indexOfWorker;
	IOCP* server;
	volatile DWORD dwThreadID;
};

#define IDLE_MAX_TIME			(0xFFFFFFFF)

class IOCP{

	static WSADATA m_wsa;
	static bool m_isInitWSA;

	int m_num_threads;
	contextPool m_contextPool;
	SOCKET	m_slisten;
	unsigned short m_port;
	HANDLE  m_completionPort;
	DWORD	m_dwNumberOfWorkerThread;
	workerDetail* m_pdWorkerDetail;
	bool	m_isStartup;
	volatile bool m_looperIsRunning;
	volatile bool m_looperIsExit;
	unsigned int m_clientIdleTimeout;
	volatile bool m_idleThreadRunning;
	volatile bool m_idleIsExit;

public:
	~IOCP();
	IOCP();

	static DWORD WINAPI WorkerThread(LPVOID lpParam);
	static DWORD WINAPI IdleWatchThread(LPVOID lpParam);
	static void Initialize();
	static void Uninitialize();

	bool Setup(unsigned short port, unsigned int client_idle_timeout = 60000, int num_threads = -1);
	void Loop();
	void Stop();
};