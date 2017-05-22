#include "server_iocp.h"
#include "client_context.h"
#pragma comment(lib, "ws2_32.lib")

bool IOCP::m_isInitWSA = false;
WSADATA IOCP::m_wsa;
IOCP::IOCP()
{
	m_isStartup = false;
	m_dwNumberOfWorkerThread = fcGetNumberOfProcessors() * 2;
	m_pdWorkerDetail = (workerDetail*)malloc(sizeof(workerDetail) * m_dwNumberOfWorkerThread);
	m_completionPort = INVALID_HANDLE_VALUE;
	m_slisten = 0;
	m_looperIsRunning = false;
	m_looperIsExit = true;
	m_idleThreadRunning = false;
	m_idleIsExit = true;
}

IOCP::~IOCP()
{
	Stop();
	free(m_pdWorkerDetail);
}

void IOCP::Initialize()
{
	if(m_isInitWSA) return;
	WSAStartup(MAKEWORD(2,2), &m_wsa);
	fcInitialize();
	clientContext::Initialize();
	m_isInitWSA = true;
}

void IOCP::Uninitialize()
{
	if(!m_isInitWSA) return;
	fcUninitialize();
	clientContext::Uninitialize();
	WSACleanup();
	m_isInitWSA = false;
}

bool IOCP::Setup(unsigned short port, unsigned int client_idle_timeout /*= 60000*/, int num_threads)
{
	Stop();

	SOCKADDR_IN local;
	int iAddrSize = sizeof(SOCKADDR_IN); 

	m_num_threads = num_threads;
	m_num_threads = m_num_threads < 1 ? m_dwNumberOfWorkerThread : m_num_threads;
	m_dwNumberOfWorkerThread = m_num_threads;
	fcLog("启动服务，线程数：%d", m_dwNumberOfWorkerThread);

	m_clientIdleTimeout = client_idle_timeout;
	m_port = port;
	m_completionPort = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
	m_slisten = socket(AF_INET, SOCK_STREAM, 0); 
	memset(&local,0,sizeof(SOCKADDR_IN)); 
	local.sin_family = AF_INET; 
	local.sin_port = htons(m_port); 
	local.sin_addr.s_addr = htonl(INADDR_ANY); 
	if(bind(m_slisten, (SOCKADDR*)&local, sizeof(SOCKADDR_IN)) == -1){
		return false;
	}

	if(listen(m_slisten, SOMAXCONN) != 0){
		closesocket(m_slisten);
		return false;
	}

	m_isStartup = true;
	return true;
}

void IOCP::Loop()
{
	SOCKADDR_IN client_addr;
	int iAddrSize = sizeof(SOCKADDR_IN);
	SOCKET sClient;
	clientContext* context = 0;
	m_looperIsRunning = true;
	m_looperIsExit = false;
	if(!m_isStartup) return;

	for(int i = 0; i < m_dwNumberOfWorkerThread; i++) 
	{
		m_pdWorkerDetail[i].indexOfWorker = i;
		m_pdWorkerDetail[i].server = this;
		CreateThread(NULL, 0, WorkerThread, &m_pdWorkerDetail[i], 0, (DWORD*)&m_pdWorkerDetail[i].dwThreadID);
	}

	//如果为最大空闲，就是说不空闲了呗
	if (this->m_clientIdleTimeout != IDLE_MAX_TIME){
		m_idleThreadRunning = true;
		CreateThread(NULL, 0, IdleWatchThread, this, 0, 0);
	}

	while(1)
	{ 
		sClient = accept(m_slisten, (SOCKADDR*)&client_addr, &iAddrSize);
		if(!m_looperIsRunning) break;

		HANDLE hd = CreateIoCompletionPort((HANDLE)sClient, m_completionPort, (DWORD)sClient, 0);
		if(hd == INVALID_HANDLE_VALUE){
			fcLog("error... = %d", WSAGetLastError());
			continue;
		}

		context = m_contextPool.getContext();
		if(context == 0){
			fcLog("Alloc clientContext error...");
			continue;
		}

		context->init(sClient, client_addr, this);
		context->postRecvSignal();
	}
	m_looperIsExit = true;
}

//执行stop的一定不能是worker线程和主线程
void IOCP::Stop()
{
	if(!m_isStartup) return;

	if(m_looperIsRunning){
		this->m_contextPool.each([&](clientContext* c){
			if (c->isConnected){
				if (c->doClose()){
					c->OnClose();
					c->destroy();
					this->m_contextPool.releaseContext(c);
				}
			}
		});

		m_looperIsRunning = false;
		m_idleThreadRunning = false;
		for (int i = 0; i < m_dwNumberOfWorkerThread; ++i)
			PostQueuedCompletionStatus(m_completionPort, 0, SERVER_WORKER_EXIT_TOKEN, NULL);

		bool workerIsExit = true;
		while(1){
			workerIsExit = true;
			for(int i = 0; i < m_dwNumberOfWorkerThread; ++i){
				if(m_pdWorkerDetail[i].dwThreadID != -1){
					workerIsExit = false;
					break;
				}
			}

			if(workerIsExit) break;
			Sleep(1);
		}

		while(!m_idleIsExit) Sleep(1);
		CloseHandle(m_completionPort);
		closesocket(m_slisten);
		while(!m_looperIsExit) Sleep(1);
	}else{
		//CloseHandle(m_completionPort);
		//closesocket(m_slisten);
	}
	m_isStartup = false;
}

DWORD WINAPI IOCP::IdleWatchThread(LPVOID lpParam)
{
	IOCP* p = (IOCP*)lpParam;
	p->m_idleIsExit = false;

	auto isExpired = [&](unsigned int t)->bool{
		unsigned int current = fcGetTickTime();
		unsigned int idleTime = 0;
		if(t > current){
			//说明时间轮询了
			idleTime = 0xFFFFFFFF - t + current;
		}else{
			idleTime = current - t;
		}

		return idleTime > p->m_clientIdleTimeout;
	};

	while(p->m_idleThreadRunning){
		p->m_contextPool.each([&](clientContext* c){
			if(c->isConnected){
				if(isExpired(c->lastOperTime)){
					if(c->doClose()){
						c->OnClose();
						c->destroy();
						p->m_contextPool.releaseContext(c);
					}
				}
			}
		});
		Sleep(SERVER_IDLE_WATCH_CYCLE_TIME);
	}

	p->m_idleIsExit = true;
	return 0;
}

DWORD WINAPI IOCP::WorkerThread(LPVOID lpParam) 
{
	workerDetail* worker = (workerDetail*)lpParam;
	HANDLE completionPort = worker->server->m_completionPort;
	DWORD dwBytesTransferred;
	SOCKET c; 
	ioPerData* io = 0;
	clientContext* context = 0;
	while(TRUE)
	{
		//等待和完成端口關聯的任意套接字上的I/O完成 
		GetQueuedCompletionStatus(completionPort, 
			&dwBytesTransferred, 
			(PULONG_PTR)&c,
			(LPOVERLAPPED*)&io, 
			INFINITE); 
		if(c == SERVER_WORKER_EXIT_TOKEN) 
		{
			fcLog("exit io = %p", io);
			break;
		}

		if(io == 0) continue;
		context = io->context;

		//先檢查一下，看是否在套接字上發生錯誤﹔ 
		//如果發生了，關閉套接字，并清除和這個套接字關聯的單句柄數据和單I/O操作數据 
		if(dwBytesTransferred == 0)
		{
			//BytesTransferred為0時，表明套接字已被通信對方關閉，因此我們也要關閉套接字 
			//注意﹕單句柄數据用來引用和I/O關聯的套接字 
			if(context->doClose()){
				context->OnClose();
				context->destroy();
				worker->server->m_contextPool.releaseContext(context);
			}
			continue;  
		}

		switch(io->type)
		{
		case ioRecv:
			{
				context->updateOperTime();
				if(context->OnRecv(dwBytesTransferred)){
					context->postRecvSignal();
					/*
					WSARecv(c, 
						&context->recv.buffer, 
						1,
						&context->recv.numOfBytes, 
						&context->recv.flags, 
						&context->recv.io,
						NULL);
				   */
				}
				break;
			}
		case ioSend:
			{
				context->updateOperTime();

				//这里的dwBytesTransferred一定是你发送出去的量
				context->OnSend(dwBytesTransferred);
				
				/*
				if(context->OnSend(dwBytesTransferred)){
					//context->postSendSignal();
					WSASend(c, 
						&context->send.buffer, 
						1,
						&context->send.numOfBytes, 
						context->send.flags, 
						&context->send.io,
						NULL);
				}
				*/
				break;
			}
		}
	} 

	//fcLog("退出啦...%d", worker->dwThreadID);
	worker->dwThreadID = -1;
	return 0; 
}