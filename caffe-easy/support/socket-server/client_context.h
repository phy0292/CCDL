
#pragma once
#include<stdio.h>
#include<winsock2.h> 
#include "func.h"
#include "def.h"
#include <vector>

class IOCP;

typedef enum{
	ioRecv,
	ioSend
}ioType;

typedef enum{
	packageBegin,
	packageBuild
}packageType;

struct clientContext;
struct ioPerData : public OVERLAPPED{
	clientContext* context;
	ioType type;
};

struct subContext{
	WSABUF buffer;
	char   data[SERVER_PACK_LEN];
	ioPerData io;
	DWORD lastOperTime;
	DWORD numOfBytes;			//已经发送出去的量
	DWORD flags;
};

struct clientContext{
	static CRITICAL_SECTION closeSection;
	std::vector<char>* package;
	int packageRemainLength;
	subContext send;
	subContext recv;
	SOCKET socket;
	SOCKADDR_IN addr;
	packageType packageType;
	unsigned int lastOperTime;
	IOCP* server;
	volatile bool isConnected;

	//static
	static void Initialize();
	static void Uninitialize();

	//method
	void updateOperTime();
	void init(SOCKET c, const SOCKADDR_IN& a, IOCP* iocp);
	void destroy();
	bool doClose();
	void postSendSignal();
	void postRecvSignal();

	//event 
	void OnClose();
	void OnAppect();
	bool OnRecv(DWORD dwBytesTransferred);
	bool OnSend(DWORD dwBytesTransferred);
};

typedef void(*procdoRecvData)(clientContext* context, int numOfBytes);

void setDoRecvDataListener(procdoRecvData proc);