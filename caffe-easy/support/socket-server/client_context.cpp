//#include <afx.h>
#include "client_context.h"
#include "server_iocp.h"
#include "liaojie_server.h"

//void* AFX_CDECL operator new(size_t nSize, LPCSTR lpszFileName, int nLine);
//#define new new(__FILE__, __LINE__)

static procdoRecvData g_procdoRecvData = 0;
CRITICAL_SECTION clientContext::closeSection;
void clientContext::Initialize()
{
	InitializeCriticalSection(&closeSection);
}

void clientContext::Uninitialize()
{
	DeleteCriticalSection(&closeSection);
}

bool clientContext::doClose()
{
	bool ret = false;
	if(!isConnected) return false;

	EnterCriticalSection(&closeSection);
	ret = isConnected;
	isConnected = false;
	LeaveCriticalSection(&closeSection);
	return ret;
}

void clientContext::destroy()
{
	shutdown(socket, SD_SEND);
	closesocket(socket);
	delete this->package;

	//fcLog("destroy");
}

void clientContext::updateOperTime(){
	lastOperTime = fcGetTickTime();
}

void clientContext::init(SOCKET c, const SOCKADDR_IN& a, IOCP* iocp){
	this->package = new std::vector<char>();
	this->packageType = packageType::packageBegin;
	this->packageRemainLength = 0;
	socket = c;
	addr = a;
	server = iocp;
	updateOperTime();
	send.buffer.buf = send.data;
	send.buffer.len = SERVER_PACK_LEN;
	send.io.context = this;
	send.io.type = ioSend;
	send.numOfBytes = 0;
	send.flags = 0;

	recv.buffer.buf = recv.data;
	recv.buffer.len = SERVER_PACK_LEN;
	recv.io.context = this;
	recv.io.type = ioRecv;
	recv.numOfBytes = 0;
	recv.flags = 0;
	isConnected = true;
	OnAppect();
}

void clientContext::postSendSignal(){
	//send.buffer.buf += send.numOfBytes;
	//send.buffer.len -= send.numOfBytes;
	//WSASend保证发送buf里面全部数据出去，不会存在发送只成了一部分的情况

	WSASend(socket, 
		&send.buffer, 
		1,
		&send.numOfBytes, 
		send.flags, 
		&send.io,
		0);
}

void clientContext::postRecvSignal(){
	WSARecv(socket, 
		&recv.buffer, 
		1, 
		&recv.numOfBytes, 
		&recv.flags, 
		&recv.io,
		0); 
}

void clientContext::OnClose(){
	//char* ip = inet_ntoa(addr.sin_addr);
	//fcLog("OnClose : %s", ip);
}

void clientContext::OnAppect(){
	//char* ip = inet_ntoa(addr.sin_addr);
	//fcLog("OnAppect : %s", ip);
}

bool clientContext::OnRecv(DWORD dwBytesTransferred){
	//fcLog("OnRecv : %d", dwBytesTransferred);
	//memcpy(this->send.data, this->recv.data, dwBytesTransferred);
	//this->send.buffer.len = dwBytesTransferred;
	//this->postSendSignal();

	if (g_procdoRecvData)
		g_procdoRecvData(this, dwBytesTransferred);

	return true;
}

bool clientContext::OnSend(DWORD dwBytesTransferred){
	//fcLog("OnSend : %d", dwBytesTransferred);
	//return dwBytesTransferred < this->send.buffer.len - this->send.numOfBytes;
	return false;
}


void setDoRecvDataListener(procdoRecvData proc){
	g_procdoRecvData = proc;
}