
#pragma once
#include "client_context.h"
#include <vector>
#include "def.h"

struct pool_element{
	clientContext context;
	volatile bool isUsed;
};

struct pool_block{
	pool_element* elms;
	int count;
};

class contextPool{

private:
	std::vector<pool_block> m_blocks;
	volatile unsigned int m_useElementCount;
	int m_spaceOfElements;
	int m_currentPositionBlock;
	int m_currentPositionElm;
	int m_appendNumber;
	HANDLE m_processHeap;

private:
	void destroy();
	void nextPosition();
	void resize(unsigned int size);

public:
	contextPool();
	~contextPool();

	//不能多线程分配哦
	unsigned int getUsedCount(){return m_useElementCount;}
	clientContext* getContext();
	void releaseContext(clientContext* context);

	template<typename _DoProc>
	void each(_DoProc const& proc){
		int size = m_blocks.size();
		for (int i = 0; i < size; ++i)
		{
			for (int j = 0; j < m_blocks[i].count; ++j)
			{
				if (m_blocks[i].elms[j].isUsed)
					proc((clientContext*)&m_blocks[i].elms[j]);
			}
		}
	}
};