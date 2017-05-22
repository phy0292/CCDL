#include "context_pool.h"
#define max(a,b) (((a) > (b)) ? (a) : (b))

contextPool::contextPool(){
	m_spaceOfElements = 0;
	m_useElementCount = 0;
	m_currentPositionBlock = 0;
	m_currentPositionElm = 0;
	m_appendNumber = POOL_DEFAULT_SIZE;
	m_blocks.reserve(POOL_NUMBER_ALLOC_BLOCK);
	m_processHeap = GetProcessHeap();
	resize(POOL_DEFAULT_SIZE);
}

contextPool::~contextPool(){
	destroy();
}

void contextPool::destroy(){
	//HeapFree(m_processHeap, 0, m_elements);
	for (int i = 0; i < m_blocks.size(); ++i){
		if(m_blocks[i].elms != 0){
			HeapFree(m_processHeap, 0, m_blocks[i].elms);
		}
	}

	m_useElementCount = 0;
}

void contextPool::resize(unsigned int size){
	if (size <= m_spaceOfElements) return;

	//printf("resize...%d\n", size);
	int newSize = m_appendNumber;
	while(newSize + m_spaceOfElements < size){
		newSize *= POOL_APPEND_ACC;
	}
	m_appendNumber = newSize * POOL_APPEND_ACC;

	pool_block bk;
	bk.count = newSize;
	bk.elms = (pool_element*)HeapAlloc( 
		m_processHeap,
		HEAP_ZERO_MEMORY,
		sizeof(pool_element) * newSize);

	if(bk.elms == 0) return;
	m_spaceOfElements += newSize;
	m_blocks.push_back(bk);
	m_currentPositionBlock = m_blocks.size() - 1;
	m_currentPositionElm = 0;
}

void contextPool::nextPosition(){
	for (int i = m_currentPositionBlock; i < m_blocks.size(); ++i)
	{
		int def = (i == m_currentPositionBlock ? m_currentPositionElm : 0);
		for (int j = def; j < m_blocks[i].count; ++j)
		{
			if(!m_blocks[i].elms[j].isUsed){
				m_currentPositionBlock = i;
				m_currentPositionElm = j;
				return;
			}
		}
	}

	for (int i = 0; i <= m_currentPositionBlock; ++i)
	{
		int def = (i == m_currentPositionBlock ? m_currentPositionElm : m_blocks[i].count);
		for (int j = 0; j < def; ++j){
			if(!m_blocks[i].elms[j].isUsed){
				m_currentPositionBlock = i;
				m_currentPositionElm = j;
				return;
			}
		}
	}

	resize(m_appendNumber);
}

clientContext* contextPool::getContext(){
	clientContext* context = 0;
	pool_element* elm = &m_blocks[m_currentPositionBlock].elms[m_currentPositionElm];
	elm->isUsed = true;
	context = &elm->context;
	InterlockedIncrement(&m_useElementCount);
	nextPosition();
	return context;
}

void contextPool::releaseContext(clientContext* context){
	pool_element* elm = (pool_element*)context;
	elm->isUsed = false;
	InterlockedDecrement(&m_useElementCount);
}