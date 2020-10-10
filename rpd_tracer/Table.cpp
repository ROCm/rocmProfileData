#include "Table.h"

Table::Table(const char *basefile)
: m_connection(NULL)
{
    //pthread_mutex_init(m_mutex);
    //pthread_cond_init(m_wait);
    sqlite3_open(basefile, &m_connection);

}

Table::~Table()
{
    // FIXME: ensure these aren't in use
    //pthread_mutex_destroy(m_mutex);
    //pthread_cond_destroy(m_wait);

    sqlite3_close(m_connection);
}
