#include <sqlite3.h>
#include <mutex>
#include <condition_variable>

class Table
{
public:
    Table(const char *basefile);
    ~Table();
    
    virtual void flush() = 0;
    virtual void finalize() = 0;

protected:
    sqlite3 *m_connection;
    std::mutex m_mutex;
    std::condition_variable m_wait;
};


class StringTablePrivate;
class StringTable: public Table
{
public:
    StringTable(const char *basefile);

    struct row {
        std::string string;
        sqlite3_int64 string_id;
    };

    void insert(const row&);
    void flush();
    void finalize();

private:
    StringTablePrivate *d;
    friend class StringTablePrivate;
};


class ApiTablePrivate;
class ApiTable: public Table
{
public:
    ApiTable(const char *basefile);

    struct row {
        int pid;
        int tid;
        sqlite3_int64 start;
        sqlite3_int64 end;
        sqlite3_int64 apiName_id;
        sqlite3_int64 args_id;
        sqlite3_int64 api_id;  // correlation id
        int phase;
    };

    void insert(const row&);
    void flush();
    void finalize();

private:
    ApiTablePrivate *d;
    friend class ApiTablePrivate;
};



class OpTablePrivate;
class OpTable: public Table
{
public:
    OpTable(const char *basefile);

    struct row {
        int gpuId;
        int queueId;
        int sequenceId;
        char completionSignal[18];
        sqlite3_int64 start;
        sqlite3_int64 end;
        sqlite3_int64 description_id;
        sqlite3_int64 opType_id;
        sqlite3_int64 api_id;  // correlation id
    };

    void insert(const row&);
    void flush();
    void finalize();

private:
    OpTablePrivate *d;
    friend class OpTablePrivate;
};

