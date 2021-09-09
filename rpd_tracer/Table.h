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

    void setIdOffset(sqlite3_int64 offset);

protected:
    sqlite3 *m_connection;
    std::mutex m_mutex;
    std::condition_variable m_wait;
    sqlite3_int64 m_idOffset;
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

    //void insert(const row&);
    sqlite3_int64 getOrCreate(std::string);
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
    void insertRoctx(row&);
    void pushRoctx(const row&);
    void popRoctx(const row&);
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

    struct kernelRow {
        sqlite3_int64 gridX;
        sqlite3_int64 gridY;
        sqlite3_int64 gridZ;
        sqlite3_int64 workgroupX;
        sqlite3_int64 workgroupY;
        sqlite3_int64 workgroupZ;
        sqlite3_int64 groupSegmentSize;
        sqlite3_int64 privateSegmentSize;
        //codeObject
        //kernelName
        //kernelArgAddress
        //aquireFence
        //releaseFence
        sqlite3_int64 api_id;  // correlation id
    };

    struct copyRow {
        sqlite3_int64 size;
        sqlite3_int64 src;
        sqlite3_int64 dst;
        bool sync;
        bool pinned;
    };

    struct barrierRow {
        sqlite3_int64 signalCount;
        char aquireFence[9];
        char releaseFence[9];
    };

    void insert(const row&);
    void associateDescription(const sqlite3_int64 &api_id, const sqlite3_int64 &string_id);
    void associateKernel(const sqlite3_int64 &api_id, const kernelRow &row);
    void associateCopy(const sqlite3_int64 &api_id, const copyRow &row);
    void associateBarrier(const sqlite3_int64 &api_id, const barrierRow &row);
    void flush();
    void finalize();

private:
    OpTablePrivate *d;
    friend class OpTablePrivate;
};


class MetadataTablePrivate;
class MetadataTable: public Table
{
public:
    MetadataTable(const char *basefile);

    sqlite3_int64 sessionId();

    void flush();
    void finalize();

private:
    MetadataTablePrivate *d;
    friend class MetadataTablePrivate;
};
