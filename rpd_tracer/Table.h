/**************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 **************************************************************************/
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
    void suspendRoctx(sqlite3_int64 atTime);
    void resumeRoctx(sqlite3_int64 atTime);
    void flush();
    void finalize();

private:
    ApiTablePrivate *d;
    friend class ApiTablePrivate;
};


class KernelApiTablePrivate;
class KernelApiTable: public Table
{
public:
    KernelApiTable(const char *basefile);

    struct row {
        std::string stream;
        int gridX {0};
        int gridY {0};
        int gridZ {0};
        int workgroupX {0};
        int workgroupY {0};
        int workgroupZ {0};
        int groupSegmentSize {0};
        int privateSegmentSize {0};
        sqlite3_int64 kernelName_id;
        //codeObject
        //kernelArgAddress
        //aquireFence
        //releaseFence
        sqlite3_int64 api_id;   // Baseclass ApiTable primary key (correlation id)
    };

    void insert(const row&);
    void flush();
    void finalize();

private:
    KernelApiTablePrivate *d;
    friend class KernelApiTablePrivate;
};


class CopyApiTablePrivate;
class CopyApiTable: public Table
{
public:
    CopyApiTable(const char *basefile);

    struct row {
        std::string stream;
        int size {0};
        int width {0};
        int height {0};
        std::string dst;
        std::string src;
        int dstDevice {0};
        int srcDevice {0};
        int kind {0};
        bool sync {false};
        bool pinned {false};
        sqlite3_int64 api_id {0};   // Baseclass ApiTable primary key (correlation id)
    };
    void insert(const row&);
    void flush();
    void finalize();

private:
    CopyApiTablePrivate *d;
    friend class CopyApiTablePrivate;
};


#if 0
class BarrierOpTablePrivate;
class BarrierOpTable: public Table
{
public:
    BarrierOpTable(const char *basefile);

    struct row {
        sqlite3_int64 signalCount;
        char aquireFence[9];
        char releaseFence[9];
        sqlite3_int64 api_id;  // correlation id
    };
    void insert(const row&);
    void flush();
    void finalize();

private:
    BarrierOpTablePrivate *d;
    friend class BarrierOpTablePrivate
};
#endif


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
    void associateDescription(const sqlite3_int64 &api_id, const sqlite3_int64 &string_id);
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
