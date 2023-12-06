/**************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 **************************************************************************/
#pragma once

#include <sqlite3.h>
#include <mutex>
#include <condition_variable>

class Table
{
public:
    Table(const char *basefile);
    virtual ~Table();
    
    virtual void flush() = 0;
    virtual void finalize() = 0;

    void setIdOffset(sqlite3_int64 offset);

protected:
    sqlite3 *m_connection;
    std::mutex m_mutex;
    //std::mutex m_writeMutex;
    std::condition_variable m_wait;
    sqlite3_int64 m_idOffset;
};

class BufferedTablePrivate;
class BufferedTable: public Table
{
public:
    void flush() override;
    void finalize() override;

protected:
    BufferedTablePrivate *d;
    friend class BufferedTablePrivate;

    BufferedTable(const char *basefile, int bufferSize, int batchsize);
    virtual ~BufferedTable();

    std::mutex m_mutex;
    std::mutex m_writeMutex;
    std::condition_variable m_wait;

    const int BUFFERSIZE;
    const int BATCHSIZE;
    int m_head {0};
    int m_tail {0};

    bool workerRunning();

    virtual void writeRows() = 0;	// "write" to buffers (cache db)
    virtual void flushRows() = 0;	// "flush" to disk (main db)
};


class StringTablePrivate;
class StringTable: public BufferedTable
{
public:
    StringTable(const char *basefile);
    virtual ~StringTable();

    struct row {
        std::string string;
        sqlite3_int64 string_id;
    };

    //void insert(const row&);
    sqlite3_int64 getOrCreate(const std::string&);

private:
    StringTablePrivate *d;
    friend class StringTablePrivate;

    virtual void writeRows() override;
    virtual void flushRows() override;
};


class ApiTablePrivate;
class ApiTable: public BufferedTable
{
public:
    ApiTable(const char *basefile);
    virtual ~ApiTable();

    struct row {
        int pid;
        int tid;
        sqlite3_int64 start;
        sqlite3_int64 end;
        sqlite3_int64 apiName_id;
        sqlite3_int64 args_id;
        sqlite3_int64 api_id;  // correlation id
    };

    void insert(const row&);
    void insertRoctx(row&);
    void pushRoctx(const row&);
    void popRoctx(const row&);
    void suspendRoctx(sqlite3_int64 atTime);
    void resumeRoctx(sqlite3_int64 atTime);

private:
    ApiTablePrivate *d;
    friend class ApiTablePrivate;

    virtual void writeRows() override;
    virtual void flushRows() override;
};


class KernelApiTablePrivate;
class KernelApiTable: public BufferedTable
{
public:
    KernelApiTable(const char *basefile);
    virtual ~KernelApiTable();

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

private:
    KernelApiTablePrivate *d;
    friend class KernelApiTablePrivate;

    virtual void writeRows() override;
    virtual void flushRows() override;
};


class CopyApiTablePrivate;
class CopyApiTable: public BufferedTable
{
public:
    CopyApiTable(const char *basefile);
    virtual ~CopyApiTable();

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

private:
    CopyApiTablePrivate *d;
    friend class CopyApiTablePrivate;

    virtual void writeRows() override;
    virtual void flushRows() override;
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
class OpTable: public BufferedTable
{
public:
    OpTable(const char *basefile);
    virtual ~OpTable();

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

private:
    OpTablePrivate *d;
    friend class OpTablePrivate;

    virtual void writeRows() override;
    virtual void flushRows() override;
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


class MonitorTablePrivate;
class MonitorTable: public BufferedTable
{
public:
    MonitorTable(const char *basefile);
    virtual ~MonitorTable();

    struct row {
        std::string deviceType;
        std::string monitorType;
        sqlite3_int64 deviceId;
        sqlite3_int64 start;
        sqlite3_int64 end;
        std::string value;
    };

    void insert(const row&);
    void endCurrentRuns(sqlite3_int64 endTimestamp);

private:
    MonitorTablePrivate *d;
    friend class MonitorTablePrivate;

    virtual void writeRows() override;
    virtual void flushRows() override;
};
