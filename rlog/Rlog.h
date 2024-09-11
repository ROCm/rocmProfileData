

namespace rlog {

class Logger {
public:
    virtual void mark(const char *apiname, const char *args) = 0;
    virtual void mark(const char *domain, const char *apiname, const char *args) = 0;
    virtual void mark(const char *domain, const char *category, const char *apiname, const char *args) = 0;

    virtual void push(const char *apiname, const char *args) = 0;
    virtual void push(const char *domain, const char *apiname, const char *args) = 0;
    virtual void push(const char *domain, const char *category, const char *apiname, const char *args) = 0;

    virtual void pop();
};


}  // namespace rlog
