/* waka_runtime.h — Wakawaka compiled-mode runtime
 * Every Wakawaka value is a WkVal (tagged union).
 * Heap types are reference-counted; scalar types live on the stack.
 */
#ifndef WAKA_RUNTIME_H
#define WAKA_RUNTIME_H

/* Enable POSIX extensions before any system headers */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

/* ── Threading primitives ────────────────────────────────────────────────── */
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
typedef CRITICAL_SECTION   WkMutex;
typedef CONDITION_VARIABLE WkCondVar;
#else
#  include <pthread.h>
#  include <time.h>
#  include <errno.h>
typedef pthread_mutex_t WkMutex;
typedef pthread_cond_t  WkCondVar;
#endif

/* ── Tag enum ────────────────────────────────────────────────────────────── */

typedef enum {
    WK_NONE  = 0,
    WK_BOOL  = 1,
    WK_INT   = 2,
    WK_FLOAT = 3,
    WK_STR   = 4,
    WK_LIST  = 5,
    WK_MAP   = 6,
    WK_FUNC  = 7,
    WK_OBJ   = 8,
    WK_RANGE = 9,
    WK_SOME  = 10,
    WK_OK    = 11,
    WK_ERR   = 12,
    WK_TUPLE = 13,
    WK_CLASS = 14,
    WK_TENSOR = 15,
} WkTag;

/* ── Forward declarations ─────────────────────────────────────────────────── */

typedef struct WkVal     WkVal;
typedef struct WkStr     WkStr;
typedef struct WkList    WkList;
typedef struct WkMap     WkMap;
typedef struct WkTuple   WkTuple;
typedef struct WkFunc    WkFunc;
typedef struct WkObj     WkObj;
typedef struct WkClass   WkClass;
typedef struct WkCapture WkCapture;
typedef struct WkMethod  WkMethod;
typedef struct WkMailbox WkMailbox;  /* actor mailbox — defined after WkVal */
typedef struct WkTensor WkTensor;

/* ── WkRange (inline, no deps on WkVal) ─────────────────────────────────── */

typedef struct {
    int64_t start, end;
    int     inclusive;
} WkRange;

/* ── Heap structs with pointer-only deps on WkVal (safe before WkVal defn) ─ */

struct WkStr {
    int    refcnt;
    size_t len;
    char   data[];      /* flexible array — single malloc */
};

struct WkList {
    int    refcnt;
    size_t len, cap;
    WkVal *items;       /* pointer — forward decl sufficient */
};

struct WkMethod {
    const char *name;
    WkFunc     *fn;
};

struct WkClass {
    const char   *name;
    WkClass      *parent;
    WkMethod     *methods;
    int           nmethods;
    const char  **field_names;
    int           nfields;
};

struct WkObj {
    int         refcnt;
    WkClass    *cls;
    WkVal      *fields;    /* pointer — forward decl sufficient */
    WkMailbox  *mailbox;   /* non-NULL for actor instances */
};

/* ── Tensor (dense, row-major, float64) ──────────────────────────────────── */

struct WkTensor {
    int      refcnt;
    int      ndim;
    int64_t *shape;     /* ndim elements */
    int64_t *strides;   /* ndim elements (in element counts) */
    int64_t  len;       /* total element count = product(shape) */
    double  *data;      /* row-major float64 */
    int      owns_data; /* 1 => free(data) on dealloc */
};

typedef WkVal (*WkCFunc)(WkVal *args, int argc, WkFunc *fn);

struct WkFunc {
    int          refcnt;
    const char  *name;
    WkCFunc      cfn;
    WkCapture   *captures;  /* pointer — forward decl sufficient */
    int          ncaptures;
    const char **param_names;
    int          nparams;
};

/* ── The core value type (defined before any embedded-WkVal structs) ─────── */

struct WkVal {
    WkTag tag;
    int   refcnt;       /* 0 = stack/static, never heap-freed */
    union {
        int64_t  i;     /* WK_INT, WK_BOOL */
        double   f;     /* WK_FLOAT */
        WkStr   *str;   /* WK_STR */
        WkList  *list;  /* WK_LIST */
        WkMap   *map;   /* WK_MAP */
        WkFunc  *func;  /* WK_FUNC */
        WkObj   *obj;   /* WK_OBJ */
        WkTuple *tup;   /* WK_TUPLE */
        WkClass *cls;   /* WK_CLASS */
        WkRange  rng;   /* WK_RANGE — inline, no heap */
        WkVal   *inner; /* WK_SOME, WK_OK, WK_ERR */
        WkTensor *tensor; /* WK_TENSOR */
    } as;
};

/* ── Types that embed WkVal directly (must follow WkVal definition) ─────── */

typedef struct {
    WkVal key, val;
    int   used;
} WkMapEntry;

struct WkMap {
    int         refcnt;
    size_t      len, cap;   /* cap is always power-of-2 */
    WkMapEntry *buckets;
};

struct WkTuple {
    int    refcnt;
    size_t len;
    WkVal  items[];         /* flexible array */
};

struct WkCapture {
    const char *name;
    WkVal       val;
};

/* ── Actor mailbox (thread-safe bounded queue of WkVal) ──────────────────── */

#define WK_MAILBOX_CAP 256

struct WkMailbox {
    WkVal    buf[WK_MAILBOX_CAP];
    int      head, tail, count;
    WkMutex  lock;
    WkCondVar not_empty;
    WkCondVar not_full;
};

/* ── Stack-value constructors (inline, refcnt=0) ─────────────────────────── */

static inline WkVal wk_none(void) {
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_NONE; return v;
}
static inline WkVal wk_bool(int b) {
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_BOOL; v.as.i=b?1:0; return v;
}
static inline WkVal wk_int(int64_t i) {
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_INT; v.as.i=i; return v;
}
static inline WkVal wk_float(double f) {
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_FLOAT; v.as.f=f; return v;
}
static inline WkVal wk_range(int64_t s, int64_t e, int inc) {
    WkVal v; memset(&v,0,sizeof(v));
    v.tag=WK_RANGE; v.as.rng.start=s; v.as.rng.end=e; v.as.rng.inclusive=inc;
    return v;
}

/* ── Defer support ───────────────────────────────────────────────────────── */

#define WK_DEFER_MAX 64

typedef struct {
    WkVal fn;
} WkDeferEntry;

typedef struct WkDeferFrame {
    WkDeferEntry         entries[WK_DEFER_MAX];
    int                  count;
    struct WkDeferFrame *prev;
} WkDeferFrame;

void wk_defer_push_frame(WkDeferFrame *f);
void wk_defer_pop_frame(WkDeferFrame *f);
void wk_defer_register(WkDeferFrame *f, WkVal fn);
void wk_defer_flush(WkDeferFrame *f);

/* ── Refcount ────────────────────────────────────────────────────────────── */

void wk_incref(WkVal v);
void wk_decref(WkVal v);

/* ── Heap constructors ───────────────────────────────────────────────────── */

WkVal wk_make_str(const char *data, size_t len);
WkVal wk_make_strz(const char *cstr);
WkVal wk_make_list(void);
WkVal wk_make_map(void);
WkVal wk_make_tuple(WkVal *items, int n);
WkVal wk_make_func(const char *name, WkCFunc cfn,
                   WkCapture *caps, int ncaps,
                   const char **pnames, int nparams);
WkVal wk_make_obj(WkClass *cls);
WkVal wk_make_some(WkVal inner);
WkVal wk_make_ok(WkVal inner);
WkVal wk_make_err(WkVal inner);
WkVal wk_make_class(WkClass *cls);
WkVal wk_make_tensor(int ndim, const int64_t *shape, const double *data, int owns);

/* ── Arithmetic ──────────────────────────────────────────────────────────── */

WkVal wk_add(WkVal a, WkVal b);
WkVal wk_sub(WkVal a, WkVal b);
WkVal wk_mul(WkVal a, WkVal b);
WkVal wk_div(WkVal a, WkVal b);
WkVal wk_mod(WkVal a, WkVal b);
WkVal wk_pow(WkVal a, WkVal b);
WkVal wk_neg(WkVal a);
WkVal wk_not(WkVal a);
WkVal wk_bitnot(WkVal a);
WkVal wk_band(WkVal a, WkVal b);
WkVal wk_bor(WkVal a, WkVal b);
WkVal wk_bxor(WkVal a, WkVal b);
WkVal wk_lshift(WkVal a, WkVal b);
WkVal wk_rshift(WkVal a, WkVal b);

/* ── Comparison ──────────────────────────────────────────────────────────── */

int   wk_truthy(WkVal v);
int   wk_equal(WkVal a, WkVal b);
WkVal wk_cmp_eq(WkVal a, WkVal b);
WkVal wk_cmp_ne(WkVal a, WkVal b);
WkVal wk_cmp_lt(WkVal a, WkVal b);
WkVal wk_cmp_le(WkVal a, WkVal b);
WkVal wk_cmp_gt(WkVal a, WkVal b);
WkVal wk_cmp_ge(WkVal a, WkVal b);
WkVal wk_in(WkVal needle, WkVal haystack);

/* ── Casts ───────────────────────────────────────────────────────────────── */

WkVal wk_cast_int(WkVal v);
WkVal wk_cast_float(WkVal v);
WkVal wk_cast_bool(WkVal v);
WkVal wk_cast_str(WkVal v);
WkVal wk_cast_byte(WkVal v);

/* ── String operations ───────────────────────────────────────────────────── */

WkVal wk_str_concat(WkVal a, WkVal b);
WkVal wk_str_index(WkVal s, WkVal idx);
WkVal wk_str_slice(WkVal s, WkVal start, WkVal end);
WkVal wk_list_slice(WkVal lst, WkVal start, WkVal end, WkVal step);
WkVal wk_str_method(WkVal s, const char *name);
WkVal wk_str_fmtbuild(int nparts, ...); /* alternating: char* literal, WkVal expr */
char *wk_to_cstr(WkVal v);              /* malloc'd; caller frees */
char *wk_to_repr(WkVal v);              /* malloc'd; caller frees */

/* ── List operations ─────────────────────────────────────────────────────── */

void  wk_list_push_raw(WkList *lst, WkVal v);
WkVal wk_list_get(WkVal lst, WkVal idx);
void  wk_list_set(WkVal lst, WkVal idx, WkVal v);
WkVal wk_list_method(WkVal lst, const char *name);

/* ── Map operations ──────────────────────────────────────────────────────── */

WkVal wk_map_get_key(WkVal m, WkVal key);
void  wk_map_set_key(WkVal m, WkVal key, WkVal val);
WkVal wk_map_keys(WkVal m);
WkVal wk_map_method(WkVal m, const char *name);

/* ── Generic member / index access ──────────────────────────────────────── */

WkVal wk_member_get(WkVal obj, const char *name);
void  wk_member_set(WkVal obj, const char *name, WkVal v);
WkVal wk_index_get(WkVal obj, WkVal idx);
void  wk_index_set(WkVal obj, WkVal idx, WkVal val);

/* ── Object / class ──────────────────────────────────────────────────────── */

WkVal wk_obj_get_field(WkVal obj, const char *name);
void  wk_obj_set_field(WkVal obj, const char *name, WkVal v);
WkVal wk_obj_find_method(WkVal obj, const char *name);

/* ── Function call ───────────────────────────────────────────────────────── */

WkVal wk_call(WkVal fn, WkVal *args, int argc);
WkVal wk_call0(WkVal fn);
WkVal wk_call1(WkVal fn, WkVal a0);
WkVal wk_call2(WkVal fn, WkVal a0, WkVal a1);

/* ── Built-in functions ──────────────────────────────────────────────────── */

WkVal wk_builtin_println(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_print(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_eprintln(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_readln(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_len(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_str(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_int(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_float(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_bool(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_typeof(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_isNone(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_isSome(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_isOk(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_isErr(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_sum(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_min(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_max(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_map_fn(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_filter(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_reduce(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_sorted(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_reversed(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_any(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_all(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_zip(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_enumerate(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_range(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_sleep(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_exit(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_assert(WkVal *args, int argc, WkFunc *fn);
WkVal wk_deep_copy(WkVal v);
WkVal wk_builtin_copy(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_chr(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_ord(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_hash(WkVal *args, int argc, WkFunc *fn);
WkVal wk_builtin_repr(WkVal *args, int argc, WkFunc *fn);

/* ── Shell execution ─────────────────────────────────────────────────────── */

WkVal wk_shell_exec(WkVal cmd);   /* run shell cmd, return captured stdout */

/* ── Goroutines (detached threads) ──────────────────────────────────────── */

void wk_go(WkVal fn, WkVal *args, int argc);

/* ── Actor mailbox ───────────────────────────────────────────────────────── */

WkMailbox *wk_mailbox_new(void);
void       wk_mailbox_send(WkMailbox *m, WkVal v);
/* Returns 1 if a message was received, 0 if timed out.
   timeout_ms < 0 means block indefinitely. */
int        wk_mailbox_recv(WkMailbox *m, int timeout_ms, WkVal *out);
void       wk_mailbox_free(WkMailbox *m);

/* ── Channels (WkObj with a mailbox) ─────────────────────────────────────── */

WkVal wk_make_chan(int capacity);       /* create a channel (obj+mailbox) */
void  wk_chan_send(WkVal ch, WkVal v);  /* ch <- v */
WkVal wk_chan_recv(WkVal ch);           /* <-ch */

/* ── SQL (only when compiled with -DWK_HAVE_SQL and linked with -lsqlite3) ─ */

#ifdef WK_HAVE_SQL
WkVal wk_sql_exec(const char *db_path, const char *query,
                  WkVal *params, int nparams);
#endif

/* ── Panic / error ───────────────────────────────────────────────────────── */

#ifdef _MSC_VER
__declspec(noreturn)
#else
__attribute__((noreturn))
#endif
void wk_panic(const char *fmt, ...);

/* ── Global init ─────────────────────────────────────────────────────────── */

/* Global WkVal slots for built-in names — set by wk_runtime_init() */
extern WkVal wk_g_println;
extern WkVal wk_g_print;
extern WkVal wk_g_eprintln;
extern WkVal wk_g_readln;
extern WkVal wk_g_len;
extern WkVal wk_g_str;
extern WkVal wk_g_int;
extern WkVal wk_g_float;
extern WkVal wk_g_bool;
extern WkVal wk_g_typeof;
extern WkVal wk_g_isNone;
extern WkVal wk_g_isSome;
extern WkVal wk_g_isOk;
extern WkVal wk_g_isErr;
extern WkVal wk_g_sum;
extern WkVal wk_g_min;
extern WkVal wk_g_max;
extern WkVal wk_g_map;
extern WkVal wk_g_filter;
extern WkVal wk_g_reduce;
extern WkVal wk_g_sorted;
extern WkVal wk_g_reversed;
extern WkVal wk_g_any;
extern WkVal wk_g_all;
extern WkVal wk_g_zip;
extern WkVal wk_g_enumerate;
extern WkVal wk_g_range;
extern WkVal wk_g_sleep;
extern WkVal wk_g_exit;
extern WkVal wk_g_assert;
extern WkVal wk_g_copy;
extern WkVal wk_g_chr;
extern WkVal wk_g_ord;
extern WkVal wk_g_hash;
extern WkVal wk_g_repr;
extern WkVal wk_g_panic;
extern WkVal wk_g_math;
extern WkVal wk_g_tensor;
extern WkVal wk_g_ad;
extern WkVal wk_g_gpu;
extern WkVal wk_g_pipeline;
extern WkVal wk_g_model;
extern WkVal wk_g_fs;
extern WkVal wk_g_py;
extern WkVal wk_g_jvm;
extern WkVal wk_g_node;

/* CLI argv (set by main before wk_runtime_init) */
extern int    wk_argc;
extern char **wk_argv;

void wk_runtime_init(void);

#endif /* WAKA_RUNTIME_H */
