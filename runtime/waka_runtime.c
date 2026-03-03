/* waka_runtime.c — Wakawaka compiled-mode runtime implementation */

/* Enable POSIX extensions (strdup, popen, pclose, nanosleep, clock_gettime) */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "waka_runtime.h"
#include <time.h>
#include <ctype.h>
#include <errno.h>

/* Suppress warn_unused_result for fread in model loaders */
static inline void FREAD(void *ptr, size_t sz, size_t n, FILE *fp) {
    size_t r = fread(ptr, sz, n, fp); (void)r;
}

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════
   FORWARD DECLARATIONS (for functions defined later but used early)
   ═══════════════════════════════════════════════════════════════════════════ */

/* Dual number helpers (autodiff) */
static WkVal _wk_make_dual(double val, double deriv);
static int _wk_is_dual(WkVal v);
static double _dual_val(WkVal v);
static double _dual_der(WkVal v);
static double _wk_val_to_double(WkVal v);

/* Dual number class */
static WkClass _wk_cls_dual;

/* Tensor method dispatch */
static WkVal wk_tensor_getmethod(WkVal obj, const char *mname);

/* Pipeline class + methods */
static WkClass _wk_cls_pipeline;
static WkVal _wkp_map(WkVal *args, int argc, WkFunc *fn);
static WkVal _wkp_filter(WkVal *args, int argc, WkFunc *fn);
static WkVal _wkp_batch(WkVal *args, int argc, WkFunc *fn);
static WkVal _wkp_flatten(WkVal *args, int argc, WkFunc *fn);
static WkVal _wkp_take(WkVal *args, int argc, WkFunc *fn);
static WkVal _wkp_skip(WkVal *args, int argc, WkFunc *fn);
static WkVal _wkp_shuffle(WkVal *args, int argc, WkFunc *fn);
static WkVal _wkp_zip(WkVal *args, int argc, WkFunc *fn);
static WkVal _wkp_collect(WkVal *args, int argc, WkFunc *fn);
static WkVal _wkp_reduce(WkVal *args, int argc, WkFunc *fn);
static WkVal _wkp_forEach(WkVal *args, int argc, WkFunc *fn);
static WkVal _wkp_count(WkVal *args, int argc, WkFunc *fn);

/* ═══════════════════════════════════════════════════════════════════════════
   PANIC
   ═══════════════════════════════════════════════════════════════════════════ */

void wk_panic(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "\033[31m[Wakawaka panic] ");
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\033[0m\n");
    exit(1);
}

/* ═══════════════════════════════════════════════════════════════════════════
   MEMORY HELPERS
   ═══════════════════════════════════════════════════════════════════════════ */

static void *wk_malloc(size_t n) {
    void *p = malloc(n);
    if (!p) wk_panic("out of memory");
    return p;
}
static void *wk_realloc(void *p, size_t n) {
    void *q = realloc(p, n);
    if (!q) wk_panic("out of memory (realloc)");
    return q;
}

/* ═══════════════════════════════════════════════════════════════════════════
   REFCOUNT
   ═══════════════════════════════════════════════════════════════════════════ */

/* Forward declarations */
static void wk_free_val(WkVal v);
void wk_mailbox_free(WkMailbox *m); /* defined later in this file */

void wk_incref(WkVal v) {
    if (v.refcnt == 0) return; /* stack value */
    switch (v.tag) {
        case WK_STR:   v.as.str->refcnt++;  break;
        case WK_LIST:  v.as.list->refcnt++; break;
        case WK_MAP:   v.as.map->refcnt++;  break;
        case WK_FUNC:  v.as.func->refcnt++; break;
        case WK_OBJ:   v.as.obj->refcnt++;  break;
        case WK_TUPLE: v.as.tup->refcnt++;  break;
        case WK_TENSOR: v.as.tensor->refcnt++; break;
        default: break;
    }
}

void wk_decref(WkVal v) {
    if (v.refcnt == 0) return;
    switch (v.tag) {
        case WK_STR:
            if (--v.as.str->refcnt == 0) free(v.as.str); break;
        case WK_LIST:
            if (--v.as.list->refcnt == 0) {
                for (size_t i = 0; i < v.as.list->len; i++)
                    wk_decref(v.as.list->items[i]);
                free(v.as.list->items);
                free(v.as.list);
            } break;
        case WK_MAP:
            if (--v.as.map->refcnt == 0) {
                for (size_t i = 0; i < v.as.map->cap; i++)
                    if (v.as.map->buckets[i].used) {
                        wk_decref(v.as.map->buckets[i].key);
                        wk_decref(v.as.map->buckets[i].val);
                    }
                free(v.as.map->buckets);
                free(v.as.map);
            } break;
        case WK_FUNC:
            if (--v.as.func->refcnt == 0) {
                if (v.as.func->captures) free(v.as.func->captures);
                free(v.as.func);
            } break;
        case WK_OBJ:
            if (--v.as.obj->refcnt == 0) {
                if (v.as.obj->fields) {
                    for (int i = 0; i < v.as.obj->cls->nfields; i++)
                        wk_decref(v.as.obj->fields[i]);
                    free(v.as.obj->fields);
                }
                if (v.as.obj->mailbox) wk_mailbox_free(v.as.obj->mailbox);
                free(v.as.obj);
            } break;
        case WK_TUPLE:
            if (--v.as.tup->refcnt == 0) {
                for (size_t i = 0; i < v.as.tup->len; i++)
                    wk_decref(v.as.tup->items[i]);
                free(v.as.tup);
            } break;
        case WK_TENSOR:
            if (--v.as.tensor->refcnt == 0) {
                free(v.as.tensor->shape);
                free(v.as.tensor->strides);
                if (v.as.tensor->owns_data) free(v.as.tensor->data);
                free(v.as.tensor);
            } break;
        case WK_SOME: case WK_OK: case WK_ERR:
            /* inner is a malloc'd WkVal */
            free(v.as.inner); break;
        default: break;
    }
}

static void wk_free_val(WkVal v) { wk_decref(v); }

/* ═══════════════════════════════════════════════════════════════════════════
   HEAP CONSTRUCTORS
   ═══════════════════════════════════════════════════════════════════════════ */

WkVal wk_make_str(const char *data, size_t len) {
    WkStr *s = (WkStr*)wk_malloc(sizeof(WkStr) + len + 1);
    s->refcnt = 1; s->len = len;
    if (data) memcpy(s->data, data, len);
    s->data[len] = '\0';
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_STR; v.refcnt=1; v.as.str=s;
    return v;
}
WkVal wk_make_strz(const char *cstr) {
    return wk_make_str(cstr, cstr ? strlen(cstr) : 0);
}
WkVal wk_make_list(void) {
    WkList *l = (WkList*)wk_malloc(sizeof(WkList));
    l->refcnt=1; l->len=0; l->cap=8;
    l->items = (WkVal*)wk_malloc(8*sizeof(WkVal));
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_LIST; v.refcnt=1; v.as.list=l;
    return v;
}
void wk_list_push_raw(WkList *l, WkVal item) {
    if (l->len == l->cap) {
        l->cap *= 2;
        l->items = (WkVal*)wk_realloc(l->items, l->cap*sizeof(WkVal));
    }
    l->items[l->len++] = item;
}
WkVal wk_make_map(void) {
    WkMap *m = (WkMap*)wk_malloc(sizeof(WkMap));
    m->refcnt=1; m->len=0; m->cap=8;
    m->buckets = (WkMapEntry*)wk_malloc(8*sizeof(WkMapEntry));
    memset(m->buckets, 0, 8*sizeof(WkMapEntry));
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_MAP; v.refcnt=1; v.as.map=m;
    return v;
}
WkVal wk_make_tuple(WkVal *items, int n) {
    WkTuple *t = (WkTuple*)wk_malloc(sizeof(WkTuple) + n*sizeof(WkVal));
    t->refcnt=1; t->len=(size_t)n;
    for (int i=0;i<n;i++) t->items[i]=items[i];
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_TUPLE; v.refcnt=1; v.as.tup=t;
    return v;
}
WkVal wk_make_func(const char *name, WkCFunc cfn,
                   WkCapture *caps, int ncaps,
                   const char **pnames, int nparams) {
    WkFunc *f = (WkFunc*)wk_malloc(sizeof(WkFunc));
    f->refcnt=1; f->name=name; f->cfn=cfn;
    f->ncaptures=ncaps; f->nparams=nparams; f->param_names=pnames;
    if (ncaps > 0) {
        f->captures = (WkCapture*)wk_malloc(ncaps*sizeof(WkCapture));
        memcpy(f->captures, caps, ncaps*sizeof(WkCapture));
    } else { f->captures=NULL; }
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_FUNC; v.refcnt=1; v.as.func=f;
    return v;
}
WkVal wk_make_obj(WkClass *cls) {
    WkObj *o = (WkObj*)wk_malloc(sizeof(WkObj));
    o->refcnt=1; o->cls=cls; o->mailbox=NULL;
    o->fields = (cls->nfields > 0)
        ? (WkVal*)wk_malloc(cls->nfields*sizeof(WkVal))
        : NULL;
    for (int i=0; i<cls->nfields; i++) o->fields[i] = wk_none();
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_OBJ; v.refcnt=1; v.as.obj=o;
    return v;
}
WkVal wk_make_some(WkVal inner) {
    WkVal *p = (WkVal*)wk_malloc(sizeof(WkVal)); *p=inner;
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_SOME; v.refcnt=1; v.as.inner=p;
    return v;
}
WkVal wk_make_ok(WkVal inner) {
    WkVal *p = (WkVal*)wk_malloc(sizeof(WkVal)); *p=inner;
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_OK; v.refcnt=1; v.as.inner=p;
    return v;
}
WkVal wk_make_err(WkVal inner) {
    WkVal *p = (WkVal*)wk_malloc(sizeof(WkVal)); *p=inner;
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_ERR; v.refcnt=1; v.as.inner=p;
    return v;
}
WkVal wk_make_class(WkClass *cls) {
    WkVal v; memset(&v,0,sizeof(v)); v.tag=WK_CLASS; v.refcnt=0; v.as.cls=cls;
    return v;
}

/* ═══════════════════════════════════════════════════════════════════════════
   STRING CONVERSION
   ═══════════════════════════════════════════════════════════════════════════ */

char *wk_to_cstr(WkVal v) {
    char buf[128];
    switch (v.tag) {
        case WK_NONE:  return strdup("none");
        case WK_BOOL:  return strdup(v.as.i ? "true" : "false");
        case WK_INT:   snprintf(buf,sizeof(buf),"%lld",(long long)v.as.i); return strdup(buf);
        case WK_FLOAT: {
            snprintf(buf,sizeof(buf),"%.15g",v.as.f);
            /* ensure there's a decimal point for floats */
            if (!strchr(buf,'.') && !strchr(buf,'e')) strncat(buf,".0",sizeof(buf)-strlen(buf)-1);
            return strdup(buf);
        }
        case WK_STR:   return strdup(v.as.str->data);
        case WK_SOME: {
            char *inner = wk_to_cstr(*v.as.inner);
            size_t len = strlen(inner)+8;
            char *out = (char*)wk_malloc(len);
            snprintf(out,len,"some(%s)",inner); free(inner); return out;
        }
        case WK_OK: {
            char *inner = wk_to_cstr(*v.as.inner);
            size_t len = strlen(inner)+6;
            char *out = (char*)wk_malloc(len);
            snprintf(out,len,"ok(%s)",inner); free(inner); return out;
        }
        case WK_ERR: {
            char *inner = wk_to_cstr(*v.as.inner);
            size_t len = strlen(inner)+7;
            char *out = (char*)wk_malloc(len);
            snprintf(out,len,"err(%s)",inner); free(inner); return out;
        }
        case WK_LIST: {
            char *out = strdup("["); size_t outlen=1, outcap=64;
            out = (char*)wk_realloc(out,outcap);
            for (size_t i=0; i<v.as.list->len; i++) {
                char *item = wk_to_repr(v.as.list->items[i]);
                size_t ilen=strlen(item);
                if (i>0) { if(outlen+2>outcap){outcap*=2;out=(char*)wk_realloc(out,outcap);} out[outlen++]=','; out[outlen++]=' '; }
                while(outlen+ilen+2>outcap){outcap*=2;out=(char*)wk_realloc(out,outcap);}
                memcpy(out+outlen,item,ilen); outlen+=ilen; free(item);
            }
            if(outlen+2>outcap){outcap+=4;out=(char*)wk_realloc(out,outcap);}
            out[outlen++]=']'; out[outlen]='\0'; return out;
        }
        case WK_MAP: {
            char *out = strdup("{"); size_t outlen=1, outcap=64;
            out = (char*)wk_realloc(out,outcap);
            int first=1;
            for (size_t i=0; i<v.as.map->cap; i++) {
                if (!v.as.map->buckets[i].used) continue;
                char *k=wk_to_repr(v.as.map->buckets[i].key);
                char *val=wk_to_repr(v.as.map->buckets[i].val);
                size_t needed=strlen(k)+strlen(val)+5;
                if(!first){needed+=2;}
                while(outlen+needed>outcap){outcap*=2;out=(char*)wk_realloc(out,outcap);}
                if(!first){out[outlen++]=',';out[outlen++]=' ';}
                first=0;
                size_t kl=strlen(k); memcpy(out+outlen,k,kl); outlen+=kl; free(k);
                out[outlen++]=':'; out[outlen++]=' ';
                size_t vl=strlen(val); memcpy(out+outlen,val,vl); outlen+=vl; free(val);
            }
            while(outlen+2>outcap){outcap+=4;out=(char*)wk_realloc(out,outcap);}
            out[outlen++]='}'; out[outlen]='\0'; return out;
        }
        case WK_TUPLE: {
            char *out = strdup("("); size_t outlen=1, outcap=64;
            out = (char*)wk_realloc(out,outcap);
            for (size_t i=0; i<v.as.tup->len; i++) {
                char *item = wk_to_repr(v.as.tup->items[i]);
                size_t ilen=strlen(item);
                if(i>0){if(outlen+2>outcap){outcap*=2;out=(char*)wk_realloc(out,outcap);}out[outlen++]=',';out[outlen++]=' ';}
                while(outlen+ilen+2>outcap){outcap*=2;out=(char*)wk_realloc(out,outcap);}
                memcpy(out+outlen,item,ilen); outlen+=ilen; free(item);
            }
            if(outlen+2>outcap){outcap+=4;out=(char*)wk_realloc(out,outcap);}
            out[outlen++]=')'; out[outlen]='\0'; return out;
        }
        case WK_RANGE: {
            snprintf(buf,sizeof(buf),"%lld%s%lld",
                (long long)v.as.rng.start, v.as.rng.inclusive?"..=":"...",
                (long long)v.as.rng.end);
            return strdup(buf);
        }
        case WK_OBJ:
            snprintf(buf,sizeof(buf),"<%s>", v.as.obj->cls->name);
            return strdup(buf);
        case WK_CLASS:
            snprintf(buf,sizeof(buf),"<class %s>", v.as.cls->name);
            return strdup(buf);
        case WK_FUNC:
            snprintf(buf,sizeof(buf),"<fn %s>", v.as.func->name ? v.as.func->name : "?");
            return strdup(buf);
        case WK_TENSOR: {
            WkTensor *t = v.as.tensor;
            char *out = strdup("tensor(["); size_t olen=strlen(out), ocap=128;
            out = (char*)wk_realloc(out, ocap);
            int64_t show = t->len < 10 ? t->len : 6;
            for (int64_t i = 0; i < show; i++) {
                char num[32]; snprintf(num,sizeof(num),"%.6g",t->data[i]);
                size_t nl=strlen(num);
                if(i>0){while(olen+2>ocap){ocap*=2;out=(char*)wk_realloc(out,ocap);}out[olen++]=',';out[olen++]=' ';}
                while(olen+nl+1>ocap){ocap*=2;out=(char*)wk_realloc(out,ocap);}
                memcpy(out+olen,num,nl); olen+=nl;
            }
            if (t->len > 10) {
                const char *dots = ", ...";
                while(olen+6>ocap){ocap*=2;out=(char*)wk_realloc(out,ocap);}
                memcpy(out+olen,dots,5); olen+=5;
            }
            const char *tail = "])";
            while(olen+3>ocap){ocap*=2;out=(char*)wk_realloc(out,ocap);}
            memcpy(out+olen,tail,3); olen+=2; out[olen]='\0';
            /* append shape info */
            char shp[128]; int sp=0; sp+=snprintf(shp+sp,sizeof(shp)-sp," shape=(");
            for(int d=0;d<t->ndim;d++){sp+=snprintf(shp+sp,sizeof(shp)-sp,"%s%lld",d?",":"",(long long)t->shape[d]);}
            sp+=snprintf(shp+sp,sizeof(shp)-sp,")");
            while(olen+(size_t)sp+1>ocap){ocap*=2;out=(char*)wk_realloc(out,ocap);}
            memcpy(out+olen,shp,sp+1); olen+=sp;
            return out;
        }
        default:
            return strdup("?");
    }
}

char *wk_to_repr(WkVal v) {
    if (v.tag == WK_STR) {
        /* quote the string */
        size_t len = v.as.str->len;
        char *out = (char*)wk_malloc(len*2+3);
        size_t p=0;
        out[p++]='"';
        for (size_t i=0; i<len; i++) {
            char c=v.as.str->data[i];
            if(c=='"'){out[p++]='\\';out[p++]='"';}
            else if(c=='\\'){out[p++]='\\';out[p++]='\\';}
            else if(c=='\n'){out[p++]='\\';out[p++]='n';}
            else if(c=='\r'){out[p++]='\\';out[p++]='r';}
            else if(c=='\t'){out[p++]='\\';out[p++]='t';}
            else out[p++]=c;
        }
        out[p++]='"'; out[p]='\0';
        return out;
    }
    return wk_to_cstr(v);
}

/* ═══════════════════════════════════════════════════════════════════════════
   TRUTHINESS & EQUALITY
   ═══════════════════════════════════════════════════════════════════════════ */

int wk_truthy(WkVal v) {
    switch (v.tag) {
        case WK_NONE:  return 0;
        case WK_BOOL:  return (int)v.as.i;
        case WK_INT:   return v.as.i != 0;
        case WK_FLOAT: return v.as.f != 0.0;
        case WK_STR:   return v.as.str->len > 0;
        case WK_LIST:  return v.as.list->len > 0;
        case WK_MAP:   return v.as.map->len > 0;
        case WK_TUPLE: return v.as.tup->len > 0;
        case WK_RANGE: return v.as.rng.start != v.as.rng.end;
        case WK_SOME:  return 1;
        case WK_OK:    return 1;
        case WK_ERR:   return 0;
        default:       return 1;
    }
}

int wk_equal(WkVal a, WkVal b) {
    /* numeric cross-type comparison */
    if (a.tag==WK_INT && b.tag==WK_FLOAT)   return (double)a.as.i == b.as.f;
    if (a.tag==WK_FLOAT && b.tag==WK_INT)   return a.as.f == (double)b.as.i;
    if (a.tag != b.tag) return 0;
    switch (a.tag) {
        case WK_NONE:  return 1;
        case WK_BOOL:
        case WK_INT:   return a.as.i == b.as.i;
        case WK_FLOAT: return a.as.f == b.as.f;
        case WK_STR:   return a.as.str->len == b.as.str->len &&
                              memcmp(a.as.str->data, b.as.str->data, a.as.str->len)==0;
        case WK_SOME: case WK_OK: case WK_ERR:
                       return wk_equal(*a.as.inner, *b.as.inner);
        default: return a.as.i == b.as.i; /* pointer equality for objects */
    }
}

WkVal wk_cmp_eq(WkVal a, WkVal b) {
    if (a.tag==WK_OBJ) { WkVal m=wk_obj_find_method(a,"operator=="); if(m.tag==WK_FUNC) return wk_call1(m,b); }
    return wk_bool(wk_equal(a,b));
}
WkVal wk_cmp_ne(WkVal a, WkVal b) {
    if (a.tag==WK_OBJ) { WkVal m=wk_obj_find_method(a,"operator!="); if(m.tag==WK_FUNC) return wk_call1(m,b); }
    return wk_bool(!wk_equal(a,b));
}

static double wk_to_num(WkVal v) {
    if (v.tag==WK_INT)   return (double)v.as.i;
    if (v.tag==WK_FLOAT) return v.as.f;
    if (v.tag==WK_BOOL)  return (double)v.as.i;
    wk_panic("expected number, got tag %d", v.tag);
}

WkVal wk_cmp_lt(WkVal a, WkVal b) {
    if (a.tag==WK_OBJ) { WkVal m=wk_obj_find_method(a,"operator<");  if(m.tag==WK_FUNC) return wk_call1(m,b); }
    if (a.tag==WK_STR && b.tag==WK_STR)
        return wk_bool(strcmp(a.as.str->data, b.as.str->data) < 0);
    return wk_bool(wk_to_num(a) < wk_to_num(b));
}
WkVal wk_cmp_le(WkVal a, WkVal b) {
    if (a.tag==WK_OBJ) { WkVal m=wk_obj_find_method(a,"operator<="); if(m.tag==WK_FUNC) return wk_call1(m,b); }
    if (a.tag==WK_STR && b.tag==WK_STR)
        return wk_bool(strcmp(a.as.str->data, b.as.str->data) <= 0);
    return wk_bool(wk_to_num(a) <= wk_to_num(b));
}
WkVal wk_cmp_gt(WkVal a, WkVal b) {
    if (a.tag==WK_OBJ) { WkVal m=wk_obj_find_method(a,"operator>");  if(m.tag==WK_FUNC) return wk_call1(m,b); }
    if (a.tag==WK_STR && b.tag==WK_STR)
        return wk_bool(strcmp(a.as.str->data, b.as.str->data) > 0);
    return wk_bool(wk_to_num(a) > wk_to_num(b));
}
WkVal wk_cmp_ge(WkVal a, WkVal b) {
    if (a.tag==WK_OBJ) { WkVal m=wk_obj_find_method(a,"operator>="); if(m.tag==WK_FUNC) return wk_call1(m,b); }
    if (a.tag==WK_STR && b.tag==WK_STR)
        return wk_bool(strcmp(a.as.str->data, b.as.str->data) >= 0);
    return wk_bool(wk_to_num(a) >= wk_to_num(b));
}

WkVal wk_in(WkVal needle, WkVal haystack) {
    switch (haystack.tag) {
        case WK_LIST:
            for (size_t i=0; i<haystack.as.list->len; i++)
                if (wk_equal(needle, haystack.as.list->items[i])) return wk_bool(1);
            return wk_bool(0);
        case WK_MAP: {
            WkMap *m = haystack.as.map;
            for (size_t i=0; i<m->cap; i++)
                if (m->buckets[i].used && wk_equal(needle, m->buckets[i].key))
                    return wk_bool(1);
            return wk_bool(0);
        }
        case WK_STR:
            if (needle.tag==WK_STR)
                return wk_bool(strstr(haystack.as.str->data, needle.as.str->data) != NULL);
            return wk_bool(0);
        case WK_RANGE: {
            WkRange r=haystack.as.rng;
            int64_t n;
            if (needle.tag==WK_INT) n=needle.as.i;
            else if (needle.tag==WK_FLOAT) n=(int64_t)needle.as.f;
            else return wk_bool(0);
            if (r.inclusive) return wk_bool(n>=r.start && n<=r.end);
            return wk_bool(n>=r.start && n<r.end);
        }
        default: return wk_bool(0);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
   ARITHMETIC
   ═══════════════════════════════════════════════════════════════════════════ */

/* Forward declarations for tensor ops */
static WkVal wk_tensor_binop(WkVal a, WkVal b, int op);

WkVal wk_add(WkVal a, WkVal b) {
    /* Operator overloading for objects (self is in captures, only pass 'other') */
    if (a.tag==WK_OBJ) { WkVal m=wk_obj_find_method(a,"operator+"); if(m.tag==WK_FUNC) return wk_call1(m,b); }
    /* Dual number arithmetic (autodiff) */
    if (_wk_is_dual(a) || _wk_is_dual(b)) {
        double av = _wk_is_dual(a)?_dual_val(a):_wk_val_to_double(a);
        double ad = _wk_is_dual(a)?_dual_der(a):0;
        double bv = _wk_is_dual(b)?_dual_val(b):_wk_val_to_double(b);
        double bd = _wk_is_dual(b)?_dual_der(b):0;
        return _wk_make_dual(av+bv, ad+bd);
    }
    /* Tensor arithmetic */
    if (a.tag==WK_TENSOR || b.tag==WK_TENSOR) return wk_tensor_binop(a,b,0);
    /* String concatenation */
    if (a.tag==WK_STR || b.tag==WK_STR) return wk_str_concat(a,b);
    /* List concatenation */
    if (a.tag==WK_LIST && b.tag==WK_LIST) {
        WkVal r = wk_make_list();
        for (size_t i=0;i<a.as.list->len;i++) wk_list_push_raw(r.as.list, a.as.list->items[i]);
        for (size_t i=0;i<b.as.list->len;i++) wk_list_push_raw(r.as.list, b.as.list->items[i]);
        return r;
    }
    if (a.tag==WK_INT && b.tag==WK_INT) return wk_int(a.as.i + b.as.i);
    if (a.tag==WK_FLOAT || b.tag==WK_FLOAT) return wk_float(wk_to_num(a)+wk_to_num(b));
    if (a.tag==WK_INT) return wk_int(a.as.i + (int64_t)wk_to_num(b));
    wk_panic("'+' not supported between %d and %d", a.tag, b.tag);
}
WkVal wk_sub(WkVal a, WkVal b) {
    if (a.tag==WK_OBJ) { WkVal m=wk_obj_find_method(a,"operator-"); if(m.tag==WK_FUNC) return wk_call1(m,b); }
    if (_wk_is_dual(a) || _wk_is_dual(b)) {
        double av=_wk_is_dual(a)?_dual_val(a):_wk_val_to_double(a), ad=_wk_is_dual(a)?_dual_der(a):0;
        double bv=_wk_is_dual(b)?_dual_val(b):_wk_val_to_double(b), bd=_wk_is_dual(b)?_dual_der(b):0;
        return _wk_make_dual(av-bv, ad-bd);
    }
    if (a.tag==WK_TENSOR || b.tag==WK_TENSOR) return wk_tensor_binop(a,b,1);
    if (a.tag==WK_INT && b.tag==WK_INT) return wk_int(a.as.i - b.as.i);
    return wk_float(wk_to_num(a)-wk_to_num(b));
}
WkVal wk_mul(WkVal a, WkVal b) {
    if (a.tag==WK_OBJ) { WkVal m=wk_obj_find_method(a,"operator*"); if(m.tag==WK_FUNC) return wk_call1(m,b); }
    if (_wk_is_dual(a) || _wk_is_dual(b)) {
        double av=_wk_is_dual(a)?_dual_val(a):_wk_val_to_double(a), ad=_wk_is_dual(a)?_dual_der(a):0;
        double bv=_wk_is_dual(b)?_dual_val(b):_wk_val_to_double(b), bd=_wk_is_dual(b)?_dual_der(b):0;
        return _wk_make_dual(av*bv, av*bd+ad*bv);
    }
    if (a.tag==WK_TENSOR || b.tag==WK_TENSOR) return wk_tensor_binop(a,b,2);
    /* string * int */
    if (a.tag==WK_STR && b.tag==WK_INT) {
        size_t slen=a.as.str->len; int64_t n=b.as.i<0?0:b.as.i;
        char *buf=(char*)wk_malloc(slen*n+1); size_t p=0;
        for (int64_t i=0;i<n;i++){memcpy(buf+p,a.as.str->data,slen);p+=slen;}
        buf[p]='\0'; WkVal r=wk_make_str(buf,p); free(buf); return r;
    }
    if (a.tag==WK_INT && b.tag==WK_INT) return wk_int(a.as.i * b.as.i);
    return wk_float(wk_to_num(a)*wk_to_num(b));
}
WkVal wk_div(WkVal a, WkVal b) {
    if (a.tag==WK_OBJ) { WkVal m=wk_obj_find_method(a,"operator/"); if(m.tag==WK_FUNC) return wk_call1(m,b); }
    if (_wk_is_dual(a) || _wk_is_dual(b)) {
        double av=_wk_is_dual(a)?_dual_val(a):_wk_val_to_double(a), ad=_wk_is_dual(a)?_dual_der(a):0;
        double bv=_wk_is_dual(b)?_dual_val(b):_wk_val_to_double(b), bd=_wk_is_dual(b)?_dual_der(b):0;
        return _wk_make_dual(av/bv, (ad*bv-av*bd)/(bv*bv));
    }
    if (a.tag==WK_TENSOR || b.tag==WK_TENSOR) return wk_tensor_binop(a,b,3);
    double bv=wk_to_num(b);
    if (bv==0.0) wk_panic("division by zero");
    if (a.tag==WK_INT && b.tag==WK_INT) return wk_int(a.as.i / b.as.i);
    return wk_float(wk_to_num(a)/bv);
}
WkVal wk_mod(WkVal a, WkVal b) {
    if (a.tag==WK_OBJ) { WkVal m=wk_obj_find_method(a,"operator%"); if(m.tag==WK_FUNC) return wk_call1(m,b); }
    if (a.tag==WK_INT && b.tag==WK_INT) {
        if (b.as.i==0) wk_panic("modulo by zero");
        return wk_int(a.as.i % b.as.i);
    }
    return wk_float(fmod(wk_to_num(a),wk_to_num(b)));
}
WkVal wk_pow(WkVal a, WkVal b) {
    return wk_float(pow(wk_to_num(a),wk_to_num(b)));
}
WkVal wk_neg(WkVal a) {
    if (a.tag==WK_INT)   return wk_int(-a.as.i);
    if (a.tag==WK_FLOAT) return wk_float(-a.as.f);
    wk_panic("unary '-' on non-number");
}
WkVal wk_not(WkVal a) { return wk_bool(!wk_truthy(a)); }
WkVal wk_bitnot(WkVal a) {
    if (a.tag==WK_INT) return wk_int(~a.as.i);
    wk_panic("bitwise '~' on non-int");
}
WkVal wk_band(WkVal a,WkVal b){ if(a.tag==WK_INT&&b.tag==WK_INT)return wk_int(a.as.i&b.as.i); wk_panic("'&' on non-int"); }
WkVal wk_bor(WkVal a,WkVal b) { if(a.tag==WK_INT&&b.tag==WK_INT)return wk_int(a.as.i|b.as.i); wk_panic("'|' on non-int"); }
WkVal wk_bxor(WkVal a,WkVal b){ if(a.tag==WK_INT&&b.tag==WK_INT)return wk_int(a.as.i^b.as.i); wk_panic("'^' on non-int"); }
WkVal wk_lshift(WkVal a,WkVal b){ if(a.tag==WK_INT&&b.tag==WK_INT)return wk_int(a.as.i<<b.as.i); wk_panic("'<<' on non-int"); }
WkVal wk_rshift(WkVal a,WkVal b){ if(a.tag==WK_INT&&b.tag==WK_INT)return wk_int(a.as.i>>b.as.i); wk_panic("'>>' on non-int"); }

/* ═══════════════════════════════════════════════════════════════════════════
   CASTS
   ═══════════════════════════════════════════════════════════════════════════ */

WkVal wk_cast_int(WkVal v) {
    switch(v.tag){
        case WK_INT:   return v;
        case WK_FLOAT: return wk_int((int64_t)v.as.f);
        case WK_BOOL:  return wk_int(v.as.i);
        case WK_STR: { char *e; int64_t n=strtoll(v.as.str->data,&e,10);
                       if(*e) wk_panic("cannot convert '%s' to int",v.as.str->data);
                       return wk_int(n); }
        default: wk_panic("cannot cast to int");
    }
}
WkVal wk_cast_float(WkVal v) {
    switch(v.tag){
        case WK_FLOAT: return v;
        case WK_INT:   return wk_float((double)v.as.i);
        case WK_BOOL:  return wk_float((double)v.as.i);
        case WK_STR: { char *e; double f=strtod(v.as.str->data,&e);
                       if(*e) wk_panic("cannot convert '%s' to float",v.as.str->data);
                       return wk_float(f); }
        default: wk_panic("cannot cast to float");
    }
}
WkVal wk_cast_bool(WkVal v) { return wk_bool(wk_truthy(v)); }
WkVal wk_cast_str(WkVal v)  { char *s=wk_to_cstr(v); WkVal r=wk_make_strz(s); free(s); return r; }
WkVal wk_cast_byte(WkVal v) { return wk_cast_int(v); }

/* ═══════════════════════════════════════════════════════════════════════════
   STRING OPERATIONS
   ═══════════════════════════════════════════════════════════════════════════ */

WkVal wk_str_concat(WkVal a, WkVal b) {
    char *sa=wk_to_cstr(a), *sb=wk_to_cstr(b);
    size_t la=strlen(sa), lb=strlen(sb);
    char *buf=(char*)wk_malloc(la+lb+1);
    memcpy(buf,sa,la); memcpy(buf+la,sb,lb); buf[la+lb]='\0';
    WkVal r=wk_make_str(buf,la+lb); free(buf); free(sa); free(sb);
    return r;
}

WkVal wk_str_index(WkVal s, WkVal idx) {
    if (s.tag!=WK_STR) wk_panic("str index on non-str");
    int64_t i=idx.as.i;
    if (i<0) i+=s.as.str->len;
    if (i<0||(size_t)i>=s.as.str->len) wk_panic("string index out of range");
    return wk_make_str(s.as.str->data+i, 1);
}
WkVal wk_str_slice(WkVal s, WkVal start, WkVal end) {
    if (s.tag!=WK_STR) wk_panic("str slice on non-str");
    int64_t len=(int64_t)s.as.str->len;
    int64_t a= (start.tag==WK_NONE)?0:start.as.i;
    int64_t b= (end.tag==WK_NONE)?len:end.as.i;
    if(a<0)a+=len; if(b<0)b+=len;
    if(a<0)a=0; if(b>len)b=len;
    if(a>=b) return wk_make_strz("");
    return wk_make_str(s.as.str->data+a, (size_t)(b-a));
}

WkVal wk_list_slice(WkVal lst, WkVal start, WkVal end, WkVal step) {
    if (lst.tag != WK_LIST) wk_panic("list slice on non-list");
    int64_t len = (int64_t)lst.as.list->len;
    int64_t st = (step.tag == WK_NONE) ? 1 : step.as.i;
    if (st == 0) wk_panic("slice step cannot be zero");
    int64_t a, b;
    if (st > 0) {
        a = (start.tag == WK_NONE) ? 0   : start.as.i;
        b = (end.tag   == WK_NONE) ? len  : end.as.i;
        if (a < 0) a += len; if (b < 0) b += len;
        if (a < 0) a = 0; if (b > len) b = len;
    } else {
        a = (start.tag == WK_NONE) ? len - 1 : start.as.i;
        b = (end.tag   == WK_NONE) ? -len - 1 : end.as.i;
        if (a < 0) a += len; if (b < 0) b += len;
        if (a >= len) a = len - 1;
    }
    WkVal r = wk_make_list();
    if (st > 0) {
        for (int64_t i = a; i < b; i += st)
            wk_list_push_raw(r.as.list, lst.as.list->items[i]);
    } else {
        for (int64_t i = a; i > b; i += st)
            if (i >= 0 && i < len)
                wk_list_push_raw(r.as.list, lst.as.list->items[i]);
    }
    return r;
}

WkVal wk_str_fmtbuild(int nparts, ...) {
    va_list ap; va_start(ap, nparts);
    /* nparts pairs: (char* | NULL, WkVal if NULL) */
    size_t total=0;
    char **parts=(char**)wk_malloc(nparts*sizeof(char*));
    int *owned=(int*)wk_malloc(nparts*sizeof(int));
    /* First pass: collect parts, measure total length */
    /* Format: nparts = number of items; items alternate literal-str and WkVal
       represented as: pass NULL for "next arg is WkVal", char* for literal */
    /* Actually: va args are (char* literal, WkVal expr) interleaved.
       literal can be NULL if no literal before an expr.
       The caller passes: nparts literal+expr pairs, each pair is (char*, WkVal).
       If char* is NULL, no literal for that slot. */
    /* Re-design: nparts = total argument count (char* or WkVal interleaved)
       We pass: char* pieces and WkVal pieces tagged by a sentinel.
       Simplest: just collect all as strings. */
    /* Implementation: caller passes nparts (char*, WkVal) pairs */
    for (int i=0; i<nparts; i++) {
        const char *lit = va_arg(ap, const char*);
        WkVal expr = va_arg(ap, WkVal);
        char *litpart = lit ? strdup(lit) : strdup("");
        char *exprpart = (expr.tag == WK_NONE) ? strdup("") : wk_to_cstr(expr);
        size_t litlen=strlen(litpart), exprlen=strlen(exprpart);
        char *combined=(char*)wk_malloc(litlen+exprlen+1);
        memcpy(combined,litpart,litlen);
        memcpy(combined+litlen,exprpart,exprlen);
        combined[litlen+exprlen]='\0';
        parts[i]=combined; owned[i]=1; total+=litlen+exprlen;
        free(litpart); free(exprpart);
    }
    va_end(ap);
    char *result=(char*)wk_malloc(total+1); size_t p=0;
    for (int i=0;i<nparts;i++){
        size_t l=strlen(parts[i]);
        memcpy(result+p,parts[i],l); p+=l;
        if(owned[i]) free(parts[i]);
    }
    result[p]='\0';
    WkVal r=wk_make_str(result,p); free(result); free(parts); free(owned);
    return r;
}

/* str_fmtbuild_tail — appends a trailing literal after all expr pairs */
WkVal wk_str_fmtbuild_tail(int nparts, const char *tail, ...) {
    /* not used directly; fmtbuild handles everything */
    (void)nparts; (void)tail; return wk_none();
}

static WkVal wk_str_getmethod(WkVal s, const char *name);

WkVal wk_str_method(WkVal s, const char *name) {
    return wk_str_getmethod(s, name);
}

/* ═══════════════════════════════════════════════════════════════════════════
   LIST OPERATIONS
   ═══════════════════════════════════════════════════════════════════════════ */

WkVal wk_list_get(WkVal lst, WkVal idx) {
    if (lst.tag!=WK_LIST) wk_panic("index on non-list");
    int64_t i=idx.as.i;
    if(i<0)i+=(int64_t)lst.as.list->len;
    if(i<0||(size_t)i>=lst.as.list->len) wk_panic("list index %lld out of range",(long long)i);
    return lst.as.list->items[i];
}
void wk_list_set(WkVal lst, WkVal idx, WkVal v) {
    if(lst.tag!=WK_LIST) wk_panic("index-set on non-list");
    int64_t i=idx.as.i;
    if(i<0)i+=(int64_t)lst.as.list->len;
    if(i<0||(size_t)i>=lst.as.list->len) wk_panic("list index out of range");
    lst.as.list->items[i]=v;
}

static WkVal wk_list_getmethod(WkVal lst, const char *name);
WkVal wk_list_method(WkVal lst, const char *name) { return wk_list_getmethod(lst, name); }

/* ═══════════════════════════════════════════════════════════════════════════
   MAP OPERATIONS
   ═══════════════════════════════════════════════════════════════════════════ */

static size_t wk_hash_val(WkVal v) {
    switch(v.tag){
        case WK_NONE:  return 0;
        case WK_BOOL:
        case WK_INT:   return (size_t)v.as.i;
        case WK_FLOAT: { uint64_t u; memcpy(&u,&v.as.f,8); return (size_t)u; }
        case WK_STR: { size_t h=5381; for(size_t i=0;i<v.as.str->len;i++) h=((h<<5)+h)+v.as.str->data[i]; return h; }
        default: return (size_t)(uintptr_t)v.as.str; /* pointer as hash for others */
    }
}

static void wk_map_grow(WkMap *m) {
    size_t newcap = m->cap * 2;
    WkMapEntry *nb = (WkMapEntry*)wk_malloc(newcap*sizeof(WkMapEntry));
    memset(nb,0,newcap*sizeof(WkMapEntry));
    for (size_t i=0;i<m->cap;i++) {
        if(!m->buckets[i].used) continue;
        size_t h=wk_hash_val(m->buckets[i].key)&(newcap-1);
        while(nb[h].used) h=(h+1)&(newcap-1);
        nb[h]=m->buckets[i];
    }
    free(m->buckets);
    m->buckets=nb; m->cap=newcap;
}

WkVal wk_map_get_key(WkVal mv, WkVal key) {
    if(mv.tag!=WK_MAP) wk_panic("map get on non-map");
    WkMap *m=mv.as.map;
    size_t h=wk_hash_val(key)&(m->cap-1);
    for(size_t i=0;i<m->cap;i++){
        size_t idx=(h+i)&(m->cap-1);
        if(!m->buckets[idx].used) return wk_none();
        if(wk_equal(m->buckets[idx].key,key)) return m->buckets[idx].val;
    }
    return wk_none();
}
void wk_map_set_key(WkVal mv, WkVal key, WkVal val) {
    if(mv.tag!=WK_MAP) wk_panic("map set on non-map");
    WkMap *m=mv.as.map;
    if(m->len*2 >= m->cap) wk_map_grow(m);
    size_t h=wk_hash_val(key)&(m->cap-1);
    for(size_t i=0;i<m->cap;i++){
        size_t idx=(h+i)&(m->cap-1);
        if(!m->buckets[idx].used){
            m->buckets[idx].key=key; m->buckets[idx].val=val;
            m->buckets[idx].used=1; m->len++; return;
        }
        if(wk_equal(m->buckets[idx].key,key)){
            m->buckets[idx].val=val; return;
        }
    }
}
static WkVal wk_map_getmethod(WkVal mv, const char *name);
WkVal wk_map_method(WkVal mv, const char *name) { return wk_map_getmethod(mv, name); }

/* ═══════════════════════════════════════════════════════════════════════════
   GENERIC MEMBER / INDEX ACCESS
   ═══════════════════════════════════════════════════════════════════════════ */

WkVal wk_index_get(WkVal obj, WkVal idx) {
    switch(obj.tag){
        case WK_LIST:  return wk_list_get(obj,idx);
        case WK_STR:   return wk_str_index(obj,idx);
        case WK_MAP:   return wk_map_get_key(obj,idx);
        case WK_TUPLE:
            if(idx.tag!=WK_INT) wk_panic("tuple index must be int");
            { int64_t i=idx.as.i; if(i<0)i+=(int64_t)obj.as.tup->len;
              if(i<0||(size_t)i>=obj.as.tup->len) wk_panic("tuple index out of range");
              return obj.as.tup->items[i]; }
        default: wk_panic("cannot index type %d", obj.tag);
    }
}
void wk_index_set(WkVal obj, WkVal idx, WkVal val) {
    switch(obj.tag){
        case WK_LIST: wk_list_set(obj,idx,val); return;
        case WK_MAP:  wk_map_set_key(obj,idx,val); return;
        default: wk_panic("cannot index-set type %d", obj.tag);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
   OBJECT / CLASS MEMBER ACCESS
   ═══════════════════════════════════════════════════════════════════════════ */

WkVal wk_obj_get_field(WkVal obj, const char *name) {
    if(obj.tag!=WK_OBJ) wk_panic("field access on non-object");
    WkObj *o=obj.as.obj;
    for(int i=0;i<o->cls->nfields;i++)
        if(strcmp(o->cls->field_names[i],name)==0) return o->fields[i];
    wk_panic("object of class '%s' has no field '%s'", o->cls->name, name);
}
void wk_obj_set_field(WkVal obj, const char *name, WkVal v) {
    if(obj.tag!=WK_OBJ) wk_panic("field set on non-object");
    WkObj *o=obj.as.obj;
    for(int i=0;i<o->cls->nfields;i++)
        if(strcmp(o->cls->field_names[i],name)==0){ o->fields[i]=v; return; }
    wk_panic("object of class '%s' has no field '%s'", o->cls->name, name);
}
WkVal wk_obj_find_method(WkVal obj, const char *name) {
    if(obj.tag!=WK_OBJ && obj.tag!=WK_CLASS) return wk_none();
    WkClass *cls = (obj.tag==WK_OBJ) ? obj.as.obj->cls : obj.as.cls;
    WkClass *cur=cls;
    while(cur){
        for(int i=0;i<cur->nmethods;i++)
            if(strcmp(cur->methods[i].name,name)==0){
                WkFunc *mfn=cur->methods[i].fn;
                /* Build a bound-method WkFunc with self as capture[0] */
                WkCapture caps[1];
                caps[0].name="self"; caps[0].val=obj;
                return wk_make_func(mfn->name, mfn->cfn, caps, 1,
                                    mfn->param_names, mfn->nparams);
            }
        cur=cur->parent;
    }
    return wk_none();
}

WkVal wk_member_get(WkVal obj, const char *name) {
    switch(obj.tag){
        case WK_STR:   return wk_str_getmethod(obj, name);
        case WK_LIST:  return wk_list_getmethod(obj, name);
        case WK_MAP:   return wk_map_getmethod(obj, name);
        case WK_OBJ: {
            WkObj *o=obj.as.obj;
            /* Pipeline objects get special method dispatch */
            if (o->cls == &_wk_cls_pipeline) {
                WkCapture *cap = (WkCapture*)wk_malloc(sizeof(WkCapture));
                cap->name = "self"; cap->val = obj;
                if(strcmp(name,"map")==0) return wk_make_func("map",_wkp_map,cap,1,NULL,-1);
                if(strcmp(name,"filter")==0) return wk_make_func("filter",_wkp_filter,cap,1,NULL,-1);
                if(strcmp(name,"batch")==0) return wk_make_func("batch",_wkp_batch,cap,1,NULL,-1);
                if(strcmp(name,"flatten")==0) return wk_make_func("flatten",_wkp_flatten,cap,1,NULL,-1);
                if(strcmp(name,"take")==0) return wk_make_func("take",_wkp_take,cap,1,NULL,-1);
                if(strcmp(name,"skip")==0) return wk_make_func("skip",_wkp_skip,cap,1,NULL,-1);
                if(strcmp(name,"shuffle")==0) return wk_make_func("shuffle",_wkp_shuffle,cap,1,NULL,-1);
                if(strcmp(name,"zip")==0) return wk_make_func("zip",_wkp_zip,cap,1,NULL,-1);
                if(strcmp(name,"collect")==0) return wk_make_func("collect",_wkp_collect,cap,1,NULL,-1);
                if(strcmp(name,"reduce")==0) return wk_make_func("reduce",_wkp_reduce,cap,1,NULL,-1);
                if(strcmp(name,"forEach")==0) return wk_make_func("forEach",_wkp_forEach,cap,1,NULL,-1);
                if(strcmp(name,"count")==0) return wk_make_func("count",_wkp_count,cap,1,NULL,-1);
                free(cap);
            }
            /* Dual number field access */
            if (o->cls == &_wk_cls_dual) {
                if(strcmp(name,"value")==0) return o->fields[0];
                if(strcmp(name,"deriv")==0) return o->fields[1];
            }
            /* Try field first, then method */
            for(int i=0;i<o->cls->nfields;i++)
                if(strcmp(o->cls->field_names[i],name)==0) return o->fields[i];
            WkVal m=wk_obj_find_method(obj, name);
            if(m.tag!=WK_NONE) return m;
            wk_panic("object of class '%s' has no member '%s'", o->cls->name, name);
        }
        case WK_CLASS: {
            WkVal m=wk_obj_find_method(obj, name);
            if(m.tag!=WK_NONE) return m;
            wk_panic("class '%s' has no member '%s'", obj.as.cls->name, name);
        }
        case WK_SOME: if(strcmp(name,"value")==0) return *obj.as.inner; break;
        case WK_OK:   if(strcmp(name,"value")==0) return *obj.as.inner; break;
        case WK_ERR:  if(strcmp(name,"value")==0) return *obj.as.inner; break;
        case WK_TENSOR: return wk_tensor_getmethod(obj, name);
        default: break;
    }
    wk_panic("type %d has no member '%s'", obj.tag, name);
}
void wk_member_set(WkVal obj, const char *name, WkVal v) {
    if(obj.tag==WK_OBJ){ wk_obj_set_field(obj,name,v); return; }
    if(obj.tag==WK_MAP){ wk_map_set_key(obj, wk_make_strz(name), v); return; }
    wk_panic("cannot set member '%s' on type %d", name, obj.tag);
}

/* ═══════════════════════════════════════════════════════════════════════════
   FUNCTION CALL
   ═══════════════════════════════════════════════════════════════════════════ */

WkVal wk_call(WkVal fn, WkVal *args, int argc) {
    if(fn.tag!=WK_FUNC) wk_panic("call on non-function (tag=%d)", fn.tag);
    return fn.as.func->cfn(args, argc, fn.as.func);
}
WkVal wk_call0(WkVal fn) { return wk_call(fn, NULL, 0); }
WkVal wk_call1(WkVal fn, WkVal a0) { return wk_call(fn, &a0, 1); }
WkVal wk_call2(WkVal fn, WkVal a0, WkVal a1) {
    WkVal args[2]={a0,a1}; return wk_call(fn, args, 2);
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEFER
   ═══════════════════════════════════════════════════════════════════════════ */

static WkDeferFrame *_defer_top = NULL;

void wk_defer_push_frame(WkDeferFrame *f) {
    f->count=0; f->prev=_defer_top; _defer_top=f;
}
void wk_defer_pop_frame(WkDeferFrame *f) {
    _defer_top = f->prev;
}
void wk_defer_register(WkDeferFrame *f, WkVal fn) {
    if(f->count>=WK_DEFER_MAX) wk_panic("too many defers in one scope");
    f->entries[f->count++].fn = fn;
}
void wk_defer_flush(WkDeferFrame *f) {
    for(int i=f->count-1; i>=0; i--) {
        if(f->entries[i].fn.tag==WK_FUNC)
            wk_call0(f->entries[i].fn);
    }
    f->count=0;
    _defer_top = f->prev;
}

/* ═══════════════════════════════════════════════════════════════════════════
   STRING METHODS (returned as bound WkFunc closures)
   ═══════════════════════════════════════════════════════════════════════════ */

#define STR_METHOD_BODY(name, impl) \
static WkVal _wksm_##name(WkVal *args, int argc, WkFunc *fn) { \
    WkVal self = fn->captures[0].val; (void)args; (void)argc; \
    impl \
} \

STR_METHOD_BODY(len,   return wk_int((int64_t)self.as.str->len); )
STR_METHOD_BODY(upper, {
    char *s=strdup(self.as.str->data);
    for(size_t i=0;s[i];i++) s[i]=(char)toupper((unsigned char)s[i]);
    WkVal r=wk_make_strz(s); free(s); return r;
})
STR_METHOD_BODY(lower, {
    char *s=strdup(self.as.str->data);
    for(size_t i=0;s[i];i++) s[i]=(char)tolower((unsigned char)s[i]);
    WkVal r=wk_make_strz(s); free(s); return r;
})
STR_METHOD_BODY(trim, {
    const char *s=self.as.str->data; size_t len=self.as.str->len;
    size_t start=0; while(start<len && isspace((unsigned char)s[start])) start++;
    size_t end=len; while(end>start && isspace((unsigned char)s[end-1])) end--;
    return wk_make_str(s+start, end-start);
})
STR_METHOD_BODY(isEmpty, return wk_bool(self.as.str->len==0); )
STR_METHOD_BODY(toInt,   return wk_cast_int(self); )
STR_METHOD_BODY(toFloat, return wk_cast_float(self); )

static WkVal _wksm_split(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    const char *sep = (argc>0 && args[0].tag==WK_STR) ? args[0].as.str->data : " ";
    size_t seplen=strlen(sep);
    WkVal result=wk_make_list();
    const char *s=self.as.str->data;
    const char *p; const char *cur=s;
    while((p=strstr(cur,sep))!=NULL){
        wk_list_push_raw(result.as.list, wk_make_str(cur,(size_t)(p-cur)));
        cur=p+seplen;
    }
    wk_list_push_raw(result.as.list, wk_make_strz(cur));
    return result;
}
static WkVal _wksm_contains(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1||args[0].tag!=WK_STR) return wk_bool(0);
    return wk_bool(strstr(self.as.str->data, args[0].as.str->data)!=NULL);
}
static WkVal _wksm_startsWith(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1||args[0].tag!=WK_STR) return wk_bool(0);
    size_t l=args[0].as.str->len;
    return wk_bool(self.as.str->len>=l && memcmp(self.as.str->data,args[0].as.str->data,l)==0);
}
static WkVal _wksm_endsWith(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1||args[0].tag!=WK_STR) return wk_bool(0);
    size_t sl=self.as.str->len, ll=args[0].as.str->len;
    if(sl<ll) return wk_bool(0);
    return wk_bool(memcmp(self.as.str->data+sl-ll, args[0].as.str->data, ll)==0);
}
static WkVal _wksm_replace(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<2||args[0].tag!=WK_STR||args[1].tag!=WK_STR) return self;
    const char *s=self.as.str->data;
    const char *old=args[0].as.str->data; size_t oldlen=strlen(old);
    const char *new_=args[1].as.str->data; size_t newlen=strlen(new_);
    char *out=(char*)wk_malloc(self.as.str->len*4+1); size_t p=0;
    while(*s){
        if(strncmp(s,old,oldlen)==0){ memcpy(out+p,new_,newlen); p+=newlen; s+=oldlen; }
        else out[p++]=*s++;
    }
    out[p]='\0';
    WkVal r=wk_make_str(out,p); free(out); return r;
}
static WkVal _wksm_indexOf(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1||args[0].tag!=WK_STR) return wk_int(-1);
    const char *p=strstr(self.as.str->data, args[0].as.str->data);
    return p ? wk_int((int64_t)(p-self.as.str->data)) : wk_int(-1);
}
static WkVal _wksm_repeat(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    int64_t n=(argc>0)?args[0].as.i:0;
    return wk_mul(self, wk_int(n));
}
static WkVal _wksm_lines(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val; (void)args;(void)argc;
    WkVal result=wk_make_list();
    const char *s=self.as.str->data;
    const char *cur=s;
    while(*cur){
        const char *nl=strchr(cur,'\n');
        if(!nl){ wk_list_push_raw(result.as.list,wk_make_strz(cur)); break; }
        size_t len=(size_t)(nl-cur);
        if(len>0&&cur[len-1]=='\r') len--;
        wk_list_push_raw(result.as.list, wk_make_str(cur,len));
        cur=nl+1;
    }
    return result;
}
static WkVal _wksm_chars(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val; (void)args;(void)argc;
    WkVal result=wk_make_list();
    for(size_t i=0;i<self.as.str->len;i++)
        wk_list_push_raw(result.as.list, wk_make_str(self.as.str->data+i,1));
    return result;
}
static WkVal _wksm_slice_str(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    WkVal start=(argc>0)?args[0]:wk_none();
    WkVal end=(argc>1)?args[1]:wk_none();
    return wk_str_slice(self,start,end);
}
static WkVal _wksm_bytes(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val; (void)args;(void)argc;
    WkVal result=wk_make_list();
    for(size_t i=0;i<self.as.str->len;i++)
        wk_list_push_raw(result.as.list, wk_int((unsigned char)self.as.str->data[i]));
    return result;
}

#define MAKE_STR_METHOD(methname, fnptr) \
    if(strcmp(name,methname)==0){ \
        WkCapture cap; cap.name="self"; cap.val=s; \
        return wk_make_func(methname, fnptr, &cap, 1, NULL, 0); \
    }

static WkVal wk_str_getmethod(WkVal s, const char *name) {
    MAKE_STR_METHOD("len",        _wksm_len)
    MAKE_STR_METHOD("upper",      _wksm_upper)
    MAKE_STR_METHOD("lower",      _wksm_lower)
    MAKE_STR_METHOD("trim",       _wksm_trim)
    MAKE_STR_METHOD("isEmpty",    _wksm_isEmpty)
    MAKE_STR_METHOD("toInt",      _wksm_toInt)
    MAKE_STR_METHOD("toFloat",    _wksm_toFloat)
    MAKE_STR_METHOD("split",      _wksm_split)
    MAKE_STR_METHOD("contains",   _wksm_contains)
    MAKE_STR_METHOD("startsWith", _wksm_startsWith)
    MAKE_STR_METHOD("endsWith",   _wksm_endsWith)
    MAKE_STR_METHOD("replace",    _wksm_replace)
    MAKE_STR_METHOD("indexOf",    _wksm_indexOf)
    MAKE_STR_METHOD("repeat",     _wksm_repeat)
    MAKE_STR_METHOD("lines",      _wksm_lines)
    MAKE_STR_METHOD("chars",      _wksm_chars)
    MAKE_STR_METHOD("slice",      _wksm_slice_str)
    MAKE_STR_METHOD("bytes",      _wksm_bytes)
    wk_panic("str has no method '%s'", name);
}

/* ═══════════════════════════════════════════════════════════════════════════
   LIST METHODS
   ═══════════════════════════════════════════════════════════════════════════ */

#define LIST_METHOD_BODY(name, impl) \
static WkVal _wklm_##name(WkVal *args, int argc, WkFunc *fn) { \
    WkVal self=fn->captures[0].val; \
    impl \
}

LIST_METHOD_BODY(len,     (void)args;(void)argc; return wk_int((int64_t)self.as.list->len); )
LIST_METHOD_BODY(isEmpty, (void)args;(void)argc; return wk_bool(self.as.list->len==0); )
LIST_METHOD_BODY(first,   (void)args;(void)argc;
    if(self.as.list->len==0) return wk_none();
    return self.as.list->items[0]; )
LIST_METHOD_BODY(last,    (void)args;(void)argc;
    if(self.as.list->len==0) return wk_none();
    return self.as.list->items[self.as.list->len-1]; )
LIST_METHOD_BODY(pop,     (void)args;(void)argc;
    if(self.as.list->len==0) wk_panic("pop on empty list");
    return self.as.list->items[--self.as.list->len]; )
LIST_METHOD_BODY(clear,   (void)args;(void)argc;
    self.as.list->len=0; return wk_none(); )
LIST_METHOD_BODY(reverse, (void)args;(void)argc;{
    WkList *l=self.as.list; size_t n=l->len;
    for(size_t i=0;i<n/2;i++){WkVal t=l->items[i];l->items[i]=l->items[n-1-i];l->items[n-1-i]=t;}
    return wk_none(); })
LIST_METHOD_BODY(sum, (void)args;(void)argc;{
    WkVal acc=wk_int(0);
    for(size_t i=0;i<self.as.list->len;i++) acc=wk_add(acc,self.as.list->items[i]);
    return acc; })
LIST_METHOD_BODY(min, (void)args;(void)argc;{
    if(self.as.list->len==0) return wk_none();
    WkVal m=self.as.list->items[0];
    for(size_t i=1;i<self.as.list->len;i++)
        if(wk_truthy(wk_cmp_lt(self.as.list->items[i],m))) m=self.as.list->items[i];
    return m; })
LIST_METHOD_BODY(max, (void)args;(void)argc;{
    if(self.as.list->len==0) return wk_none();
    WkVal m=self.as.list->items[0];
    for(size_t i=1;i<self.as.list->len;i++)
        if(wk_truthy(wk_cmp_gt(self.as.list->items[i],m))) m=self.as.list->items[i];
    return m; })

static WkVal _wklm_push(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc>0) wk_list_push_raw(self.as.list, args[0]);
    return wk_none();
}
static WkVal _wklm_append(WkVal *args, int argc, WkFunc *fn) {
    return _wklm_push(args,argc,fn);
}
static WkVal _wklm_insert(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<2) return wk_none();
    int64_t idx=args[0].as.i; WkVal val=args[1];
    WkList *l=self.as.list;
    if(idx<0) idx+=(int64_t)l->len;
    if(idx<0) idx=0; if((size_t)idx>l->len) idx=(int64_t)l->len;
    wk_list_push_raw(l, wk_none()); /* grow */
    for(int64_t i=(int64_t)l->len-1;i>idx;i--) l->items[i]=l->items[i-1];
    l->items[idx]=val;
    return wk_none();
}
static WkVal _wklm_remove(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1) return wk_none();
    WkList *l=self.as.list;
    for(size_t i=0;i<l->len;i++)
        if(wk_equal(l->items[i],args[0])){
            for(size_t j=i;j<l->len-1;j++) l->items[j]=l->items[j+1];
            l->len--; return wk_none();
        }
    return wk_none();
}
static WkVal _wklm_contains(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1) return wk_bool(0);
    for(size_t i=0;i<self.as.list->len;i++)
        if(wk_equal(self.as.list->items[i],args[0])) return wk_bool(1);
    return wk_bool(0);
}
static WkVal _wklm_indexOf(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1) return wk_int(-1);
    for(size_t i=0;i<self.as.list->len;i++)
        if(wk_equal(self.as.list->items[i],args[0])) return wk_int((int64_t)i);
    return wk_int(-1);
}
static WkVal _wklm_join(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    const char *sep = (argc>0&&args[0].tag==WK_STR) ? args[0].as.str->data : "";
    size_t seplen=strlen(sep);
    size_t total=0; size_t n=self.as.list->len;
    char **parts=(char**)wk_malloc(n*sizeof(char*));
    for(size_t i=0;i<n;i++){parts[i]=wk_to_cstr(self.as.list->items[i]);total+=strlen(parts[i]);}
    if(n>0) total+=seplen*(n-1);
    char *buf=(char*)wk_malloc(total+1); size_t p=0;
    for(size_t i=0;i<n;i++){
        if(i>0){memcpy(buf+p,sep,seplen);p+=seplen;}
        size_t l=strlen(parts[i]); memcpy(buf+p,parts[i],l); p+=l; free(parts[i]);
    }
    buf[p]='\0'; free(parts);
    WkVal r=wk_make_str(buf,p); free(buf); return r;
}
static WkVal _wklm_map(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1) return self;
    WkVal result=wk_make_list();
    for(size_t i=0;i<self.as.list->len;i++)
        wk_list_push_raw(result.as.list, wk_call1(args[0],self.as.list->items[i]));
    return result;
}
static WkVal _wklm_filter(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1) return self;
    WkVal result=wk_make_list();
    for(size_t i=0;i<self.as.list->len;i++)
        if(wk_truthy(wk_call1(args[0],self.as.list->items[i])))
            wk_list_push_raw(result.as.list, self.as.list->items[i]);
    return result;
}
static WkVal _wklm_reduce(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1) return wk_none();
    WkVal acc=(argc>=2)?args[1]:(self.as.list->len>0?self.as.list->items[0]:wk_none());
    size_t start=(argc>=2)?0:1;
    for(size_t i=start;i<self.as.list->len;i++)
        acc=wk_call2(args[0],acc,self.as.list->items[i]);
    return acc;
}
static WkVal _wklm_forEach(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1) return wk_none();
    for(size_t i=0;i<self.as.list->len;i++) wk_call1(args[0],self.as.list->items[i]);
    return wk_none();
}
static WkVal _wklm_any(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    for(size_t i=0;i<self.as.list->len;i++){
        WkVal v=(argc>0)?wk_call1(args[0],self.as.list->items[i]):self.as.list->items[i];
        if(wk_truthy(v)) return wk_bool(1);
    }
    return wk_bool(0);
}
static WkVal _wklm_all(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    for(size_t i=0;i<self.as.list->len;i++){
        WkVal v=(argc>0)?wk_call1(args[0],self.as.list->items[i]):self.as.list->items[i];
        if(!wk_truthy(v)) return wk_bool(0);
    }
    return wk_bool(1);
}
static WkVal _wklm_slice(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    int64_t len=(int64_t)self.as.list->len;
    int64_t a=(argc>0&&args[0].tag!=WK_NONE)?args[0].as.i:0;
    int64_t b=(argc>1&&args[1].tag!=WK_NONE)?args[1].as.i:len;
    if(a<0)a+=len; if(b<0)b+=len;
    if(a<0)a=0; if(b>len)b=len;
    WkVal result=wk_make_list();
    for(int64_t i=a;i<b;i++) wk_list_push_raw(result.as.list,self.as.list->items[i]);
    return result;
}
static void _wklm_sort_impl(WkList *l, WkVal key_fn) {
    /* Bubble sort with optional key function */
    for(size_t i=0;i<l->len;i++)
        for(size_t j=0;j+1<l->len-i;j++){
            WkVal a=l->items[j], b=l->items[j+1];
            if(key_fn.tag==WK_FUNC){ a=wk_call1(key_fn,a); b=wk_call1(key_fn,b); }
            if(wk_truthy(wk_cmp_gt(a,b))){
                WkVal t=l->items[j];l->items[j]=l->items[j+1];l->items[j+1]=t;
            }
        }
}
static WkVal _wklm_sort(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    WkVal key_fn=(argc>0&&args[0].tag==WK_FUNC)?args[0]:wk_none();
    _wklm_sort_impl(self.as.list, key_fn);
    return wk_none();
}
static WkVal _wklm_sorted(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    WkVal key_fn=(argc>0&&args[0].tag==WK_FUNC)?args[0]:wk_none();
    WkVal copy=wk_make_list();
    for(size_t i=0;i<self.as.list->len;i++)
        wk_list_push_raw(copy.as.list,self.as.list->items[i]);
    _wklm_sort_impl(copy.as.list, key_fn);
    return copy;
}
static WkVal _wklm_unique(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val; (void)args;(void)argc;
    WkVal result=wk_make_list();
    for(size_t i=0;i<self.as.list->len;i++){
        int found=0;
        for(size_t j=0;j<result.as.list->len;j++)
            if(wk_equal(result.as.list->items[j],self.as.list->items[i])){found=1;break;}
        if(!found) wk_list_push_raw(result.as.list,self.as.list->items[i]);
    }
    return result;
}
static WkVal _wklm_flat(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val; (void)args;(void)argc;
    WkVal result=wk_make_list();
    for(size_t i=0;i<self.as.list->len;i++){
        WkVal item=self.as.list->items[i];
        if(item.tag==WK_LIST)
            for(size_t j=0;j<item.as.list->len;j++)
                wk_list_push_raw(result.as.list,item.as.list->items[j]);
        else wk_list_push_raw(result.as.list,item);
    }
    return result;
}
static WkVal _wklm_zip(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1||args[0].tag!=WK_LIST) return wk_make_list();
    WkList *a=self.as.list, *b=args[0].as.list;
    size_t n=(a->len<b->len)?a->len:b->len;
    WkVal result=wk_make_list();
    for(size_t i=0;i<n;i++){
        WkVal pair[2]={a->items[i],b->items[i]};
        wk_list_push_raw(result.as.list, wk_make_tuple(pair,2));
    }
    return result;
}
static WkVal _wklm_enumerate(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val; (void)args;(void)argc;
    WkVal result=wk_make_list();
    for(size_t i=0;i<self.as.list->len;i++){
        WkVal pair[2]={wk_int((int64_t)i),self.as.list->items[i]};
        wk_list_push_raw(result.as.list, wk_make_tuple(pair,2));
    }
    return result;
}
static WkVal _wklm_count(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1) return wk_int((int64_t)self.as.list->len);
    int64_t c=0;
    for(size_t i=0;i<self.as.list->len;i++)
        if(wk_equal(self.as.list->items[i],args[0])) c++;
    return wk_int(c);
}
static WkVal _wklm_extend(WkVal *args, int argc, WkFunc *fn) {
    WkVal self=fn->captures[0].val;
    if(argc<1||args[0].tag!=WK_LIST) return wk_none();
    for(size_t i=0;i<args[0].as.list->len;i++)
        wk_list_push_raw(self.as.list,args[0].as.list->items[i]);
    return wk_none();
}

#define MAKE_LIST_METHOD(methname, fnptr) \
    if(strcmp(name,methname)==0){ \
        WkCapture cap; cap.name="self"; cap.val=lst; \
        return wk_make_func(methname, fnptr, &cap, 1, NULL, 0); \
    }

static WkVal wk_list_getmethod(WkVal lst, const char *name) {
    MAKE_LIST_METHOD("len",       _wklm_len)
    MAKE_LIST_METHOD("isEmpty",   _wklm_isEmpty)
    MAKE_LIST_METHOD("push",      _wklm_push)
    MAKE_LIST_METHOD("append",    _wklm_append)
    MAKE_LIST_METHOD("pop",       _wklm_pop)
    MAKE_LIST_METHOD("insert",    _wklm_insert)
    MAKE_LIST_METHOD("remove",    _wklm_remove)
    MAKE_LIST_METHOD("contains",  _wklm_contains)
    MAKE_LIST_METHOD("indexOf",   _wklm_indexOf)
    MAKE_LIST_METHOD("first",     _wklm_first)
    MAKE_LIST_METHOD("last",      _wklm_last)
    MAKE_LIST_METHOD("clear",     _wklm_clear)
    MAKE_LIST_METHOD("reverse",   _wklm_reverse)
    MAKE_LIST_METHOD("sort",      _wklm_sort)
    MAKE_LIST_METHOD("sorted",    _wklm_sorted)
    MAKE_LIST_METHOD("join",      _wklm_join)
    MAKE_LIST_METHOD("map",       _wklm_map)
    MAKE_LIST_METHOD("filter",    _wklm_filter)
    MAKE_LIST_METHOD("reduce",    _wklm_reduce)
    MAKE_LIST_METHOD("forEach",   _wklm_forEach)
    MAKE_LIST_METHOD("any",       _wklm_any)
    MAKE_LIST_METHOD("all",       _wklm_all)
    MAKE_LIST_METHOD("sum",       _wklm_sum)
    MAKE_LIST_METHOD("min",       _wklm_min)
    MAKE_LIST_METHOD("max",       _wklm_max)
    MAKE_LIST_METHOD("slice",     _wklm_slice)
    MAKE_LIST_METHOD("unique",    _wklm_unique)
    MAKE_LIST_METHOD("flat",      _wklm_flat)
    MAKE_LIST_METHOD("zip",       _wklm_zip)
    MAKE_LIST_METHOD("enumerate", _wklm_enumerate)
    MAKE_LIST_METHOD("count",     _wklm_count)
    MAKE_LIST_METHOD("extend",    _wklm_extend)
    wk_panic("list has no method '%s'", name);
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAP METHODS
   ═══════════════════════════════════════════════════════════════════════════ */

static WkVal _wkmm_len(WkVal *a,int c,WkFunc *f){(void)a;(void)c; return wk_int((int64_t)f->captures[0].val.as.map->len);}
static WkVal _wkmm_isEmpty(WkVal *a,int c,WkFunc *f){(void)a;(void)c; return wk_bool(f->captures[0].val.as.map->len==0);}
static WkVal _wkmm_has(WkVal *args,int argc,WkFunc *fn){
    WkVal self=fn->captures[0].val;
    if(argc<1) return wk_bool(0);
    WkMap *m=self.as.map;
    size_t h=wk_hash_val(args[0])&(m->cap-1);
    for(size_t i=0;i<m->cap;i++){
        size_t idx=(h+i)&(m->cap-1);
        if(!m->buckets[idx].used) return wk_bool(0);
        if(wk_equal(m->buckets[idx].key,args[0])) return wk_bool(1);
    }
    return wk_bool(0);
}
static WkVal _wkmm_get(WkVal *args,int argc,WkFunc *fn){
    WkVal self=fn->captures[0].val;
    if(argc<1) return wk_none();
    WkVal r=wk_map_get_key(self,args[0]);
    if(r.tag==WK_NONE && argc>=2) return args[1]; /* default */
    return r;
}
static WkVal _wkmm_set(WkVal *args,int argc,WkFunc *fn){
    WkVal self=fn->captures[0].val;
    if(argc>=2) wk_map_set_key(self,args[0],args[1]);
    return wk_none();
}
static WkVal _wkmm_delete(WkVal *args,int argc,WkFunc *fn){
    WkVal self=fn->captures[0].val;
    if(argc<1) return wk_none();
    WkMap *m=self.as.map;
    size_t h=wk_hash_val(args[0])&(m->cap-1);
    for(size_t i=0;i<m->cap;i++){
        size_t idx=(h+i)&(m->cap-1);
        if(!m->buckets[idx].used) return wk_none();
        if(wk_equal(m->buckets[idx].key,args[0])){ m->buckets[idx].used=0; m->len--; return wk_none(); }
    }
    return wk_none();
}
static WkVal _wkmm_keys(WkVal *a,int c,WkFunc *fn){
    (void)a;(void)c;
    WkVal self=fn->captures[0].val; WkMap *m=self.as.map;
    WkVal r=wk_make_list();
    for(size_t i=0;i<m->cap;i++) if(m->buckets[i].used) wk_list_push_raw(r.as.list,m->buckets[i].key);
    return r;
}
static WkVal _wkmm_values(WkVal *a,int c,WkFunc *fn){
    (void)a;(void)c;
    WkVal self=fn->captures[0].val; WkMap *m=self.as.map;
    WkVal r=wk_make_list();
    for(size_t i=0;i<m->cap;i++) if(m->buckets[i].used) wk_list_push_raw(r.as.list,m->buckets[i].val);
    return r;
}
static WkVal _wkmm_entries(WkVal *a,int c,WkFunc *fn){
    (void)a;(void)c;
    WkVal self=fn->captures[0].val; WkMap *m=self.as.map;
    WkVal r=wk_make_list();
    for(size_t i=0;i<m->cap;i++) if(m->buckets[i].used){
        WkVal pair[2]={m->buckets[i].key, m->buckets[i].val};
        wk_list_push_raw(r.as.list, wk_make_tuple(pair,2));
    }
    return r;
}
static WkVal _wkmm_clear(WkVal *a,int c,WkFunc *fn){
    (void)a;(void)c;
    WkVal self=fn->captures[0].val; WkMap *m=self.as.map;
    memset(m->buckets,0,m->cap*sizeof(WkMapEntry));
    m->len=0; return wk_none();
}
static WkVal _wkmm_merge(WkVal *args,int argc,WkFunc *fn){
    WkVal self=fn->captures[0].val;
    if(argc<1||args[0].tag!=WK_MAP) return wk_none();
    WkMap *src=args[0].as.map;
    for(size_t i=0;i<src->cap;i++)
        if(src->buckets[i].used) wk_map_set_key(self,src->buckets[i].key,src->buckets[i].val);
    return wk_none();
}

#define MAKE_MAP_METHOD(methname, fnptr) \
    if(strcmp(name,methname)==0){ \
        WkCapture cap; cap.name="self"; cap.val=mv; \
        return wk_make_func(methname, fnptr, &cap, 1, NULL, 0); \
    }

static WkVal wk_map_getmethod(WkVal mv, const char *name) {
    /* First: check if key exists in map data (supports module-style maps like math) */
    { WkVal k = wk_make_strz(name);
      WkVal v = wk_map_get_key(mv, k);
      if (v.tag != WK_NONE) return v; }
    MAKE_MAP_METHOD("len",     _wkmm_len)
    MAKE_MAP_METHOD("isEmpty", _wkmm_isEmpty)
    MAKE_MAP_METHOD("has",     _wkmm_has)
    MAKE_MAP_METHOD("get",     _wkmm_get)
    MAKE_MAP_METHOD("set",     _wkmm_set)
    MAKE_MAP_METHOD("delete",  _wkmm_delete)
    MAKE_MAP_METHOD("keys",    _wkmm_keys)
    MAKE_MAP_METHOD("values",  _wkmm_values)
    MAKE_MAP_METHOD("entries", _wkmm_entries)
    MAKE_MAP_METHOD("clear",   _wkmm_clear)
    MAKE_MAP_METHOD("merge",   _wkmm_merge)
    wk_panic("map has no member '%s'", name);
}

WkVal wk_map_keys(WkVal m) {
    if (m.tag != WK_MAP) wk_panic("map_keys: not a map");
    WkMap *map = m.as.map;
    WkVal r = wk_make_list();
    for (size_t i = 0; i < map->cap; i++)
        if (map->buckets[i].used) wk_list_push_raw(r.as.list, map->buckets[i].key);
    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
   BUILT-IN FUNCTIONS
   ═══════════════════════════════════════════════════════════════════════════ */

WkVal wk_builtin_println(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    for(int i=0;i<argc;i++){
        if(i>0) putchar(' ');
        char *s=wk_to_cstr(args[i]); fputs(s,stdout); free(s);
    }
    putchar('\n'); fflush(stdout);
    return wk_none();
}
WkVal wk_builtin_print(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    for(int i=0;i<argc;i++){
        if(i>0) putchar(' ');
        char *s=wk_to_cstr(args[i]); fputs(s,stdout); free(s);
    }
    fflush(stdout);
    return wk_none();
}
WkVal wk_builtin_eprintln(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    for(int i=0;i<argc;i++){
        if(i>0) fputc(' ',stderr);
        char *s=wk_to_cstr(args[i]); fputs(s,stderr); free(s);
    }
    fputc('\n',stderr);
    return wk_none();
}
WkVal wk_builtin_readln(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if(argc>0){ char *s=wk_to_cstr(args[0]); fputs(s,stdout); fflush(stdout); free(s); }
    char buf[4096]; if(!fgets(buf,sizeof(buf),stdin)) return wk_none();
    size_t l=strlen(buf); if(l>0&&buf[l-1]=='\n') buf[--l]='\0';
    return wk_make_str(buf,l);
}
WkVal wk_builtin_len(WkVal *args, int argc, WkFunc *fn) {
    (void)fn; if(argc<1) return wk_int(0);
    WkVal v=args[0];
    switch(v.tag){
        case WK_STR:   return wk_int((int64_t)v.as.str->len);
        case WK_LIST:  return wk_int((int64_t)v.as.list->len);
        case WK_MAP:   return wk_int((int64_t)v.as.map->len);
        case WK_TUPLE: return wk_int((int64_t)v.as.tup->len);
        case WK_RANGE: {
            WkRange r=v.as.rng;
            int64_t n=r.end-r.start+(r.inclusive?1:0);
            return wk_int(n>0?n:0);
        }
        default: return wk_int(1);
    }
}
WkVal wk_builtin_str(WkVal *args,int argc,WkFunc *fn){(void)fn;if(argc<1)return wk_make_strz("");return wk_cast_str(args[0]);}
WkVal wk_builtin_int(WkVal *args,int argc,WkFunc *fn){(void)fn;if(argc<1)return wk_int(0);return wk_cast_int(args[0]);}
WkVal wk_builtin_float(WkVal *args,int argc,WkFunc *fn){
    (void)fn;if(argc<1)return wk_float(0.0);
    WkVal v=args[0];
    switch(v.tag){
        case WK_FLOAT: return v;
        case WK_INT:   return wk_float((double)v.as.i);
        case WK_BOOL:  return wk_float((double)v.as.i);
        case WK_STR: { char *e; double f=strtod(v.as.str->data,&e);
                       if(*e) return wk_none();
                       return wk_float(f); }
        default: return wk_none();
    }
}
WkVal wk_builtin_bool(WkVal *args,int argc,WkFunc *fn){(void)fn;if(argc<1)return wk_bool(0);return wk_cast_bool(args[0]);}
WkVal wk_builtin_typeof(WkVal *args,int argc,WkFunc *fn){
    (void)fn; if(argc<1) return wk_make_strz("none");
    const char *names[]={"none","bool","int","float","str","list","map","func","object","range","some","ok","err","tuple","class","tensor"};
    int t=args[0].tag; if(t<0||t>15) return wk_make_strz("unknown");
    return wk_make_strz(names[t]);
}
WkVal wk_builtin_isNone(WkVal *a,int c,WkFunc *f){(void)f;return wk_bool(c>0&&a[0].tag==WK_NONE);}
WkVal wk_builtin_isSome(WkVal *a,int c,WkFunc *f){(void)f;return wk_bool(c>0&&a[0].tag==WK_SOME);}
WkVal wk_builtin_isOk(WkVal *a,int c,WkFunc *f){(void)f;return wk_bool(c>0&&a[0].tag==WK_OK);}
WkVal wk_builtin_isErr(WkVal *a,int c,WkFunc *f){(void)f;return wk_bool(c>0&&a[0].tag==WK_ERR);}
WkVal wk_builtin_sum(WkVal *args,int argc,WkFunc *fn){
    (void)fn; WkVal a1[1]={argc>0?args[0]:wk_make_list()};
    return _wklm_sum(a1,1,&(WkFunc){.captures=&(WkCapture){.name="self",.val=a1[0]},.ncaptures=1});
}
/* stub min/max for single arg */
WkVal wk_builtin_min(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc==0) return wk_none();
    if(argc==1 && args[0].tag==WK_LIST) return _wklm_min(NULL,0,&(WkFunc){.captures=&(WkCapture){.name="self",.val=args[0]},.ncaptures=1});
    WkVal m=args[0]; for(int i=1;i<argc;i++) if(wk_truthy(wk_cmp_lt(args[i],m))) m=args[i];
    return m;
}
WkVal wk_builtin_max(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc==0) return wk_none();
    if(argc==1 && args[0].tag==WK_LIST) return _wklm_max(NULL,0,&(WkFunc){.captures=&(WkCapture){.name="self",.val=args[0]},.ncaptures=1});
    WkVal m=args[0]; for(int i=1;i<argc;i++) if(wk_truthy(wk_cmp_gt(args[i],m))) m=args[i];
    return m;
}
WkVal wk_builtin_map_fn(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<2) return wk_make_list();
    /* map(list, fn) — list first, fn second (matches transpiler pipe convention) */
    WkVal lst=args[0], f=args[1];
    WkVal result=wk_make_list();
    if(lst.tag==WK_LIST) for(size_t i=0;i<lst.as.list->len;i++) wk_list_push_raw(result.as.list, wk_call1(f,lst.as.list->items[i]));
    return result;
}
WkVal wk_builtin_filter(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<2) return wk_make_list();
    /* filter(list, fn) — list first, fn second */
    WkVal lst=args[0], f=args[1];
    WkVal result=wk_make_list();
    if(lst.tag==WK_LIST) for(size_t i=0;i<lst.as.list->len;i++) if(wk_truthy(wk_call1(f,lst.as.list->items[i]))) wk_list_push_raw(result.as.list,lst.as.list->items[i]);
    return result;
}
WkVal wk_builtin_reduce(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<2) return wk_none();
    WkVal f=args[0],lst=args[1];
    WkVal init=(argc>=3)?args[2]:(lst.tag==WK_LIST&&lst.as.list->len>0?lst.as.list->items[0]:wk_none());
    size_t start=(argc>=3)?0:1;
    if(lst.tag==WK_LIST) for(size_t i=start;i<lst.as.list->len;i++) init=wk_call2(f,init,lst.as.list->items[i]);
    return init;
}
WkVal wk_builtin_sorted(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<1||args[0].tag!=WK_LIST) return wk_make_list();
    WkVal key_fn=(argc>=2&&args[1].tag==WK_FUNC)?args[1]:wk_none();
    WkVal copy=wk_make_list();
    for(size_t i=0;i<args[0].as.list->len;i++) wk_list_push_raw(copy.as.list,args[0].as.list->items[i]);
    _wklm_sort_impl(copy.as.list, key_fn);
    return copy;
}
WkVal wk_builtin_reversed(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<1||args[0].tag!=WK_LIST) return wk_make_list();
    WkVal r=wk_make_list();
    for(int64_t i=(int64_t)args[0].as.list->len-1;i>=0;i--) wk_list_push_raw(r.as.list,args[0].as.list->items[i]);
    return r;
}
WkVal wk_builtin_any(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<1||args[0].tag!=WK_LIST) return wk_bool(0);
    WkVal fn2=(argc>=2)?args[1]:wk_none();
    for(size_t i=0;i<args[0].as.list->len;i++){
        WkVal v=(fn2.tag==WK_FUNC)?wk_call1(fn2,args[0].as.list->items[i]):args[0].as.list->items[i];
        if(wk_truthy(v)) return wk_bool(1);
    }
    return wk_bool(0);
}
WkVal wk_builtin_all(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<1||args[0].tag!=WK_LIST) return wk_bool(1);
    WkVal fn2=(argc>=2)?args[1]:wk_none();
    for(size_t i=0;i<args[0].as.list->len;i++){
        WkVal v=(fn2.tag==WK_FUNC)?wk_call1(fn2,args[0].as.list->items[i]):args[0].as.list->items[i];
        if(!wk_truthy(v)) return wk_bool(0);
    }
    return wk_bool(1);
}
WkVal wk_builtin_zip(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<2||args[0].tag!=WK_LIST||args[1].tag!=WK_LIST) return wk_make_list();
    WkList *a=args[0].as.list,*b=args[1].as.list;
    size_t n=(a->len<b->len)?a->len:b->len;
    WkVal r=wk_make_list();
    for(size_t i=0;i<n;i++){WkVal p[2]={a->items[i],b->items[i]};wk_list_push_raw(r.as.list,wk_make_tuple(p,2));}
    return r;
}
WkVal wk_builtin_enumerate(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<1||args[0].tag!=WK_LIST) return wk_make_list();
    WkVal r=wk_make_list();
    for(size_t i=0;i<args[0].as.list->len;i++){
        WkVal p[2]={wk_int((int64_t)i),args[0].as.list->items[i]};
        wk_list_push_raw(r.as.list,wk_make_tuple(p,2));
    }
    return r;
}
WkVal wk_builtin_range(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<2) return wk_range(0,argc>0?args[0].as.i:0,0);
    return wk_range(args[0].as.i,args[1].as.i,0);
}
WkVal wk_builtin_sleep(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    double secs=(argc>0)?wk_to_num(args[0]):0.0;
#ifdef _WIN32
    Sleep((DWORD)(secs*1000));
#else
    struct timespec ts; ts.tv_sec=(time_t)secs; ts.tv_nsec=(long)((secs-(long)secs)*1e9);
    nanosleep(&ts,NULL);
#endif
    return wk_none();
}
WkVal wk_builtin_exit(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    exit(argc>0?(int)args[0].as.i:0);
}
WkVal wk_builtin_assert(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<1||wk_truthy(args[0])) return wk_none();
    const char *msg=(argc>=2&&args[1].tag==WK_STR)?args[1].as.str->data:"assertion failed";
    wk_panic("%s", msg);
}
WkVal wk_deep_copy(WkVal v) {
    switch (v.tag) {
    case WK_NONE: case WK_BOOL: case WK_INT: case WK_FLOAT:
    case WK_RANGE: case WK_CLASS: case WK_FUNC:
        return v;
    case WK_STR:
        return wk_make_str(v.as.str->data, v.as.str->len);
    case WK_LIST: {
        WkVal r = wk_make_list();
        for (size_t i = 0; i < v.as.list->len; i++)
            wk_list_push_raw(r.as.list, wk_deep_copy(v.as.list->items[i]));
        return r;
    }
    case WK_MAP: {
        WkVal r = wk_make_map();
        for (size_t i = 0; i < v.as.map->cap; i++) {
            if (v.as.map->buckets[i].used)
                wk_map_set_key(r, wk_deep_copy(v.as.map->buckets[i].key),
                               wk_deep_copy(v.as.map->buckets[i].val));
        }
        return r;
    }
    case WK_TUPLE: {
        size_t n = v.as.tup->len;
        WkVal *items = (WkVal*)wk_malloc(n * sizeof(WkVal));
        for (size_t i = 0; i < n; i++)
            items[i] = wk_deep_copy(v.as.tup->items[i]);
        WkVal r = wk_make_tuple(items, (int)n);
        free(items);
        return r;
    }
    case WK_OBJ: {
        WkVal r = wk_make_obj(v.as.obj->cls);
        for (int i = 0; i < v.as.obj->cls->nfields; i++)
            r.as.obj->fields[i] = wk_deep_copy(v.as.obj->fields[i]);
        return r;
    }
    case WK_SOME: return wk_make_some(wk_deep_copy(*v.as.inner));
    case WK_OK:   return wk_make_ok(wk_deep_copy(*v.as.inner));
    case WK_ERR:  return wk_make_err(wk_deep_copy(*v.as.inner));
    default: return v;
    }
}
WkVal wk_builtin_copy(WkVal *args,int argc,WkFunc *fn){(void)fn;if(argc<1)return wk_none();return wk_deep_copy(args[0]);}
WkVal wk_builtin_chr(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<1||args[0].tag!=WK_INT) return wk_make_strz("");
    char c=(char)args[0].as.i;
    return wk_make_str(&c,1);
}
WkVal wk_builtin_ord(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<1||args[0].tag!=WK_STR||args[0].as.str->len==0) return wk_int(0);
    return wk_int((unsigned char)args[0].as.str->data[0]);
}
WkVal wk_builtin_hash(WkVal *args,int argc,WkFunc *fn){(void)fn;if(argc<1)return wk_int(0);return wk_int((int64_t)wk_hash_val(args[0]));}
WkVal wk_builtin_repr(WkVal *args,int argc,WkFunc *fn){
    (void)fn;
    if(argc<1) return wk_make_strz("");
    char *s=wk_to_repr(args[0]); WkVal r=wk_make_strz(s); free(s); return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
   TENSOR OPERATIONS
   ═══════════════════════════════════════════════════════════════════════════ */

WkVal wk_make_tensor(int ndim, const int64_t *shape, const double *data, int owns) {
    WkTensor *t = (WkTensor*)wk_malloc(sizeof(WkTensor));
    t->refcnt = 1;
    t->ndim = ndim;
    t->shape = (int64_t*)wk_malloc(ndim * sizeof(int64_t));
    t->strides = (int64_t*)wk_malloc(ndim * sizeof(int64_t));
    memcpy(t->shape, shape, ndim * sizeof(int64_t));
    int64_t total = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->strides[i] = total;
        total *= shape[i];
    }
    t->len = total;
    if (data && owns) {
        t->data = (double*)data;
        t->owns_data = 1;
    } else {
        t->data = (double*)wk_malloc(total * sizeof(double));
        t->owns_data = 1;
        if (data) memcpy(t->data, data, total * sizeof(double));
        else memset(t->data, 0, total * sizeof(double));
    }
    WkVal v; memset(&v,0,sizeof(v));
    v.tag = WK_TENSOR; v.refcnt = 1; v.as.tensor = t;
    return v;
}

static double _wk_val_to_double(WkVal v) {
    if (v.tag == WK_INT) return (double)v.as.i;
    if (v.tag == WK_FLOAT) return v.as.f;
    return 0.0;
}

static WkVal wk_tensor_binop(WkVal a, WkVal b, int op) {
    /* op: 0=add, 1=sub, 2=mul, 3=div */
    /* scalar + tensor or tensor + scalar */
    if (a.tag == WK_TENSOR && b.tag == WK_TENSOR) {
        WkTensor *ta = a.as.tensor, *tb = b.as.tensor;
        if (ta->len != tb->len) wk_panic("tensor size mismatch: %lld vs %lld", (long long)ta->len, (long long)tb->len);
        double *rd = (double*)wk_malloc(ta->len * sizeof(double));
        for (int64_t i = 0; i < ta->len; i++) {
            switch(op) {
                case 0: rd[i] = ta->data[i] + tb->data[i]; break;
                case 1: rd[i] = ta->data[i] - tb->data[i]; break;
                case 2: rd[i] = ta->data[i] * tb->data[i]; break;
                case 3: rd[i] = ta->data[i] / tb->data[i]; break;
            }
        }
        return wk_make_tensor(ta->ndim, ta->shape, rd, 1);
    }
    /* scalar broadcast */
    WkTensor *t; double s;
    int scalar_left = 0;
    if (a.tag == WK_TENSOR) { t = a.as.tensor; s = _wk_val_to_double(b); }
    else { t = b.as.tensor; s = _wk_val_to_double(a); scalar_left = 1; }
    double *rd = (double*)wk_malloc(t->len * sizeof(double));
    for (int64_t i = 0; i < t->len; i++) {
        double tv = t->data[i];
        double lv = scalar_left ? s : tv;
        double rv = scalar_left ? tv : s;
        switch(op) {
            case 0: rd[i] = lv + rv; break;
            case 1: rd[i] = lv - rv; break;
            case 2: rd[i] = lv * rv; break;
            case 3: rd[i] = lv / rv; break;
        }
    }
    return wk_make_tensor(t->ndim, t->shape, rd, 1);
}

/* Tensor creation wrappers */
static WkVal _wk_tensor_zeros(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    /* Accept list of dims or individual dim args */
    int ndim; int64_t shape[8];
    if (argc == 1 && args[0].tag == WK_LIST) {
        ndim = (int)args[0].as.list->len;
        for (int i = 0; i < ndim && i < 8; i++)
            shape[i] = (int64_t)_wk_val_to_double(args[0].as.list->items[i]);
    } else {
        ndim = argc;
        for (int i = 0; i < argc && i < 8; i++)
            shape[i] = (int64_t)_wk_val_to_double(args[i]);
    }
    if (ndim == 0) { ndim = 1; shape[0] = 0; }
    return wk_make_tensor(ndim, shape, NULL, 0);
}

static WkVal _wk_tensor_ones(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    int ndim; int64_t shape[8];
    if (argc == 1 && args[0].tag == WK_LIST) {
        ndim = (int)args[0].as.list->len;
        for (int i = 0; i < ndim && i < 8; i++)
            shape[i] = (int64_t)_wk_val_to_double(args[0].as.list->items[i]);
    } else {
        ndim = argc;
        for (int i = 0; i < argc && i < 8; i++)
            shape[i] = (int64_t)_wk_val_to_double(args[i]);
    }
    if (ndim == 0) { ndim = 1; shape[0] = 0; }
    WkVal t = wk_make_tensor(ndim, shape, NULL, 0);
    for (int64_t i = 0; i < t.as.tensor->len; i++) t.as.tensor->data[i] = 1.0;
    return t;
}

static void _wk_flatten_list(WkVal lst, double *out, int64_t *idx) {
    for (size_t i = 0; i < lst.as.list->len; i++) {
        WkVal item = lst.as.list->items[i];
        if (item.tag == WK_LIST) _wk_flatten_list(item, out, idx);
        else { out[*idx] = _wk_val_to_double(item); (*idx)++; }
    }
}

static int _wk_infer_shape(WkVal lst, int64_t *shape, int depth) {
    if (lst.tag != WK_LIST || depth >= 8) return depth;
    shape[depth] = lst.as.list->len;
    if (lst.as.list->len > 0 && lst.as.list->items[0].tag == WK_LIST)
        return _wk_infer_shape(lst.as.list->items[0], shape, depth + 1);
    return depth + 1;
}

static WkVal _wk_tensor_from(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 1) return wk_none();
    if (args[0].tag == WK_TENSOR) return args[0]; /* already a tensor */
    if (args[0].tag != WK_LIST) {
        int64_t sh[1] = {1};
        double d = _wk_val_to_double(args[0]);
        return wk_make_tensor(1, sh, &d, 0);
    }
    int64_t shape[8]; memset(shape, 0, sizeof(shape));
    int ndim = _wk_infer_shape(args[0], shape, 0);
    int64_t total = 1;
    for (int i = 0; i < ndim; i++) total *= shape[i];
    double *data = (double*)wk_malloc(total * sizeof(double));
    int64_t idx = 0;
    _wk_flatten_list(args[0], data, &idx);
    return wk_make_tensor(ndim, shape, data, 1);
}

static WkVal _wk_tensor_arange(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    double start = 0, stop = 0, step = 1;
    if (argc == 1) { stop = _wk_val_to_double(args[0]); }
    else if (argc >= 2) { start = _wk_val_to_double(args[0]); stop = _wk_val_to_double(args[1]); }
    if (argc >= 3) step = _wk_val_to_double(args[2]);
    if (step == 0) step = 1;
    int64_t n = (int64_t)ceil((stop - start) / step);
    if (n < 0) n = 0;
    double *data = (double*)wk_malloc(n * sizeof(double));
    for (int64_t i = 0; i < n; i++) data[i] = start + i * step;
    int64_t sh[1] = {n};
    return wk_make_tensor(1, sh, data, 1);
}

static WkVal _wk_tensor_linspace(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 3) return wk_none();
    double start = _wk_val_to_double(args[0]);
    double stop = _wk_val_to_double(args[1]);
    int64_t n = (int64_t)_wk_val_to_double(args[2]);
    if (n < 1) n = 1;
    double *data = (double*)wk_malloc(n * sizeof(double));
    double step = (n > 1) ? (stop - start) / (n - 1) : 0;
    for (int64_t i = 0; i < n; i++) data[i] = start + i * step;
    int64_t sh[1] = {n};
    return wk_make_tensor(1, sh, data, 1);
}

static WkVal _wk_tensor_eye(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    int64_t n = (argc > 0) ? (int64_t)_wk_val_to_double(args[0]) : 1;
    int64_t sh[2] = {n, n};
    WkVal t = wk_make_tensor(2, sh, NULL, 0);
    for (int64_t i = 0; i < n; i++) t.as.tensor->data[i * n + i] = 1.0;
    return t;
}

static WkVal _wk_tensor_rand(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    int ndim; int64_t shape[8];
    if (argc == 1 && args[0].tag == WK_LIST) {
        ndim = (int)args[0].as.list->len;
        for (int i = 0; i < ndim && i < 8; i++)
            shape[i] = (int64_t)_wk_val_to_double(args[0].as.list->items[i]);
    } else {
        ndim = argc;
        for (int i = 0; i < argc && i < 8; i++)
            shape[i] = (int64_t)_wk_val_to_double(args[i]);
    }
    if (ndim == 0) { ndim = 1; shape[0] = 1; }
    WkVal t = wk_make_tensor(ndim, shape, NULL, 0);
    for (int64_t i = 0; i < t.as.tensor->len; i++)
        t.as.tensor->data[i] = (double)rand() / RAND_MAX;
    return t;
}

/* Tensor reductions */
static WkVal _wkt_sum(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    double s = 0; for (int64_t i = 0; i < t->len; i++) s += t->data[i];
    return wk_float(s);
}
static WkVal _wkt_mean(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    if (t->len == 0) return wk_float(0);
    double s = 0; for (int64_t i = 0; i < t->len; i++) s += t->data[i];
    return wk_float(s / t->len);
}
static WkVal _wkt_max(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    if (t->len == 0) return wk_float(0);
    double m = t->data[0]; for (int64_t i = 1; i < t->len; i++) if (t->data[i] > m) m = t->data[i];
    return wk_float(m);
}
static WkVal _wkt_min(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    if (t->len == 0) return wk_float(0);
    double m = t->data[0]; for (int64_t i = 1; i < t->len; i++) if (t->data[i] < m) m = t->data[i];
    return wk_float(m);
}
static WkVal _wkt_argmax(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    if (t->len == 0) return wk_int(0);
    int64_t mi = 0; for (int64_t i = 1; i < t->len; i++) if (t->data[i] > t->data[mi]) mi = i;
    return wk_int(mi);
}
static WkVal _wkt_argmin(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    if (t->len == 0) return wk_int(0);
    int64_t mi = 0; for (int64_t i = 1; i < t->len; i++) if (t->data[i] < t->data[mi]) mi = i;
    return wk_int(mi);
}
static WkVal _wkt_shape(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    WkVal lst = wk_make_list();
    for (int i = 0; i < t->ndim; i++) wk_list_push_raw(lst.as.list, wk_int(t->shape[i]));
    return lst;
}
static WkVal _wkt_toList(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    WkVal lst = wk_make_list();
    for (int64_t i = 0; i < t->len; i++) wk_list_push_raw(lst.as.list, wk_float(t->data[i]));
    return lst;
}
static WkVal _wkt_reshape(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor;
    int ndim; int64_t shape[8];
    if (argc == 1 && args[0].tag == WK_LIST) {
        ndim = (int)args[0].as.list->len;
        for (int i = 0; i < ndim && i < 8; i++)
            shape[i] = (int64_t)_wk_val_to_double(args[0].as.list->items[i]);
    } else {
        ndim = argc;
        for (int i = 0; i < argc && i < 8; i++)
            shape[i] = (int64_t)_wk_val_to_double(args[i]);
    }
    int64_t total = 1; for (int i = 0; i < ndim; i++) total *= shape[i];
    if (total != t->len) wk_panic("reshape: size mismatch %lld vs %lld", (long long)total, (long long)t->len);
    return wk_make_tensor(ndim, shape, t->data, 0);
}
static WkVal _wkt_transpose(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    if (t->ndim != 2) wk_panic("transpose requires 2D tensor");
    int64_t rows = t->shape[0], cols = t->shape[1];
    int64_t sh[2] = {cols, rows};
    double *data = (double*)wk_malloc(t->len * sizeof(double));
    for (int64_t r = 0; r < rows; r++)
        for (int64_t c = 0; c < cols; c++)
            data[c * rows + r] = t->data[r * cols + c];
    return wk_make_tensor(2, sh, data, 1);
}
static WkVal _wkt_flatten(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    int64_t sh[1] = {t->len};
    return wk_make_tensor(1, sh, t->data, 0);
}
static WkVal _wkt_matmul(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *a = fn->captures[0].val.as.tensor;
    if (argc < 1 || args[0].tag != WK_TENSOR) wk_panic("matmul requires a tensor argument");
    WkTensor *b = args[0].as.tensor;
    if (a->ndim != 2 || b->ndim != 2) wk_panic("matmul requires 2D tensors");
    if (a->shape[1] != b->shape[0]) wk_panic("matmul shape mismatch: %lld vs %lld", (long long)a->shape[1], (long long)b->shape[0]);
    int64_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    int64_t sh[2] = {M, N};
    double *data = (double*)wk_malloc(M * N * sizeof(double));
    memset(data, 0, M * N * sizeof(double));
    for (int64_t i = 0; i < M; i++)
        for (int64_t k = 0; k < K; k++) {
            double aik = a->data[i * K + k];
            for (int64_t j = 0; j < N; j++)
                data[i * N + j] += aik * b->data[k * N + j];
        }
    return wk_make_tensor(2, sh, data, 1);
}
static WkVal _wkt_item(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    if (t->len == 0) return wk_float(0);
    if (argc > 0) {
        int64_t idx = (int64_t)_wk_val_to_double(args[0]);
        if (idx < 0) idx += t->len;
        if (idx < 0 || idx >= t->len) wk_panic("tensor index out of range");
        return wk_float(t->data[idx]);
    }
    if (t->len == 1) return wk_float(t->data[0]);
    wk_panic("item() on tensor with more than one element requires an index");
}
static WkVal _wkt_abs(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    double *d = (double*)wk_malloc(t->len * sizeof(double));
    for (int64_t i = 0; i < t->len; i++) d[i] = fabs(t->data[i]);
    return wk_make_tensor(t->ndim, t->shape, d, 1);
}
static WkVal _wkt_sqrt(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *t = fn->captures[0].val.as.tensor; (void)args; (void)argc;
    double *d = (double*)wk_malloc(t->len * sizeof(double));
    for (int64_t i = 0; i < t->len; i++) d[i] = sqrt(t->data[i]);
    return wk_make_tensor(t->ndim, t->shape, d, 1);
}
static WkVal _wkt_dot(WkVal *args, int argc, WkFunc *fn) {
    WkTensor *a = fn->captures[0].val.as.tensor;
    if (argc < 1 || args[0].tag != WK_TENSOR) wk_panic("dot requires a tensor argument");
    WkTensor *b = args[0].as.tensor;
    if (a->len != b->len) wk_panic("dot: size mismatch");
    double s = 0;
    for (int64_t i = 0; i < a->len; i++) s += a->data[i] * b->data[i];
    return wk_float(s);
}

#define _WKT_BIND(meth_name, cfn) \
    if (strcmp(mname, meth_name) == 0) { \
        WkCapture *cap = (WkCapture*)wk_malloc(sizeof(WkCapture)); \
        cap->name = "self"; cap->val = obj; \
        return wk_make_func(meth_name, cfn, cap, 1, NULL, -1); \
    }

static WkVal wk_tensor_getmethod(WkVal obj, const char *mname) {
    _WKT_BIND("sum", _wkt_sum)
    _WKT_BIND("mean", _wkt_mean)
    _WKT_BIND("max", _wkt_max)
    _WKT_BIND("min", _wkt_min)
    _WKT_BIND("argmax", _wkt_argmax)
    _WKT_BIND("argmin", _wkt_argmin)
    _WKT_BIND("shape", _wkt_shape)
    _WKT_BIND("toList", _wkt_toList)
    _WKT_BIND("reshape", _wkt_reshape)
    _WKT_BIND("transpose", _wkt_transpose)
    _WKT_BIND("T", _wkt_transpose)
    _WKT_BIND("flatten", _wkt_flatten)
    _WKT_BIND("matmul", _wkt_matmul)
    _WKT_BIND("dot", _wkt_dot)
    _WKT_BIND("item", _wkt_item)
    _WKT_BIND("abs", _wkt_abs)
    _WKT_BIND("sqrt", _wkt_sqrt)
    if (strcmp(mname, "ndim") == 0) return wk_int(obj.as.tensor->ndim);
    if (strcmp(mname, "size") == 0) return wk_int(obj.as.tensor->len);
    wk_panic("tensor has no member '%s'", mname);
}

/* ═══════════════════════════════════════════════════════════════════════════
   AUTODIFF (Forward-mode dual numbers)
   Mitigation: uses 1970s academic dual-number technique, no patent risk.
   ═══════════════════════════════════════════════════════════════════════════ */

static WkClass _wk_cls_dual;
static const char *_dual_fnames[] = {"value", "deriv"};

static WkVal _wk_make_dual(double val, double deriv) {
    WkVal obj = wk_make_obj(&_wk_cls_dual);
    obj.as.obj->fields[0] = wk_float(val);
    obj.as.obj->fields[1] = wk_float(deriv);
    return obj;
}

static int _wk_is_dual(WkVal v) {
    return v.tag == WK_OBJ && v.as.obj->cls == &_wk_cls_dual;
}

static double _dual_val(WkVal v) { return v.as.obj->fields[0].as.f; }
static double _dual_der(WkVal v) { return v.as.obj->fields[1].as.f; }

static WkVal _wk_ad_dual(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    double v = (argc > 0) ? _wk_val_to_double(args[0]) : 0;
    double d = (argc > 1) ? _wk_val_to_double(args[1]) : 0;
    return _wk_make_dual(v, d);
}

static WkVal _wk_ad_value(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc > 0 && _wk_is_dual(args[0])) return wk_float(_dual_val(args[0]));
    return (argc > 0) ? args[0] : wk_float(0);
}
static WkVal _wk_ad_deriv(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc > 0 && _wk_is_dual(args[0])) return wk_float(_dual_der(args[0]));
    return wk_float(0);
}

static WkVal _wk_ad_grad(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 2) wk_panic("ad.grad(fn, x) requires 2 args");
    WkVal func = args[0];
    double x = _wk_val_to_double(args[1]);
    WkVal dual_x = _wk_make_dual(x, 1.0);
    WkVal result = wk_call1(func, dual_x);
    if (_wk_is_dual(result)) return wk_float(_dual_der(result));
    return wk_float(0);
}

/* ad.sin, ad.cos, ad.exp, ad.log, ad.sqrt — dual-aware */
static WkVal _wk_ad_sin(WkVal *args, int argc, WkFunc *fn) {
    (void)fn; if (argc < 1) return wk_float(0);
    if (_wk_is_dual(args[0])) {
        double v = _dual_val(args[0]), d = _dual_der(args[0]);
        return _wk_make_dual(sin(v), cos(v) * d);
    }
    return wk_float(sin(_wk_val_to_double(args[0])));
}
static WkVal _wk_ad_cos(WkVal *args, int argc, WkFunc *fn) {
    (void)fn; if (argc < 1) return wk_float(0);
    if (_wk_is_dual(args[0])) {
        double v = _dual_val(args[0]), d = _dual_der(args[0]);
        return _wk_make_dual(cos(v), -sin(v) * d);
    }
    return wk_float(cos(_wk_val_to_double(args[0])));
}
static WkVal _wk_ad_exp(WkVal *args, int argc, WkFunc *fn) {
    (void)fn; if (argc < 1) return wk_float(0);
    if (_wk_is_dual(args[0])) {
        double v = _dual_val(args[0]), d = _dual_der(args[0]);
        double ev = exp(v);
        return _wk_make_dual(ev, ev * d);
    }
    return wk_float(exp(_wk_val_to_double(args[0])));
}
static WkVal _wk_ad_log(WkVal *args, int argc, WkFunc *fn) {
    (void)fn; if (argc < 1) return wk_float(0);
    if (_wk_is_dual(args[0])) {
        double v = _dual_val(args[0]), d = _dual_der(args[0]);
        return _wk_make_dual(log(v), d / v);
    }
    return wk_float(log(_wk_val_to_double(args[0])));
}
static WkVal _wk_ad_sqrt(WkVal *args, int argc, WkFunc *fn) {
    (void)fn; if (argc < 1) return wk_float(0);
    if (_wk_is_dual(args[0])) {
        double v = _dual_val(args[0]), d = _dual_der(args[0]);
        double sv = sqrt(v);
        return _wk_make_dual(sv, d / (2.0 * sv));
    }
    return wk_float(sqrt(_wk_val_to_double(args[0])));
}
static WkVal _wk_ad_pow(WkVal *args, int argc, WkFunc *fn) {
    (void)fn; if (argc < 2) return wk_float(0);
    double n = _wk_val_to_double(args[1]);
    if (_wk_is_dual(args[0])) {
        double v = _dual_val(args[0]), d = _dual_der(args[0]);
        return _wk_make_dual(pow(v, n), n * pow(v, n - 1) * d);
    }
    return wk_float(pow(_wk_val_to_double(args[0]), n));
}

/* ═══════════════════════════════════════════════════════════════════════════
   GPU DISPATCH (OpenCL, dynamically loaded)
   Mitigation: Uses OpenCL (Khronos, royalty-free open standard).
   ═══════════════════════════════════════════════════════════════════════════ */

#ifndef _WIN32
#include <dlfcn.h>
#endif

static int _wk_gpu_avail = 0;
static void *_wk_ocl_lib = NULL;

static void _wk_gpu_init(void) {
#ifndef _WIN32
    _wk_ocl_lib = dlopen("libOpenCL.so", RTLD_LAZY);
    if (!_wk_ocl_lib) _wk_ocl_lib = dlopen("libOpenCL.so.1", RTLD_LAZY);
#ifdef __APPLE__
    if (!_wk_ocl_lib) _wk_ocl_lib = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_LAZY);
#endif
    if (_wk_ocl_lib) _wk_gpu_avail = 1;
#endif
}

static WkVal _wk_gpu_available(WkVal *args, int argc, WkFunc *fn) {
    (void)args; (void)argc; (void)fn;
    return wk_bool(_wk_gpu_avail);
}

static WkVal _wk_gpu_devices(WkVal *args, int argc, WkFunc *fn) {
    (void)args; (void)argc; (void)fn;
    WkVal lst = wk_make_list();
    if (!_wk_gpu_avail) return lst;
    /* Minimal device enumeration through dlsym */
    wk_list_push_raw(lst.as.list, wk_make_strz("OpenCL device (use gpu.run to execute kernels)"));
    return lst;
}

static WkVal _wk_gpu_run(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (!_wk_gpu_avail) return wk_make_err(wk_make_strz("OpenCL not available on this system"));
    if (argc < 1) return wk_make_err(wk_make_strz("gpu.run requires kernel source string"));
    /* Placeholder: a real implementation would compile and execute the kernel */
    return wk_make_err(wk_make_strz("gpu.run: full OpenCL kernel execution requires runtime linking (kernel source accepted)"));
}

/* ═══════════════════════════════════════════════════════════════════════════
   DATA PIPELINE (lazy transform chains)
   ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    WK_PIPE_MAP=0, WK_PIPE_FILTER, WK_PIPE_BATCH, WK_PIPE_FLATTEN,
    WK_PIPE_TAKE, WK_PIPE_SKIP, WK_PIPE_SHUFFLE, WK_PIPE_ZIP
} WkPipeKind;

typedef struct _WkPipeStep {
    WkPipeKind kind;
    WkVal arg;  /* fn or int or other pipeline */
    struct _WkPipeStep *next;
} WkPipeStep;

static WkClass _wk_cls_pipeline;
static const char *_pipe_fnames[] = {"source", "steps"};

static WkPipeStep *_pipe_clone_chain(WkPipeStep *s) {
    if (!s) return NULL;
    WkPipeStep *n = (WkPipeStep*)wk_malloc(sizeof(WkPipeStep));
    *n = *s;
    n->next = _pipe_clone_chain(s->next);
    return n;
}
static void _pipe_free_chain(WkPipeStep *s) {
    while (s) { WkPipeStep *nx = s->next; free(s); s = nx; }
}

static WkVal _pipe_new(WkVal source, WkPipeStep *steps) {
    WkVal obj = wk_make_obj(&_wk_cls_pipeline);
    obj.as.obj->fields[0] = source;
    /* Store step chain pointer packed into an int */
    obj.as.obj->fields[1] = wk_int((int64_t)(intptr_t)steps);
    return obj;
}
static WkPipeStep *_pipe_steps(WkVal obj) {
    return (WkPipeStep*)(intptr_t)obj.as.obj->fields[1].as.i;
}
static WkVal _pipe_source(WkVal obj) {
    return obj.as.obj->fields[0];
}

static WkVal _pipe_append(WkVal pipe, WkPipeKind kind, WkVal arg) {
    WkPipeStep *chain = _pipe_clone_chain(_pipe_steps(pipe));
    WkPipeStep *step = (WkPipeStep*)wk_malloc(sizeof(WkPipeStep));
    step->kind = kind; step->arg = arg; step->next = NULL;
    if (!chain) { chain = step; }
    else { WkPipeStep *p = chain; while (p->next) p = p->next; p->next = step; }
    return _pipe_new(_pipe_source(pipe), chain);
}

static WkVal _pipe_collect(WkVal pipe) {
    WkVal src = _pipe_source(pipe);
    /* Build initial list from source */
    WkVal data = wk_make_list();
    if (src.tag == WK_LIST) {
        for (size_t i = 0; i < src.as.list->len; i++)
            wk_list_push_raw(data.as.list, src.as.list->items[i]);
    } else if (src.tag == WK_TENSOR) {
        for (int64_t i = 0; i < src.as.tensor->len; i++)
            wk_list_push_raw(data.as.list, wk_float(src.as.tensor->data[i]));
    }
    /* Apply step chain */
    WkPipeStep *s = _pipe_steps(pipe);
    while (s) {
        switch (s->kind) {
        case WK_PIPE_MAP: {
            WkVal out = wk_make_list();
            for (size_t i = 0; i < data.as.list->len; i++)
                wk_list_push_raw(out.as.list, wk_call1(s->arg, data.as.list->items[i]));
            data = out;
        } break;
        case WK_PIPE_FILTER: {
            WkVal out = wk_make_list();
            for (size_t i = 0; i < data.as.list->len; i++) {
                WkVal r = wk_call1(s->arg, data.as.list->items[i]);
                if (wk_truthy(r)) wk_list_push_raw(out.as.list, data.as.list->items[i]);
            }
            data = out;
        } break;
        case WK_PIPE_BATCH: {
            int64_t n = s->arg.as.i;
            WkVal out = wk_make_list();
            for (size_t i = 0; i < data.as.list->len; i += n) {
                WkVal batch = wk_make_list();
                for (int64_t j = 0; j < n && i + j < data.as.list->len; j++)
                    wk_list_push_raw(batch.as.list, data.as.list->items[i + j]);
                wk_list_push_raw(out.as.list, batch);
            }
            data = out;
        } break;
        case WK_PIPE_FLATTEN: {
            WkVal out = wk_make_list();
            for (size_t i = 0; i < data.as.list->len; i++) {
                WkVal item = data.as.list->items[i];
                if (item.tag == WK_LIST)
                    for (size_t j = 0; j < item.as.list->len; j++)
                        wk_list_push_raw(out.as.list, item.as.list->items[j]);
                else
                    wk_list_push_raw(out.as.list, item);
            }
            data = out;
        } break;
        case WK_PIPE_TAKE: {
            int64_t n = s->arg.as.i;
            WkVal out = wk_make_list();
            for (int64_t i = 0; i < n && i < (int64_t)data.as.list->len; i++)
                wk_list_push_raw(out.as.list, data.as.list->items[i]);
            data = out;
        } break;
        case WK_PIPE_SKIP: {
            int64_t n = s->arg.as.i;
            WkVal out = wk_make_list();
            for (size_t i = (size_t)n; i < data.as.list->len; i++)
                wk_list_push_raw(out.as.list, data.as.list->items[i]);
            data = out;
        } break;
        case WK_PIPE_SHUFFLE: {
            /* Fisher-Yates shuffle */
            for (size_t i = data.as.list->len; i > 1; i--) {
                size_t j = (size_t)rand() % i;
                WkVal tmp = data.as.list->items[i-1];
                data.as.list->items[i-1] = data.as.list->items[j];
                data.as.list->items[j] = tmp;
            }
        } break;
        case WK_PIPE_ZIP: {
            WkVal other = s->arg;
            WkVal out = wk_make_list();
            size_t olen = (other.tag == WK_LIST) ? other.as.list->len : 0;
            size_t mlen = data.as.list->len < olen ? data.as.list->len : olen;
            for (size_t i = 0; i < mlen; i++) {
                WkVal pair[2] = { data.as.list->items[i], other.as.list->items[i] };
                wk_list_push_raw(out.as.list, wk_make_tuple(pair, 2));
            }
            data = out;
        } break;
        }
        s = s->next;
    }
    return data;
}

/* Pipeline method dispatch via WkObj */
static WkVal _wkp_map(WkVal *args, int argc, WkFunc *fn) { return _pipe_append(fn->captures[0].val, WK_PIPE_MAP, argc>0?args[0]:wk_none()); }
static WkVal _wkp_filter(WkVal *args, int argc, WkFunc *fn) { return _pipe_append(fn->captures[0].val, WK_PIPE_FILTER, argc>0?args[0]:wk_none()); }
static WkVal _wkp_batch(WkVal *args, int argc, WkFunc *fn) { return _pipe_append(fn->captures[0].val, WK_PIPE_BATCH, argc>0?args[0]:wk_int(1)); }
static WkVal _wkp_flatten(WkVal *args, int argc, WkFunc *fn) { (void)args;(void)argc; return _pipe_append(fn->captures[0].val, WK_PIPE_FLATTEN, wk_none()); }
static WkVal _wkp_take(WkVal *args, int argc, WkFunc *fn) { return _pipe_append(fn->captures[0].val, WK_PIPE_TAKE, argc>0?args[0]:wk_int(0)); }
static WkVal _wkp_skip(WkVal *args, int argc, WkFunc *fn) { return _pipe_append(fn->captures[0].val, WK_PIPE_SKIP, argc>0?args[0]:wk_int(0)); }
static WkVal _wkp_shuffle(WkVal *args, int argc, WkFunc *fn) { (void)args;(void)argc; return _pipe_append(fn->captures[0].val, WK_PIPE_SHUFFLE, wk_none()); }
static WkVal _wkp_zip(WkVal *args, int argc, WkFunc *fn) { return _pipe_append(fn->captures[0].val, WK_PIPE_ZIP, argc>0?args[0]:wk_make_list()); }
static WkVal _wkp_collect(WkVal *args, int argc, WkFunc *fn) { (void)args;(void)argc; return _pipe_collect(fn->captures[0].val); }
static WkVal _wkp_count(WkVal *args, int argc, WkFunc *fn) {
    (void)args; (void)argc;
    WkVal lst = _pipe_collect(fn->captures[0].val);
    return wk_int((int64_t)lst.as.list->len);
}
static WkVal _wkp_reduce(WkVal *args, int argc, WkFunc *fn) {
    WkVal lst = _pipe_collect(fn->captures[0].val);
    if (argc < 1) return wk_none();
    WkVal reducer = args[0];
    WkVal acc = (argc > 1) ? args[1] : (lst.as.list->len > 0 ? lst.as.list->items[0] : wk_none());
    size_t start = (argc > 1) ? 0 : 1;
    for (size_t i = start; i < lst.as.list->len; i++)
        acc = wk_call2(reducer, acc, lst.as.list->items[i]);
    return acc;
}
static WkVal _wkp_forEach(WkVal *args, int argc, WkFunc *fn) {
    WkVal lst = _pipe_collect(fn->captures[0].val);
    if (argc < 1) return wk_none();
    for (size_t i = 0; i < lst.as.list->len; i++) wk_call1(args[0], lst.as.list->items[i]);
    return wk_none();
}

static WkVal _wk_pipeline_from(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    WkVal source = (argc > 0) ? args[0] : wk_make_list();
    return _pipe_new(source, NULL);
}

/* Pipeline object method dispatch — hook into wk_member_get for pipeline objects */
/* This is done by checking cls == &_wk_cls_pipeline in wk_member_get's WK_OBJ case */

/* ═══════════════════════════════════════════════════════════════════════════
   MODEL SERIALIZATION + PROVENANCE METADATA
   Binary format: WKTM v1
   ═══════════════════════════════════════════════════════════════════════════ */

/* Minimal JSON encoder for metadata (strings, ints, floats, bools, maps) */
static char *_wk_json_encode(WkVal v) {
    if (v.tag == WK_NONE) return strdup("null");
    if (v.tag == WK_BOOL) return strdup(v.as.i ? "true" : "false");
    if (v.tag == WK_INT) {
        char buf[32]; snprintf(buf, sizeof(buf), "%lld", (long long)v.as.i);
        return strdup(buf);
    }
    if (v.tag == WK_FLOAT) {
        char buf[32]; snprintf(buf, sizeof(buf), "%.15g", v.as.f);
        return strdup(buf);
    }
    if (v.tag == WK_STR) {
        size_t sl = v.as.str->len;
        char *out = (char*)wk_malloc(sl * 2 + 3);
        size_t p = 0;
        out[p++] = '"';
        for (size_t i = 0; i < sl; i++) {
            char c = v.as.str->data[i];
            if (c == '"' || c == '\\') out[p++] = '\\';
            out[p++] = c;
        }
        out[p++] = '"'; out[p] = '\0';
        return out;
    }
    if (v.tag == WK_MAP) {
        char *out = strdup("{"); size_t olen = 1, ocap = 128;
        out = (char*)wk_realloc(out, ocap);
        int first = 1;
        for (size_t i = 0; i < v.as.map->cap; i++) {
            if (!v.as.map->buckets[i].used) continue;
            char *k = _wk_json_encode(v.as.map->buckets[i].key);
            char *val = _wk_json_encode(v.as.map->buckets[i].val);
            size_t need = strlen(k) + strlen(val) + 4;
            if (!first) need += 1;
            while (olen + need > ocap) { ocap *= 2; out = (char*)wk_realloc(out, ocap); }
            if (!first) out[olen++] = ',';
            first = 0;
            size_t kl = strlen(k); memcpy(out + olen, k, kl); olen += kl; free(k);
            out[olen++] = ':';
            size_t vl = strlen(val); memcpy(out + olen, val, vl); olen += vl; free(val);
        }
        while (olen + 2 > ocap) { ocap += 4; out = (char*)wk_realloc(out, ocap); }
        out[olen++] = '}'; out[olen] = '\0';
        return out;
    }
    return strdup("null");
}

/* Minimal JSON decoder */
static WkVal _wk_json_parse_value(const char **p, const char *end);

static void _wk_json_skip_ws(const char **p, const char *end) {
    while (*p < end && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r')) (*p)++;
}

static WkVal _wk_json_parse_string(const char **p, const char *end) {
    if (**p != '"') return wk_make_strz("");
    (*p)++;
    const char *start = *p;
    char *buf = (char*)wk_malloc(end - start + 1);
    size_t len = 0;
    while (*p < end && **p != '"') {
        if (**p == '\\' && *p + 1 < end) {
            (*p)++;
            switch (**p) {
                case 'n': buf[len++] = '\n'; break;
                case 't': buf[len++] = '\t'; break;
                case 'r': buf[len++] = '\r'; break;
                default: buf[len++] = **p; break;
            }
        } else {
            buf[len++] = **p;
        }
        (*p)++;
    }
    if (*p < end) (*p)++; /* skip closing quote */
    buf[len] = '\0';
    WkVal r = wk_make_str(buf, len);
    free(buf);
    return r;
}

static WkVal _wk_json_parse_value(const char **p, const char *end) {
    _wk_json_skip_ws(p, end);
    if (*p >= end) return wk_none();
    if (**p == '"') return _wk_json_parse_string(p, end);
    if (**p == '{') {
        (*p)++;
        WkVal map = wk_make_map();
        _wk_json_skip_ws(p, end);
        if (*p < end && **p == '}') { (*p)++; return map; }
        while (*p < end) {
            _wk_json_skip_ws(p, end);
            WkVal key = _wk_json_parse_string(p, end);
            _wk_json_skip_ws(p, end);
            if (*p < end && **p == ':') (*p)++;
            WkVal val = _wk_json_parse_value(p, end);
            wk_map_set_key(map, key, val);
            _wk_json_skip_ws(p, end);
            if (*p < end && **p == ',') (*p)++;
            else break;
        }
        _wk_json_skip_ws(p, end);
        if (*p < end && **p == '}') (*p)++;
        return map;
    }
    if (**p == '[') {
        (*p)++;
        WkVal lst = wk_make_list();
        _wk_json_skip_ws(p, end);
        if (*p < end && **p == ']') { (*p)++; return lst; }
        while (*p < end) {
            WkVal val = _wk_json_parse_value(p, end);
            wk_list_push_raw(lst.as.list, val);
            _wk_json_skip_ws(p, end);
            if (*p < end && **p == ',') (*p)++;
            else break;
        }
        _wk_json_skip_ws(p, end);
        if (*p < end && **p == ']') (*p)++;
        return lst;
    }
    if (strncmp(*p, "true", 4) == 0) { *p += 4; return wk_bool(1); }
    if (strncmp(*p, "false", 5) == 0) { *p += 5; return wk_bool(0); }
    if (strncmp(*p, "null", 4) == 0) { *p += 4; return wk_none(); }
    /* number */
    char *numend;
    double dv = strtod(*p, &numend);
    if (numend > *p) {
        int is_int = 1;
        for (const char *c = *p; c < numend; c++)
            if (*c == '.' || *c == 'e' || *c == 'E') { is_int = 0; break; }
        *p = numend;
        return is_int ? wk_int((int64_t)dv) : wk_float(dv);
    }
    return wk_none();
}

static WkVal _wk_model_save(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 2 || args[0].tag != WK_MAP || args[1].tag != WK_STR)
        wk_panic("model.save(model_map, path) requires a map and a string path");
    char *path = wk_to_cstr(args[1]);
    FILE *f = fopen(path, "wb");
    free(path);
    if (!f) return wk_make_err(wk_make_strz("cannot open file for writing"));

    fwrite("WKTM", 1, 4, f);
    uint8_t ver = 1; fwrite(&ver, 1, 1, f);

    /* Count tensors */
    uint32_t n = 0;
    WkMap *m = args[0].as.map;
    for (size_t i = 0; i < m->cap; i++)
        if (m->buckets[i].used && m->buckets[i].val.tag == WK_TENSOR) n++;
    fwrite(&n, 4, 1, f);

    /* Write each tensor */
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->buckets[i].used || m->buckets[i].val.tag != WK_TENSOR) continue;
        char *name = wk_to_cstr(m->buckets[i].key);
        uint32_t nlen = (uint32_t)strlen(name);
        fwrite(&nlen, 4, 1, f);
        fwrite(name, 1, nlen, f);
        free(name);
        WkTensor *t = m->buckets[i].val.as.tensor;
        uint8_t ndim = (uint8_t)t->ndim;
        fwrite(&ndim, 1, 1, f);
        fwrite(t->shape, 8, ndim, f);
        fwrite(t->data, 8, t->len, f);
    }

    /* Metadata (JSON) */
    WkVal meta_key = wk_make_strz("__meta__");
    WkVal meta = wk_map_get_key(args[0], meta_key);
    if (meta.tag == WK_MAP) {
        char *json = _wk_json_encode(meta);
        uint32_t jlen = (uint32_t)strlen(json);
        fwrite(&jlen, 4, 1, f);
        fwrite(json, 1, jlen, f);
        free(json);
    } else {
        uint32_t zero = 0;
        fwrite(&zero, 4, 1, f);
    }

    fclose(f);
    return wk_make_ok(wk_none());
}

static WkVal _wk_model_load(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 1 || args[0].tag != WK_STR) wk_panic("model.load(path) requires a string path");
    char *path = wk_to_cstr(args[0]);
    FILE *f = fopen(path, "rb");
    free(path);
    if (!f) return wk_make_err(wk_make_strz("cannot open model file"));

    char magic[4]; FREAD(magic, 1, 4, f);
    if (memcmp(magic, "WKTM", 4) != 0) { fclose(f); return wk_make_err(wk_make_strz("not a WKTM file")); }
    uint8_t ver; FREAD(&ver, 1, 1, f);
    uint32_t n_tensors; FREAD(&n_tensors, 4, 1, f);

    WkVal result = wk_make_map();
    for (uint32_t ti = 0; ti < n_tensors; ti++) {
        uint32_t nlen; FREAD(&nlen, 4, 1, f);
        char *name = (char*)wk_malloc(nlen + 1);
        FREAD(name, 1, nlen, f); name[nlen] = '\0';
        uint8_t ndim; FREAD(&ndim, 1, 1, f);
        int64_t shape[8]; FREAD(shape, 8, ndim, f);
        int64_t total = 1; for (int d = 0; d < ndim; d++) total *= shape[d];
        double *data = (double*)wk_malloc(total * sizeof(double));
        FREAD(data, 8, total, f);
        WkVal tensor = wk_make_tensor(ndim, shape, data, 1);
        wk_map_set_key(result, wk_make_strz(name), tensor);
        free(name);
    }

    /* Read metadata */
    uint32_t meta_len = 0;
    if (fread(&meta_len, 4, 1, f) == 1 && meta_len > 0) {
        char *json = (char*)wk_malloc(meta_len + 1);
        FREAD(json, 1, meta_len, f); json[meta_len] = '\0';
        const char *p = json;
        WkVal meta = _wk_json_parse_value(&p, json + meta_len);
        wk_map_set_key(result, wk_make_strz("__meta__"), meta);
        free(json);
    }

    fclose(f);
    return result;
}

static WkVal _wk_model_meta(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 1 || args[0].tag != WK_MAP) return wk_make_map();
    WkVal meta_key = wk_make_strz("__meta__");
    WkVal meta = wk_map_get_key(args[0], meta_key);
    if (meta.tag != WK_MAP) return wk_none();
    /* If a second arg (key) is provided, return that specific meta value */
    if (argc >= 2) return wk_map_get_key(meta, args[1]);
    return meta;
}

static WkVal _wk_model_setMeta(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 3 || args[0].tag != WK_MAP) return wk_none();
    WkVal meta_key = wk_make_strz("__meta__");
    WkVal meta = wk_map_get_key(args[0], meta_key);
    if (meta.tag != WK_MAP) { meta = wk_make_map(); wk_map_set_key(args[0], meta_key, meta); }
    wk_map_set_key(meta, args[1], args[2]);
    return wk_none();
}

/* ═══════════════════════════════════════════════════════════════════════════
   ONNX / GGUF MODEL LOADING
   Mitigation: ONNX (Linux Foundation), GGUF (llama.cpp) — both open formats.
   ═══════════════════════════════════════════════════════════════════════════ */

/* GGUF loader — parse the binary header format */
static void _gguf_skip_string(FILE *f) {
    uint64_t len; FREAD(&len, 8, 1, f);
    fseek(f, (long)len, SEEK_CUR);
}

static char *_gguf_read_string(FILE *f) {
    uint64_t len; FREAD(&len, 8, 1, f);
    char *s = (char*)wk_malloc(len + 1);
    FREAD(s, 1, len, f); s[len] = '\0';
    return s;
}

static void _gguf_skip_kv_value(FILE *f, uint32_t vtype) {
    /* GGUF value types: 0=uint8,1=int8,2=uint16,...,6=float32,7=bool,8=string,9=array,10=uint64,11=int64,12=float64 */
    switch (vtype) {
        case 0: case 1: case 7: fseek(f, 1, SEEK_CUR); break;
        case 2: case 3: fseek(f, 2, SEEK_CUR); break;
        case 4: case 5: case 6: fseek(f, 4, SEEK_CUR); break;
        case 10: case 11: case 12: fseek(f, 8, SEEK_CUR); break;
        case 8: _gguf_skip_string(f); break;
        case 9: { /* array: type(u32) + len(u64) + values */
            uint32_t atype; FREAD(&atype, 4, 1, f);
            uint64_t alen; FREAD(&alen, 8, 1, f);
            for (uint64_t i = 0; i < alen; i++) _gguf_skip_kv_value(f, atype);
        } break;
        default: break;
    }
}

static WkVal _wk_model_loadGGUF(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 1 || args[0].tag != WK_STR) return wk_make_err(wk_make_strz("loadGGUF requires a path string"));
    char *path = wk_to_cstr(args[0]);
    FILE *f = fopen(path, "rb");
    free(path);
    if (!f) return wk_make_err(wk_make_strz("cannot open GGUF file"));

    char magic[4]; FREAD(magic, 1, 4, f);
    if (memcmp(magic, "GGUF", 4) != 0) { fclose(f); return wk_make_err(wk_make_strz("not a GGUF file")); }

    uint32_t version; FREAD(&version, 4, 1, f);
    uint64_t n_tensors, n_kv;
    FREAD(&n_tensors, 8, 1, f);
    FREAD(&n_kv, 8, 1, f);

    /* Skip metadata KV pairs */
    for (uint64_t i = 0; i < n_kv; i++) {
        _gguf_skip_string(f); /* key */
        uint32_t vtype; FREAD(&vtype, 4, 1, f);
        _gguf_skip_kv_value(f, vtype);
    }

    /* Read tensor infos */
    typedef struct { char *name; int ndim; int64_t shape[8]; uint32_t dtype; uint64_t offset; } _GGUFTInfo;
    _GGUFTInfo *tinfos = (_GGUFTInfo*)wk_malloc(n_tensors * sizeof(_GGUFTInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfos[i].name = _gguf_read_string(f);
        uint32_t ndim; FREAD(&ndim, 4, 1, f);
        tinfos[i].ndim = ndim;
        for (uint32_t d = 0; d < ndim && d < 8; d++) FREAD(&tinfos[i].shape[d], 8, 1, f);
        FREAD(&tinfos[i].dtype, 4, 1, f);
        FREAD(&tinfos[i].offset, 8, 1, f);
    }

    /* Align to 32 bytes for tensor data */
    long pos = ftell(f);
    long align = ((pos + 31) / 32) * 32;
    fseek(f, align, SEEK_SET);
    long data_start = ftell(f);

    WkVal result = wk_make_map();
    for (uint64_t i = 0; i < n_tensors; i++) {
        int64_t total = 1;
        for (int d = 0; d < tinfos[i].ndim; d++) total *= tinfos[i].shape[d];
        /* Seek to tensor data */
        fseek(f, data_start + (long)tinfos[i].offset, SEEK_SET);
        /* Read as float32 (most common GGUF type) and convert to float64 */
        double *data = (double*)wk_malloc(total * sizeof(double));
        if (tinfos[i].dtype == 0) { /* F32 */
            float *tmp = (float*)wk_malloc(total * sizeof(float));
            FREAD(tmp, 4, total, f);
            for (int64_t j = 0; j < total; j++) data[j] = (double)tmp[j];
            free(tmp);
        } else if (tinfos[i].dtype == 1) { /* F16 — approximate read as zeros */
            memset(data, 0, total * sizeof(double));
        } else {
            memset(data, 0, total * sizeof(double));
        }
        WkVal tensor = wk_make_tensor(tinfos[i].ndim, tinfos[i].shape, data, 1);
        wk_map_set_key(result, wk_make_strz(tinfos[i].name), tensor);
        free(tinfos[i].name);
    }
    free(tinfos);
    fclose(f);
    return wk_make_ok(result);
}

/* Simplified ONNX loader — reads protobuf wire format */
static WkVal _wk_model_loadONNX(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 1 || args[0].tag != WK_STR) return wk_make_err(wk_make_strz("loadONNX requires a path string"));
    char *path = wk_to_cstr(args[0]);
    FILE *f = fopen(path, "rb");
    free(path);
    if (!f) return wk_make_err(wk_make_strz("cannot open ONNX file"));

    /* Get file size */
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t*)wk_malloc(fsize);
    FREAD(buf, 1, fsize, f);
    fclose(f);

    /* Very simplified: scan for float tensor data blocks
       Real ONNX parsing would need full protobuf, but this extracts raw_data fields */
    WkVal result = wk_make_map();
    int tensor_idx = 0;

    /* Look for TensorProto patterns in the protobuf stream */
    /* This is a best-effort parser for the most common case */
    for (long pos = 0; pos < fsize - 8; pos++) {
        /* Skip to significant structures — look for raw_data field tag (field 13, wire type 2 = 0x6A) */
        if (buf[pos] == 0x6A) {
            /* Read varint length */
            long p = pos + 1;
            uint64_t len = 0; int shift = 0;
            while (p < fsize && (buf[p] & 0x80)) { len |= (uint64_t)(buf[p] & 0x7F) << shift; shift += 7; p++; }
            if (p < fsize) { len |= (uint64_t)(buf[p] & 0x7F) << shift; p++; }
            if (len > 0 && len % 4 == 0 && p + (long)len <= fsize && len < (uint64_t)fsize) {
                int64_t n = len / 4; /* float32 elements */
                if (n > 1 && n < 100000000) {
                    double *data = (double*)wk_malloc(n * sizeof(double));
                    float *fdata = (float*)(buf + p);
                    for (int64_t i = 0; i < n; i++) data[i] = (double)fdata[i];
                    int64_t sh[1] = {n};
                    char name[32]; snprintf(name, sizeof(name), "tensor_%d", tensor_idx++);
                    wk_map_set_key(result, wk_make_strz(name), wk_make_tensor(1, sh, data, 1));
                    pos = p + len - 1; /* skip past this data */
                }
            }
        }
    }
    free(buf);

    if (tensor_idx == 0) return wk_make_err(wk_make_strz("no tensor data found in ONNX file"));
    return wk_make_ok(result);
}

/* ═══════════════════════════════════════════════════════════════════════════
   GLOBAL BUILTINS TABLE
   ═══════════════════════════════════════════════════════════════════════════ */

WkVal wk_g_println, wk_g_print, wk_g_eprintln, wk_g_readln;
WkVal wk_g_len, wk_g_str, wk_g_int, wk_g_float, wk_g_bool, wk_g_typeof;
WkVal wk_g_isNone, wk_g_isSome, wk_g_isOk, wk_g_isErr;
WkVal wk_g_sum, wk_g_min, wk_g_max, wk_g_map, wk_g_filter, wk_g_reduce;
WkVal wk_g_sorted, wk_g_reversed, wk_g_any, wk_g_all;
WkVal wk_g_zip, wk_g_enumerate, wk_g_range;
WkVal wk_g_sleep, wk_g_exit, wk_g_assert, wk_g_copy;
WkVal wk_g_chr, wk_g_ord, wk_g_hash, wk_g_repr, wk_g_panic;
WkVal wk_g_math;
WkVal wk_g_tensor, wk_g_ad, wk_g_gpu, wk_g_pipeline, wk_g_model;
WkVal wk_g_fs, wk_g_py, wk_g_jvm, wk_g_node;
int    wk_argc = 0;
char **wk_argv = NULL;

/* ── fs module ─────────────────────────────────────────────────────────── */

static WkVal _wk_fs_read(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 1 || args[0].tag != WK_STR)
        wk_panic("fs.read: expected string path");
    char *path = wk_to_cstr(args[0]);
    FILE *f = fopen(path, "rb");
    if (!f) { free(path); return wk_make_strz(""); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); free(path); wk_panic("fs.read: out of memory"); }
    size_t nr = fread(buf, 1, sz, f);
    buf[nr] = '\0';
    fclose(f);
    free(path);
    WkVal result = wk_make_str(buf, nr);
    free(buf);
    return result;
}

static WkVal _wk_fs_write(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 2 || args[0].tag != WK_STR || args[1].tag != WK_STR)
        wk_panic("fs.write: expected (path, content) strings");
    char *path = wk_to_cstr(args[0]);
    FILE *f = fopen(path, "wb");
    if (!f) { free(path); wk_panic("fs.write: cannot open '%s'", path); }
    fwrite(args[1].as.str->data, 1, args[1].as.str->len, f);
    fclose(f);
    free(path);
    return wk_none();
}

static WkVal _wk_fs_exists(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 1 || args[0].tag != WK_STR)
        wk_panic("fs.exists: expected string path");
    char *path = wk_to_cstr(args[0]);
    FILE *f = fopen(path, "r");
    free(path);
    if (f) { fclose(f); return wk_bool(1); }
    return wk_bool(0);
}

static WkVal _wk_fs_args(WkVal *args, int argc, WkFunc *fn) {
    (void)args; (void)argc; (void)fn;
    WkVal lst = wk_make_list();
    for (int i = 0; i < wk_argc; i++)
        wk_list_push_raw(lst.as.list, wk_make_strz(wk_argv[i]));
    return lst;
}

/* ── Foreign Module FFI (dlopen-based) ─────────────────────────────────── */

#ifndef _WIN32
#include <dlfcn.h>

/* ── Python FFI ────────────────────────────────────────────────────────── */

static void *_py_lib = NULL;

/* Function pointer types for CPython API */
typedef void   (*PFN_Py_Initialize)(void);
typedef void*  (*PFN_PyImport_ImportModule)(const char *);
typedef void*  (*PFN_PyObject_GetAttrString)(void *, const char *);
typedef void*  (*PFN_PyObject_CallObject)(void *, void *);
typedef void*  (*PFN_PyObject_Str)(void *);
typedef const char* (*PFN_PyUnicode_AsUTF8)(void *);
typedef long   (*PFN_PyLong_AsLong)(void *);
typedef double (*PFN_PyFloat_AsDouble)(void *);
typedef void*  (*PFN_PyTuple_New)(int);
typedef int    (*PFN_PyTuple_SetItem)(void *, int, void *);
typedef void*  (*PFN_PyUnicode_FromString)(const char *);
typedef void*  (*PFN_PyLong_FromLong)(long);
typedef void*  (*PFN_PyFloat_FromDouble)(double);
typedef int    (*PFN_PyLong_Check)(void *);
typedef int    (*PFN_PyFloat_Check)(void *);
typedef int    (*PFN_PyUnicode_Check)(void *);
typedef void*  (*PFN_PyObject_Call)(void *, void *, void *);
typedef void   (*PFN_Py_DecRef)(void *);

static struct {
    PFN_Py_Initialize         Initialize;
    PFN_PyImport_ImportModule ImportModule;
    PFN_PyObject_GetAttrString GetAttrString;
    PFN_PyObject_CallObject   CallObject;
    PFN_PyObject_Str          Str;
    PFN_PyUnicode_AsUTF8      AsUTF8;
    PFN_PyLong_AsLong         LongAsLong;
    PFN_PyFloat_AsDouble      FloatAsDouble;
    PFN_PyTuple_New           TupleNew;
    PFN_PyTuple_SetItem       TupleSetItem;
    PFN_PyUnicode_FromString  UnicodeFromString;
    PFN_PyLong_FromLong       LongFromLong;
    PFN_PyFloat_FromDouble    FloatFromDouble;
    PFN_Py_DecRef             DecRef;
    int ready;
} _pyapi = {0};

static int _wk_py_ensure(void) {
    if (_pyapi.ready) return 1;
    if (_py_lib) return 0; /* already tried, failed */
    /* Try common library names */
    const char *names[] = {
        "libpython3.so", "libpython3.12.so", "libpython3.11.so",
        "libpython3.10.so", "libpython3.9.so", "libpython3.8.so",
        "libpython3.12.so.1", "libpython3.11.so.1", "libpython3.10.so.1",
        NULL
    };
    for (int i = 0; names[i]; i++) {
        _py_lib = dlopen(names[i], RTLD_LAZY | RTLD_GLOBAL);
        if (_py_lib) break;
    }
    if (!_py_lib) { _py_lib = (void*)1; return 0; } /* sentinel: tried, failed */
    /* Resolve symbols */
    #define PY_SYM(field, sym) _pyapi.field = (void*)dlsym(_py_lib, #sym); if(!_pyapi.field) return 0
    PY_SYM(Initialize,       Py_Initialize);
    PY_SYM(ImportModule,     PyImport_ImportModule);
    PY_SYM(GetAttrString,    PyObject_GetAttrString);
    PY_SYM(CallObject,       PyObject_CallObject);
    PY_SYM(Str,              PyObject_Str);
    PY_SYM(AsUTF8,           PyUnicode_AsUTF8);
    PY_SYM(LongAsLong,       PyLong_AsLong);
    PY_SYM(FloatAsDouble,    PyFloat_AsDouble);
    PY_SYM(TupleNew,         PyTuple_New);
    PY_SYM(TupleSetItem,     PyTuple_SetItem);
    PY_SYM(UnicodeFromString, PyUnicode_FromString);
    PY_SYM(LongFromLong,     PyLong_FromLong);
    PY_SYM(FloatFromDouble,  PyFloat_FromDouble);
    PY_SYM(DecRef,           Py_DecRef);
    #undef PY_SYM
    _pyapi.Initialize();
    _pyapi.ready = 1;
    return 1;
}

/* Convert a PyObject* to WkVal — best-effort */
static WkVal _wk_py_to_wkval(void *pyobj) {
    if (!pyobj) return wk_none();
    /* Try string first (via PyObject_Str) */
    void *s = _pyapi.Str(pyobj);
    if (s) {
        const char *utf8 = _pyapi.AsUTF8(s);
        if (utf8) {
            WkVal r = wk_make_strz(utf8);
            _pyapi.DecRef(s);
            return r;
        }
        _pyapi.DecRef(s);
    }
    return wk_none();
}

/* Convert WkVal → PyObject* */
static void *_wk_py_from_wkval(WkVal v) {
    switch (v.tag) {
        case WK_INT:   return _pyapi.LongFromLong((long)v.as.i);
        case WK_FLOAT: return _pyapi.FloatFromDouble(v.as.f);
        case WK_STR:   return _pyapi.UnicodeFromString(v.as.str->data);
        default: return _pyapi.UnicodeFromString("");
    }
}

static WkVal _wk_py_import(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (!_wk_py_ensure())
        wk_panic("py.import: Python not available (libpython3.so not found)");
    if (argc < 1 || args[0].tag != WK_STR)
        wk_panic("py.import: expected module name string");
    void *mod = _pyapi.ImportModule(args[0].as.str->data);
    if (!mod) wk_panic("py.import: cannot import '%s'", args[0].as.str->data);
    /* Store as int (pointer cast) — opaque handle */
    WkVal result = wk_make_map();
    wk_map_set_key(result, wk_make_strz("__pyobj__"), wk_int((int64_t)(intptr_t)mod));
    wk_map_set_key(result, wk_make_strz("__type__"), wk_make_strz("py_module"));
    return result;
}

static WkVal _wk_py_call(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (!_pyapi.ready) wk_panic("py.call: Python not initialized");
    if (argc < 2) wk_panic("py.call: expected (module, method, ...args)");
    /* args[0] = module map with __pyobj__, args[1] = method name string */
    WkVal pyobjv = wk_map_get_key(args[0], wk_make_strz("__pyobj__"));
    if (pyobjv.tag != WK_INT) wk_panic("py.call: invalid module object");
    void *pymod = (void *)(intptr_t)pyobjv.as.i;
    if (args[1].tag != WK_STR) wk_panic("py.call: method name must be string");
    void *method = _pyapi.GetAttrString(pymod, args[1].as.str->data);
    if (!method) wk_panic("py.call: no attribute '%s'", args[1].as.str->data);
    /* Build args tuple */
    int nargs = argc - 2;
    void *pytuple = _pyapi.TupleNew(nargs);
    for (int i = 0; i < nargs; i++)
        _pyapi.TupleSetItem(pytuple, i, _wk_py_from_wkval(args[i + 2]));
    void *result = _pyapi.CallObject(method, pytuple);
    _pyapi.DecRef(pytuple);
    WkVal wkresult = _wk_py_to_wkval(result);
    if (result) _pyapi.DecRef(result);
    return wkresult;
}

static WkVal _wk_py_eval(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (!_wk_py_ensure())
        wk_panic("py.eval: Python not available");
    if (argc < 1 || args[0].tag != WK_STR)
        wk_panic("py.eval: expected string expression");
    /* Use PyRun_SimpleString — we'd need to resolve this separately */
    /* For now, import builtins and call eval() */
    void *builtins = _pyapi.ImportModule("builtins");
    if (!builtins) wk_panic("py.eval: cannot import builtins");
    void *evalfn = _pyapi.GetAttrString(builtins, "eval");
    if (!evalfn) wk_panic("py.eval: cannot find eval");
    void *pytuple = _pyapi.TupleNew(1);
    _pyapi.TupleSetItem(pytuple, 0, _pyapi.UnicodeFromString(args[0].as.str->data));
    void *result = _pyapi.CallObject(evalfn, pytuple);
    _pyapi.DecRef(pytuple);
    WkVal wkresult = _wk_py_to_wkval(result);
    if (result) _pyapi.DecRef(result);
    return wkresult;
}

static WkVal _wk_py_available(WkVal *args, int argc, WkFunc *fn) {
    (void)args; (void)argc; (void)fn;
    return wk_bool(_wk_py_ensure());
}

/* ── JVM FFI ───────────────────────────────────────────────────────────── */

static void *_jvm_lib = NULL;
static int _jvm_tried = 0;

static int _wk_jvm_ensure(void) {
    if (_jvm_tried) return _jvm_lib != NULL && _jvm_lib != (void*)1;
    _jvm_tried = 1;
    /* Try to find libjvm.so */
    const char *java_home = getenv("JAVA_HOME");
    char path[1024];
    if (java_home) {
        snprintf(path, sizeof(path), "%s/lib/server/libjvm.so", java_home);
        _jvm_lib = dlopen(path, RTLD_LAZY);
        if (!_jvm_lib) {
            snprintf(path, sizeof(path), "%s/lib/libjvm.so", java_home);
            _jvm_lib = dlopen(path, RTLD_LAZY);
        }
    }
    if (!_jvm_lib) _jvm_lib = dlopen("libjvm.so", RTLD_LAZY);
    if (!_jvm_lib) { _jvm_lib = (void*)1; return 0; }
    return 1;
}

static WkVal _wk_jvm_available(WkVal *args, int argc, WkFunc *fn) {
    (void)args; (void)argc; (void)fn;
    return wk_bool(_wk_jvm_ensure());
}

static WkVal _wk_jvm_import(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (!_wk_jvm_ensure())
        wk_panic("jvm.import: JVM not available (libjvm.so not found — set JAVA_HOME)");
    if (argc < 1 || args[0].tag != WK_STR)
        wk_panic("jvm.import: expected class name string");
    /* Stub: JNI integration is complex — return a placeholder */
    WkVal result = wk_make_map();
    wk_map_set_key(result, wk_make_strz("__type__"), wk_make_strz("jvm_class"));
    wk_map_set_key(result, wk_make_strz("class"), args[0]);
    return result;
}

static WkVal _wk_jvm_call(WkVal *args, int argc, WkFunc *fn) {
    (void)fn; (void)args; (void)argc;
    wk_panic("jvm.call: JVM call interface not yet fully implemented — use py or node for now");
    return wk_none();
}

/* ── Node.js FFI (subprocess-based) ───────────────────────────────────── */

static WkVal _wk_node_available(WkVal *args, int argc, WkFunc *fn) {
    (void)args; (void)argc; (void)fn;
    /* Check if node is in PATH */
    int ok = (system("which node > /dev/null 2>&1") == 0);
    return wk_bool(ok);
}

static WkVal _wk_node_require(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 1 || args[0].tag != WK_STR)
        wk_panic("node.require: expected module name string");
    WkVal result = wk_make_map();
    wk_map_set_key(result, wk_make_strz("__type__"), wk_make_strz("node_module"));
    wk_map_set_key(result, wk_make_strz("module"), args[0]);
    return result;
}

static WkVal _wk_node_eval(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 1 || args[0].tag != WK_STR)
        wk_panic("node.eval: expected string expression");
    /* Build a node -e command and capture output */
    char *expr = wk_to_cstr(args[0]);
    size_t cmdlen = strlen(expr) + 64;
    char *cmd = (char *)malloc(cmdlen);
    snprintf(cmd, cmdlen, "node -e \"process.stdout.write(String(%s))\"", expr);
    WkVal cmdv = wk_make_strz(cmd);
    free(cmd);
    free(expr);
    return wk_shell_exec(cmdv);
}

static WkVal _wk_node_call(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if (argc < 2)
        wk_panic("node.call: expected (module, method, ...args)");
    /* Build JSON-formatted node one-liner */
    if (args[0].tag != WK_MAP || args[1].tag != WK_STR)
        wk_panic("node.call: expected (module_map, method_str, ...)");
    WkVal modname = wk_map_get_key(args[0], wk_make_strz("module"));
    if (modname.tag != WK_STR) wk_panic("node.call: invalid module");
    char *mod = wk_to_cstr(modname);
    char *method = wk_to_cstr(args[1]);
    /* Simple: require(mod).method(args...) — serialized via JSON */
    size_t cmdlen = strlen(mod) + strlen(method) + 256;
    char *cmd = (char *)malloc(cmdlen);
    snprintf(cmd, cmdlen,
        "node -e \"const m=require('%s');process.stdout.write(String(m.%s()))\"",
        mod, method);
    WkVal cmdv = wk_make_strz(cmd);
    free(cmd); free(mod); free(method);
    return wk_shell_exec(cmdv);
}

#else
/* Windows stubs */
static WkVal _wk_py_available(WkVal *a,int c,WkFunc *f){(void)a;(void)c;(void)f;return wk_bool(0);}
static WkVal _wk_py_import(WkVal *a,int c,WkFunc *f){(void)a;(void)c;(void)f;wk_panic("py: not available on Windows yet");return wk_none();}
static WkVal _wk_py_call(WkVal *a,int c,WkFunc *f){(void)a;(void)c;(void)f;wk_panic("py: not available on Windows yet");return wk_none();}
static WkVal _wk_py_eval(WkVal *a,int c,WkFunc *f){(void)a;(void)c;(void)f;wk_panic("py: not available on Windows yet");return wk_none();}
static WkVal _wk_jvm_available(WkVal *a,int c,WkFunc *f){(void)a;(void)c;(void)f;return wk_bool(0);}
static WkVal _wk_jvm_import(WkVal *a,int c,WkFunc *f){(void)a;(void)c;(void)f;wk_panic("jvm: not available on Windows yet");return wk_none();}
static WkVal _wk_jvm_call(WkVal *a,int c,WkFunc *f){(void)a;(void)c;(void)f;wk_panic("jvm: not available on Windows yet");return wk_none();}
static WkVal _wk_node_available(WkVal *a,int c,WkFunc *f){(void)a;(void)c;(void)f;return wk_bool(0);}
static WkVal _wk_node_require(WkVal *a,int c,WkFunc *f){(void)a;(void)c;(void)f;wk_panic("node: not available on Windows yet");return wk_none();}
static WkVal _wk_node_eval(WkVal *a,int c,WkFunc *f){(void)a;(void)c;(void)f;wk_panic("node: not available on Windows yet");return wk_none();}
static WkVal _wk_node_call(WkVal *a,int c,WkFunc *f){(void)a;(void)c;(void)f;wk_panic("node: not available on Windows yet");return wk_none();}
#endif /* !_WIN32 */

#define REGISTER_BUILTIN(slot, cfn, nm, np) \
    slot = wk_make_func(nm, cfn, NULL, 0, NULL, np);

/* math module wrappers */
#define _MATH_WRAP1(fname, cfn) \
    static WkVal _wk_math_##fname(WkVal *a,int c,WkFunc *f){ \
        (void)f; double x=(c>0&&a[0].tag==WK_INT)?(double)a[0].as.i: \
                          (c>0&&a[0].tag==WK_FLOAT)?a[0].as.f:0.0; \
        return wk_float(cfn(x)); }
_MATH_WRAP1(sqrt,  sqrt)
_MATH_WRAP1(floor, floor)
_MATH_WRAP1(ceil,  ceil)
_MATH_WRAP1(sin,   sin)
_MATH_WRAP1(cos,   cos)
_MATH_WRAP1(tan,   tan)
_MATH_WRAP1(asin,  asin)
_MATH_WRAP1(acos,  acos)
_MATH_WRAP1(atan,  atan)
_MATH_WRAP1(exp,   exp)
_MATH_WRAP1(log,   log)
_MATH_WRAP1(log2,  log2)
_MATH_WRAP1(log10, log10)
_MATH_WRAP1(fabs,  fabs)
static WkVal _wk_math_atan2(WkVal *a,int c,WkFunc *f){
    (void)f;
    double y=(c>0&&a[0].tag==WK_INT)?(double)a[0].as.i:(c>0&&a[0].tag==WK_FLOAT)?a[0].as.f:0.0;
    double x=(c>1&&a[1].tag==WK_INT)?(double)a[1].as.i:(c>1&&a[1].tag==WK_FLOAT)?a[1].as.f:0.0;
    return wk_float(atan2(y,x));
}
static WkVal _wk_math_pow(WkVal *a,int c,WkFunc *f){
    (void)f;
    double base=(c>0&&a[0].tag==WK_INT)?(double)a[0].as.i:(c>0&&a[0].tag==WK_FLOAT)?a[0].as.f:0.0;
    double exp_=(c>1&&a[1].tag==WK_INT)?(double)a[1].as.i:(c>1&&a[1].tag==WK_FLOAT)?a[1].as.f:0.0;
    return wk_float(pow(base,exp_));
}
#define _MATH_WRAP_REGISTER(slot, fname, nm, np) \
    wk_member_set(wk_g_math, nm, wk_make_func(nm, _wk_math_##fname, NULL, 0, NULL, np));

static WkVal _wk_panic_wrapper(WkVal *args, int argc, WkFunc *fn) {
    (void)fn;
    if(argc>0){ char *s=wk_to_cstr(args[0]); wk_panic("%s",s); }
    wk_panic("panic");
}

void wk_runtime_init(void) {
    REGISTER_BUILTIN(wk_g_println,   wk_builtin_println,   "println",   -1)
    REGISTER_BUILTIN(wk_g_print,     wk_builtin_print,     "print",     -1)
    REGISTER_BUILTIN(wk_g_eprintln,  wk_builtin_eprintln,  "eprintln",  -1)
    REGISTER_BUILTIN(wk_g_readln,    wk_builtin_readln,    "readln",    -1)
    REGISTER_BUILTIN(wk_g_len,       wk_builtin_len,       "len",        1)
    REGISTER_BUILTIN(wk_g_str,       wk_builtin_str,       "str",        1)
    REGISTER_BUILTIN(wk_g_int,       wk_builtin_int,       "int",        1)
    REGISTER_BUILTIN(wk_g_float,     wk_builtin_float,     "float",      1)
    REGISTER_BUILTIN(wk_g_bool,      wk_builtin_bool,      "bool",       1)
    REGISTER_BUILTIN(wk_g_typeof,    wk_builtin_typeof,    "typeof",     1)
    REGISTER_BUILTIN(wk_g_isNone,    wk_builtin_isNone,    "isNone",     1)
    REGISTER_BUILTIN(wk_g_isSome,    wk_builtin_isSome,    "isSome",     1)
    REGISTER_BUILTIN(wk_g_isOk,      wk_builtin_isOk,      "isOk",       1)
    REGISTER_BUILTIN(wk_g_isErr,     wk_builtin_isErr,      "isErr",      1)
    REGISTER_BUILTIN(wk_g_sum,       wk_builtin_sum,       "sum",        1)
    REGISTER_BUILTIN(wk_g_min,       wk_builtin_min,       "min",       -1)
    REGISTER_BUILTIN(wk_g_max,       wk_builtin_max,       "max",       -1)
    REGISTER_BUILTIN(wk_g_map,       wk_builtin_map_fn,    "map",        2)
    REGISTER_BUILTIN(wk_g_filter,    wk_builtin_filter,    "filter",     2)
    REGISTER_BUILTIN(wk_g_reduce,    wk_builtin_reduce,    "reduce",    -1)
    REGISTER_BUILTIN(wk_g_sorted,    wk_builtin_sorted,    "sorted",     1)
    REGISTER_BUILTIN(wk_g_reversed,  wk_builtin_reversed,  "reversed",   1)
    REGISTER_BUILTIN(wk_g_any,       wk_builtin_any,       "any",       -1)
    REGISTER_BUILTIN(wk_g_all,       wk_builtin_all,       "all",       -1)
    REGISTER_BUILTIN(wk_g_zip,       wk_builtin_zip,       "zip",        2)
    REGISTER_BUILTIN(wk_g_enumerate, wk_builtin_enumerate, "enumerate",  1)
    REGISTER_BUILTIN(wk_g_range,     wk_builtin_range,     "range",      2)
    REGISTER_BUILTIN(wk_g_sleep,     wk_builtin_sleep,     "sleep",      1)
    REGISTER_BUILTIN(wk_g_exit,      wk_builtin_exit,      "exit",       1)
    REGISTER_BUILTIN(wk_g_assert,    wk_builtin_assert,    "assert",    -1)
    REGISTER_BUILTIN(wk_g_copy,      wk_builtin_copy,      "copy",       1)
    REGISTER_BUILTIN(wk_g_chr,       wk_builtin_chr,       "chr",        1)
    REGISTER_BUILTIN(wk_g_ord,       wk_builtin_ord,       "ord",        1)
    REGISTER_BUILTIN(wk_g_hash,      wk_builtin_hash,      "hash",       1)
    REGISTER_BUILTIN(wk_g_repr,      wk_builtin_repr,      "repr",       1)
    REGISTER_BUILTIN(wk_g_panic,     _wk_panic_wrapper,    "panic",      1)

    /* math module */
    wk_g_math = wk_make_map();
    wk_member_set(wk_g_math, "PI",      wk_float(3.14159265358979323846));
    wk_member_set(wk_g_math, "E",       wk_float(2.71828182845904523536));
    wk_member_set(wk_g_math, "TAU",     wk_float(6.28318530717958647693));
    wk_member_set(wk_g_math, "INF",     wk_float(1.0/0.0));
    wk_member_set(wk_g_math, "NAN",     wk_float(0.0/0.0));
    _MATH_WRAP_REGISTER(sqrt,  sqrt,  "sqrt",  1)
    _MATH_WRAP_REGISTER(floor, floor, "floor", 1)
    _MATH_WRAP_REGISTER(ceil,  ceil,  "ceil",  1)
    _MATH_WRAP_REGISTER(sin,   sin,   "sin",   1)
    _MATH_WRAP_REGISTER(cos,   cos,   "cos",   1)
    _MATH_WRAP_REGISTER(tan,   tan,   "tan",   1)
    _MATH_WRAP_REGISTER(asin,  asin,  "asin",  1)
    _MATH_WRAP_REGISTER(acos,  acos,  "acos",  1)
    _MATH_WRAP_REGISTER(atan,  atan,  "atan",  1)
    _MATH_WRAP_REGISTER(atan2, atan2, "atan2", 2)
    _MATH_WRAP_REGISTER(pow,   pow,   "pow",   2)
    _MATH_WRAP_REGISTER(exp,   exp,   "exp",   1)
    _MATH_WRAP_REGISTER(log,   log,   "log",   1)
    _MATH_WRAP_REGISTER(log2,  log2,  "log2",  1)
    _MATH_WRAP_REGISTER(log10, log10, "log10", 1)
    _MATH_WRAP_REGISTER(fabs,  fabs,  "abs",   1)

    /* ── tensor module ──────────────────────────────────────────────────── */
    wk_g_tensor = wk_make_map();
    wk_member_set(wk_g_tensor, "zeros",    wk_make_func("zeros",    _wk_tensor_zeros,    NULL, 0, NULL, -1));
    wk_member_set(wk_g_tensor, "ones",     wk_make_func("ones",     _wk_tensor_ones,     NULL, 0, NULL, -1));
    wk_member_set(wk_g_tensor, "from",     wk_make_func("from",     _wk_tensor_from,     NULL, 0, NULL, -1));
    wk_member_set(wk_g_tensor, "arange",   wk_make_func("arange",   _wk_tensor_arange,   NULL, 0, NULL, -1));
    wk_member_set(wk_g_tensor, "linspace", wk_make_func("linspace", _wk_tensor_linspace, NULL, 0, NULL, -1));
    wk_member_set(wk_g_tensor, "eye",      wk_make_func("eye",      _wk_tensor_eye,      NULL, 0, NULL, -1));
    wk_member_set(wk_g_tensor, "rand",     wk_make_func("rand",     _wk_tensor_rand,     NULL, 0, NULL, -1));

    /* ── autodiff module ────────────────────────────────────────────────── */
    _wk_cls_dual.name = "Dual";
    _wk_cls_dual.parent = NULL;
    _wk_cls_dual.methods = NULL;
    _wk_cls_dual.nmethods = 0;
    _wk_cls_dual.field_names = _dual_fnames;
    _wk_cls_dual.nfields = 2;

    wk_g_ad = wk_make_map();
    wk_member_set(wk_g_ad, "dual",     wk_make_func("dual",     _wk_ad_dual,  NULL, 0, NULL, -1));
    wk_member_set(wk_g_ad, "grad",     wk_make_func("grad",     _wk_ad_grad,  NULL, 0, NULL,  2));
    wk_member_set(wk_g_ad, "value",    wk_make_func("value",    _wk_ad_value, NULL, 0, NULL,  1));
    wk_member_set(wk_g_ad, "deriv",    wk_make_func("deriv",    _wk_ad_deriv, NULL, 0, NULL,  1));
    wk_member_set(wk_g_ad, "sin",      wk_make_func("sin",      _wk_ad_sin,   NULL, 0, NULL,  1));
    wk_member_set(wk_g_ad, "cos",      wk_make_func("cos",      _wk_ad_cos,   NULL, 0, NULL,  1));
    wk_member_set(wk_g_ad, "exp",      wk_make_func("exp",      _wk_ad_exp,   NULL, 0, NULL,  1));
    wk_member_set(wk_g_ad, "log",      wk_make_func("log",      _wk_ad_log,   NULL, 0, NULL,  1));
    wk_member_set(wk_g_ad, "sqrt",     wk_make_func("sqrt",     _wk_ad_sqrt,  NULL, 0, NULL,  1));
    wk_member_set(wk_g_ad, "pow",      wk_make_func("pow",      _wk_ad_pow,   NULL, 0, NULL,  2));

    /* ── gpu module ─────────────────────────────────────────────────────── */
    _wk_gpu_init();
    wk_g_gpu = wk_make_map();
    wk_member_set(wk_g_gpu, "available", wk_make_func("available", _wk_gpu_available, NULL, 0, NULL, 0));
    wk_member_set(wk_g_gpu, "devices",   wk_make_func("devices",   _wk_gpu_devices,   NULL, 0, NULL, 0));
    wk_member_set(wk_g_gpu, "run",       wk_make_func("run",       _wk_gpu_run,       NULL, 0, NULL, -1));

    /* ── pipeline module ────────────────────────────────────────────────── */
    _wk_cls_pipeline.name = "Pipeline";
    _wk_cls_pipeline.parent = NULL;
    _wk_cls_pipeline.methods = NULL;
    _wk_cls_pipeline.nmethods = 0;
    _wk_cls_pipeline.field_names = _pipe_fnames;
    _wk_cls_pipeline.nfields = 2;

    wk_g_pipeline = wk_make_map();
    wk_member_set(wk_g_pipeline, "from", wk_make_func("from", _wk_pipeline_from, NULL, 0, NULL, 1));

    /* ── model module (serialization + provenance + ONNX/GGUF) ──────── */
    wk_g_model = wk_make_map();
    wk_member_set(wk_g_model, "save",     wk_make_func("save",     _wk_model_save,     NULL, 0, NULL, 2));
    wk_member_set(wk_g_model, "load",     wk_make_func("load",     _wk_model_load,     NULL, 0, NULL, 1));
    wk_member_set(wk_g_model, "meta",     wk_make_func("meta",     _wk_model_meta,     NULL, 0, NULL, 1));
    wk_member_set(wk_g_model, "setMeta",  wk_make_func("setMeta",  _wk_model_setMeta,  NULL, 0, NULL, 3));
    wk_member_set(wk_g_model, "loadGGUF", wk_make_func("loadGGUF", _wk_model_loadGGUF, NULL, 0, NULL, 1));
    wk_member_set(wk_g_model, "loadONNX", wk_make_func("loadONNX", _wk_model_loadONNX, NULL, 0, NULL, 1));

    /* ── fs module ─────────────────────────────────────────────────────── */
    wk_g_fs = wk_make_map();
    wk_member_set(wk_g_fs, "read",   wk_make_func("read",   _wk_fs_read,   NULL, 0, NULL, 1));
    wk_member_set(wk_g_fs, "write",  wk_make_func("write",  _wk_fs_write,  NULL, 0, NULL, 2));
    wk_member_set(wk_g_fs, "exists", wk_make_func("exists", _wk_fs_exists, NULL, 0, NULL, 1));
    wk_member_set(wk_g_fs, "args",   wk_make_func("args",   _wk_fs_args,   NULL, 0, NULL, 0));

    /* ── py module (Python FFI via dlopen) ─────────────────────────────── */
    wk_g_py = wk_make_map();
    wk_member_set(wk_g_py, "available", wk_make_func("available", _wk_py_available, NULL, 0, NULL, 0));
    wk_member_set(wk_g_py, "import",    wk_make_func("import",    _wk_py_import,    NULL, 0, NULL, 1));
    wk_member_set(wk_g_py, "call",      wk_make_func("call",      _wk_py_call,      NULL, 0, NULL, -1));
    wk_member_set(wk_g_py, "eval",      wk_make_func("eval",      _wk_py_eval,      NULL, 0, NULL, 1));

    /* ── jvm module (Java FFI via dlopen) ──────────────────────────────── */
    wk_g_jvm = wk_make_map();
    wk_member_set(wk_g_jvm, "available", wk_make_func("available", _wk_jvm_available, NULL, 0, NULL, 0));
    wk_member_set(wk_g_jvm, "import",    wk_make_func("import",    _wk_jvm_import,    NULL, 0, NULL, 1));
    wk_member_set(wk_g_jvm, "call",      wk_make_func("call",      _wk_jvm_call,      NULL, 0, NULL, -1));

    /* ── node module (Node.js FFI via subprocess) ──────────────────────── */
    wk_g_node = wk_make_map();
    wk_member_set(wk_g_node, "available", wk_make_func("available", _wk_node_available, NULL, 0, NULL, 0));
    wk_member_set(wk_g_node, "require",   wk_make_func("require",   _wk_node_require,   NULL, 0, NULL, 1));
    wk_member_set(wk_g_node, "call",      wk_make_func("call",      _wk_node_call,      NULL, 0, NULL, -1));
    wk_member_set(wk_g_node, "eval",      wk_make_func("eval",      _wk_node_eval,      NULL, 0, NULL, 1));
}

/* ═══════════════════════════════════════════════════════════════════════════
   SHELL EXECUTION
   ═══════════════════════════════════════════════════════════════════════════ */

WkVal wk_shell_exec(WkVal cmd) {
    char *cstr = wk_to_cstr(cmd);
#ifdef _WIN32
    FILE *f = _popen(cstr, "r");
#else
    FILE *f = popen(cstr, "r");
#endif
    free(cstr);
    if (!f) return wk_make_strz("");

    char    chunk[4096];
    char   *out     = NULL;
    size_t  out_len = 0;
    size_t  n;
    while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0) {
        out = (char *)realloc(out, out_len + n + 1);
        if (!out) wk_panic("wk_shell_exec: out of memory");
        memcpy(out + out_len, chunk, n);
        out_len += n;
    }
#ifdef _WIN32
    _pclose(f);
#else
    pclose(f);
#endif

    if (!out) return wk_make_strz("");
    out[out_len] = '\0';
    /* strip single trailing newline (POSIX convention) */
    if (out_len > 0 && out[out_len - 1] == '\n') { out_len--; out[out_len] = '\0'; }
    WkVal result = wk_make_str(out, out_len);
    free(out);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
   GOROUTINES  (detached background threads)
   ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    WkVal  fn;
    WkVal *args;
    int    argc;
} _WkGoArg;

#ifdef _WIN32
static DWORD WINAPI _wk_go_thread(LPVOID p) {
    _WkGoArg *a = (_WkGoArg *)p;
    wk_call(a->fn, a->args, a->argc);
    if (a->args) free(a->args);
    free(a);
    return 0;
}
#else
static void *_wk_go_thread(void *p) {
    _WkGoArg *a = (_WkGoArg *)p;
    wk_call(a->fn, a->args, a->argc);
    if (a->args) free(a->args);
    free(a);
    return NULL;
}
#endif

void wk_go(WkVal fn, WkVal *args, int argc) {
    _WkGoArg *a = (_WkGoArg *)wk_malloc(sizeof(_WkGoArg));
    a->fn   = fn;
    a->argc = argc;
    if (argc > 0) {
        a->args = (WkVal *)wk_malloc((size_t)argc * sizeof(WkVal));
        memcpy(a->args, args, (size_t)argc * sizeof(WkVal));
    } else {
        a->args = NULL;
    }
#ifdef _WIN32
    HANDLE h = CreateThread(NULL, 0, _wk_go_thread, a, 0, NULL);
    if (h) CloseHandle(h); /* detach immediately */
#else
    pthread_t tid;
    pthread_create(&tid, NULL, _wk_go_thread, a);
    pthread_detach(tid);
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
   ACTOR MAILBOX
   ═══════════════════════════════════════════════════════════════════════════ */

WkMailbox *wk_mailbox_new(void) {
    WkMailbox *m = (WkMailbox *)wk_malloc(sizeof(WkMailbox));
    m->head = m->tail = m->count = 0;
#ifdef _WIN32
    InitializeCriticalSection(&m->lock);
    InitializeConditionVariable(&m->not_empty);
    InitializeConditionVariable(&m->not_full);
#else
    pthread_mutex_init(&m->lock, NULL);
    pthread_cond_init(&m->not_empty, NULL);
    pthread_cond_init(&m->not_full, NULL);
#endif
    return m;
}

void wk_mailbox_send(WkMailbox *m, WkVal v) {
#ifdef _WIN32
    EnterCriticalSection(&m->lock);
    while (m->count == WK_MAILBOX_CAP)
        SleepConditionVariableCS(&m->not_full, &m->lock, INFINITE);
    m->buf[m->tail] = v;
    m->tail = (m->tail + 1) % WK_MAILBOX_CAP;
    m->count++;
    WakeConditionVariable(&m->not_empty);
    LeaveCriticalSection(&m->lock);
#else
    pthread_mutex_lock(&m->lock);
    while (m->count == WK_MAILBOX_CAP)
        pthread_cond_wait(&m->not_full, &m->lock);
    m->buf[m->tail] = v;
    m->tail = (m->tail + 1) % WK_MAILBOX_CAP;
    m->count++;
    pthread_cond_signal(&m->not_empty);
    pthread_mutex_unlock(&m->lock);
#endif
}

int wk_mailbox_recv(WkMailbox *m, int timeout_ms, WkVal *out) {
    int got = 0;
#ifdef _WIN32
    EnterCriticalSection(&m->lock);
    DWORD wait_ms = (timeout_ms < 0) ? INFINITE : (DWORD)timeout_ms;
    if (m->count == 0)
        SleepConditionVariableCS(&m->not_empty, &m->lock, wait_ms);
    if (m->count > 0) {
        *out   = m->buf[m->head];
        m->head = (m->head + 1) % WK_MAILBOX_CAP;
        m->count--;
        WakeConditionVariable(&m->not_full);
        got = 1;
    }
    LeaveCriticalSection(&m->lock);
#else
    pthread_mutex_lock(&m->lock);
    if (timeout_ms < 0) {
        while (m->count == 0)
            pthread_cond_wait(&m->not_empty, &m->lock);
    } else if (m->count == 0) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec  += timeout_ms / 1000;
        ts.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
        if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
        pthread_cond_timedwait(&m->not_empty, &m->lock, &ts);
    }
    if (m->count > 0) {
        *out   = m->buf[m->head];
        m->head = (m->head + 1) % WK_MAILBOX_CAP;
        m->count--;
        pthread_cond_signal(&m->not_full);
        got = 1;
    }
    pthread_mutex_unlock(&m->lock);
#endif
    return got;
}

void wk_mailbox_free(WkMailbox *m) {
    if (!m) return;
#ifdef _WIN32
    DeleteCriticalSection(&m->lock);
#else
    pthread_mutex_destroy(&m->lock);
    pthread_cond_destroy(&m->not_empty);
    pthread_cond_destroy(&m->not_full);
#endif
    free(m);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Channels (WkObj wrapping a WkMailbox)
   ═══════════════════════════════════════════════════════════════════════════ */

static WkClass _wk_chan_class = {"Channel", NULL, NULL, 0, NULL, 0};

WkVal wk_make_chan(int capacity) {
    (void)capacity; /* mailbox uses fixed WK_MAILBOX_CAP */
    WkObj *o = (WkObj*)wk_malloc(sizeof(WkObj));
    o->refcnt  = 1;
    o->cls     = &_wk_chan_class;
    o->fields  = NULL;
    o->mailbox = wk_mailbox_new();
    WkVal v; memset(&v,0,sizeof(v));
    v.tag = WK_OBJ; v.refcnt = 1; v.as.obj = o;
    return v;
}

void wk_chan_send(WkVal ch, WkVal val) {
    if (ch.tag != WK_OBJ || !ch.as.obj->mailbox)
        wk_panic("channel send: not a channel");
    wk_mailbox_send(ch.as.obj->mailbox, val);
}

WkVal wk_chan_recv(WkVal ch) {
    if (ch.tag != WK_OBJ || !ch.as.obj->mailbox)
        wk_panic("channel recv: not a channel");
    WkVal out = wk_none();
    wk_mailbox_recv(ch.as.obj->mailbox, -1, &out); /* block indefinitely */
    return out;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SQL (sqlite3 — only compiled when -DWK_HAVE_SQL is passed)
   ═══════════════════════════════════════════════════════════════════════════ */

#ifdef WK_HAVE_SQL
#include <sqlite3.h>

WkVal wk_sql_exec(const char *db_path, const char *query,
                  WkVal *params, int nparams) {
    sqlite3 *db;
    if (sqlite3_open(db_path, &db) != SQLITE_OK)
        wk_panic("SQL: cannot open '%s': %s", db_path, sqlite3_errmsg(db));

    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, query, -1, &stmt, NULL) != SQLITE_OK)
        wk_panic("SQL: bad query: %s", sqlite3_errmsg(db));

    for (int i = 0; i < nparams; i++) {
        WkVal p = params[i];
        int col = i + 1;
        switch (p.tag) {
            case WK_INT:   sqlite3_bind_int64(stmt, col, p.as.i); break;
            case WK_FLOAT: sqlite3_bind_double(stmt, col, p.as.f); break;
            case WK_STR:
                sqlite3_bind_text(stmt, col, p.as.str->data,
                                  (int)p.as.str->len, SQLITE_STATIC); break;
            default: sqlite3_bind_null(stmt, col);
        }
    }

    WkVal rows = wk_make_list();
    int rc;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        int ncols = sqlite3_column_count(stmt);
        WkVal row = wk_make_map();
        for (int c = 0; c < ncols; c++) {
            const char *col_name = sqlite3_column_name(stmt, c);
            WkVal key = wk_make_strz(col_name);
            WkVal val;
            switch (sqlite3_column_type(stmt, c)) {
                case SQLITE_INTEGER: val = wk_int(sqlite3_column_int64(stmt, c)); break;
                case SQLITE_FLOAT:   val = wk_float(sqlite3_column_double(stmt, c)); break;
                case SQLITE_TEXT:
                    val = wk_make_strz((const char *)sqlite3_column_text(stmt, c)); break;
                default: val = wk_none();
            }
            wk_map_set_key(row, key, val);
        }
        wk_list_push_raw(rows.as.list, row);
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return rows;
}
#endif /* WK_HAVE_SQL */
