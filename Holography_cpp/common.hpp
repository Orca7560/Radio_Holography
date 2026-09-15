#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
namespace holo {
inline std::string join(const std::string&a,const std::string&b){return a.empty()?b:(a[a.size()-1]=='/'?a+b:a+"/"+b);}
inline bool is_dir(const std::string&p){struct stat s;return stat(p.c_str(),&s)==0&&S_ISDIR(s.st_mode);}
inline bool is_file(const std::string&p){struct stat s;return stat(p.c_str(),&s)==0&&S_ISREG(s.st_mode);}
inline void mkdir_p(const std::string&p){std::string cur;for(size_t i=0;i<p.size();++i){cur+=p[i];if(p[i]=='/'&&cur.size()>1)mkdir(cur.c_str(),0755);}if(!p.empty()&&mkdir(p.c_str(),0755)!=0&&!is_dir(p))throw std::runtime_error("Cannot create directory: "+p);}
inline std::string quote(const std::string&s){std::string r="'";for(size_t i=0;i<s.size();++i){if(s[i]=='\'')r+="'\\''";else r+=s[i];}return r+"'";}
inline int run(const std::string&c){std::cout<<"[RUN] "<<c<<"\n";return std::system(c.c_str());}
inline std::string capture(const std::string&c){std::string p="/tmp/holo_"+std::to_string((long long)getpid())+".txt";int rc=run(c+" > "+quote(p)+" 2>&1");std::ifstream in(p.c_str());std::ostringstream o;o<<in.rdbuf();std::remove(p.c_str());return rc==0?o.str():"";}
inline std::vector<std::string> split(const std::string&s,char sep=','){std::vector<std::string>v;std::string x;bool q=false;for(size_t i=0;i<s.size();++i){char c=s[i];if(c=='"')q=!q;else if(c==sep&&!q){v.push_back(x);x.clear();}else x+=c;}v.push_back(x);return v;}
inline std::string trim(std::string s){size_t a=s.find_first_not_of(" \t\r\n"),b=s.find_last_not_of(" \t\r\n");return a==std::string::npos?"":s.substr(a,b-a+1);}
inline bool has_flag(int n,char**v,const std::string&k){for(int i=1;i<n;++i)if(v[i]==k)return true;return false;}
inline std::string arg(int n,char**v,const std::string&a,const std::string&b="",const std::string&d=""){for(int i=1;i+1<n;++i)if(v[i]==a||(!b.empty()&&v[i]==b))return v[i+1];return d;}
inline std::string filename(const std::string&p){size_t q=p.find_last_of('/');return q==std::string::npos?p:p.substr(q+1);}
inline std::string stem(const std::string&p){std::string f=filename(p);size_t q=f.find_last_of('.');return q==std::string::npos?f:f.substr(0,q);}
inline std::string extension(const std::string&p){std::string f=filename(p);size_t q=f.find_last_of('.');return q==std::string::npos?"":f.substr(q);}
inline std::vector<std::string> files_matching(const std::string&dir,const std::string&contains,const std::string&ext){std::vector<std::string>r;DIR*d=opendir(dir.c_str());if(!d)return r;dirent*e;while((e=readdir(d))){std::string n=e->d_name;if(n=="."||n=="..")continue;std::string p=join(dir,n);if(is_file(p)&&n.find(contains)!=std::string::npos&&extension(n)==ext)r.push_back(p);}closedir(d);std::sort(r.begin(),r.end());return r;}
inline int clampi(int x,int lo,int hi){return x<lo?lo:(x>hi?hi:x);}
}