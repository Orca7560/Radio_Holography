#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace holo {
namespace fs = std::filesystem;
inline std::string quote(const std::string& s) {
    std::string r="'"; for (char c: s) { if (c=='\'') r += "'\\''"; else r += c; } return r + "'";
}
inline int run(const std::string& command) { std::cout << "[RUN] " << command << "\n"; return std::system(command.c_str()); }
inline std::string capture(const std::string& command) {
    const auto p = fs::temp_directory_path() / ("holo_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
    int rc = run(command + " > " + quote(p.string()) + " 2>&1");
    std::ifstream in(p); std::ostringstream out; out << in.rdbuf(); std::error_code ec; fs::remove(p, ec);
    if (rc != 0) return ""; return out.str();
}
inline std::vector<std::string> split(const std::string& s, char sep=',') {
    std::vector<std::string> v; std::string x; bool q=false;
    for(char c:s) { if(c=='"') q=!q; else if(c==sep&&!q){v.push_back(x);x.clear();} else x+=c; } v.push_back(x); return v;
}
inline std::string trim(std::string s) {
    auto a=s.find_first_not_of(" \t\r\n"), b=s.find_last_not_of(" \t\r\n");
    return a==std::string::npos ? "" : s.substr(a,b-a+1);
}
inline bool has_flag(int argc,char**argv,const std::string& key) {
    for(int i=1;i<argc;++i) if(argv[i]==key) return true; return false;
}
inline std::string arg(int argc,char**argv,const std::string& a,const std::string& b="",const std::string& def="") {
    for(int i=1;i+1<argc;++i) if(argv[i]==a || (!b.empty()&&argv[i]==b)) return argv[i+1]; return def;
}
inline std::vector<fs::path> files_matching(const fs::path& dir,const std::string& contains,const std::string& ext) {
    std::vector<fs::path> r; if(!fs::is_directory(dir)) return r;
    for(auto& e:fs::directory_iterator(dir)) if(e.is_regular_file() && e.path().filename().string().find(contains)!=std::string::npos && e.path().extension()==ext) r.push_back(e.path());
    std::sort(r.begin(),r.end()); return r;
}
inline void require_file(const fs::path& p) { if(!fs::is_regular_file(p)) throw std::runtime_error("File not found: "+p.string()); }
}