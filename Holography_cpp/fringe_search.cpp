#include "common.hpp"
#include <iostream>
#include <regex>
using namespace holo;
static void help(){std::cout<<"fringe_search [--workdir DIR] [--max-iterations N]\n";}
static bool set_delay(const fs::path& file,double seconds,char sign) {
 std::ifstream in(file); std::ostringstream all; all<<in.rdbuf(); std::string s=all.str();
 std::regex tag(R"((<clock[^>]*key=["']K["'][^>]*>[\s\S]*?<delay>)[^<]*(</delay>)");
 std::ostringstream v; v<<sign<<std::scientific<<std::setprecision(5)<<std::abs(seconds);
 if(!std::regex_search(s,tag)) return false; s=std::regex_replace(s,tag,"$1"+v.str()+"$2");
 std::ofstream out(file); out<<s; return true;
}
static double residual(const std::string& t) {
 std::regex r(R"(Res-Delay\s*[:=]\s*([+-]?[0-9.eE]+))"); std::smatch m;
 if(std::regex_search(t,m,r)) return std::stod(m[1]); throw std::runtime_error("Res-Delay not found");
}
static std::string cor_for(const fs::path& wd,const fs::path& xml) {
 std::string stem=xml.stem().string(); auto p=files_matching(wd/"stepcor","",".cor");
 for(auto& x:p) if(x.filename().string().find(stem.substr(0,std::min<size_t>(stem.size(),12)))!=std::string::npos)return x.string();
 return p.empty()?"":p.front().string();
}
int main(int argc,char**argv){
 try{
  if(has_flag(argc,argv,"-h")||has_flag(argc,argv,"--help")){help();return 0;}
  fs::path wd=arg(argc,argv,"--workdir","","."); int maxit=std::stoi(arg(argc,argv,"--max-iterations","", "20"));
  auto xs=files_matching(wd,"_fringe_search",".xml"); if(xs.empty())throw std::runtime_error("No fringe-search XML found.");
  std::ofstream log(wd/"fringe_search_result.log"); log<<"xml\titerations\tsign\tres_delay_sample\tdelay_s\n";
  for(auto x:xs){
   set_delay(x,0,'+'); auto measure=[&](){if(run("gico3 --schedule "+quote(x.string())+" --raw-file "+quote((wd/"raw").string())+" --cor-file "+quote((wd/"stepcor").string()))!=0)throw std::runtime_error("gico3 failed"); auto c=cor_for(wd,x);if(c.empty())throw std::runtime_error("cor not found");return residual(capture("fringe --in "+quote(c)));};
   double r0=measure(), mag=std::abs(r0); char sign='+'; double rplus;
   set_delay(x,mag/1024000000.0,'+'); rplus=measure();
   double rnow=rplus; if(std::abs(rplus)>=mag){set_delay(x,0,'+');set_delay(x,mag/1024000000.0,'-');rnow=measure();sign='-';if(std::abs(rnow)>=mag){log<<x.filename().string()<<"\tFAILED\n";set_delay(x,0,'+');continue;}}
   double total=mag; int it=2; while(std::abs(rnow)>0&&it<maxit){total+=std::abs(rnow);set_delay(x,total/1024000000.0,sign);rnow=measure();++it;}
   if(std::abs(rnow)>0){log<<x.filename().string()<<"\tFAILED\n";set_delay(x,0,'+');continue;}
   fs::path target=x; std::string n=target.string(); n=std::regex_replace(n,std::regex("_fringe_search\\.xml$"),".xml"); set_delay(n,total/1024000000.0,sign);
   log<<x.filename().string()<<"\t"<<it<<"\t"<<sign<<"\t"<<rnow<<"\t"<<sign<<total/1024000000.0<<"\n";set_delay(x,0,'+');
  } return 0;
 }catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 2;}
}