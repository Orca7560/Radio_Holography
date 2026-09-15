#include "common.hpp"
#include <iostream>
using namespace holo;
static void help() {
 std::cout<<"corr_fringe --workdir DIR [--only-corr|--only-fringe|--only-frinz] [--cpu N] [--band-split N]\n"
             "  workdir defaults to the current directory. gico3/fringe/frinZ must be in PATH.\n";
}
int main(int argc,char**argv) {
 try {
  if(has_flag(argc,argv,"-h")||has_flag(argc,argv,"--help")) {help();return 0;}
  std::string wd=arg(argc,argv,"--workdir","", ".");
  if(!is_dir(wd)) throw std::runtime_error("workdir not found: "+wd);
  const bool only_corr=has_flag(argc,argv,"--only-corr"), only_fringe=has_flag(argc,argv,"--only-fringe");
  const bool only_frinz=has_flag(argc,argv,"--only-frinz")||has_flag(argc,argv,"--only-frinZ");
  const std::string raw=arg(argc,argv,"--raw-dir","", "raw");
  const std::string cor=arg(argc,argv,"--cor-dir","", "stepcor");
  const std::string cpu=arg(argc,argv,"--cpu");
  auto xml=files_matching(wd,"_KL_X_step",".xml");
  xml.erase(std::remove_if(xml.begin(),xml.end(),[](const std::string&p){return p.find("_fringe_search")!=std::string::npos;}),xml.end());
  if(xml.empty()) throw std::runtime_error("No *_KL_X_step*.xml files found.");
  std::string rawdir=join(wd,raw), cordir=join(wd,cor); mkdir_p(cordir);
  if(!only_fringe&&!only_frinz) for(const auto& x:xml) {
    std::string c="gico3 --schedule "+quote(x)+" --raw-file "+quote(rawdir)+" --cor-file "+quote(cordir);
    if(!cpu.empty()) c+=" --cpu "+cpu; if(run(c)!=0) return 2;
  }
  if(only_corr) return 0;
  const std::string processor=only_frinz?"frinZ":"fringe";
  std::string out=join(wd, only_frinz?"frinz_results":"fringe_results"); mkdir_p(out);
  for(const auto& corfile: files_matching(cordir,"",".cor")) {
    std::string result=capture(processor+" --in "+quote(corfile));
    std::ofstream f(join(out,stem(corfile)+"_output.txt")); f<<result;
  }
  std::cout<<"Results: "<<out<<"\n"; return 0;
 } catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 2;}
}