#include "common.hpp"
#include <iostream>
#include <map>
using namespace holo;
struct Prd { double t,az,el; };
static void help(){
 std::cout <<
"Usage:\n"
"  group_up_txt [--input DIR] [--output FILE] [OPTIONS]\n\n"
"Merge fringe-result text files into a beam file. By default it reads\n"
"./fringe_results/ and writes ./beam.txt.\n\n"
"Input and output:\n"
"  --input, --in DIR    Observation directory (default: current directory).\n"
"  --output, --out FILE Output beam file (default: beam.txt).\n"
"  --prd FILE           PRD file for Az/El coordinates. If omitted, a unique\n"
"                       *32.prd in the input directory is used automatically.\n\n"
"Processing options:\n"
"  --add-delay          Add the Res-Delay column.\n"
"  --maser              Store Frequency instead of SNR.\n"
"  -h, --help           Show this help.\n\n"
"Example:\n"
"  group_up_txt --in I26184Y --out I26184Y/beam.txt --prd I26184Y/I26184Y32.prd\n";
}
static std::vector<Prd> read_prd(const std::string&p){
 std::ifstream f(p); if(!f)throw std::runtime_error("Cannot open PRD: "+p);std::vector<Prd>r;std::string l;
 while(std::getline(f,l)){auto v=split(trim(l),' ');v.erase(std::remove(v.begin(),v.end(),""),v.end());if(v.size()<3)continue;try{r.push_back({std::stod(v[v.size()-3]),std::stod(v[v.size()-2]),std::stod(v[v.size()-1])});}catch(...){}}
 return r;
}
int main(int argc,char**argv){
 try{
  if(argc == 1){ help(); return 1; }
  if(has_flag(argc,argv,"-h")||has_flag(argc,argv,"--help")){help();return 0;}
  std::string root=arg(argc,argv,"--input","--in",".");std::string out=arg(argc,argv,"--output","--out","beam.txt");
  std::string indir=join(root,"fringe_results"); if(!is_dir(indir)) indir=root;
  std::string prdarg=arg(argc,argv,"--prd"); std::vector<Prd>prd;
  if(!prdarg.empty())prd=read_prd(prdarg); else {auto c=files_matching(root,"32",".prd");if(c.size()==1)prd=read_prd(c[0]);}
  auto files=files_matching(indir,"",".txt");if(files.empty())throw std::runtime_error("No .txt fringe results found.");
  std::ofstream o(out); bool maser=has_flag(argc,argv,"--maser"), delay=has_flag(argc,argv,"--add-delay");
  o<<"Epoch, Length, Amp, Phase, "<<(maser?"Frequency":"SNR");if(delay)o<<", Res-Delay";if(!prd.empty())o<<", Az_Offset, El_Offset";o<<"\n";
  for(auto file:files){std::ifstream f(file);std::string l;while(std::getline(f,l)){if(l.empty()||l[0]=='#')continue;std::istringstream is(l);std::vector<std::string>v;std::string q;while(is>>q)v.push_back(q);if(v.size()<17)continue;try{
   double amp=std::stod(v[5]), snr=std::stod(v[6]), ph=std::stod(v[7]), len=std::stod(v[4]);bool modern=v.size()>=20;if(!modern)amp/=100.0;
   o<<v[0]<<" "<<v[1]<<", "<<len<<", "<<amp<<", "<<ph<<", "<<(maser?v[8]:std::to_string(snr));if(delay)o<<", "<<v[modern?8:9];
   if(!prd.empty()){auto it=prd.front(); o<<", "<<it.az<<", "<<it.el;} o<<"\n";
  }catch(...){}}}
  std::cout<<"Output: "<<out<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 2;}
}