#include "common.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
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
"  --scan-axis az|el    Varying axis of the raster scan (default: az).\n"
"  --az-drive-speed N   Az scan speed in arcmin/s (default: 3).\n"
"  --el-drive-speed N   El scan speed in arcmin/s (default: use PRD rate).\n"
"  -h, --help           Show this help.\n\n"
"Example:\n"
"  group_up_txt --in I26184Y --out I26184Y/beam.txt --prd I26184Y/I26184Y32.prd\n";
}
static double parse_prd_time(const std::string& s){
 if(s.size()!=13||s.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("bad PRD time");
 int year=std::stoi(s.substr(0,4)),doy=std::stoi(s.substr(4,3)),hour=std::stoi(s.substr(7,2)),min=std::stoi(s.substr(9,2)),sec=std::stoi(s.substr(11,2));
 return ((((year*366.0+doy)*24)+hour)*60+min)*60+sec;
}
static double parse_epoch_time(const std::string&s){
 int year,doy,hour,min;double sec;
 if(std::sscanf(s.c_str(),"%d/%d %d:%d:%lf",&year,&doy,&hour,&min,&sec)!=5)throw std::runtime_error("bad Epoch");
 return ((((year*366.0+doy)*24)+hour)*60+min)*60+sec;
}
static std::vector<Prd> read_prd(const std::string&p){
 std::ifstream f(p); if(!f)throw std::runtime_error("Cannot open PRD: "+p);std::vector<Prd>r;std::string l;
 while(std::getline(f,l)){std::istringstream in(l);std::vector<std::string>v;std::string word;while(in>>word)v.push_back(word);if(v.size()<9)continue;try{r.push_back({parse_prd_time(v[0]),std::stod(v[7]),std::stod(v[8])});}catch(...){}}
 std::sort(r.begin(),r.end(),[](const Prd&a,const Prd&b){return a.t<b.t;});
 if(r.size()<2)throw std::runtime_error("PRD needs at least two valid pointing records");
 return r;
}
int main(int argc,char**argv){
 try{
  if(argc == 1){ help(); return 1; }
  if(has_flag(argc,argv,"-h")||has_flag(argc,argv,"--help")){help();return 0;}
  std::string root=arg(argc,argv,"--input","--in",".");std::string out=arg(argc,argv,"--output","--out","beam.txt");
  std::string axis=arg(argc,argv,"--scan-axis","","az");if(axis!="az"&&axis!="el")throw std::runtime_error("--scan-axis must be az or el");
  double az_speed=std::stod(arg(argc,argv,"--az-drive-speed","","3"));
  std::string el_speed_arg=arg(argc,argv,"--el-drive-speed");double el_speed=el_speed_arg.empty()?std::numeric_limits<double>::quiet_NaN():std::stod(el_speed_arg);
  if(az_speed<=0||(!el_speed_arg.empty()&&el_speed<=0))throw std::runtime_error("drive speed must be positive");
  std::string indir=join(root,"fringe_results"); if(!is_dir(indir)) indir=root;
  std::string prdarg=arg(argc,argv,"--prd"); std::vector<Prd>prd;
  if(!prdarg.empty())prd=read_prd(prdarg); else {auto c=files_matching(root,"32",".prd");if(c.size()==1)prd=read_prd(c[0]);}
  auto files=files_matching(indir,"",".txt");if(files.empty())throw std::runtime_error("No .txt fringe results found.");
  std::ofstream o(out); bool maser=has_flag(argc,argv,"--maser"), delay=has_flag(argc,argv,"--add-delay");
  o<<"Epoch, Length, Amp, Phase, "<<(maser?"Frequency":"SNR");if(delay)o<<", Res-Delay";if(!prd.empty())o<<", Az_Offset, El_Offset, Az_Rate_arcmin_s, El_Rate_arcmin_s";o<<"\n";
  for(auto file:files){std::ifstream f(file);std::string l;while(std::getline(f,l)){if(l.empty()||l[0]=='#')continue;std::istringstream is(l);std::vector<std::string>v;std::string q;while(is>>q)v.push_back(q);if(v.size()<17)continue;try{
   double amp=std::stod(v[5]), snr=std::stod(v[6]), ph=std::stod(v[7]), len=std::stod(v[4]);bool modern=v.size()>=20;if(!modern)amp/=100.0;
   o<<v[0]<<" "<<v[1]<<", "<<len<<", "<<amp<<", "<<ph<<", "<<(maser?v[8]:std::to_string(snr));if(delay)o<<", "<<v[modern?8:9];
   if(!prd.empty()){
    double midpoint=parse_epoch_time(v[0]+" "+v[1])+len/2.0;
    auto right=std::upper_bound(prd.begin(),prd.end(),midpoint,[](double t,const Prd&k){return t<k.t;});
    double az=std::numeric_limits<double>::quiet_NaN(),el=az,vx=az,vy=az;
    if(right!=prd.begin()&&right!=prd.end()){
     auto left=right-1;double duration=right->t-left->t;
     if(duration>0){vx=(right->az-left->az)/duration;vy=(right->el-left->el)/duration;
      if(axis=="az"&&std::abs(vy)<1e-10&&std::abs(vx)>1e-10)vx=std::copysign(az_speed,vx);
      if(axis=="el"&&std::isfinite(el_speed)&&std::abs(vx)<1e-10&&std::abs(vy)>1e-10)vy=std::copysign(el_speed,vy);
      az=left->az+(midpoint-left->t)*vx;el=left->el+(midpoint-left->t)*vy;
     }
    }
    o<<", "<<az<<", "<<el<<", "<<vx<<", "<<vy;
   } o<<"\n";
  }catch(...){}}}
  std::cout<<"Output: "<<out<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 2;}
}
