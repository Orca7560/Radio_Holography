#include "common.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <map>
using namespace holo;
struct S {double t,amp,phase,snr,az,el,vx=0,vy=0;int dir=0;};
static void help(){std::cout<<"scanning_effect --input FILE --skd FILE [--output DIR] [--max-lag-ms MS] [--lag-step-ms MS] [--min-snr N] [--row-gap S]\n";}
static double epoch(const std::string&s){int y,d,h,m;double z;if(std::sscanf(s.c_str(),"%d/%d %d:%d:%lf",&y,&d,&h,&m,&z)==5)return (((y*366.+d)*24+h)*60+m)*60+z;throw std::runtime_error("Bad Epoch: "+s);}
int main(int argc,char**argv){
 try{
  if(has_flag(argc,argv,"-h")||has_flag(argc,argv,"--help")){help();return 0;}
  std::string ip=arg(argc,argv,"--input","--in"), sp=arg(argc,argv,"--skd");if(ip.empty()||sp.empty())throw std::runtime_error("--input and --skd are required.");
  std::string out=arg(argc,argv,"--output","--out","scanning_result");mkdir_p(out);
  double maxlag=std::stod(arg(argc,argv,"--max-lag-ms","", "1000")), step=std::stod(arg(argc,argv,"--lag-step-ms","", "5")), mins=std::stod(arg(argc,argv,"--min-snr","", "3"));
  std::ifstream f(ip);std::string l;if(!std::getline(f,l))throw std::runtime_error("empty input");auto hd=split(l);std::map<std::string,int>col;for(int i=0;i<(int)hd.size();++i)col[trim(hd[i])]=i;
  for(auto n:{"Epoch","Amp","Phase","SNR"})if(!col.count(n))throw std::runtime_error("missing column "+std::string(n));
  std::vector<S>a;while(std::getline(f,l)){auto v=split(l);if((int)v.size()<(int)hd.size())continue;try{S sample; sample.t=epoch(v[col["Epoch"]]); sample.amp=std::stod(v[col["Amp"]]); sample.phase=std::stod(v[col["Phase"]]); sample.snr=std::stod(v[col["SNR"]]); a.push_back(sample);}catch(...){}}
  std::ifstream sk(sp);std::vector<std::array<double,3>> knots;bool in=false;while(std::getline(sk,l)){auto q=trim(l);if(!q.empty()&&q[0]=='$'){in=q.rfind("$SKED",0)==0;continue;}auto v=split(q,' ');v.erase(std::remove(v.begin(),v.end(),""),v.end());if(in&&v.size()>=5)try{int yy=std::stoi(v[1].substr(0,2)),dd=std::stoi(v[1].substr(2,3)),hh=std::stoi(v[1].substr(5,2)),mm=std::stoi(v[1].substr(7,2));double ss=std::stod(v[1].substr(9));knots.push_back({(((2000+yy)*366.+dd)*24+hh)*60*60+mm*60+ss,std::stod(v[3]),std::stod(v[4])});}catch(...){}}
  if(knots.size()<2)throw std::runtime_error("No SKED rows.");
  for(auto&x:a){auto it=std::lower_bound(knots.begin(),knots.end(),x.t,[](const std::array<double,3>& k,double t){return k[0]<t;});if(it==knots.begin()||it==knots.end())continue;auto p=it-1;double u=(x.t-(*p)[0])/((*it)[0]-(*p)[0]);x.az=(*p)[1]*(1-u)+(*it)[1]*u;x.el=(*p)[2]*(1-u)+(*it)[2]*u;x.vx=((*it)[1]-(*p)[1])/((*it)[0]-(*p)[0]);x.vy=((*it)[2]-(*p)[2])/((*it)[0]-(*p)[0]);x.dir=(x.vx>1e-5)-(x.vx<-1e-5);}
  double best=0,bscore=1e300;for(double lag=-maxlag;lag<=maxlag+1e-9;lag+=step){double sp=0,sm=0,wp=0,wm=0;for(auto&x:a)if(x.snr>=mins&&x.dir){double z=x.amp; if(x.dir>0){sp+=z;wp++;}else{sm+=z;wm++;}}if(wp&&wm){double score=std::abs(sp/wp-sm/wm);if(score<bscore){bscore=score;best=lag;}}}
  std::ofstream o(join(out,"matched_and_corrected.csv"));o<<"Epoch,Amp,Phase,SNR,Az_Offset,El_Offset,Az_Corrected,El_Corrected\n";for(auto&x:a)o<<x.t<<","<<x.amp<<","<<x.phase<<","<<x.snr<<","<<x.az<<","<<x.el<<","<<x.az-x.vx*best/1000<<","<<x.el-x.vy*best/1000<<"\n";
  std::ofstream q(join(out,"lag_search.csv"));q<<"best_lag_ms,mismatch\n"<<best<<","<<bscore<<"\n";std::cout<<"Best scan lag: "<<best<<" ms\nOutputs: "<<out<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 2;}
}