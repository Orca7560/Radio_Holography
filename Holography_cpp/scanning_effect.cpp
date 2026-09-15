#include "common.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <map>\n#include <limits>
using namespace holo;
struct S {double t,amp,phase,snr,az,el,vx=0,vy=0;int dir=0;};
static void help(){std::cout <<
"Usage:\n"
"  scanning_effect --input FILE --skd FILE [OPTIONS]\n\n"
"Estimate the scan lag that makes +Az and -Az beam maps agree, then write\n"
"corrected coordinates and the lag-search result.\n\n"
"Required:\n"
"  --input, --in FILE  Correlation summary CSV (Epoch, Amp, Phase, SNR).\n"
"  --skd FILE          SKD schedule containing the $SKED block.\n\n"
"Output:\n"
"  --output, --out DIR Output directory (default: scanning_result).\n"
"                       Creates matched_and_corrected.csv and lag_search.csv.\n\n"
"Lag search:\n"
"  --max-lag-ms MS     Search range: -MS to +MS (default: 1000).\n"
"  --lag-step-ms MS    Search interval (default: 5).\n"
"  --min-snr N         Minimum SNR used for the fit (default: 3).\n"
"  --row-gap S         Reserved row-separation setting (default: 2).\n"
"  -h, --help          Show this help.\n\n"
"Example:\n"
"  scanning_effect --in summary.txt --skd schedule.skd --out scan_result\n";}
static double epoch(const std::string&s){int y,d,h,m;double z;if(std::sscanf(s.c_str(),"%d/%d %d:%d:%lf",&y,&d,&h,&m,&z)==5)return (((y*366.+d)*24+h)*60+m)*60+z;throw std::runtime_error("Bad Epoch: "+s);}
int main(int argc,char**argv){
 try{
  if(argc == 1){ help(); return 1; }
  if(has_flag(argc,argv,"-h")||has_flag(argc,argv,"--help")){help();return 0;}
  std::string ip=arg(argc,argv,"--input","--in"), sp=arg(argc,argv,"--skd"); if(ip.empty()||sp.empty()){ help(); throw std::runtime_error("--input and --skd are required."); }
  std::string out=arg(argc,argv,"--output","--out","scanning_result");mkdir_p(out);
  double maxlag=std::stod(arg(argc,argv,"--max-lag-ms","", "1000")), step=std::stod(arg(argc,argv,"--lag-step-ms","", "5")), mins=std::stod(arg(argc,argv,"--min-snr","", "3"));
  std::ifstream f(ip);std::string l;if(!std::getline(f,l))throw std::runtime_error("empty input");auto hd=split(l);std::map<std::string,int>col;for(int i=0;i<(int)hd.size();++i)col[trim(hd[i])]=i;
  for(auto n:{"Epoch","Amp","Phase","SNR"})if(!col.count(n))throw std::runtime_error("missing column "+std::string(n));
  std::vector<S>a;while(std::getline(f,l)){auto v=split(l);if((int)v.size()<(int)hd.size())continue;try{S sample; sample.t=epoch(v[col["Epoch"]]); sample.amp=std::stod(v[col["Amp"]]); sample.phase=std::stod(v[col["Phase"]]); sample.snr=std::stod(v[col["SNR"]]); a.push_back(sample);}catch(...){}}
  std::ifstream sk(sp);std::vector<std::array<double,3>> knots;bool in=false;while(std::getline(sk,l)){auto q=trim(l);if(!q.empty()&&q[0]=='$'){in=q.rfind("$SKED",0)==0;continue;}auto v=split(q,' ');v.erase(std::remove(v.begin(),v.end(),""),v.end());if(in&&v.size()>=5)try{int yy=std::stoi(v[1].substr(0,2)),dd=std::stoi(v[1].substr(2,3)),hh=std::stoi(v[1].substr(5,2)),mm=std::stoi(v[1].substr(7,2));double ss=std::stod(v[1].substr(9));knots.push_back({(((2000+yy)*366.+dd)*24+hh)*60*60+mm*60+ss,std::stod(v[3]),std::stod(v[4])});}catch(...){}}
  if(knots.size()<2)throw std::runtime_error("No SKED rows.");
  for(auto&x:a){auto it=std::lower_bound(knots.begin(),knots.end(),x.t,[](const std::array<double,3>& k,double t){return k[0]<t;});if(it==knots.begin()||it==knots.end())continue;auto p=it-1;double u=(x.t-(*p)[0])/((*it)[0]-(*p)[0]);x.az=(*p)[1]*(1-u)+(*it)[1]*u;x.el=(*p)[2]*(1-u)+(*it)[2]*u;x.vx=((*it)[1]-(*p)[1])/((*it)[0]-(*p)[0]);x.vy=((*it)[2]-(*p)[2])/((*it)[0]-(*p)[0]);x.dir=(x.vx>1e-5)-(x.vx<-1e-5);}
  // For every trial lag, make separate +Az/-Az maps on a common grid and
  // compare only cells occupied by both directions (the same quantity used by
  // scanning_effect.py, without its SciPy interpolation dependency).
  double xmin=a[0].az,xmax=xmin,ymin=a[0].el,ymax=ymin;for(auto&x:a){xmin=std::min(xmin,x.az);xmax=std::max(xmax,x.az);ymin=std::min(ymin,x.el);ymax=std::max(ymax,x.el);}const int ng=121;double best=0,bscore=1e300;std::vector<std::pair<double,double>> curve;
  for(double lag=-maxlag;lag<=maxlag+1e-9;lag+=step){std::map<int,std::array<double,4>> cell;for(auto&x:a)if(x.snr>=mins&&x.dir){double xx=x.az-x.vx*lag/1000,yy=x.el-x.vy*lag/1000;int ix=std::max(0,std::min(ng-1,(int)((xx-xmin)/(xmax-xmin+1e-12)*ng))),iy=std::max(0,std::min(ng-1,(int)((yy-ymin)/(ymax-ymin+1e-12)*ng)));auto&z=cell[iy*ng+ix];if(x.dir>0){z[0]+=x.amp;z[1]+=1;}else{z[2]+=x.amp;z[3]+=1;}}double pmax=0,mmax=0;for(auto&kv:cell){if(kv.second[1])pmax=std::max(pmax,kv.second[0]/kv.second[1]);if(kv.second[3])mmax=std::max(mmax,kv.second[2]/kv.second[3]);}double sum=0;int n=0;for(auto&kv:cell)if(kv.second[1]&&kv.second[3]&&pmax>0&&mmax>0){double p=kv.second[0]/kv.second[1]/pmax,m=kv.second[2]/kv.second[3]/mmax;if((p+m)*.5>.05){sum+=(p-m)*(p-m);++n;}}double score=n>=10?sum/n:std::numeric_limits<double>::quiet_NaN();curve.push_back({lag,score});if(std::isfinite(score)&&score<bscore){bscore=score;best=lag;}}
  if(!std::isfinite(bscore))throw std::runtime_error("lag cannot be fitted: both +Az and -Az rows with enough SNR are required");
  std::ofstream o(join(out,"matched_and_corrected.csv"));o<<"Epoch,Amp,Phase,SNR,Az_Offset,El_Offset,Az_Corrected,El_Corrected\n";for(auto&x:a)o<<x.t<<","<<x.amp<<","<<x.phase<<","<<x.snr<<","<<x.az<<","<<x.el<<","<<x.az-x.vx*best/1000<<","<<x.el-x.vy*best/1000<<"\n";
  std::ofstream q(join(out,"lag_search.csv"));q<<"lag_ms,mismatch\n";for(auto&v:curve)q<<v.first<<","<<v.second<<"\n";std::ofstream b(join(out,"best_lag_ms.txt"));b<<best<<"\n";std::cout<<"Best scan lag: "<<best<<" ms\nOutputs: "<<out<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 2;}
}