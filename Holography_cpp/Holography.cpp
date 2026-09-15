// Holography.cpp
// Python を使わない電波ホログラフィ解析器。
// 数値解析はC++、PNG出力は gnuplot を使用する。
// 必要: C++17 コンパイラ / 実行時に gnuplot が PATH 上にあること。

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using cd = std::complex<double>;
constexpr double PI = 3.1415926535897932384626433832795;
constexpr double ARC_MIN_TO_RAD = PI / 10800.0;
constexpr double C_LIGHT = 3.0e8;
constexpr double FREQ_HZ = 8.448e9;
constexpr double WAVELENGTH = C_LIGHT / FREQ_HZ;
constexpr double DIAMETER_M = 32.0;
constexpr double FIT_R_MIN_M = 2.0;
constexpr double FIT_R_MAX_M = 14.0;
constexpr double SURFACE_LIMIT_MM = 3.0;

struct Options {
    std::string input, output;
    bool slice_beam = false, slice_aperture = false, polar = false;
    double db_min = -45.0, zoom_size = 50.0, center_block_size = 3.0;
};

struct Sample {
    double time = 0.0, az = 0.0, el = 0.0, amp = 0.0, phase_deg = 0.0, snr = 0.0;
    cd e;
};

struct Grid {
    int nx = 0, ny = 0;
    double xmin = 0.0, ymin = 0.0, dx = 0.0, dy = 0.0;
    std::vector<cd> e;
    std::vector<double> snr;
    cd& at(int y, int x) { return e[y * nx + x]; }
    const cd& at(int y, int x) const { return e[y * nx + x]; }
    double x(int i) const { return xmin + i * dx; }
    double y(int i) const { return ymin + i * dy; }
};

static void usage(const char* exe) {
    std::cout <<
R"(Usage:
  )" << exe << R"( --input FILE --output DIR [options]

Required:
  --input, --in FILE          Input beam.txt / beam_test.txt
  --output, --out DIR         Output directory

Optional:
  --slice-beam                Write beam Az slices
  --slice-aperture            Write aperture phase slices
  --slice-apperture           Alias of --slice-aperture
  --db-min DB                 Lower limit of dB maps (default: -45)
  --zoom-size ARCMIN          Beam-map zoom width (default: 50)
  --center-block-size M       Central square block size (default: 3.0)
  --polar                     Write angle-radius aperture map
  -h, --help                  Show this help

Example:
  )" << exe << R"( --in I26184Y/beam.txt --out I26184Y/output
)";
}

static Options parse_args(int argc, char** argv) {
    Options o;
    auto value = [&](int& i, const std::string& name) -> std::string {
        if (++i >= argc) throw std::runtime_error(name + " requires a value.");
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
        else if (a == "--input" || a == "--in") o.input = value(i, a);
        else if (a == "--output" || a == "--out") o.output = value(i, a);
        else if (a == "--slice-beam") o.slice_beam = true;
        else if (a == "--slice-aperture" || a == "--slice-apperture") o.slice_aperture = true;
        else if (a == "--polar") o.polar = true;
        else if (a == "--db-min") o.db_min = std::stod(value(i, a));
        else if (a == "--zoom-size") o.zoom_size = std::stod(value(i, a));
        else if (a == "--center-block-size") o.center_block_size = std::stod(value(i, a));
        else throw std::runtime_error("Unknown option: " + a);
    }
    if (o.input.empty() || o.output.empty())
        throw std::runtime_error("--input/--in and --output/--out are required. Use --help.");
    if (!(o.db_min < 0.0)) throw std::runtime_error("--db-min must be negative.");
    if (o.zoom_size <= 0.0 || o.center_block_size <= 0.0)
        throw std::runtime_error("--zoom-size and --center-block-size must be positive.");
    return o;
}

static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out; std::string field; bool quote = false;
    for (char c : s) {
        if (c == '"') quote = !quote;
        else if (c == ',' && !quote) { out.push_back(field); field.clear(); }
        else field += c;
    }
    out.push_back(field);
    for (auto& v : out) {
        auto first = v.find_first_not_of(" \t\r\n");
        auto last = v.find_last_not_of(" \t\r\n");
        v = first == std::string::npos ? "" : v.substr(first, last - first + 1);
    }
    return out;
}

static double epoch_seconds(const std::string& s, double fallback) {
    int year = 0, day = 0, hh = 0, mm = 0; double ss = 0.0;
    if (std::sscanf(s.c_str(), "%d/%d %d:%d:%lf", &year, &day, &hh, &mm, &ss) == 5)
        return (((year * 366.0 + day) * 24.0 + hh) * 60.0 + mm) * 60.0 + ss;
    return fallback;
}

static std::vector<Sample> read_beam(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open input: " + path);
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Input is empty.");
    auto header = split_csv(line);
    std::map<std::string, size_t> col;
    for (size_t i = 0; i < header.size(); ++i) col[header[i]] = i;
    for (const auto& name : {"Epoch","Amp","Phase","SNR","Az_Offset","El_Offset"}) {
        if (!col.count(name)) throw std::runtime_error("Missing column: " + std::string(name));
    }
    std::vector<Sample> data; size_t rowno = 1;
    while (std::getline(in, line)) {
        ++rowno; if (line.empty()) continue;
        auto row = split_csv(line);
        try {
            auto get = [&](const char* n) { return row.at(col.at(n)); };
            Sample p;
            p.time = epoch_seconds(get("Epoch"), static_cast<double>(data.size()));
            p.amp = std::stod(get("Amp")); p.phase_deg = std::stod(get("Phase"));
            p.snr = std::stod(get("SNR")); p.az = std::stod(get("Az_Offset")); p.el = std::stod(get("El_Offset"));
            p.e = std::polar(p.amp, p.phase_deg * PI / 180.0);
            data.push_back(p);
        } catch (const std::exception&) {
            std::cerr << "[WARN] Skip invalid CSV row " << rowno << "\n";
        }
    }
    if (data.empty()) throw std::runtime_error("No valid samples.");
    return data;
}

static double median_positive(std::vector<double> v, double fallback) {
    v.erase(std::remove_if(v.begin(), v.end(), [](double a){ return a <= 1e-6 || !std::isfinite(a); }), v.end());
    if (v.empty()) return fallback;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void correct_reference_phase(std::vector<Sample>& p) {
    std::vector<int> on;
    for (int i = 0; i < static_cast<int>(p.size()); ++i)
        if (std::abs(p[i].az) < 1e-9 && std::abs(p[i].el) < 1e-9 && p[i].amp > 1.0) on.push_back(i);
    if (on.size() < 2) { std::cerr << "[WARN] Fewer than two ON points; phase reference correction skipped.\n"; return; }
    std::vector<double> phase(on.size());
    phase[0] = std::arg(p[on[0]].e);
    for (size_t i = 1; i < on.size(); ++i) {
        double raw = std::arg(p[on[i]].e), d = raw - std::arg(p[on[i - 1]].e);
        while (d > PI) d -= 2.0 * PI;
        while (d < -PI) d += 2.0 * PI;
        phase[i] = phase[i - 1] + d;
    }
    for (auto& q : p) {
        size_t j = 1;
        while (j < on.size() && p[on[j]].time < q.time) ++j;
        double ref;
        if (j == 0) ref = phase.front();
        else if (j == on.size()) ref = phase.back();
        else {
            double t0 = p[on[j - 1]].time, t1 = p[on[j]].time;
            double a = (q.time - t0) / std::max(1e-12, t1 - t0);
            ref = phase[j - 1] * (1.0 - a) + phase[j] * a;
        }
        q.e *= std::polar(1.0, -ref);
    }
}

static Grid make_grid(std::vector<Sample> p) {
    auto peak = std::max_element(p.begin(), p.end(), [](const Sample& a, const Sample& b){ return a.amp < b.amp; });
    const double az0 = peak->az, el0 = peak->el;
    for (auto& q : p) { q.az -= az0; q.el -= el0; }

    std::vector<double> daz, del;
    for (size_t i = 1; i < p.size(); ++i) {
        daz.push_back(std::abs(p[i].az - p[i - 1].az));
        del.push_back(std::abs(p[i].el - p[i - 1].el));
    }
    double sx = median_positive(daz, 3.0), sy = median_positive(del, 3.0);
    double xmin = p[0].az, xmax = p[0].az, ymin = p[0].el, ymax = p[0].el;
    for (const auto& q : p) { xmin = std::min(xmin,q.az); xmax = std::max(xmax,q.az); ymin = std::min(ymin,q.el); ymax = std::max(ymax,q.el); }
    int nx = std::max(2, static_cast<int>(std::llround((xmax - xmin) / sx)) + 1);
    int ny = std::max(2, static_cast<int>(std::llround((ymax - ymin) / sy)) + 1);
    Grid g; g.nx=nx; g.ny=ny; g.xmin=xmin; g.ymin=ymin; g.dx=sx; g.dy=sy;
    g.e.assign(nx*ny, cd(0,0)); g.snr.assign(nx*ny, 0.0);
    std::vector<int> n(nx*ny, 0);
    for (const auto& q : p) {
        int ix = std::clamp(static_cast<int>(std::llround((q.az-xmin)/sx)),0,nx-1);
        int iy = std::clamp(static_cast<int>(std::llround((q.el-ymin)/sy)),0,ny-1);
        int k=iy*nx+ix; g.e[k]+=q.e; g.snr[k]+=q.snr; ++n[k];
    }
    std::vector<int> valid;
    for (int k=0;k<nx*ny;++k) if(n[k]) { g.e[k]/=n[k]; g.snr[k]/=n[k]; valid.push_back(k); }
    if (valid.empty()) throw std::runtime_error("Grid construction failed.");
    for (int k=0;k<nx*ny;++k) if(!n[k]) {
        int iy=k/nx, ix=k%nx, best=valid.front(); double bd=1e300;
        for(int q:valid) { int qy=q/nx,qx=q%nx; double d=(qx-ix)*(qx-ix)+(qy-iy)*(qy-iy); if(d<bd){bd=d;best=q;} }
        g.e[k]=g.e[best]; g.snr[k]=g.snr[best];
    }
    std::cout << "Grid: " << nx << " x " << ny << ", step " << sx << "' x " << sy << "'\n";
    return g;
}

static std::vector<cd> dft_inverse(const std::vector<cd>& a) {
    const int n = static_cast<int>(a.size()); std::vector<cd> out(n);
    for(int k=0;k<n;++k) for(int t=0;t<n;++t)
        out[k] += a[t] * std::polar(1.0, 2.0*PI*k*t/n);
    for(auto& v:out) v/=n;
    return out;
}

static Grid aperture_from_beam(const Grid& b) {
    Grid t=b, out=b;
    for(int y=0;y<b.ny;++y) for(int x=0;x<b.nx;++x)
        t.at(y,x)=b.at((y+b.ny/2)%b.ny,(x+b.nx/2)%b.nx);
    for(int y=0;y<b.ny;++y) {
        std::vector<cd> row(b.nx);
        for(int x=0;x<b.nx;++x) row[x]=t.at(y,x);
        row=dft_inverse(row); for(int x=0;x<b.nx;++x) t.at(y,x)=row[x];
    }
    for(int x=0;x<b.nx;++x) {
        std::vector<cd> col(b.ny);
        for(int y=0;y<b.ny;++y) col[y]=t.at(y,x);
        col=dft_inverse(col); for(int y=0;y<b.ny;++y) t.at(y,x)=col[y];
    }
    for(int y=0;y<b.ny;++y) for(int x=0;x<b.nx;++x)
        out.at(y,x)=t.at((y+(b.ny+1)/2)%b.ny,(x+(b.nx+1)/2)%b.nx);
    out.dx=WAVELENGTH/(b.nx*b.dx*ARC_MIN_TO_RAD);
    out.dy=WAVELENGTH/(b.ny*b.dy*ARC_MIN_TO_RAD);
    out.xmin=-(b.nx/2)*out.dx; out.ymin=-(b.ny/2)*out.dy;
    return out;
}

static std::array<double,3> solve3(std::array<std::array<double,4>,3> a) {
    for(int i=0;i<3;++i) {
        int pivot=i; for(int j=i+1;j<3;++j) if(std::abs(a[j][i])>std::abs(a[pivot][i])) pivot=j;
        if(std::abs(a[pivot][i])<1e-14) throw std::runtime_error("Plane fit is singular.");
        std::swap(a[i],a[pivot]); double d=a[i][i]; for(int k=i;k<4;++k)a[i][k]/=d;
        for(int j=0;j<3;++j) if(j!=i) { double q=a[j][i]; for(int k=i;k<4;++k)a[j][k]-=q*a[i][k]; }
    }
    return {a[0][3],a[1][3],a[2][3]};
}

static std::vector<double> surface_error(const Grid& a, double block, double& rms) {
    std::vector<double> ph(a.nx*a.ny), z(a.nx*a.ny);
    std::array<std::array<double,4>,3> m{}; int count=0;
    for(int y=0;y<a.ny;++y) for(int x=0;x<a.nx;++x) {
        int k=y*a.nx+x; ph[k]=std::arg(a.at(y,x));
        if (ph[k] > 3 * PI / 4) ph[k] -= PI;
        if (ph[k] < -3 * PI / 4) ph[k] += PI;
        double X=a.x(x),Y=a.y(y),r=std::hypot(X,Y);
        if(r>=FIT_R_MIN_M && r<=FIT_R_MAX_M) {
            double v[3]={X,Y,1.0};
            for(int i=0;i<3;++i){ for(int j=0;j<3;++j)m[i][j]+=v[i]*v[j]; m[i][3]+=v[i]*ph[k]; }
        }
    }
    auto c=solve3(m); double ss=0.0; count=0;
    for(int y=0;y<a.ny;++y) for(int x=0;x<a.nx;++x) {
        int k=y*a.nx+x; double X=a.x(x),Y=a.y(y),r=std::hypot(X,Y);
        z[k]=WAVELENGTH*(ph[k]-(c[0]*X+c[1]*Y+c[2]))/(4*PI)*1000.0;
        bool outside_block=std::abs(X)>block/2 || std::abs(Y)>block/2;
        if(r<DIAMETER_M/2 && outside_block && std::abs(z[k])<=SURFACE_LIMIT_MM) {ss+=z[k]*z[k];++count;}
    }
    rms=count?std::sqrt(ss/count):std::numeric_limits<double>::quiet_NaN();
    return z;
}

static std::string gp_path(const fs::path& p) {
    std::string s=fs::absolute(p).generic_string(), out;
    for(char c:s) { if(c=='\'') out+="\\\\'"; else out+=c; }
    return out;
}

static void write_map(const fs::path& p, const Grid& g, const std::vector<double>& v, bool mask_circle=false) {
    std::ofstream o(p); o<<std::setprecision(12);
    for(int y=0;y<g.ny;++y) { for(int x=0;x<g.nx;++x) {
        double z=v[y*g.nx+x]; if(mask_circle && std::hypot(g.x(x),g.y(y))>DIAMETER_M/2) z=std::numeric_limits<double>::quiet_NaN();
        o<<g.x(x)<<" "<<g.y(y)<<" "<<z<<"\n";
    } o<<"\n"; }
}
static std::vector<double> amplitude(const Grid& g){std::vector<double> v(g.nx*g.ny);for(size_t i=0;i<v.size();++i)v[i]=std::abs(g.e[i]);return v;}
static std::vector<double> phase_deg(const Grid& g){std::vector<double> v(g.nx*g.ny);for(size_t i=0;i<v.size();++i)v[i]=std::arg(g.e[i])*180/PI;return v;}

static void write_gnuplot(const fs::path& out, const Options& o, double beam_peak, double ap_peak) {
    std::ofstream g(out/"plot.gp");
    auto p=[&](const std::string& n){return gp_path(out/n);};
    g<<"set terminal pngcairo size 1200,900 enhanced font 'Arial,14'\nset view map\nset pm3d map\nset key off\n";
    auto map=[&](const std::string& data,const std::string& png,const std::string& title,const std::string& cb,double lo,double hi){
      g<<"set output '"<<p(png)<<"'\nset title '"<<title<<"'\nset xlabel 'x [arcmin]'\nset ylabel 'y [arcmin]'\nset cblabel '"<<cb<<"'\nset cbrange ["<<lo<<":"<<hi<<"]\nsplot '"<<p(data)<<"' using 1:2:3 with pm3d\n";
    };
    map("beam_amp.dat","beam_amplitude.png","Complex beam amplitude","Amplitude",0,beam_peak);
    map("beam_db.dat","beam_amplitude_db.png","Complex beam amplitude (dB)","dB",o.db_min,0);
    map("beam_phase.dat","beam_phase.png","Complex beam phase","Phase [deg]",-180,180);
    map("aperture_amp.dat","aperture_amplitude.png","Aperture amplitude","Amplitude",0,ap_peak);
    map("aperture_phase.dat","aperture_phase.png","Aperture phase (tilt removed)","Phase [deg]",-180,180);
    map("surface_error.dat","surface_error.png","Surface error","Surface error [mm]",-SURFACE_LIMIT_MM,SURFACE_LIMIT_MM);
    if(o.polar) map("aperture_rtheta.dat","aperture_rtheta.png","Aperture field: angle-radius","Amplitude",0,ap_peak);
    g<<"unset output\n";
}

static void write_slices(const fs::path& out, const Grid& b, const Grid& a, const Options& o) {
    if(o.slice_beam) {
        std::ofstream f(out/"beam_slice_el0.dat"); int y=std::clamp(static_cast<int>(std::llround(-b.ymin/b.dy)),0,b.ny-1);
        for(int x=0;x<b.nx;++x) f<<b.x(x)<<" "<<std::abs(b.at(y,x))<<" "<<std::arg(b.at(y,x))*180/PI<<"\n";
    }
    if(o.slice_aperture) {
        std::ofstream f(out/"aperture_slice_y0.dat"); int y=std::clamp(static_cast<int>(std::llround(-a.ymin/a.dy)),0,a.ny-1);
        for(int x=0;x<a.nx;++x) f<<a.x(x)<<" "<<std::arg(a.at(y,x))*180/PI<<"\n";
    }
}

static void write_polar(const fs::path& out, const Grid& a) {
    std::ofstream f(out/"aperture_rtheta.dat"); const int nr=160, nt=360;
    for(int ir=0;ir<nr;++ir) { double r=(ir+.5)*DIAMETER_M/(2*nr);
        for(int it=0;it<nt;++it) { double th=(it+.5)*2*PI/nt, X=r*std::cos(th),Y=r*std::sin(th);
            int ix=std::clamp(static_cast<int>(std::llround((X-a.xmin)/a.dx)),0,a.nx-1);
            int iy=std::clamp(static_cast<int>(std::llround((Y-a.ymin)/a.dy)),0,a.ny-1);
            f<<th*180/PI<<" "<<r<<" "<<std::abs(a.at(iy,ix))<<"\n";
        } f<<"\n";
    }
}

int main(int argc, char** argv) {
    try {
        Options o=parse_args(argc,argv);
        fs::create_directories(o.output);
        std::cout<<"Input : "<<o.input<<"\nOutput: "<<o.output<<"\n";
        auto data=read_beam(o.input); correct_reference_phase(data);
        Grid beam=make_grid(data), ap=aperture_from_beam(beam);
        double rms; auto surface=surface_error(ap,o.center_block_size,rms);
        auto ba=amplitude(beam), bp=phase_deg(beam), aa=amplitude(ap), pp=phase_deg(ap);
        double bpeak=*std::max_element(ba.begin(),ba.end()), apeak=*std::max_element(aa.begin(),aa.end());
        std::vector<double> db(ba.size()); for(size_t i=0;i<db.size();++i) db[i]=std::max(o.db_min,20*std::log10(std::max(ba[i],1e-300)/bpeak));
        fs::path out=o.output;
        write_map(out/"beam_amp.dat",beam,ba); write_map(out/"beam_db.dat",beam,db); write_map(out/"beam_phase.dat",beam,bp);
        write_map(out/"aperture_amp.dat",ap,aa,true); write_map(out/"aperture_phase.dat",ap,pp,true); write_map(out/"surface_error.dat",ap,surface,true);
        write_slices(out,beam,ap,o); if(o.polar) write_polar(out,ap);
        { std::ofstream r(out/"result.txt"); r<<std::setprecision(8)<<"Surface RMS [mm] = "<<rms<<"\n"; }
        write_gnuplot(out,o,bpeak,apeak);
        std::string cmd="gnuplot \""+fs::absolute(out/"plot.gp").string()+"\"";
        int rc=std::system(cmd.c_str());
        if(rc!=0) std::cerr<<"[WARN] gnuplot failed. Data files and plot.gp were still written.\n";
        else std::cout<<"PNG files written by gnuplot.\n";
        std::cout<<"Surface RMS = "<<rms<<" mm\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"Error: "<<e.what()<<"\n"; return 2; }
}
