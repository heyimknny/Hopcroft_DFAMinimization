// HopcroftMinimizer.cpp – ultra‑verbose debug build (C++17‑compatible)
//   * Removed the single C++20 `views::iota` call so it now compiles fine
//     with `-std=c++17` (or `-std=c++14`).
//   * Everything else is identical: colourised logging, diff viewer, etc.
//
// Compile: g++ -std=c++17 -O0 -g HopcroftMinimizer.cpp -o hopcroft
// Run    : ./hopcroft 2> debug.log
//
#include <bits/stdc++.h>
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;

static const string Y="\033[33m", C="\033[36m", G="\033[32m", R="\033[31m", B="\033[34m", X="\033[0m";

static vector<string> splitCSV(const string& line) {
    vector<string> out; string cell; stringstream ss(line);
    while (getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

struct DFA {
    int n{}, k{};
    vector<vector<int>> delta;
    int start{};
    vector<char> acc;

    DFA(int N=0,int K=0){reset(N,K);}    
    void reset(int N, int K){n=N;k=K;delta.assign(N,vector<int>(K,-1));acc.assign(N,0);}    

    void debugDump(const string& t) const {
        cerr<<B<<"\n==== "<<t<<" ====="<<X<<"\n";
        cerr<<"n="<<n<<", k="<<k<<", start="<<start<<"\naccept:";
        for(int i=0;i<n;++i) if(acc[i]) cerr<<' '<<i; cerr<<"\nstate |"; for(int c=0;c<k;++c) cerr<<' '<<c; cerr<<"\n";
        for(int s=0;s<n;++s){ cerr<<setw(5)<<s<<" |"; for(int c=0;c<k;++c) cerr<<' '<<delta[s][c]; cerr<<"\n"; }
    }

    void removeUnreachable(){
        vector<char> vis(n); queue<int> q; vis[start]=1; q.push(start);
        while(!q.empty()){
            int s=q.front();q.pop();
            for(int c=0;c<k;++c){int t=delta[s][c]; if(t>=0&&!vis[t]){vis[t]=1; q.push(t);} }
        }
        cerr<<C<<"Reachable:"; for(int i=0;i<n;++i) if(vis[i]) cerr<<' '<<i; cerr<<X<<"\n";
        vector<int> map(n,-1); int m=0; for(int i=0;i<n;++i) if(vis[i]) map[i]=m++;
        if(m==n){ cerr<<C<<"No unreachable removed."<<X<<"\n"; return; }
        cerr<<C<<"Compacting "<<n<<"→"<<m<<X<<"\n";
        vector<vector<int>> nd(m,vector<int>(k,-1)); vector<char> na(m);
        for(int i=0;i<n;++i) if(map[i]!=-1){ int ni=map[i]; na[ni]=acc[i];
            for(int c=0;c<k;++c){int t=delta[i][c]; nd[ni][c]=(t==-1?-1:map[t]); }}
        start=map[start]; n=m; delta.swap(nd); acc.swap(na);
    }

    void makeTotal(){
        bool need=false; for(int s=0;s<n&&!need;++s) for(int c=0;c<k;++c) if(delta[s][c]==-1) need=true;
        if(!need){ cerr<<C<<"Already total."<<X<<"\n"; return; }
        int trap=n++; delta.push_back(vector<int>(k,0)); acc.push_back(0);
        cerr<<C<<"Added trap state "<<trap<<X<<"\n";
        for(int c=0;c<k;++c) delta[trap][c]=trap;
        for(int s=0;s<trap;++s) for(int c=0;c<k;++c) if(delta[s][c]==-1) delta[s][c]=trap;
    }

    DFA minimize() const {
        vector<int> cls(n); vector<vector<int>> blk(2);
        for(int i=0;i<n;++i){ int id=acc[i]?0:1; cls[i]=id; blk[id].push_back(i);} 
        if(blk[0].empty()){ blk.erase(blk.begin()); for(int&x:cls) if(x) --x; }
        deque<int> W; for(int i=0;i<(int)blk.size();++i) W.push_back(i);
        vector<vector<vector<int>>> inv(k, vector<vector<int>>(n));
        for(int s=0;s<n;++s) for(int c=0;c<k;++c) inv[c][delta[s][c]].push_back(s);
        int iter=0;
        while(!W.empty()){
            int A=W.front(); W.pop_front();
            cerr<<Y<<"Iter "<<iter++<<" splitter="<<A<<" size="<<blk[A].size()<<X<<"\n";
            for(int c=0;c<k;++c){
                vector<int> Xvec; for(int t:blk[A]) for(int s:inv[c][t]) Xvec.push_back(s);
                if(Xvec.empty()) continue; cerr<<"  sym="<<c<<" X= "<<Xvec.size()<<"\n";
                unordered_map<int, vector<int>> part; for(int s:Xvec) part[cls[s]].push_back(s);
                for(auto& pr:part){ int bid=pr.first; auto& vec=pr.second; auto& Yblk=blk[bid]; if(vec.size()==Yblk.size()) continue;
                    cerr<<"    split block "<<bid<<" Y="<<Yblk.size()<<" vec="<<vec.size()<<"\n";
                    vector<int> R; R.reserve(Yblk.size()-vec.size()); vector<char> mark(n);
                    for(int s:vec) mark[s]=1; for(int s:Yblk) if(!mark[s]) R.push_back(s);
                    if(vec.size()>R.size()) vec.swap(R);
                    Yblk.swap(vec); int newId=blk.size(); blk.push_back(R);
                    for(int s:R) cls[s]=newId;
                    if(find(W.begin(),W.end(),bid)!=W.end()) W.push_back(newId); else W.push_back(Yblk.size()<=R.size()?bid:newId);
                }
            }
        }
        cerr<<G<<"Minimized: "<<blk.size()<<" states"<<X<<"\n";
        DFA M(blk.size(),k);
        vector<int> rep(n); for(int i=0;i<(int)blk.size();++i) for(int s:blk[i]) rep[s]=i;
        M.start=rep[start]; for(int i=0;i<(int)blk.size();++i) M.acc[i]=acc[blk[i][0]];
        for(int i=0;i<(int)blk.size();++i) for(int c=0;c<k;++c) M.delta[i][c]=rep[delta[blk[i][0]][c]];
        return M;
    }
};

static bool readCSV(const string& p,DFA& d,vector<string>& alpha){
    cerr<<B<<"Reading "<<p<<X<<"\n"; ifstream fin(p); if(!fin){ cerr<<R<<"Open fail"<<X<<"\n"; return false; }
    string header; getline(fin,header); auto h=splitCSV(header); if(h.size()<4){ cerr<<R<<"Header short"<<X<<"\n"; return false; }
    alpha.assign(h.begin()+3,h.end()); int k=alpha.size(); cerr<<"α="<<k<<"\n";
    vector<vector<string>> rows; string line; while(getline(fin,line)) if(!line.empty()) rows.push_back(splitCSV(line));
    int n=rows.size(); if(!n){ cerr<<R<<"No rows"<<X<<"\n"; return false; }
    d.reset(n,k);
    unordered_map<string,int> id; for(int i=0;i<n;++i) id[rows[i][0]]=i;
    for(int i=0;i<n;++i){ auto& r=rows[i]; if((int)r.size()!=3+k){ cerr<<R<<"Rowlen mis"<<X<<"\n"; return false; }
        if(r[1]=="1") d.start=i; if(r[2]=="1") d.acc[i]=1;
        for(int c=0;c<k;++c){ string tgt=r[3+c]; d.delta[i][c]= tgt.empty()?-1:id[tgt]; }
    }
    return true;
}

static void writeCSV(const string& p,const DFA& d,const vector<string>& alpha){
    cerr<<B<<"Writing "<<p<<X<<"\n"; ofstream out(p);
    out<<"state,is_start,is_accept"; for(auto&s:alpha) out<<','<<s; out<<'\n';
    for(int s=0;s<d.n;++s){ string name="q"+to_string(s); out<<name<<','<<(s==d.start)<<','<<(int)d.acc[s];
        for(int c=0;c<d.k;++c){int t=d.delta[s][c]; out<<','<<(t==-1?"":"q"+to_string(t));} out<<'\n'; }
}

static bool diff(const string&a,const string&b){ ifstream f1(a),f2(b); string l1,l2; bool ok=true; int ln=1;
    while(true){ bool e1=!getline(f1,l1); bool e2=!getline(f2,l2); if(e1||e2){ if(!(e1&&e2)){ ok=false; cerr<<R<<"EOF mis"<<X<<"\n";} break; } if(l1!=l2){ ok=false; cerr<<R<<"line"<<ln<<X<<"\nexp:"<<l2<<"\nget:"<<l1<<"\n";} ++ln; } return ok; }

int main(){ ios::sync_with_stdio(false); cin.tie(nullptr);
    cerr<<Y<<"Dir ./io"<<X<<"\n"; for(auto& p:fs::directory_iterator("io")) cerr<<"  • "<<p.path().filename().string()<<"\n";
    string base="io"+string(1,fs::path::preferred_separator);
    for(int idx=1;;++idx){ string in=base+"sample.in."+to_string(idx); if(!fs::exists(in)){ if(idx==1) cerr<<R<<"No tests"<<X<<"\n"; break; }
        cout<<G<<"Test"<<idx<<X<<"\n"; DFA d; vector<string> alpha; if(!readCSV(in,d,alpha)){continue;} d.debugDump("orig"); d.removeUnreachable(); d.makeTotal(); d.debugDump("tot"); DFA m=d.minimize(); m.debugDump("min"); string gen=base+"generated.out."+to_string(idx); writeCSV(gen,m,alpha); string exp=base+"sample.out."+to_string(idx); if(fs::exists(exp)){ bool ok=diff(gen,exp); cout<<(ok?G+"✅":"❌")<<X<<"\n";} }
    return 0; }
