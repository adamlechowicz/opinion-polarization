#include <iostream> 
#include <unordered_map> 
#include <fstream>
#include <string> 

using namespace std;

// Author: C.E. Tsourakakis
// Date: 19 October 2017
// Preprocess Twitter opinion data

int main(int argc, char** argv){

	unordered_map<long,long> hm;
    
    string edgef = "edgelist.txt";
    string ndf = "nodelist.txt";
    string opinionf = "opinion.txt";
    
    
    string opinions_out = "opinion_twitter.txt";
    string nodes_out = "node_map_twitter.txt";
    string edges_out = "edges_twitter.txt";
    
    
    ofstream node_out;
    node_out.open(nodes_out.c_str());
    ofstream opinion_out;
    opinion_out.open(opinions_out.c_str());
    ofstream edge_out;
    edge_out.open(edges_out.c_str());

    
    // treating nodes
    ifstream nodeFile;
    nodeFile.open(ndf.c_str());
    if(!nodeFile){
        cerr<<"Unable to open node file "<<ndf<<endl;
        return -1;
    }
    
    long u,counter=1;
    while(nodeFile >> u){
        node_out<<u<<"\t"<<counter<<endl;
        hm[u] = counter;
        counter++;
    }
    
    
    // treating edges
    ifstream edgeFile;
    edgeFile.open(edgef.c_str());
    if(!edgeFile){
        cerr<<"Unable to open edge file "<<edgef<<endl;
        return -1;
    }
    
    
    long v;
    while(edgeFile >> u >> v)
        edge_out<<hm[u]<<"\t"<<hm[v]<<endl;
        
    
    
    edgeFile.close();
    edge_out.close();
    
    
    // opinions time ids (node id, time tick,
    
    unordered_map<long,long> timeh;
    ifstream opFile;
    opFile.open(opinionf.c_str());
    
    long timetick;
    double val;
    
    long T = 0;
    double minop=100000.0, maxop=-100000.0; // large enough values
    while(opFile>>u>>timetick>>val){
        
        if( val < minop ){
            minop = val;
        }
        if(val>maxop){
            maxop = val;
        }
        std::unordered_map<long,long>::const_iterator got = timeh.find (timetick);

        if(got == timeh.end()) {
            timeh[timetick] = T;
            T++;
        }
    }
    
    opFile.clear();
    opFile.seekg(0, opFile.beg);
    while(opFile>>u>>timetick>>val){
        
        val = (val-minop)/(maxop - minop);
        opinion_out << hm[u] <<"\t"<< timeh[timetick] << "\t"<<val<<endl;
    }
    opinion_out.close();
    opFile.close();
    return 0;
}
