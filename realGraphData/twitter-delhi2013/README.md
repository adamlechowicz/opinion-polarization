Preprocesses a Twitter dataset on Delhi Assembly elections 2013. 

- Compile using g++ preprocess.cc.
- Then,  run ./a.out from a terminal. 

The code generates three files. 
1. opinion_twitter.txt formatted as node id, time tick, opinion value (in [0,1])
2. node_map_twitter.txt formatted as original node id, node id (in 1..n) 
3. edges_twitter.txt that contains the twitter connections using node ids from 1 to n. 
