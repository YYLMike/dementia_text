import cluster
import numpy as numpy


def main():
    test_cluster = cluster.Cluster()
    test_cluster.syntactic_analysis('jieba')
    test_cluster.load_sentence_vector('s2v_array_zhs_500dim.pickle')
    _, _ = test_cluster.evaluate('Syntactic')
    test_cluster.plot_cluster()
    _, _ = test_cluster.evaluate('Semantic')
    test_cluster.plot_cluster()
    _, _ = test_cluster.evaluate('Syntactic_Semantic')
    test_cluster.plot_cluster()
    
if __name__ == '__main__':
    main()
