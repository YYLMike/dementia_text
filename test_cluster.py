import cluster
import numpy as numpy

W2V_MODEL_ZHT = '500features_20context_20mincount_zht'

def main():
        print('Start Scenario 1, Kmean Clustering with semi-labeled data\nUsing jieba syntactic, jieba semantic features ...')
        test_cluster = cluster.Cluster()
        test_cluster.syntactic_analysis('jieba')
        test_cluster.write_syntactic_feature('syntactic_score.txt')
        test_cluster.prior_semantic_analysis()
        _, result_1_syntactic = test_cluster.evaluate('Syntactic', 'result_1_syntactic.pickle')
        test_cluster.plot_cluster('syntactic_scenario_1')
        # _, result_1_semantic = test_cluster.evaluate('Semantic', 'result_1_semantic.pickle')
        # test_cluster.plot_cluster('semantic_scenario_1')
        # _, result_1_both = test_cluster.evaluate('Syntactic_Prior_Semantic', 'result_1_both.pickle')
        # test_cluster.plot_cluster('both_scenario_1')

        # print('Start Scenario 2, Kmean Clustering with semi-labeled data\nUsing ckip syntactic, ckip semantic features ...')
        # test_cluster2 = cluster.Cluster(W2V_MODEL_ZHT)
        # test_cluster2.syntactic_ckip_load()
        # test_cluster2.semantic_analysis(segment_tool='ckip')
        # _, result_2_syntactic = test_cluster2.evaluate('Syntactic', 'result_2_syntactic.pickle')
        # # test_cluster2.plot_cluster('syntactic_scenario_2')
        # _, result_2_semantic = test_cluster2.evaluate('Semantic', 'result_2_semantic.pickle')
        # # test_cluster2.plot_cluster('semantic_scenario_2')
        # _, result_2_both = test_cluster2.evaluate('Syntactic_Prior_Semantic', 'result_2_both.pickle')
        # # test_cluster2.plot_cluster('both_scenario_2')

        # print('Start Scenario 3, Kmean Clustering with semi-labeled data\nUsing jieba syntactic, ckip semantic features ...')
        # test_cluster3 = cluster.Cluster()
        # test_cluster3.syntactic_analysis('jieba')
        # test_cluster3.semantic_analysis(segment_tool='ckip')
        # _, result_3_both = test_cluster3.evaluate('Syntactic_Prior_Semantic', 'result_3_both.pickle')
        # test_cluster3.plot_cluster('both_scenario_3')

        # print('Start Scenario 4, Kmean Clustering with semi-labeled data\nUsing ckip syntactic, jieba semantic features ...')
        # test_cluster4 = cluster.Cluster()
        # test_cluster4.syntactic_ckip_load()
        # test_cluster4.semantic_analysis()
        # # test_cluster4.load_sentence_vector('s2v_array_zhs_500dim.pickle')
        # _, result_4_both = test_cluster4.evaluate('Syntactic_Prior_Semantic', 'result_4_both.pickle')
        # test_cluster4.plot_cluster('both_scenario_4')
        
if __name__ == '__main__':
    main()
