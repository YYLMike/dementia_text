import CNN_text_classifier

print('Paragraph level ...\n')
test_cnn_model = CNN_text_classifier.CNN_text_classifier('paragraph', cv=False)
test_cnn_model.load_data()
test_cnn_model.load_embedding_matrix()
test_cnn_model.train_model()
test_cnn_model.test_model()
test_cnn_model.save_model()

print('Sentence level with cv ...\n')
test_cnn_model = CNN_text_classifier.CNN_text_classifier('sentence', cv=True)
test_cnn_model.load_data_total()
test_cnn_model.load_embedding_matrix()
test_cnn_model.train_model()
