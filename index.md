---
layout: default
---

# 👾108.413A(002) Computational Linguistics💻


### Course Information

* Instructor: Sangah Lee (sanalee@snu.ac.kr)
* TA: Minji Kang (mnjkng@snu.ac.kr)

This course outlines the fundamental notions and theories on computational linguistics and natural language processing, dealing with current issues on deep learning models and the Transformer mechanism and large-scale language models based on them.

* Prerequisite course: Language and Computer (or, students that can implement at least the logistic regression model in Python are allowed)
* Students majoring in engineering could get separate grades.


### Resources
* [Jurafsky and Martin (2023 draft), “Speech and Language Processing”](https://web.stanford.edu/~jurafsky/slp3/)
* [Practical Deep Learning with PyTorch](https://www.deeplearningwizard.com/deep_learning/course_progression/)
* [Jupyter Notebook](https://jupyter.org/)
  * [Jupyter notebook for beginners-A tutorial](https://towardsdatascience.com/jupyter-notebook-for-beginners-a-tutorial-f55b57c23ada)
* [Google Colabatory](https://colab.research.google.com/notebooks/welcome.ipynb)
  * [Primer for Learning Google CoLab](https://medium.com/dair-ai/primer-for-learning-google-colab-bb4cabca5dd6)
  * [Google Colab - Quick Guide](https://www.tutorialspoint.com/google_colab/google_colab_quick_guide.htm)

* [Numpy and Data Representation](https://jalammar.github.io/visual-numpy/)
* [NLTK (Natural Language Toolkit)](https://www.nltk.org/)
* [SpaCy](https://spacy.io/)
* [textacy](https://textacy.readthedocs.io/en/latest/)
* [csv](https://docs.python.org/3/library/csv.html)
* [json](https://docs.python.org/3/library/json.html)
* [Python "Class"](https://docs.python.org/3/tutorial/classes.html)


### Syllabus

* **Week 0 (3/2 Thu)** Course Introduction
  * [slide](https://github.com/sanajlee/cl2023u/raw/main/cl0_courseintro.pdf)

* **Week 1 (3/7, 3/9)** Basics of Text Processing
  * [Natural Language Processing is Fun!](https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e)
  * [Natural Langauge Processing with Python](https://www.nltk.org/book/)
  * [Hands-on-nltk-tutorial](https://github.com/hb20007/hands-on-nltk-tutorial)
  * PyTorch
    * [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
    * [Tutorial](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)
    * [Matrices](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_matrices/)
    * [Gradients](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_gradients/)
    * [Linear Regression](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/)
    * [Logistic Regression](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_logistic_regression/)
  
* **Week 2 (3/14, 3/16)** PyTorch: Feed Forward Neural Network, Recurrent Neural Networks
  * [Feed Forward Neural Network (FFN)](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/)
  * [Recurrent Neural Network (RNN)](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_recurrent_neuralnetwork/)

* **Week 3 (3/21, 3/23)** PyTorch: Convolutional Neural Networks, Long Short Term Neural Network (LSTM)
  * [Convolutional Neural Network (CNN)](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_convolutional_neuralnetwork/)
  * [Long Short-Term Memory (LSTM)](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/)

* **Week 4 (3/28, 3/30)** Language Model I: Statistical Language Model (N-gram)
  * [Textbook](https://web.stanford.edu/~jurafsky/slp3/3.pdf)

* **Week 5 (4/4, 4/6)** N-gram and Entropy
  * [Entropy is a measure of uncertainty](https://towardsdatascience.com/entropy-is-a-measure-of-uncertainty-e2c000301c2c)

* **Week 6 (4/11, 4/13)** Text Classification
  * [Textbook](https://web.stanford.edu/~jurafsky/slp3/4.pdf)

* **Week 7 (4/18, 4/20)** Language Model II: Word Embedding
  * [Textbook](https://web.stanford.edu/~jurafsky/slp3/6.pdf)
  * [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
  * [On Word Embeddings (Sebastian Ruder)](https://www.ruder.io/word-embeddings-1/)
  * [📚The Current Best of Universal Word Embeddings and Sentence Embeddings](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
  * [Python Tutorial](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
  * [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py)

* **Week 8 (4/25, 4/27)** Midterm Exam

* **Week 9 (5/2, 5/4)** Sequence-to-Sequence Model (Encoder-Decoder)
  * [Encoder-Decoder Long Short-Term Memory Networks](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)
  * [A Gentle Introduction to LSTM Autoencoders](https://machinelearningmastery.com/lstm-autoencoders/)
  * [Step-by-step understanding LSTM Autoencoder layers](https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352)
  * [PyTorch: Sequence to Sequence Learning with Neural Networks.ipynb](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)

* **Week 10 (5/9, 5/11)** Attention
  * [Introduction to Attention Mechanism](https://ai.plainenglish.io/introduction-to-attention-mechanism-bahdanau-and-luong-attention-e2efd6ce22da)
  * [Attn: Illustrated Attention](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
  * [Attention and Memory in Deep Learning and NLP](https://dennybritz.com/posts/wildml/attention-and-memory-in-deep-learning-and-nlp/)
  * [PyTorch: Translation with Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

* **Week 11 (5/16, 5/18)** Transformer
  * [Vaswani et al. (2017), Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
  * [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  * [Transformers Illustrated!](https://tamoghnasaha-22.medium.com/transformers-illustrated-5c9205a6c70f)
  * Seq2seq pay Attention to Self Attention: [Part 1](https://bgg.medium.com/seq2seq-pay-attention-to-self-attention-part-1-d332e85e9aad) [Part 2](https://bgg.medium.com/seq2seq-pay-attention-to-self-attention-part-2-cf81bf32c73d)
  * [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

* **Week 12 (5/23, 5/25)** Transformer-based Pre-trained Models
  * [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)
  * [FROM Pre-trained Word Embeddings TO Pre-trained Language Models — Focus on BERT](https://towardsdatascience.com/from-pre-trained-word-embeddings-to-pre-trained-language-models-focus-on-bert-343815627598)
  * Dissecting BERT [Part 1](https://medium.com/@mromerocalvo/dissecting-bert-part1-6dcf5360b07f) [Part 2](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73) [Part 3](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f)
  * [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
  * [BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)
  * [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)
  * [How GPT3 Works - Visualizations and Animations](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)
  * [Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
 
* **Week 13 (5/30, 6/1)** HuggingFace Transformer
  * [GitHub](https://github.com/huggingface/transformers)
  * [Full Documentation](https://huggingface.co/docs/transformers/index)

* **Week 14 (6/8 Thu, 6/13 Tue)** Various NLP Tasks based on Transformer

* **Week 15 (6/15 Thu)** Final Project Presentations


