# QA-Net
Implementation by Dewal Gupta

This repository contains all the tools necessary to downloading, preprocessing, 
training and testing the QA-Net model. 

<img src="/model.png"></img>

## Downloading the data

Downloading script used was adapted from @NLPLearn (https://github.com/NLPLearn/QANet). This script
downloads the training and dev data from the SQuAD links and downloads the Glove embeddings as well. 

Files: ```download.py```

## Preprocessing

The preprocessing script was taken from @NLPLearn who adapted
 it from https://github.com/HKUST-KnowComp/R-Net. This script builds several JSON files
 that contain data preprocessed with Glove Embeddings for the SQuAD dataset. This allows
 us to load word and character level embeddings directly into our model, where we can do
 what we want with them. 
 
This script also created TFRecords file to make input into the model easier, and helps with the
 creation of the tf.data.Dataset. 
 
Files: ```util.py``` and ```prepro.py```

## Model Implementation

The model was implemented in 5 different parts as the original paper had broken them into. The model
is implemented in the class ```QANet``` and the main, big picture implementation is inside the 
method ```forward(self, context, ques, context_char, ques_char)```. As shown, it takes in both the 
word and character embeddings for both the context and the question. This function implements all 5
layers but uses helper functions and classes to do so. These specifics are detailed below.

Files: ```model.py```

###Input Embedding Layer
The input embedding layer is similar to Seo et al's embedding layers. Essentially, 
the Glove embeddings per word and the character embeddings are obtained by using 
a 1D convolution on the character embeddings and finding the largest element per row, then 
reducing the vector to those numbers (tf.reduce_max). We use this representation for our character
level embeddings. After calculating these for both the context and question, we pass them through
the same highway network (Srivastava et al., 2015). The weights are shared in this highway network, 
and it returns to a vector with of length 96. We concatenate this to our word embedding and use this
representation as our final embedding. 

The paper originally uses an character embedding vector of length 200, but for the purposes of faster
training times, this was shortened to 96. Also other users (@NLPLearn) have reported similar enough 
performances showing that a 200 dimensional embedding isn't entirely necessary. 

See function ```embed(self, word, char, is_context=False)```

###Embedding Encoding Layer
The encoding layer is built using the stacked modules that are novel to this paper. We used
the same hyperparameters as the paper suggested and implemented this stacked block as a separate
class, ```EncoderBlk```. Each block consists of a positional encoding, convolutions, a self attention
mechanism, and then a feedforward layer. 

The self attention layer was a little tricky to implement due to the fact that it required key-value-query triplets
to learn attention over, but it was not entirely clear from this work what those corresponded to in 
the question-answer encodings. As a result, to compensate, we build a learnable fully connected layer to 
transform our original embeddings to each of the K, V or Q embeddings over which the attention can work. The hope
is that the network can correctly project the embedding so that attention is able to work. 

The self attention mechanism is taken from another work (as defined in Vaswani et al., 2017) and 
the implementation is not my own. It was originally implemented by @DongjunLee (https://github.com/DongjunLee/transformer-tensorflow/).

See ```class EncoderBlk```

###Context-Query Layer
This layer was implemented using tf.map_fn due to the fact that we are working in batches, and I could not find a
more efficient, batch-friendly way to implement these transformations. 

In this layer, the authors generate two attention matrices, a context-to-query and a query-to-context matrix
using the same transformations as defined by Seo et al. in their BiDAF network.  

See function ```context_query_att(self, c, q):```

###Model Encoder Layer
The model encoder layer is composed of 3 of the same encoder set of blocks with 7 blocks in one set. The same
encoder block class is used (initialized in the QANet constructor). The same hyperparameters are used for the
implementations of these blocks, and the weights are shared between all 3 sets of blocks. 

###Output Layer
This is a relatively simple layer just responsible for predicting the start and end indices of the answer. 
There are two branches (one to predict start, and the other the end). The start branch takes the concatenated
output of the first 2 sets of encoder blocks, whereas the end branch takes the concatenated output from the 
first and third sets of the blocks. These features are linearly projected and then softmaxed to output predictions.

See ```class EncoderBlk```

## Training
The training script builds a dataset pipeline using the methods from ```util.py``` and uses it to input
into the model. The training script is designed to be used with multiple GPUs, though this can be adjusted
in the config. It runs a validation set every ```VAL_ITER``` and saves the model every ```SAVE_ITER``` both of 
which can be edited in config.ini. 

See ```train.py```

## Evaluation
The evaluation function is adapted from @NLPLearn as well, and is used to determine the F1 and EM scores
of the model. 

## Results