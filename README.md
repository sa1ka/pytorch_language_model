# lstm_language_model

a pytorch version lstm language model, support class-based softmax for speeding up (Following the [paper](https://arxiv.org/pdf/1602.01576.pdf)).

In class-based softmax, each word is assigned to one class, hence the probability of a word become:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(w|h)&space;=&space;p(c|h)p(w|c,h)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(w|h)&space;=&space;p(c|h)p(w|c,h)" title="p(w|h) = p(c|h)p(w|c,h)" /></a>

Theoretically, the computational cost can be reduced from O(dk) to O(d\sqrt{k}), where d is the size of last hidden layer and k is the size of vocabulary.

But in pratice, there are too many overhead (especially in GPU).

## Usage 

Run the following script the build a vocab with class:

```
python build_vocab_with_class.py --ncls 30 --min_count 0
```

The vocab built above is based on the frequence, you can also build your own vocab using other methods. (see example in ./data/penn/vocab.c.txt, Notice that the class should be a integer.)

Run training script:
```
python train.py --cuda --data [data_path] --cls
```

Or run the standard softmax model:
```
python train.py --cuda --data [data_path]
```

## Performance
