# NLP_Project
 Course project of CSCI 544: Applied Natural Language Processing

## How to run
```shell script
python train.py --model_name aen_simple 
                --dataset laptop 
                --bert_type roberta 
                --pretrained_bert_name roberta-base 
                --lstm_hid 300 
                --lstm_layer 1 
                --lstm_bidir 
                --mha_heads 4
```
+ On the first run, it may need to download pre-trained RoBERTa files through internet.