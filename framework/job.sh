#!/bin/sh



for i in 0.01 0.0001 0.000001 0.00000001
do
  python3 svm_training.py -f ../features2019/libsvm.txt -p ../model2019 -i 1000000000 -t "$i"
  python3 model_ranking.py -m ../model2019/model -f ../features2019/libsvm.txt -g ../data2019/fair-TREC-sample-author-groups.csv -l ../features2019/linker.txt
  # python3 adjust_for_exposure.py -m ../model2020/model -f ../features2020/libsvm.txt -g ../data2020/fair-TREC-sample-author-groups.csv -l ../features2020/linker.txt
  # python3 model_ranking.py -m ../model2020/model -f ../features2020/libsvm.txt -g ../data2020/fair-TREC-sample-author-groups.csv -l ../features2020/linker.txt
done



# python3 svm_training.py -f ../features2020/libsvm.txt -p ../model2020 -i 1000000000 -t "$i"
# python3 model_ranking.py -m ../model2020/model -f ../features2020/libsvm.txt -g ../data2020/fair-TREC-sample-author-groups.csv -l ../features2020/linker.txt
