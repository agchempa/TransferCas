cat data/raw/train_*.tsv > data/train.tsv
cat data/raw/val_*.tsv > data/val.tsv
cat data/raw/test_*.tsv > data/test.tsv

awk '{ print $1 }' data/val.tsv > data/val_test_guides.txt
awk '{ print $1 }' data/test.tsv >> data/val_test_guides.txt
grep -Fv -f data/val_test_guides.txt data/train.tsv > data/temp
cp data/temp data/train.tsv
