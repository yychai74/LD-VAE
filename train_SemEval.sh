# Train for SemEval dataset

# 1. Train text filter first.
python bert_main.py --dataset "SemEval" --train --eval --train_filter

# 2. Train text generator LD-VAE/LS-PT, and generate augmentation data.
python main.py --dataset "SemEval" --model "LD_VAE" --train --eval
# python main.py --dataset "SemEval" --model "LS_PT" --train --eval

# 3. Train classifier with augmentation data.
python bert_main.py --dataset "SemEval" --train --eval
