OUTDIR=models/librispeech
mkdir -p $OUTDIR

python train.py \
  --rnn-type gru \
  --hidden-size 800 \
  --hidden-layers 5 \
  --checkpoint \
  --train-manifest data/libri_train_manifest.csv \
  --val-manifest data/libri_val_manifest.csv \
  --epochs 15 \
  --num-workers 2 \
  --cuda --checkpoint \
  --checkpoint-per-batch 10000 \
  --save-folder $OUTDIR \
  --model-path $OUTDIR/best.pth \
  --batch-size 10 --learning-anneal 1.1
