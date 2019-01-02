OUTDIR=models/librispeech_norm
mkdir -p $OUTDIR

python train.py \
  --rnn-type gru \
  --hidden-size 800 \
  --hidden-layers 5 \
  --lr 3e-4 \
  --checkpoint \
  --train-manifest data/libri_train_manifest.csv \
  --val-manifest data/libri_val_manifest.csv \
  --epochs 15 \
  --num-workers 4 \
  --cuda --checkpoint \
  --save-folder $OUTDIR \
  --model-path $OUTDIR/best.pth \
  --normalize_by_frame \
  --learning-anneal 1.1 | tee $OUTDIR/train.log
