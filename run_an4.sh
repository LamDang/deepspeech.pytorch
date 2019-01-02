OUTDIR=models/an4
mkdir -p $OUTDIR

python train.py \
  --rnn-type gru \
  --hidden-size 800 \
  --hidden-layers 5 \
  --checkpoint \
  --train-manifest data/an4_train_manifest.csv \
  --val-manifest data/an4_val_manifest.csv \
  --epochs 50 \
  --num-workers 4 \
  --cuda \
  --augment \
  --model-path $OUTDIR/best.pth \
  --batch-size 32 --learning-anneal 1.01 | tee $OUTDIR/train.log
