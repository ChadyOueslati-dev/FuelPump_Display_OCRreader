# Training the number reader on the GPU

PaddleOCR reads text in two steps: one model finds where the text is, a
second one (the recognition model) reads each piece. The second one is what
misreads pump digits: it was trained on printed text, not on 7-segment
displays. This folder fine-tunes it on pump numbers.

Everything large lives outside the project, in `C:\ocr-train`, so it is not
synced to OneDrive:

    C:\ocr-train\PaddleOCR     PaddleOCR v3.7.0 source (the training code)
    C:\ocr-train\pretrained    the released weights training starts from
    C:\ocr-train\data          training pictures
    C:\ocr-train\output        what training produces

Training uses its own environment, `C:\venvs\pump-ocr-train`, with the GPU
build of PaddlePaddle. The app's environment is not touched.

## Steps

1. **Generate numbers with known values.** Draws 7-segment numbers in many
   styles and applies camera effects.

       python scripts\train\make_rec_dataset.py --out C:\ocr-train\data\synthetic --train 30000 --val 1500

2. **Harvest real crops.** Cuts the numbers out of labelled photos, with the
   true text taken from the labels. Only from photos that are NOT used to
   measure accuracy.

       python scripts\train\harvest_crops.py --photos FOLDER --labels labels.csv --out C:\ocr-train\data\real_NAME

3. **Merge** into the two list files the config reads.

       python scripts\train\merge_lists.py --data C:\ocr-train\data

4. **Train**, from the PaddleOCR folder, in the training environment.

       cd C:\ocr-train\PaddleOCR
       C:\venvs\pump-ocr-train\Scripts\python.exe tools\train.py -c <project>\scripts\train\pump_rec_finetune.yml

5. **Export** the best checkpoint as an inference model.

       C:\venvs\pump-ocr-train\Scripts\python.exe tools\export_model.py -c <project>\scripts\train\pump_rec_finetune.yml -o Global.pretrained_model=C:/ocr-train/output/pump_rec/best_accuracy Global.save_inference_dir=C:/ocr-train/output/pump_rec_infer

6. **Use it.** Set `recognition_model_dir` in `config\settings.yaml` to the
   exported folder, then measure with `scripts\evaluate.py`.

## Which photos are for what

- **Training:** the generated numbers, plus crops from `data\pump_ads_dataset`.
- **Measuring:** `data\Pump Displays_Large` and `data\Pump Displays_Small`.
  These must never be used for training, or the measured accuracy means
  nothing.

All of these photos come from the same image generator. A high score on them
shows the model learned that generator's pumps. It does not prove accuracy
on real pumps; only labelled real photos can show that.
