This project is a prototype for a parallel model runner built on Apache Beam. It performs inference using BertForMaskedLM library in Transformers with the `bert-base-uncased` model.

Steps to run are:

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Generate the model files (.pth files) by running:

```
python generate_model_files.py
```

3. Perform inference:

```
python pytorch_language_modeling.py --output output.json > console.log 2>&1
```

4. Check the console log:

```
cat console.log
```

You should see the models getting loaded:

```
INFO:root:Loading state_dict_path model_a.pth onto a cpu device
INFO:root:Finished loading PyTorch model.
INFO:root:Loading state_dict_path model_c.pth onto a cpu device
INFO:root:Finished loading PyTorch model.
```

Then the output being generated:

```
Model Name: model_d.pth - Output: The biggest animal in the world is the whale .;elephant
Model Name: model_e.pth - Output: The secret ingredient to his wonderful life was gratitude .;love
Model Name: model_e.pth - Output: The biggest animal in the world is the whale .;elephant
Model Name: model_a.pth - Output: The secret ingredient to his wonderful life was gratitude .;love
```

5. Check the inference output:

```
cat output.json | jq
```

You should see lines like the following (ordering is not guaranteed):

```
{
  "model_id": "model_b",
  "text": "The secret ingredient to his wonderful life was gratitude .",
  "decoded_word": "love"
}
{
  "model_id": "model_b",
  "text": "The biggest animal in the world is the whale .",
  "decoded_word": "elephant"
}
```