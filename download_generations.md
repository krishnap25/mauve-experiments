# Download Text Generations

We provide the text generations from GPT-2 (various sizes) on the webtext datasets.
These samples were used in much of the empirical evaluation in our [NeurIPS 2021 paper](https://arxiv.org/pdf/2102.01454.pdf)
and the subsequent [longer version](https://arxiv.org/pdf/2212.14578.pdf) (under review as of June 2023).


**Trigger Warning**: 
The generated text is sampled from GPT-2 models of various sizes. It could be biased, harmful, racist, sexist, toxic, and potentially upsetting.
Please use at your own risk.
Do not treat model outputs as substitutes for human judgment or as sources of truth. Please use responsibly.


## Download the data
The data can be found at [this Google Drive link](https://drive.google.com/file/d/1DlmEQ3zgaBMKDRA-Yu5VFD-xw0JvJOft/view?usp=sharing).
The MD5 checksum of `mauve_generations.tgz` is `63bae977e3ce5f3c86d9e35188c1b8e6`.

You can alternatively download it via the command line using [gdown](https://github.com/wkentaro/gdown) as 

```bash
pip install gdown  # Install gdown if you do not have it
file_id="1DlmEQ3zgaBMKDRA-Yu5VFD-xw0JvJOft"  # ID of the file on Google Drive
gdown https://drive.google.com/uc?id=${file_id}  # Download the generations
md5sum mauve_generations.tgz  # verify that it is "63bae977e3ce5f3c86d9e35188c1b8e6"
tar -zvxf mauve_generations.tgz  # Uncompress the generations
```

This downloads `mauve_generations.tgz` whose size is 992M compressed and 2.1G uncompressed.

## Data format
The folder structure is `mauve_experiments/webtext_${model_name}/sample_test_p${p}_k${k}_t1.0_seed${seed}.p`, where 
* `model_name` is the name of the model and takes values from `['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']` corresponding to the four sizes of GPT-2
* `p` is the top-p parameter of nucleus sampling, and takes values `[0.9, 0.92, 0.95, 0.99, 1.0]`
* `k` is top-k parameter for top-k sampling, and takes values `[0, 1]`  (the latter for greedy decoding)
* `seed` is the random seed and takes values in `[0, 1, 2, 3, 4]`.

The generations are stored as Python Pickle archives.
Follow the [recommended precautions](https://docs.python.org/3/library/pickle.html) when dealing with pickle archives and use at your own risk.

## Load the generations
Each pickle file can be loaded as follows:
```python
import pickle as pkl
filename = "mauve_generations/webtext_gpt2-large/sample_test_p0.95_k0_t1.0_seed1.p"  # Or choose your own
with open(filename, "rb") as f:
    generations = pkl.load(f)[0]
```

The object `generations` is a list of length 5000, one for each example of the testset of webtext (available from [here](https://github.com/openai/gpt-2-output-dataset), see also the [README](https://github.com/krishnap25/mauve-experiments/blob/main/README.md).
Each entry is a list of integers, representing the BPE tokens used by GPT-2. To get the raw detokenized text, you can run

```python
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(tokenizer.decode(generations[0]))  # => de-tokenized generations
```

## Plugging the generations into the rest of the experimental pipeline

The format of files written by the scripts of this repository is described in the [README](https://github.com/krishnap25/mauve-experiments/blob/main/README.md).
We provide only the `sample*` files, while the `sentences*` and `feats*` files will still have to be created.

To this end, first move each file from `mauve_generations/webtext_{model_name}/` to `./outputs/webtext_{model_name}/generations/basic/`. Then, follow the instructions [here](https://github.com/krishnap25/mauve-experiments/blob/main/README.md#experimental-pipeline).
This will skip the generation and proceed to featurizing the samples directly (as enforced by [this check](https://github.com/krishnap25/mauve-experiments/blob/main/generate_basic.py#L43)).
