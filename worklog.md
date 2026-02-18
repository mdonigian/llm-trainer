## Work
* Downloaded fineweb-edu from: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/data/CC-MAIN-2021-04/train-00000-of-00022.parquet
* Ran categorize_fineweb_edu.py using gpt-5-nano
* cost about ~$80 in OpenAI batch credits
* Grabbed another 50K rows from https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/sample/10BT/004_00000.parquet so that I'm not overindexing to 2021-04, would have been ideal to take from the sample dataset to begin with, live and learn
* model is getting F1 of ~0.85, this isn't bad at all for the number of categories, it's good enough
* used the model to infer a bunch of labels for the fineweb 10b sampler
*took a random sample of 100,000 records from this batch with ~6K from each label, uploaded them to the OpenAI batch api to label them according to "complexity". This will allow me to build a fineweb dataset that is broken down by complexity and subject.
* Starting code classification pipeline with bigcode/starcoderdata. Downloading 200K samples (25K per language) across Python, JavaScript, TypeScript, Java, Go, Rust, SQL, Shell. Classifying on 3 dimensions: code quality (1-5), structured data relevance (0-3), content type (9 categories). Using gpt-5-nano via Batch API.



