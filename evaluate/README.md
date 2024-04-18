# Evaluation

To run evaluation, run

```
python evaluation_script.py --edc_output /path/to/edc_output.txt --reference /path/to/reference.txt --max_length_diff N
```

The evaluation script is adopted from the [WebNLG evaluation script](https://github.com/WebNLG/WebNLG-Text-to-triples). It is to be noted that the evaluation script works by enumerating or possible alignment between the output triplets and reference triplets, so the evaluation speed may be very slow, this is expected. You may pass a `max_length_diff` to filter out some triplets for faster evaluation.