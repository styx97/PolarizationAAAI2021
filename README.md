# Code to find alignment between two corpuses as described in [We Don't Speak the Same Language: Interpreting Polarization through Machine Translation](https://arxiv.org/abs/2010.0233)


### Format - 

```python
python find_alignment.py \
<frequent_words_filepath> \
<source_embedding_file> \
<target_embedding_file>  \ 
<results_filepath> \
<number_of_words_to_run_alignment_on>
```

### Notes - 

- This code follows the algorithm described in [Smith et al. 2019](https://arxiv.org/pdf/1702.03859)
- Thie frequent_words_filepath should contain the words to check the alignent on. 
- The source and target embedding files are expected to be fasttext binaries. 
- Terminal output shows the percentage of alignment and the number of misaligned pairs. 
- The output also shows the source word, the nearest word in the aligned space and the top 10 nearest words. 
- The saved file additionally stores the 10 nearest neighbours in aligned and non-aligned space for comparison.  
- The last parameter decides the number of words on whom to run the alignment procedure. 



Please feel free to reach out to the authors for any question. 
