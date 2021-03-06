# Code to find alignment between two corpora as described in [We Don't Speak the Same Language: Interpreting Polarization through Machine Translation](https://arxiv.org/pdf/2010.02339.pdf)


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
- The source and target embeddings should be trained on corpuses with identical token size. 
- The frequent_words_filepath should contain the words to check the alignment on. 
- The source and target embedding files are supposed to be fasttext binaries. 
- Terminal output shows the percentage of alignment and the number of misaligned pairs. 
- The output also shows the source word, the nearest word in the aligned space and the top 10 nearest words. 
- The saved file additionally stores the 10 nearest neighbours in aligned and non-aligned space for comparison.  
- The last parameter specifies the number of words on which to run the alignment procedure. 


Please feel free to reach out to the authors [Ashique Khudabukhsh](mailto:akhudabu@cs.cmu.edu) and [Rupak Sarkar](mailto:rupaksarkar.cs@gmail.com). 
