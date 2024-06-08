# Semantic Text Relatedness in Low-Resource Languages

## Running Modules in Terminal

To run python modules (python scripts) in Terminal, first navigate to the project directory - the directory that 
contains the `src` package, `data` directory, etc...
Then execute the command
```bash
py -m src.<optional_subpackage.><name_of_python_module_without_py_extension>
```

## Additional Notes

To increase memory efficiency, the models are trained and evaluated on the provided datasets only. The code can
be easily extended to work with new datasets, raw input sentence pairs, etc. If you want to test the models on
new sentence pairs, make sure to create embeddings for those sentences with the transformer that was used for
the training of the model, otherwise the accuracy might be greatly decreased (or not work at all if
the embedding dimension does not match the input dimension for the model in question)