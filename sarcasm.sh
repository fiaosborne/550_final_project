module load cobralab
module load gcc
module load python/3.9.8
module load gnu-parallel qbatch

source ~/.virtualenvs/Automl_env/bin/activate

python ./sarcasm_df.py

#data from instance 2
echo python ./get_embeddings.py > joblist_embeddings
qbatch -c1 --walltime=1:00:00 joblist_embeddings

#data from no text -> AutoML3
echo python ./run_automl.py ./features_matrix_notext.tsv > joblist_sarcasm
qbatch -c1 --walltime=7:00:00 joblist_sarcasm

#data from no text -> AutoML3
echo python ./run_automl2.py ./feature_matrix_tfidf.tsv > joblist_sarcasm_tfidf
qbatch -c1 --walltime=12:00:00 joblist_sarcasm_tfidf

python ./sarcasm_catboost.py ./features_matrix_notext.tsv ./dependence_plots/ 0.2 5 1 

python ./sarcasm_catboost_ngram.py ./feature_matrix_tfidf.tsv ./dependence_plots/tfidf/ 0.2 5 1 

squeue -l -u osso6500