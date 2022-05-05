cd src/concept_embedding
python ../train_embedding.py --model skip_gram --configSet concept_embedding/mimic_paper.json
python ../train_embedding.py --model random_interval --configSet concept_embedding/mimic_paper.json
cd ../../src/mortality_prediction
python ../train_prediction.py --model skip_gram --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model random_interval --configSet mortality_prediction/mimic_paper.json
