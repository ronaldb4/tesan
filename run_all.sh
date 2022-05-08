cd src/concept_embedding
#python ../train_embedding.py --model tesan --configSet concept_embedding/mimic_paper.json
#python ../train_embedding.py --model tesa --configSet concept_embedding/mimic_paper.json
#python ../train_embedding.py --model cbow --configSet concept_embedding/mimic_paper.json
#python ../train_embedding.py --model skip_gram --configSet concept_embedding/mimic_paper.json
#python ../train_embedding.py --model delta --configSet concept_embedding/mimic_paper.json
#python ../train_embedding.py --model sa --configSet concept_embedding/mimic_paper.json
#python ../train_embedding.py --model normal --configSet concept_embedding/mimic_paper.json
#python ../train_embedding.py --model random_interval --configSet concept_embedding/mimic_paper.json

python ../train_embedding.py --model tesan --configSet concept_embedding/cms_paper.json
python ../train_embedding.py --model tesa --configSet concept_embedding/cms_paper.json
python ../train_embedding.py --model cbow --configSet concept_embedding/cms_paper.json
python ../train_embedding.py --model skip_gram --configSet concept_embedding/cms_paper.json
python ../train_embedding.py --model delta --configSet concept_embedding/cms_paper.json
python ../train_embedding.py --model sa --configSet concept_embedding/cms_paper.json
python ../train_embedding.py --model normal --configSet concept_embedding/cms_paper.json
python ../train_embedding.py --model random_interval --configSet concept_embedding/cms_paper.json

cd ../../src/mortality_prediction
python ../train_prediction.py --model tesan --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model tesa --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model cbow --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model skip_gram --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model delta --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model sa --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model normal --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model random_interval --configSet mortality_prediction/mimic_paper.json

