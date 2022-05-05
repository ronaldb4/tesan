cd src/mortality_prediction
python ../train_prediction.py --model cbow --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model normal --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model delta --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model sa --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model tesa --configSet mortality_prediction/mimic_paper.json
python ../train_prediction.py --model tesan --configSet mortality_prediction/mimic_paper.json
