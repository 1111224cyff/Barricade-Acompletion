source /D/anaconda/bin/activate
conda activate acomp_barricade

export PYTHONPATH="/D/cyf/Barricade-Amodal-Completion:$PYTHONPATH"
python ./utils/preprocess.py
