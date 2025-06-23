python -m bootstrap --diagnose configs/manifests/standard.yaml
python -m bootstrap configs/manifests/standard.yaml

python run_explorer_test.py 

## Create artifact via Proto Engine
python .\02_proto_runner\run_proto_test.py --nl "create a sine wave plot and save it as sine_wave.png, return information about the plot"