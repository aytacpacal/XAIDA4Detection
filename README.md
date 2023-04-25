[<img src="tutorials/imgs/xaida_logo.png" width="130" />](tutorials/imgs/xaida_logo.png)

# XAIDA4Detection
Open source code for the detection and characterization of spatio-temporal extreme events

## Description
The XAIDA4Detection toolbox consists of a full pipeline for the detection and characterization of extreme events using ML and automatic image processing tools. Its purpose is to provide an ML-based generic and flexible pipeline to detect and characterize extreme events based on spatio-temporal Earth and climate observational data. The pipeline consists of three different stages for both the detection and characterization of extreme events:

1) Data loading and pre-processing
2) ML architecture selection and training
3) Evaluation and visualization of results

## Usage
```python
# 1) Create an empty pip environment
python3 -m venv ./xaida_env 


# 2) Activate environment
source ./xaida_env/bin/activate


# 3) Install dependencies
pip install -r requirements_xaida.txt install libs


# 4) Run main.py of XAIDA4Detection using a config file. Some examples:

# SYRIA Droughts database and Angle-Based Outlier Detection model (from PyOD) 
python main.py --config=/configs/config_SYRIA_ABOD.yaml

# SYRIA Droughts database and K-Nearest Neighbors model (from PyOD) 
python main.py --config=/configs/config_SYRIA_KNN.yaml

# SYRIA Droughts database and UNET model (from Segmentation Models PyTorch) 
python main.py --config=/configs/config_SYRIA_UNET.yaml

# SYRIA Droughts database and CNN-based encoder-decoder architecture (user-defined) 
python main.py --config=/configs/config_SYRIA_CNN2D.yaml
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Citation
If you use this code for your research, please cite **XAIDA4Detection: A Toolbox for the Detection and Characterization of Spatio-Temporal Extreme Events**:

Cortés-Andrés, J., Gonzalez-Calabuig, M., Zhang, M., Williams, T., Fernández-Torres, M.-Á., Pellicer-Valero, O. J., and Camps-Valls, G.: XAIDA4Detection: A Toolbox for the Detection and Characterization of Spatio-Temporal Extreme Events, EGU General Assembly 2023, Vienna, Austria, 24–28 Apr 2023, EGU23-4816, https://doi.org/10.5194/egusphere-egu23-4816, 2023.

## Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement 101003469.

## License
[MIT](https://choosealicense.com/licenses/mit/)
