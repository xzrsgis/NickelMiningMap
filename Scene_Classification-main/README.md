
# Mining Scene classification

This repo contains code on image classification for nickel mining

The purpose is to find the presence of nickel mining grid by grid (a region of approximately 2.5 km * 2.5 km degree cell).
### Key features

- This code splits the image chip labels with the target 'mining' and background 'non-mining', which were saved as csv, into training, validation and testing data.

- Train model using EfficientNet backbones

- Predict at 2.5km *23.5 grid



## Code structure:


### Prepare labels

```
python data_prepare_classification.py
```

--- :bookmark: set configs ---

config/config_classification.yaml

-------------------------------------------------------------------------------------------------------

### Train 1st model: Greenhouse image classification:

```
python main_classification.py
```


-------------------------------------------------------------------------------------------

### Test 1st model: Predict at 1km grid for large area using satellite images (e.g. PlanetScope):

```
python inference_run_classification.py
```

--- :bookmark: set configs ---

config/config_inference_planet.py





