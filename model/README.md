# model/

This folder contains the trained Keras model file:

    action_model.keras   (2 MB)

## Model Details

| Property | Value |
|----------|-------|
| Architecture | 2× LSTM + Dense |
| Input shape | (30, 99) |
| Output classes | 7 |
| Total parameters | 170,759 |
| Training epochs | 200 |
| Test accuracy | 69% |

## How to load the model in Python

```python
import tensorflow as tf

model = tf.keras.models.load_model("model/action_model.keras")
model.summary()
```

## How to re-train the model

Open `notebooks/keypoints_lab.ipynb` and run all cells.
After training finishes, save the new model with:

```python
model.save("model/action_model.keras")
```
<<<<<<< HEAD

Then copy the saved file into this `model/` folder.
