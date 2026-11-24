"""DermaLens AI: skin disease classification for Indian and South Asian skin.

The package holds every piece of logic the notebooks share, so each notebook
stays short and only tells the story of one step:

    paths     -> where every file lives (one place, no hard-coded paths in cells)
    config    -> reads configs/*.yaml
    labels    -> the six target classes and how each dataset's labels map to them
    manifest  -> builds the image table from raw data, de-duplicates it
    splits    -> patient-grouped train/val/test splits and class weights
    data      -> PyTorch Dataset, image transforms, resized-image cache
    model     -> builds pretrained models with timm
    train     -> the training loop (one loop for every experiment)
    evaluate  -> metrics, confidence intervals, calibration, plots
    explain   -> Grad-CAM heatmaps
    export    -> saves and loads the model bundle used by the web app
    utils     -> seeds, device, JSON helpers
"""

__version__ = "2.0.0"
