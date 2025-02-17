# Bio Cell Detection

This repository us Deep Learning and Computer Vision models for processing biology cell images and analyzing the data to track the cell's growing growth. 
It all started when my PhD friend asked for help so that she could calculate the area of growing cells in her Biology lab experiments.
Today this repository is helping here and here colleges to analyze the microbiology life in their lab.

# Code example

## 📌 Cell day-0 raw data

![image](https://github.com/AvivSalo/Bio-Cell-Detection/assets/121252358/358fbe02-4fc2-43c6-8c4f-1609655692ee)

## 📌 Cell day-0 postprocessing

![image](https://github.com/AvivSalo/Bio-Cell-Detection/assets/121252358/6ba7f4a2-0238-40bb-afb7-147fd252a845)

### Calculated Area: 38,048 [pixels]

# Getting started
1. Create a virtual environment

```bash
pip install -r requirements.txt
```

2. Activate your venv

```bash
source venv/bin/activate
```

4. Run `python eval.py` 

Choose the directory of images folder to analyze 
![image](https://github.com/AvivSalo/Bio-Cell-Detection/assets/121252358/ed6d8185-1007-471b-a0f7-b4184b61352f)

The output will be a CSV file with the Area calculated
