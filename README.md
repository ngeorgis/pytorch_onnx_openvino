# Pytorch to ONNX to Intel OpenVino

Validation of the "PyTorch to ONNX to Intel OpenVino workflow" using ImageNet pretrained ResNet-50

## PyTorch to ONNX

Study and run pytorch_onnx_openvino.ipynb to execute ResNet50 inference using PyTorch and also create ONNX model to be used by the OpenVino model optimizer in the next step.

Typical PyTorch output when processing dog.jpeg is 

```
Top 10 results:
===============

n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian 		 tensor(15.4252, grad_fn=<SelectBackward>) tensor(235)
n02111129_Leonberg 		 tensor(11.2401, grad_fn=<SelectBackward>) tensor(255)
n02105162_malinois 		 tensor(11.0313, grad_fn=<SelectBackward>) tensor(225)
n02091467_Norwegian_elkhound,_elkhound 		 tensor(9.7304, grad_fn=<SelectBackward>) tensor(174)
n02090721_Irish_wolfhound 		 tensor(8.9736, grad_fn=<SelectBackward>) tensor(170)
n02105056_groenendael 		 tensor(8.8621, grad_fn=<SelectBackward>) tensor(224)
n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus 		 tensor(8.5262, grad_fn=<SelectBackward>) tensor(275)
n02088094_Afghan_hound,_Afghan 		 tensor(8.4578, grad_fn=<SelectBackward>) tensor(160)
n02090622_borzoi,_Russian_wolfhound 		 tensor(7.9833, grad_fn=<SelectBackward>) tensor(169)
n02105412_kelpie 		 tensor(7.8347, grad_fn=<SelectBackward>) tensor(227)
```

## Convert ONNX model to Intel OpenVino IR

First download the OpenVino SDK from https://software.intel.com/en-us/openvino-toolkit

Set up OpenVino environment
```
source  ~/intel/computer_vision_sdk/bin/setupvars.sh
```
Convert ONNX to OpenVino IR
```
mo_onnx.py --input_model test_model.onnx --scale_values [51.5865,50.847,51.255] --mean_values [125.307,122.961,113.8575] --data_type FP32 --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP32
```
Two files should be now created: test_model.xml and test_model.bin


## Run Intel OpenVino classification

Run the modified **classification_sample.py**
```
python3 classification_sample.py --labels test_model.labels  -m test_model.xml -i dog.jpeg -d CPU
```
Typical output is

```
15.3578548 label n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian
11.2073364 label n02111129_Leonberg
10.9584856 label n02105162_malinois
9.9125824 label n02091467_Norwegian_elkhound,_elkhound
8.9993858 label n02090721_Irish_wolfhound
8.9059830 label n02105056_groenendael
8.5530519 label n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus
8.4389114 label n02088094_Afghan_hound,_Afghan
7.9750357 label n02090622_borzoi,_Russian_wolfhound
7.9166594 label n02105412_kelpie

```
The OpenVino output is similar but not identical to PyTorch due to the imaging processing pipeline that prepares the input blob. For example PyTorch 15.4252 is not equal to OpenVino 15.3578548 for top prediction German_shepherd.

## Run Intel OpenVino classification with input vector saved from PyTorch

To ensure identical input we will run again with the same input vector that was saved in the PyTorch notebook ( test_in_vector.npy ).

The modified Intel OpenVino SDK **classification_sample.py** now has a new parameter **-rf** to allow this.

```
diff classification_sample.py ~/intel/computer_vision_sdk_2018.5.445/deployment_tools/inference_engine/samples/python_samples/classification_sample.py

45d44
<     parser.add_argument("-rf", "--read_vector_from_file", help="Read input vector from file", default=False, action="store_true")
94,99d92
<     # Read input vector to compare with PyTorch
<     if args.read_vector_from_file:
<         r = np.load("test_in_vector.npy")
<         #print (r)
<         images[0] = r
<  
124,126d116
< 
<     print (res[0][0:10])
< 
```


In addition we need to run the model optimizer again with new parameters:
```
mo_onnx.py --input_model test_model.onnx --data_type FP32 --data_type=FP32
```

Run modified OpenVino classification:
```
python3 classification_sample.py --labels test_model.labels  -m test_model.xml -i dog.jpeg -d CPU -rf
[ INFO ] Loading network files:
	test_model.xml
	test_model.bin
[ INFO ] Preparing input blobs
[ WARNING ] Image dog.jpeg is resized from (216, 233) to (224, 224)
[ INFO ] Batch size is 1
[ INFO ] Loading model to the plugin
[ INFO ] Starting inference (1 iterations)
[ INFO ] Average running time of one iteration: 32.76681900024414 ms
[ INFO ] Processing output blob
[ 0.8969815  -3.0496185  -2.704152   -0.74797183 -3.5622005  -1.7981017
 -4.848625    0.10940043  0.86848116  0.05356478]
[ INFO ] Top 10 results: 
Image dog.jpeg

15.4252472 label n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian
11.2401333 label n02111129_Leonberg
11.0313206 label n02105162_malinois
9.7304420 label n02091467_Norwegian_elkhound,_elkhound
8.9735994 label n02090721_Irish_wolfhound
8.8621025 label n02105056_groenendael
8.5262289 label n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus
8.4578362 label n02088094_Afghan_hound,_Afghan
7.9833107 label n02090622_borzoi,_Russian_wolfhound
7.8347163 label n02105412_kelpie

```
We can now see we are in complete agreement between PyTorch and OpenVino as OpenVino 15.4252472 is equal to PyTorch 15.4252 value as expected assuming we have the same input.


## Run model optimizer without optimizations
Just for reference in some cases it may be useful to disable some optimizations to better debug similar discrepancy issues.
```
mo_onnx.py --input_model test_model.onnx --data_type FP32  --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP32

Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/home/ngeorgis/dl/pytorch_onnx_openvino/test_model.onnx
	- Path for generated IR: 	/home/ngeorgis/dl/pytorch_onnx_openvino/.
	- IR output name: 	test_model
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP32
	- Enable fusing: 	False
	- Enable grouped convolutions fusing: 	False
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	False
ONNX specific parameters:
Model Optimizer version: 	1.5.12.49d067a0

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/ngeorgis/dl/pytorch_onnx_openvino/./test_model.xml
[ SUCCESS ] BIN file: /home/ngeorgis/dl/pytorch_onnx_openvino/./test_model.bin
[ SUCCESS ] Total execution time: 2.36 seconds. 
```
The new output is very similar but execution maybe slower.
```
15.4252462 label n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian
11.2401333 label n02111129_Leonberg
11.0313234 label n02105162_malinois
9.7304478 label n02091467_Norwegian_elkhound,_elkhound
8.9735994 label n02090721_Irish_wolfhound
8.8621044 label n02105056_groenendael
8.5262327 label n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus
8.4578342 label n02088094_Afghan_hound,_Afghan
7.9833093 label n02090622_borzoi,_Russian_wolfhound
7.8347163 label n02105412_kelpie
```

## FP16 Validation on Intel UHD630 GPU

Run model optimizer again for FP16
```
mo_onnx.py --input_model test_model.onnx --data_type=FP16

python3 classification_sample.py --labels test_model.labels  -m test_model.xml -i dog.jpeg -d GPU -rf
```
Similar results
```
15.4296875 label n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian
11.2500000 label n02111129_Leonberg
11.0390625 label n02105162_malinois
9.7500000 label n02091467_Norwegian_elkhound,_elkhound
8.9765625 label n02090721_Irish_wolfhound
8.8515625 label n02105056_groenendael
8.5390625 label n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus
8.4531250 label n02088094_Afghan_hound,_Afghan
7.9804688 label n02090622_borzoi,_Russian_wolfhound
7.8398438 label n02105412_kelpie
```

## Test same FP16 IR on Movidius NCS2

```
python3 classification_sample.py --labels test_model.labels  -m test_model.xml -i dog.jpeg -d MYRIAD -rf
```

```
13.8125000 label n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian
9.7109375 label n02111129_Leonberg
9.2500000 label n02105162_malinois
9.1640625 label n02091467_Norwegian_elkhound,_elkhound
8.2968750 label n02088094_Afghan_hound,_Afghan
8.1718750 label n02090721_Irish_wolfhound
7.7500000 label n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus
7.6171875 label n02105056_groenendael
7.2656250 label n02090622_borzoi,_Russian_wolfhound
7.2656250 label n02112350_keeshond
```

## Conclusions
PyTorch to ONNX to OpenVino workflow validated and likely to produce identical results if same input vector is used. CPU FP32 and GPU FP32 or FP16 results are very similar with NCS FP16 having slight discrepancies. 

## References: 
- https://software.intel.com/en-us/forums/computer-vision/topic/802631
- https://software.intel.com/en-us/openvino-toolkit

