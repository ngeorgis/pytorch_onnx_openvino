# Pytorch to ONNX to Intel OpenVino

Validation of the "PyTorch to ONNX to Intel OpenVino" workflow using ImageNet pretrained ResNet.

## PyTorch to ONNX

Study and run pytorch_onnx_openvino.ipynb to execute ResNet50 inference using PyTorch and also create ONNX model to be used by the OpenVino model optimizer in the next step.

Typical PyTorch output when processing dog.jpeg is 

```
Top 10 results:
===============

n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian tensor(14.0807, grad_fn=<SelectBackward>) tensor(235)
n02105162_malinois 		 tensor(11.5884, grad_fn=<SelectBackward>) tensor(225)
n02111129_Leonberg 		 tensor(11.1329, grad_fn=<SelectBackward>) tensor(255)
n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus 		 tensor(9.3868, grad_fn=<SelectBackward>) tensor(275)
n02088094_Afghan_hound,_Afghan 		 tensor(8.8371, grad_fn=<SelectBackward>) tensor(160)
n02091467_Norwegian_elkhound,_elkhound 		 tensor(8.8061, grad_fn=<SelectBackward>) tensor(174)
n02105056_groenendael 		 tensor(8.4025, grad_fn=<SelectBackward>) tensor(224)
n02108551_Tibetan_mastiff 		 tensor(7.9561, grad_fn=<SelectBackward>) tensor(244)
n02090721_Irish_wolfhound 		 tensor(7.9554, grad_fn=<SelectBackward>) tensor(170)
n02105412_kelpie 		 tensor(7.9356, grad_fn=<SelectBackward>) tensor(227)
```

## Convert ONNX model to Intel OpenVino IR

First download the OpenVino SDK from https://software.intel.com/en-us/openvino-toolkit

Set up OpenVino environment
```
source  ~/intel/computer_vision_sdk/bin/setupvars.sh
```
Convert ONNX to OpenVino IR
```
mkdir fp16 fp32

mo_onnx.py --input_model resnet18.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP32 --output_dir fp32
mo_onnx.py --input_model resnet34.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP32 --output_dir fp32
mo_onnx.py --input_model resnet50.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP32 --output_dir fp32
mo_onnx.py --input_model resnet101.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP32 --output_dir fp32
mo_onnx.py --input_model resnet152.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP32 --output_dir fp32

mo_onnx.py --input_model resnet18.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP16 --output_dir fp16
mo_onnx.py --input_model resnet34.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP16 --output_dir fp16
mo_onnx.py --input_model resnet50.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP16 --output_dir fp16
mo_onnx.py --input_model resnet101.onnx --scale_values [51.5865,50.847,51.255] --mean_values [125.307,122.961,113.8575 --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP16 --output_dir fp16
mo_onnx.py --input_model resnet152.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --disable_resnet_optimization --disable_fusing --disable_gfusing --data_type=FP16 --output_dir fp16
```


## Run Intel OpenVino classification

Without loss of generality we will compare the ResNet50 case.

Run the modified **classification_sample.py**
```
python3 classification_sample.py --labels test_model.labels  -m fp32/resnet50.xml -i dog.jpeg -d CPU
```
Typical output is

```
14.5915499 label n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian
10.5486736 label n02105162_malinois
10.3912392 label n02111129_Leonberg
9.3196468 label n02091467_Norwegian_elkhound,_elkhound
8.3930368 label n02090721_Irish_wolfhound
8.2550011 label n02105056_groenendael
8.2503805 label n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus
8.0939283 label n02088094_Afghan_hound,_Afghan
7.5636091 label n02090622_borzoi,_Russian_wolfhound
7.5220599 label n02105412_kelpie
```
The OpenVino output is similar but not identical to PyTorch due to the imaging processing pipeline that prepares the input blob. For example PyTorch 14.0807 is not equal to OpenVino 14.5915499 for top prediction German_shepherd.

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
mo_onnx.py --input_model resnet50.onnx --data_type=FP32 --output_dir fp32
```

Run modified OpenVino classification:
```
python3 classification_sample.py --labels test_model.labels  -m fp32/resnet50.xml -i dog.jpeg -d CPU -rf
[ INFO ] Loading network files:
	fp32/resnet50.xml
	fp32/resnet50.bin
[ INFO ] Preparing input blobs
[ WARNING ] Image dog.jpeg is resized from (216, 233) to (224, 224)
[ INFO ] Batch size is 1
[ INFO ] Loading model to the plugin
[ INFO ] Starting inference (1 iterations)
[ INFO ] Average running time of one iteration: 16.2506103515625 ms
[ INFO ] Processing output blob
[ 1.4262071  -2.8766239  -2.201517   -1.2191257  -3.089852   -1.4179183
 -4.359639    0.9693046   0.83319616  0.43519974]
[ INFO ] Top 10 results: 
Image dog.jpeg

14.0807028 label n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian
11.5883627 label n02105162_malinois
11.1329060 label n02111129_Leonberg
9.3867922 label n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus
8.8370838 label n02088094_Afghan_hound,_Afghan
8.8061085 label n02091467_Norwegian_elkhound,_elkhound
8.4025440 label n02105056_groenendael
7.9560895 label n02108551_Tibetan_mastiff
7.9554348 label n02090721_Irish_wolfhound
7.9355597 label n02105412_kelpie
```
We can now see we are in agreement between PyTorch and OpenVino as OpenVino 14.0807028 is equal to PyTorch 14.0807 value as expected assuming we have the same input.


## Run model optimizer without optimizations
Just for reference in some cases it may be useful to disable some optimizations to better debug similar discrepancy issues.
```
mo_onnx.py --input_model resnet50.onnx --data_type=FP32 --output_dir fp32 --disable_resnet_optimization --disable_fusing --disable_gfusing 
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/home/ngeorgis/dl/pytorch_onnx_openvino/resnet50.onnx
	- Path for generated IR: 	/home/ngeorgis/dl/pytorch_onnx_openvino/fp32
	- IR output name: 	resnet50
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
[ SUCCESS ] XML file: /home/ngeorgis/dl/pytorch_onnx_openvino/fp32/resnet50.xml
[ SUCCESS ] BIN file: /home/ngeorgis/dl/pytorch_onnx_openvino/fp32/resnet50.bin
[ SUCCESS ] Total execution time: 2.55 seconds.
```
The new output is very similar but execution may be slower.
```
14.0807085 label n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian
11.5883684 label n02105162_malinois
11.1329079 label n02111129_Leonberg
9.3867903 label n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus
8.8370848 label n02088094_Afghan_hound,_Afghan
8.8061075 label n02091467_Norwegian_elkhound,_elkhound
8.4025459 label n02105056_groenendael
7.9560914 label n02108551_Tibetan_mastiff
7.9554353 label n02090721_Irish_wolfhound
7.9355602 label n02105412_kelpie
```

## FP16 Validation on Intel UHD630 GPU

Run model optimizer again for FP16
```
mo_onnx.py --input_model resnet50.onnx --data_type FP16  --output_dir fp16
python3 classification_sample.py --labels test_model.labels  -m fp16/resnet50.xml -i dog.jpeg -d GPU -rf
```
Similar results
```
14.0468750 label n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian
11.5625000 label n02105162_malinois
11.0937500 label n02111129_Leonberg
9.3671875 label n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus
8.7890625 label n02088094_Afghan_hound,_Afghan
8.7812500 label n02091467_Norwegian_elkhound,_elkhound
8.3750000 label n02105056_groenendael
7.9296875 label n02090721_Irish_wolfhound
7.9257812 label n02108551_Tibetan_mastiff
7.9218750 label n02105412_kelpie
```

## Test same FP16 IR on Movidius NCS2

```
python3 classification_sample.py --labels test_model.labels  -m fp16/resnet50.xml -i dog.jpeg -d MYRIAD -rf
```
MYRIAD NCS2 output
```
15.1406250 label n02106662_German_shepherd,_German_shepherd_dog,_German_police_dog,_alsatian
12.8437500 label n02105162_malinois
11.7890625 label n02111129_Leonberg
10.0703125 label n02088094_Afghan_hound,_Afghan
9.7890625 label n02116738_African_hunting_dog,_hyena_dog,_Cape_hunting_dog,_Lycaon_pictus
9.0625000 label n02090622_borzoi,_Russian_wolfhound
8.9296875 label n02091467_Norwegian_elkhound,_elkhound
8.8437500 label n02090721_Irish_wolfhound
8.7812500 label n02105056_groenendael
7.6523438 label n02106030_collie
```

## Generate optimized IR
Make sure to run again the model optimizer again to have all optimizations enabled.

```
mo_onnx.py  --input_model resnet18.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --data_type=FP32 --output_dir fp32
mo_onnx.py  --input_model resnet34.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --data_type=FP32 --output_dir fp32
mo_onnx.py  --input_model resnet50.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --data_type=FP32 --output_dir fp32
mo_onnx.py --input_model resnet101.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --data_type=FP32 --output_dir fp32
mo_onnx.py --input_model resnet152.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --data_type=FP32 --output_dir fp32

mo_onnx.py  --input_model resnet18.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --data_type=FP16 --output_dir fp16
mo_onnx.py  --input_model resnet34.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --data_type=FP16 --output_dir fp16
mo_onnx.py  --input_model resnet50.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --data_type=FP16 --output_dir fp16
mo_onnx.py --input_model resnet101.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --data_type=FP16 --output_dir fp16
mo_onnx.py --input_model resnet152.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --data_type=FP16 --output_dir fp16
```

## Conclusions
PyTorch to ONNX to OpenVino workflow validated and likely to produce identical results if same input vector is used. CPU FP32 and GPU FP32 or FP16 results are very similar with NCS FP16 having slight discrepancies. 

## References: 
- https://software.intel.com/en-us/forums/computer-vision/topic/802631
- https://software.intel.com/en-us/openvino-toolkit

