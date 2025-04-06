import onnx

# Load and verify the ONNX model
onnx_model = onnx.load("indictrans_model.onnx")
onnx.checker.check_model(onnx_model)

print("ONNX model is valid.")
