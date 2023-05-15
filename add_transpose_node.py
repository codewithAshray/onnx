import cv2
import onnx
import onnx.helper as helper

model = onnx.load("/media/ashray/D/Projects/x/yolov8n.onnx")

# output_tensor = helper.make_tensor_value_info('output_transposed', onnx.TensorProto.FLOAT, [1, 8400, 84])

# Define transpose node attributes
input_name = model.graph.output[0].name
output_name = 'output_transposed'
perm = [0, 2, 1]

# Create transpose node
transpose_node = helper.make_node('Transpose', [input_name], [output_name], perm=perm)

# Append the transpose node to the graph
model.graph.node.append(transpose_node)

# Update the graph output
model.graph.output[0].name = output_name
model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 8400
model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 84

# Save the modified model to a new file
onnx.save(model, 'modified_model.onnx')