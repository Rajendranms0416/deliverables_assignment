using Flux

# Define a simple neural network (dummy model)
model = Chain(
    Dense(224*224, 128, relu),  # Fully connected layer
    Dense(128, 64, relu),       # Another dense layer
    Dense(64, 10),              # Output layer with 10 classes
    softmax                     # Convert outputs to probabilities
)

# Generate a random input (dummy image data)
dummy_input = rand(Float32, 224*224)  # Flattened 224x224 image

# Run a forward pass
output = model(dummy_input)
println("Model Output: ", output)