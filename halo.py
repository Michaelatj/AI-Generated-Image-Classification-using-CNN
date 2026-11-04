import tensorflow as tf

# ==========================================
# 1. CHECK TENSORFLOW VERSION
# ==========================================

print("TensorFlow Version:", tf.__version__)

# ==========================================
# 2. DETECT GPU
# ==========================================

gpus = tf.config.list_physical_devices("GPU")

print("\nAvailable GPUs:", gpus)

if not gpus:
    raise SystemExit("ERROR: No GPU detected!")

# ==========================================
# 3. ENABLE MEMORY GROWTH
# ==========================================

# Prevent TensorFlow from reserving all GPU memory immediately
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ==========================================
# 4. TEST GPU COMPUTATION
# ==========================================

print("\nTesting GPU computation...")

with tf.device("/GPU:0"):

    # Create two random matrices
    a = tf.random.normal((4096, 4096))
    b = tf.random.normal((4096, 4096))

    # Matrix multiplication
    result = tf.matmul(a, b)

    # Calculate mean
    output = tf.reduce_mean(result)

print("Result:", float(output.numpy()))

# ==========================================
# 5. SHOW EXECUTION DEVICE
# ==========================================

print("\nTensor execution device:")
print(result.device)

print("\nGPU test completed!")