import coremltools as ct
import numpy as np
import os

# Import TensorFlow/Keras
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Input, Concatenate, Add, Multiply, Average, Maximum

# Create wrapper classes to handle old Keras 1.x parameter names
# Old Keras 1.x uses: nb_filter, nb_row, nb_col, border_mode, subsample
# Modern Keras uses: filters, kernel_size, padding, strides

def convert_old_conv_params(config):
    """Convert old Keras 1.x Conv2D parameters to modern format"""
    new_config = config.copy()
    
    # Convert nb_filter to filters
    if 'nb_filter' in new_config:
        new_config['filters'] = new_config.pop('nb_filter')
    
    # Convert nb_row, nb_col to kernel_size
    if 'nb_row' in new_config and 'nb_col' in new_config:
        new_config['kernel_size'] = (new_config.pop('nb_row'), new_config.pop('nb_col'))
    
    # Convert border_mode to padding
    if 'border_mode' in new_config:
        border_mode = new_config.pop('border_mode')
        if border_mode == 'valid':
            new_config['padding'] = 'valid'
        elif border_mode == 'same':
            new_config['padding'] = 'same'
        else:
            new_config['padding'] = 'valid'
    
    # Convert subsample to strides
    if 'subsample' in new_config:
        new_config['strides'] = new_config.pop('subsample')
    
    # Convert dim_ordering to data_format
    if 'dim_ordering' in new_config:
        dim_ordering = new_config.pop('dim_ordering')
        if dim_ordering == 'th':
            new_config['data_format'] = 'channels_first'
        else:
            new_config['data_format'] = 'channels_last'
    
    # Convert init to kernel_initializer
    if 'init' in new_config:
        init = new_config.pop('init')
        if isinstance(init, str):
            init_map = {
                'glorot_uniform': 'glorot_uniform',
                'glorot_normal': 'glorot_normal',
                'he_uniform': 'he_uniform',
                'he_normal': 'he_normal',
                'uniform': 'random_uniform',
                'normal': 'random_normal',
                'zero': 'zeros',
                'one': 'ones',
            }
            new_config['kernel_initializer'] = init_map.get(init, 'glorot_uniform')
        else:
            new_config['kernel_initializer'] = init
    
    # Convert bias to use_bias
    if 'bias' in new_config:
        new_config['use_bias'] = new_config.pop('bias')
    
    # Remove old parameters that are no longer used
    new_config.pop('W_regularizer', None)
    new_config.pop('b_regularizer', None)
    new_config.pop('activity_regularizer', None)
    new_config.pop('W_constraint', None)
    new_config.pop('b_constraint', None)
    
    return new_config

class Convolution2D(Conv2D):
    """Wrapper for old Keras 1.x Convolution2D layer"""
    def __init__(self, *args, **kwargs):
        # Convert old parameter names to new ones
        kwargs = convert_old_conv_params(kwargs)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        # Convert old config format to new format
        config = convert_old_conv_params(config)
        return super().from_config(config)

def convert_old_pool_params(config):
    """Convert old Keras 1.x MaxPooling2D parameters to modern format"""
    new_config = config.copy()
    
    # Convert border_mode to padding
    if 'border_mode' in new_config:
        border_mode = new_config.pop('border_mode')
        if border_mode == 'valid':
            new_config['padding'] = 'valid'
        elif border_mode == 'same':
            new_config['padding'] = 'same'
    
    # Convert dim_ordering to data_format
    if 'dim_ordering' in new_config:
        dim_ordering = new_config.pop('dim_ordering')
        if dim_ordering == 'th':
            new_config['data_format'] = 'channels_first'
        else:
            new_config['data_format'] = 'channels_last'
    
    return new_config

class MaxPooling2DWrapper(MaxPooling2D):
    """Wrapper for old Keras 1.x MaxPooling2D layer"""
    def __init__(self, *args, **kwargs):
        # Convert old parameter names
        kwargs = convert_old_pool_params(kwargs)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        # Convert old config format to new format
        config = convert_old_pool_params(config)
        return super().from_config(config)

class AveragePooling2DWrapper(AveragePooling2D):
    """Wrapper for old Keras 1.x AveragePooling2D layer"""
    def __init__(self, *args, **kwargs):
        # Convert old parameter names
        kwargs = convert_old_pool_params(kwargs)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        # Convert old config format to new format
        config = convert_old_pool_params(config)
        return super().from_config(config)

def convert_old_bn_params(config):
    """Convert old Keras 1.x BatchNormalization parameters to modern format"""
    new_config = config.copy()
    
    # Remove 'mode' parameter - it's no longer used in modern Keras
    new_config.pop('mode', None)
    
    # Remove old regularizer parameters that are no longer used
    new_config.pop('beta_regularizer', None)
    new_config.pop('gamma_regularizer', None)
    
    return new_config

class BatchNormalizationWrapper(BatchNormalization):
    """Wrapper for old Keras 1.x BatchNormalization layer"""
    def __init__(self, *args, **kwargs):
        # Convert old parameter names
        kwargs = convert_old_bn_params(kwargs)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        # Convert old config format to new format
        config = convert_old_bn_params(config)
        return super().from_config(config)

# Merge layer wrapper - old Keras 1.x Merge layer is replaced by specific merge layers
# Most common mode is 'concat' which maps to Concatenate
class Merge(Concatenate):
    """Wrapper for old Keras 1.x Merge layer - defaults to Concatenate (most common)"""
    def __init__(self, mode='concat', **kwargs):
        # Map old merge modes to modern layers
        mode_map = {
            'concat': Concatenate,
            'sum': Add,
            'add': Add,
            'mul': Multiply,
            'multiply': Multiply,
            'ave': Average,
            'average': Average,
            'max': Maximum,
            'maximum': Maximum,
        }
        
        # Remove old parameters
        kwargs.pop('layers', None)  # Old Merge had a 'layers' parameter
        kwargs.pop('output_shape', None)
        kwargs.pop('node_indices', None)
        kwargs.pop('tensor_indices', None)
        kwargs.pop('output_mask', None)
        kwargs.pop('output_mask_type', None)
        kwargs.pop('arguments', None)
        
        # Use the appropriate modern layer
        merge_class = mode_map.get(mode, Concatenate)
        if merge_class == Concatenate:
            # Concatenate uses 'axis' instead of 'concat_dim'
            if 'concat_dim' in kwargs:
                kwargs['axis'] = kwargs.pop('concat_dim')
            super().__init__(**kwargs)
        else:
            # For other merge types, we'd need to instantiate the correct class
            # For now, default to Concatenate
            super().__init__(**kwargs)
    
    @classmethod
    def from_config(cls, config):
        # Get the mode from config
        mode = config.get('mode', 'concat')
        
        # Map to appropriate modern layer
        mode_map = {
            'concat': Concatenate,
            'sum': Add,
            'add': Add,
            'mul': Multiply,
            'multiply': Multiply,
            'ave': Average,
            'average': Average,
            'max': Maximum,
            'maximum': Maximum,
        }
        
        # Clean up old config
        new_config = config.copy()
        new_config.pop('mode', None)
        new_config.pop('layers', None)
        new_config.pop('output_shape', None)
        new_config.pop('node_indices', None)
        new_config.pop('tensor_indices', None)
        new_config.pop('output_mask', None)
        new_config.pop('output_mask_type', None)
        new_config.pop('arguments', None)
        
        # Remove old Merge-specific parameters
        new_config.pop('mode_type', None)
        new_config.pop('dot_axes', None)
        new_config.pop('output_shape_type', None)
        
        # Convert concat_axis or concat_dim to axis
        if 'concat_axis' in new_config:
            new_config['axis'] = new_config.pop('concat_axis')
        elif 'concat_dim' in new_config:
            new_config['axis'] = new_config.pop('concat_dim')
        
        # Use appropriate merge layer
        merge_class = mode_map.get(mode, Concatenate)
        return merge_class.from_config(new_config)

class DropoutWrapper(Dropout):
    """Wrapper for old Keras 1.x Dropout layer - converts p to rate"""
    def __init__(self, p=None, rate=None, **kwargs):
        # Convert old 'p' parameter to 'rate'
        if p is not None and rate is None:
            rate = p
        super().__init__(rate=rate, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        # Convert old config format to new format
        new_config = config.copy()
        # Convert 'p' to 'rate'
        if 'p' in new_config:
            new_config['rate'] = new_config.pop('p')
        return super().from_config(new_config)

def convert_old_dense_params(config):
    """Convert old Keras 1.x Dense parameters to modern format"""
    new_config = config.copy()
    
    # Convert output_dim to units
    if 'output_dim' in new_config:
        new_config['units'] = new_config.pop('output_dim')
    
    # Remove input_dim (not used in modern Keras, inferred from input)
    new_config.pop('input_dim', None)
    
    # Convert init to kernel_initializer
    if 'init' in new_config:
        init = new_config.pop('init')
        if isinstance(init, str):
            init_map = {
                'glorot_uniform': 'glorot_uniform',
                'glorot_normal': 'glorot_normal',
                'he_uniform': 'he_uniform',
                'he_normal': 'he_normal',
                'uniform': 'random_uniform',
                'normal': 'random_normal',
                'zero': 'zeros',
                'one': 'ones',
            }
            new_config['kernel_initializer'] = init_map.get(init, 'glorot_uniform')
        else:
            new_config['kernel_initializer'] = init
    
    # Convert bias to use_bias
    if 'bias' in new_config:
        new_config['use_bias'] = new_config.pop('bias')
    
    # Remove old regularizer and constraint parameters
    new_config.pop('W_regularizer', None)
    new_config.pop('b_regularizer', None)
    new_config.pop('activity_regularizer', None)
    new_config.pop('W_constraint', None)
    new_config.pop('b_constraint', None)
    new_config.pop('quantization_config', None)
    
    return new_config

class DenseWrapper(Dense):
    """Wrapper for old Keras 1.x Dense layer"""
    def __init__(self, output_dim=None, units=None, **kwargs):
        # Convert old parameter names
        if output_dim is not None and units is None:
            units = output_dim
        kwargs = convert_old_dense_params(kwargs)
        super().__init__(units=units, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        # Convert old config format to new format
        config = convert_old_dense_params(config)
        return super().from_config(config)

# Map old Keras 1.x layer names to wrapper classes
# Note: We map both 'Convolution2D' (old name) and 'Conv2D' (new name) to our wrapper
# because Keras might try to deserialize using either name
custom_objects = {
    'Convolution2D': Convolution2D,
    'Conv2D': Convolution2D,  # Also handle if it's saved as Conv2D
    'MaxPooling2D': MaxPooling2DWrapper,
    'AveragePooling2D': AveragePooling2DWrapper,
    'Dense': DenseWrapper,
    'Flatten': Flatten,
    'Dropout': DropoutWrapper,
    'Activation': Activation,
    'BatchNormalization': BatchNormalizationWrapper,
    'GlobalAveragePooling2D': GlobalAveragePooling2D,
    'Input': Input,
    'Merge': Merge,  # Old Keras 1.x Merge layer
}

# Try to load with TensorFlow/Keras
print("Loading Keras model with TensorFlow...")
try:
    keras_model = keras.models.load_model('model4b.10-0.68.hdf5', custom_objects=custom_objects)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading with TensorFlow: {e}")
    print("\nTrying with compile=False...")
    try:
        # Sometimes old models need compile=False
        keras_model = keras.models.load_model('model4b.10-0.68.hdf5', custom_objects=custom_objects, compile=False)
        print("Model loaded successfully with compile=False!")
    except Exception as e2:
        print(f"Error with compile=False: {e2}")
        raise RuntimeError(f"Failed to load model. Please ensure the model file exists and TensorFlow is properly installed.\nOriginal error: {e}")

# Read class labels
print("Reading class labels...")
if not os.path.exists('labels.txt'):
    raise FileNotFoundError("labels.txt not found. Please ensure it exists in the current directory.")
with open('labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines() if line.strip()]
print(f"Found {len(class_labels)} class labels")

# Get the actual input name from the model
# In modern Keras, we access the input layer directly
if hasattr(keras_model, 'input_names') and keras_model.input_names:
    input_name = keras_model.input_names[0]
elif hasattr(keras_model, 'inputs') and keras_model.inputs:
    input_name = keras_model.inputs[0].name.split(':')[0]  # Remove :0 suffix if present
else:
    input_name = 'input_1'
print(f"Model input name: {input_name}")

# Get input shape from the model
if hasattr(keras_model, 'input_shape') and keras_model.input_shape:
    if isinstance(keras_model.input_shape, list) and len(keras_model.input_shape) > 0:
        input_shape = keras_model.input_shape[0][1:] if len(keras_model.input_shape[0]) > 1 else (299, 299, 3)
    else:
        input_shape = keras_model.input_shape[1:] if len(keras_model.input_shape) > 1 else (299, 299, 3)
elif hasattr(keras_model, 'inputs') and keras_model.inputs:
    input_shape = keras_model.inputs[0].shape[1:].as_list() if hasattr(keras_model.inputs[0].shape, 'as_list') else tuple(keras_model.inputs[0].shape[1:])
    # Handle None dimensions
    input_shape = tuple(s if s is not None else 299 if i < 2 else 3 for i, s in enumerate(input_shape)) if input_shape else (299, 299, 3)
else:
    input_shape = (299, 299, 3)
print(f"Model input shape: {input_shape}")

# Convert the model
# Try multiple conversion approaches due to compatibility issues with old models
print("Converting to CoreML...")

# First, try with TensorType (simpler, may work better with old models)
try:
    print("Attempting conversion with TensorType...")
    tensor_input = ct.TensorType(name=input_name, shape=(1,) + input_shape)
    coreml_model = ct.convert(
        keras_model,
        inputs=[tensor_input],
        classifier_config=ct.ClassifierConfig(class_labels),
        minimum_deployment_target=ct.target.iOS13
    )
    print("Conversion successful with TensorType!")
except Exception as e1:
    print(f"Error with TensorType: {e1}")
    try:
        print("Attempting conversion with ImageType...")
        # Try with ImageType
        image_input = ct.ImageType(
            name=input_name,
            shape=input_shape,
            scale=2.0/255.0,
            bias=[-1.0, -1.0, -1.0],
            color_layout=ct.colorlayout.RGB
        )
        coreml_model = ct.convert(
            keras_model,
            inputs=[image_input],
            classifier_config=ct.ClassifierConfig(class_labels),
            minimum_deployment_target=ct.target.iOS13
        )
        print("Conversion successful with ImageType!")
    except Exception as e2:
        print(f"Error with ImageType: {e2}")
        try:
            print("Attempting conversion without input specification...")
            # Last resort: let coremltools infer everything
            coreml_model = ct.convert(
                keras_model,
                classifier_config=ct.ClassifierConfig(class_labels),
                minimum_deployment_target=ct.target.iOS13
            )
            print("Conversion successful without input specification!")
        except Exception as e3:
            print(f"Error without input specification: {e3}")
            try:
                print("Attempting conversion without classifier config...")
                # Try without classifier config
                tensor_input = ct.TensorType(name=input_name, shape=(1,) + input_shape)
                coreml_model = ct.convert(
                    keras_model,
                    inputs=[tensor_input],
                    minimum_deployment_target=ct.target.iOS13
                )
                print("Conversion successful without classifier config!")
            except Exception as e4:
                raise RuntimeError(f"All conversion attempts failed. Last error: {e4}")

# Set metadata
coreml_model.author = 'Philipp Gabriel'
coreml_model.license = 'MIT'
coreml_model.short_description = 'This model takes a picture of a food and predicts its name'
# Use the actual input name for description
if input_name in coreml_model.input_description:
    coreml_model.input_description[input_name] = 'Image of a food'
# Try to set output description (may vary based on conversion)
if hasattr(coreml_model, 'output_description'):
    for output_name in coreml_model.output_description:
        if 'classLabel' in output_name.lower() or 'label' in output_name.lower():
            coreml_model.output_description[output_name] = 'Label of predicted food'

# Save the model
print("Saving CoreML model...")
coreml_model.save('Food101.mlmodel')
print("Done! Food101.mlmodel has been created.")
