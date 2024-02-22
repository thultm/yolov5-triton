import numpy as np
import json
from typing import Tuple
import triton_python_backend_utils as pb_utils
import cv2
from PIL import Image
import io

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get output configuration
        preprocessed_config = pb_utils.get_output_config_by_name(
            model_config, "preprocessed_image")
        
        resized_config = pb_utils.get_output_config_by_name(
            model_config, "resized_image")
        
        resized_config_dtype = pb_utils.get_output_config_by_name(
            model_config, "resized_config")
        
        
        # Convert Triton types to numpy types
        self.preprocessed_dtype = pb_utils.triton_string_to_numpy(
            preprocessed_config['data_type'])
        self.resized_dtype = pb_utils.triton_string_to_numpy(
            resized_config['data_type'])
        
        self.resized_config_dtype = pb_utils.triton_string_to_numpy(
            resized_config_dtype['data_type'])

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        # Define the data types for the output tensors
        preprocessed_dtype = self.preprocessed_dtype
        resized_dtype = self.resized_dtype
        resized_config = self.resized_config_dtype
        
        responses = []
        
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get the input tensor from the request
            input_tensor = pb_utils.get_input_tensor_by_name(request, "preprocess_input")
            images = input_tensor.as_numpy()
            images = Image.open(io.BytesIO(images.tobytes()))
            images = cv2.cvtColor(np.array(images), cv2.COLOR_RGB2BGR)
            # Define the expected image shape for the model
            expected_image_shape = self.model_config['output'][0]['dims']
            # Resize and pad the image
            padding_color: Tuple[int, int, int] = (144, 144, 144)            
            h_new, w_new = expected_image_shape[2], expected_image_shape[3]
            h_org, w_org = images.shape[:2]
            
            padd_left, padd_right, padd_top, padd_bottom = 0, 0, 0, 0
            ratio = 1.0 if h_org >= w_org else w_new/w_org
            #Padding left to right
            if h_org >= w_org:
                img_resize = cv2.resize(images, (int(w_org*h_new/h_org), h_new))
                h, w = img_resize.shape[:2]
                padd_left = (w_new-w)//2
                padd_right =  w_new - w - padd_left
                ratio = h_new/h_org

            #Padding top to bottom
            if h_org < w_org:
                img_resize = cv2.resize(images, (w_new, int(h_org*w_new/w_org)))
                h, w = img_resize.shape[:2]
                padd_top = (h_new-h)//2
                padd_bottom =  h_new - h - padd_top
            
            images = cv2.copyMakeBorder(img_resize, padd_top, padd_bottom, padd_left, padd_right, cv2.BORDER_CONSTANT,None,value=padding_color)
            
            resized_image = images.copy()
            resized_config = np.array([ratio, padd_left, padd_top], dtype=self.resized_config_dtype)
            resized_config = resized_config[np.newaxis, ...]
            
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB) #BGR to RGB
            img = images.transpose((2, 0, 1)) # HWC to CHW
            img = np.ascontiguousarray(img).astype(np.float32)
            img /=255.0
            img = img[np.newaxis, ...]
        
            # Create the output tensor
            preprocessed_tensor = pb_utils.Tensor("preprocessed_image", img.astype(preprocessed_dtype))            
            resized_tensor = pb_utils.Tensor("resized_image", resized_image.astype(resized_dtype))
            
            resized_config_tensor = pb_utils.Tensor("resized_config", resized_config)
                        
            inference_response = pb_utils.InferenceResponse(output_tensors=[preprocessed_tensor, resized_tensor, resized_config_tensor])
            
            responses.append(inference_response)
            
            return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass